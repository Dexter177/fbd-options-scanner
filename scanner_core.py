"""
scanner_core.py  —  FBD Options Analyser (return-value API)
============================================================
Refactored from fbd_stock_scanner.py to return DataFrames and dicts
instead of printing. Used by app.py Streamlit UI.

Key function: analyse_ticker()
  Returns a dict with calls_df, spreads_df, and recommendation.
"""

import math
import time
import warnings
from datetime import date, datetime
from typing import Optional

warnings.filterwarnings("ignore")


def _yf_retry(fn, retries: int = 2, wait: int = 8):
    """
    Call fn(), retrying once on Yahoo rate-limit (429) errors.
    Waits `wait` seconds between attempts so the throttle window resets.
    """
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            is_rate = "429" in msg or "too many" in msg or "rate limit" in msg
            if is_rate and attempt < retries - 1:
                time.sleep(wait)
            else:
                raise

try:
    import yfinance as yf
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
except ImportError as exc:
    raise ImportError(
        f"Missing dependency: {exc}. "
        "Run: pip install yfinance numpy pandas scipy"
    ) from exc


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MAX_RISK   = 1_000   # USD, default hard cap per trade
RISK_FREE  = 0.045   # ~4.5% risk-free rate
SPREAD_W   = 2       # sell leg: N strikes above buy leg (when no level map)

MIN_VOLUME = 20      # min option volume
MIN_OI     = 75      # min open interest
MAX_BA_PCT = 0.30    # max bid-ask as % of mid
MIN_DELTA  = 0.18
MAX_DELTA  = 0.70

# (prefer_call, rationale string)
SETUP_TYPES = {
    "FBD_CONFIRMED": (True,  "Short-dated call — momentum is confirmed, strike while it's hot"),
    "FBD_ZONE":      (False, "Spread — flush zone not fully resolved, give it room"),
    "FBD_WATCH":     (False, "Spread — not triggered yet, wait for the flush then buy the recovery"),
    "SUPPORT_TEST":  (False, "Spread — support not reconfirmed, cap downside"),
    "LONGER_TERM":   (False, "Spread — wider DTE, directional thesis not time-sensitive"),
    "KNIFE_CATCH":   (False, "Spread — starter size only, oversold but not confirmed"),
}

SETUP_LABELS = {
    "FBD_CONFIRMED": "✅ FBD Confirmed",
    "FBD_ZONE":      "🟡 FBD Zone",
    "FBD_WATCH":     "👁 FBD Watch",
    "SUPPORT_TEST":  "🔁 Support Retest",
    "LONGER_TERM":   "📈 Longer Term",
    "KNIFE_CATCH":   "⚠️ Knife Catch",
}

SETUP_WARNINGS = {
    "KNIFE_CATCH": "⚠️ Not a confirmed FBD. Consider 25–50% of max budget. High risk.",
    "FBD_WATCH":   "👁 Enter only if the anchor price is actually tapped. Don't chase.",
}


# ─────────────────────────────────────────────────────────────────────────────
#  BLACK-SCHOLES
# ─────────────────────────────────────────────────────────────────────────────

def bs_greeks(S: float, K: float, T: float, r: float, iv: float) -> Optional[dict]:
    """Call greeks via Black-Scholes. Returns dict or None on failure."""
    if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
        return None
    try:
        d1 = (math.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * math.sqrt(T))
        d2 = d1 - iv * math.sqrt(T)
        delta = norm.cdf(d1)
        theta = (
            -(S * norm.pdf(d1) * iv) / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        ) / 365
        vega  = S * norm.pdf(d1) * math.sqrt(T) / 100
        return dict(
            delta=round(delta, 4),
            theta=round(theta, 5),
            vega=round(vega, 4),
        )
    except (ValueError, ZeroDivisionError):
        return None


def _score_call(delta: float, theta: float, mid: float,
                volume: int, oi: int) -> float:
    """Score a call option 0–10. Higher = better risk/reward profile."""
    if mid <= 0:
        return 0.0
    ds = max(0, 1 - abs(delta - 0.42) / 0.25) * 3   # sweet spot ~0.42 delta
    ts = max(0, 1 - abs(theta) / mid / 0.04) * 3     # low theta relative to premium
    vs = min(1, volume / 300) * 2                      # volume
    os = min(1, oi / 1_000) * 2                        # open interest
    return round(ds + ts + vs + os, 2)


# ─────────────────────────────────────────────────────────────────────────────
#  CORE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_ticker(
    ticker:            str,
    anchor:            Optional[float],
    setup_type:        str  = "FBD_WATCH",
    max_risk:          int  = MAX_RISK,
    short_min_dte:     int  = 7,
    short_max_dte:     int  = 35,
    spread_dte_target: int  = 60,
) -> dict:
    """
    Run full options analysis for one ticker.

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g. "ORCL").
    anchor : float or None
        The price level that was broken (flush low). None = use ATM.
    setup_type : str
        One of the keys in SETUP_TYPES.
    max_risk : int
        Maximum USD risk per trade (default $1,000).
    short_min_dte / short_max_dte : int
        DTE range for directional call search.
    spread_dte_target : int
        Target DTE for debit spread search (±15 days window).

    Returns
    -------
    dict with keys:
      status          "ok" | "error"
      error           str | None
      ticker          str
      current_price   float | None
      anchor          float | None
      lo_5d           float | None
      recovery_pct    float | None
      fbd_status      str
      calls_df        pd.DataFrame (empty if none found)
      spreads_df      pd.DataFrame (empty if none found)
      recommendation  dict | None
        .type         "CALL" | "SPREAD"
        .primary      str    headline trade line
        .detail       str    outlay / B/E / max profit
        .greeks       str    delta / theta / IV or R:R
        .outlay       float  total USD cost
        .rationale    str    why this structure
        .warning      str | None
        .alt          str | None  alternative trade
    """
    prefer_call, rationale = SETUP_TYPES.get(setup_type, (False, ""))
    spread_window = 15
    today = date.today()

    out = {
        "status":        "ok",
        "error":         None,
        "ticker":        ticker,
        "setup_type":    setup_type,
        "setup_label":   SETUP_LABELS.get(setup_type, setup_type),
        "prefer_call":   prefer_call,
        "rationale":     rationale,
        "current_price": None,
        "anchor":        anchor,
        "lo_5d":         None,
        "recovery_pct":  None,
        "fbd_status":    "—",
        "calls_df":      pd.DataFrame(),
        "spreads_df":    pd.DataFrame(),
        "recommendation": None,
        "_best_call":    None,
        "_best_spread":  None,
    }

    try:
        # ── price ─────────────────────────────────────────────────────────
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="10d", interval="1d")
        if hist.empty:
            out["status"] = "error"
            out["error"]  = f"No price data for {ticker}"
            return out

        try:
            current = round(float(tk.fast_info.last_price), 4)
        except Exception:
            current = round(float(hist["Close"].iloc[-1]), 4)

        lo_5d = round(float(hist["Low"].iloc[-5:].min()), 4)
        anch  = anchor if (anchor is not None and anchor > 0) else current

        out["current_price"] = current
        out["lo_5d"]         = lo_5d
        out["anchor"]        = anch

        if anchor and anchor > 0:
            rec_pct = round((current - anch) / anch * 100, 1)
            out["recovery_pct"] = rec_pct
            if current < anch:
                out["fbd_status"] = f"⚠️ Still below anchor (${anch:.2f}) — not yet recovered"
            elif lo_5d > anch:
                out["fbd_status"] = f"⚠️ 5-day low (${lo_5d:.2f}) never touched anchor — not a flush"
            else:
                out["fbd_status"] = (
                    f"✅ Flushed below ${anch:.2f}, now {rec_pct:+.1f}% above — "
                    f"{'confirmed FBD' if rec_pct > 2 else 'early recovery'}"
                )
        else:
            out["fbd_status"] = "ATM reference (no anchor set)"

        # ── options expiries ──────────────────────────────────────────────
        try:
            exps = _yf_retry(lambda: tk.options)
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "too many" in msg or "rate" in msg:
                out["status"] = "error"
                out["error"]  = "rate_limited"
            else:
                out["status"] = "error"
                out["error"]  = f"Options unavailable: {e}"
            return out

        if not exps:
            out["status"] = "error"
            out["error"]  = "No options listed for this ticker"
            return out

        def _dte(e: str) -> int:
            return (datetime.strptime(e, "%Y-%m-%d").date() - today).days

        short_exps  = [(e, _dte(e)) for e in exps
                       if short_min_dte <= _dte(e) <= short_max_dte]
        spread_exps = [(e, _dte(e)) for e in exps
                       if abs(_dte(e) - spread_dte_target) <= spread_window]

        # ── section 1: directional calls ──────────────────────────────────
        call_rows = []
        for exp_str, dte in short_exps:
            T = dte / 365
            try:
                chain = _yf_retry(lambda: tk.option_chain(exp_str).calls)
            except Exception:
                continue

            chain = chain[
                (chain.strike >= anch * 0.85) &
                (chain.strike <= current * 1.30)
            ]

            for _, r in chain.iterrows():
                k   = float(r.strike)
                bid = float(r.get("bid", 0) or 0)
                ask = float(r.get("ask", 0) or 0)
                iv  = float(r.get("impliedVolatility", 0) or 0)
                _v  = r.get("volume", 0)
                _o  = r.get("openInterest", 0)
                vol = int(_v) if (_v == _v and _v) else 0
                oi  = int(_o) if (_o == _o and _o) else 0
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0

                if mid < 0.01 or iv <= 0:                         continue
                if vol < MIN_VOLUME or oi < MIN_OI:                continue
                if bid > 0 and ask > 0 and (ask - bid)/mid > MAX_BA_PCT: continue

                g = bs_greeks(current, k, T, RISK_FREE, iv)
                if not g or not (MIN_DELTA <= g["delta"] <= MAX_DELTA): continue

                max_c  = max(1, int(max_risk / (ask * 100)))
                outlay = round(min(max_c * ask * 100, max_risk), 0)
                sc     = _score_call(g["delta"], g["theta"], mid, vol, oi)
                be     = round(k + ask, 2)

                call_rows.append({
                    "Expiry":    f"{exp_str} ({dte}d)",
                    "Strike":    f"${k:.2f}",
                    "Ask":       f"${ask:.2f}",
                    "B/E":       f"${be:.2f}",
                    "Delta":     f"{g['delta']:.2f}",
                    "Theta":     f"-${abs(g['theta']):.3f}/d",
                    "IV":        f"{iv*100:.0f}%",
                    "Volume":    vol,
                    "OI":        oi,
                    "Contracts": max_c,
                    "Outlay":    f"${outlay:.0f}",
                    "Score":     sc,
                    # private — used for recommendation, dropped from display
                    "_score": sc, "_ask": ask, "_exp": exp_str,
                    "_dte": dte, "_k": k, "_delta": g["delta"],
                    "_theta": g["theta"], "_iv": iv, "_be": be,
                    "_outlay": outlay,
                })

        if call_rows:
            df_c = pd.DataFrame(call_rows).sort_values("_score", ascending=False)
            out["_best_call"] = df_c.iloc[0].to_dict()
            out["calls_df"]   = (
                df_c.drop(columns=[c for c in df_c.columns if c.startswith("_")])
                    .head(8).reset_index(drop=True)
            )

        # ── section 2: debit spreads ──────────────────────────────────────
        spread_rows = []
        for exp_str, dte in spread_exps:
            T = dte / 365
            try:
                chain = _yf_retry(lambda: tk.option_chain(exp_str).calls)
            except Exception:
                continue

            # build strike map
            smap: dict = {}
            for _, r in chain.iterrows():
                k   = float(r.strike)
                bid = float(r.get("bid", 0) or 0)
                ask = float(r.get("ask", 0) or 0)
                iv  = float(r.get("impliedVolatility", 0) or 0)
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                if iv > 0 and mid > 0.01:
                    smap[k] = {"bid": bid, "ask": ask, "mid": mid, "iv": iv}

            ks = sorted(smap)
            for i, bk in enumerate(ks):
                if bk < anch * 0.93 or bk > current * 1.15:
                    continue
                si = i + SPREAD_W
                sk = ks[si] if si < len(ks) else None
                if sk is None or sk == bk:
                    continue

                bd = smap[bk]
                sd = smap[sk]
                nd  = round(bd["ask"] - sd["bid"], 4)   # net debit
                if nd <= 0:
                    continue

                sw  = sk - bk                            # spread width
                mp  = round(sw - nd, 4)                  # max profit/contract
                be  = round(bk + nd, 2)                  # break-even
                rr  = round(mp / nd, 2) if nd > 0 else 0
                if rr < 0.4:
                    continue

                gb = bs_greeks(current, bk, T, RISK_FREE, bd["iv"])
                gs = bs_greeks(current, sk, T, RISK_FREE, sd["iv"])
                if not gb or not gs:
                    continue

                nd_delta = round(gb["delta"] - gs["delta"], 3)
                nd_theta = round(gb["theta"] - gs["theta"], 5)
                max_c    = max(1, int(max_risk / (nd * 100)))
                tot_mp   = round(max_c * mp * 100, 0)
                outlay   = round(max_c * nd * 100, 0)

                spread_rows.append({
                    "Expiry":     f"{exp_str} ({dte}d)",
                    "Spread":     f"${bk:.0f}/${sk:.0f}C",
                    "Debit":      f"${nd:.2f}",
                    "Max P/L":    f"${mp:.2f}",
                    "B/E":        f"${be:.2f}",
                    "R:R":        f"{rr:.1f}x",
                    "Net Δ":      f"{nd_delta:.3f}",
                    "Net θ":      f"-${abs(nd_theta):.4f}/d",
                    "Contracts":  max_c,
                    "Outlay":     f"${outlay:.0f}",
                    "Max Profit": f"${tot_mp:.0f}",
                    # private
                    "_rr": rr, "_nd": nd, "_mp": mp, "_be": be,
                    "_exp": exp_str, "_dte": dte,
                    "_bk": bk, "_sk": sk, "_nd_delta": nd_delta,
                    "_outlay": outlay, "_tot_mp": tot_mp,
                })

        if spread_rows:
            df_s = pd.DataFrame(spread_rows).sort_values("_rr", ascending=False)
            out["_best_spread"] = df_s.iloc[0].to_dict()
            out["spreads_df"]   = (
                df_s.drop(columns=[c for c in df_s.columns if c.startswith("_")])
                    .head(8).reset_index(drop=True)
            )

        # ── recommendation ────────────────────────────────────────────────
        bc = out["_best_call"]
        bs = out["_best_spread"]
        warning = SETUP_WARNINGS.get(setup_type)

        if prefer_call and bc:
            max_c  = max(1, int(max_risk / (bc["_ask"] * 100)))
            outlay = round(min(max_c * bc["_ask"] * 100, max_risk), 0)
            out["recommendation"] = {
                "type":      "CALL",
                "primary":   f"BUY {max_c}× {bc['_exp']}  ${bc['_k']:.2f} Call",
                "detail":    (
                    f"Pay ~${bc['_ask']:.2f}/contract  ·  "
                    f"Total outlay ${outlay:.0f}  ·  "
                    f"Break-even ${bc['_be']:.2f}"
                ),
                "greeks":    (
                    f"Delta {bc['_delta']:.2f}  ·  "
                    f"Theta −${abs(bc['_theta']):.3f}/day  ·  "
                    f"IV {bc['_iv']*100:.0f}%"
                ),
                "outlay":    outlay,
                "rationale": rationale,
                "warning":   warning,
                "alt": (
                    f"Alt spread: "
                    f"{max(1, int(max_risk/(bs['_nd']*100)))}× "
                    f"{bs['_exp']}  ${bs['_bk']:.0f}/${bs['_sk']:.0f}C  ·  "
                    f"Debit ${bs['_nd']:.2f}  ·  R:R {bs['_rr']}x"
                ) if bs else None,
            }

        elif bs:
            max_s  = max(1, int(max_risk / (bs["_nd"] * 100)))
            outlay = round(max_s * bs["_nd"] * 100, 0)
            tot    = round(max_s * bs["_mp"] * 100, 0)
            out["recommendation"] = {
                "type":      "SPREAD",
                "primary":   (
                    f"BUY {max_s}× {bs['_exp']}  "
                    f"${bs['_bk']:.0f}/${bs['_sk']:.0f}C spread"
                ),
                "detail":    (
                    f"Net debit ${bs['_nd']:.2f}  ·  "
                    f"Total outlay ${outlay:.0f}  ·  "
                    f"Max profit ${tot:.0f}"
                ),
                "greeks":    (
                    f"Break-even ${bs['_be']:.2f}  ·  "
                    f"R:R {bs['_rr']}x  ·  "
                    f"Net Δ {bs['_nd_delta']:.3f}"
                ),
                "outlay":    outlay,
                "rationale": rationale,
                "warning":   warning,
                "alt": (
                    f"Alt call: "
                    f"{max(1, int(max_risk/(bc['_ask']*100)))}× "
                    f"{bc['_exp']}  ${bc['_k']:.2f}C @ ${bc['_ask']:.2f}  ·  "
                    f"Delta {bc['_delta']:.2f}"
                ) if bc else None,
            }

        elif bc:
            max_c  = max(1, int(max_risk / (bc["_ask"] * 100)))
            outlay = round(min(max_c * bc["_ask"] * 100, max_risk), 0)
            out["recommendation"] = {
                "type":      "CALL",
                "primary":   f"BUY {max_c}× {bc['_exp']}  ${bc['_k']:.2f} Call  [no spread found]",
                "detail":    (
                    f"Pay ~${bc['_ask']:.2f}  ·  "
                    f"Total ${outlay:.0f}  ·  "
                    f"B/E ${bc['_be']:.2f}"
                ),
                "greeks":    f"Delta {bc['_delta']:.2f}",
                "outlay":    outlay,
                "rationale": rationale,
                "warning":   warning,
                "alt":       None,
            }

    except Exception as e:
        out["status"] = "error"
        out["error"]  = str(e)

    return out
