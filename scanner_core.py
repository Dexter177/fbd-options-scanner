"""
scanner_core.py  —  FBD Options Analyser (return-value API)
============================================================
Refactored from fbd_stock_scanner.py to return DataFrames and dicts
instead of printing. Used by app.py Streamlit UI.

Key function: analyse_ticker()
  Returns a dict with calls_df, straddles_df, leaps_df, and recommendation.
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


# ---------------------------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------------------------

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
    "FBD_CONFIRMED": (True,  "Short-dated call - momentum is confirmed, strike while it's hot"),
    "FBD_ZONE":      (False, "Spread - flush zone not fully resolved, give it room"),
    "FBD_WATCH":     (False, "Spread - not triggered yet, wait for the flush then buy the recovery"),
    "SUPPORT_TEST":  (False, "Spread - support not reconfirmed, cap downside"),
    "LONGER_TERM":   (False, "Spread - wider DTE, directional thesis not time-sensitive"),
    "KNIFE_CATCH":   (False, "Spread - starter size only, oversold but not confirmed"),
}

SETUP_LABELS = {
    "FBD_CONFIRMED": "FBD Confirmed",
    "FBD_ZONE":      "FBD Zone",
    "FBD_WATCH":     "FBD Watch",
    "SUPPORT_TEST":  "Support Retest",
    "LONGER_TERM":   "Longer Term",
    "KNIFE_CATCH":   "Knife Catch",
}

SETUP_WARNINGS = {
    "KNIFE_CATCH": "Not a confirmed FBD. Consider 25-50% of max budget. High risk.",
    "FBD_WATCH":   "Enter only if the level is actually tapped. Don't chase.",
}

SETUP_DESCRIPTIONS = {
    "FBD_CONFIRMED": (
        "Stock has flushed below a key level AND has already reclaimed above it -- "
        "reversal confirmed. Highest-conviction FBD setup. Uses short-dated calls "
        "(7-35 DTE) to capture near-term momentum while it's hot."
    ),
    "FBD_ZONE": (
        "Stock is flushing through or sitting at a key level but hasn't fully "
        "recovered yet. The setup is still developing -- a spread structure reduces "
        "premium risk and gives the trade room to resolve."
    ),
    "FBD_WATCH": (
        "You have a level to watch for a potential flush. You're NOT in yet -- "
        "wait for price to actually tap the level before entering, then buy the "
        "recovery. Uses a spread as the vehicle once triggered."
    ),
    "SUPPORT_TEST": (
        "Stock is retesting a prior support level from above but hasn't broken "
        "through it yet. Directional conviction is lower -- a spread caps max loss "
        "in case support fails."
    ),
    "LONGER_TERM": (
        "The directional thesis isn't time-sensitive. Maybe earnings are pending, "
        "or you expect a multi-week move. Wider DTE range, often pointing toward "
        "a LEAP rather than a short swing call."
    ),
    "KNIFE_CATCH": (
        "Stock is deeply oversold but there is NO confirmed FBD setup. Speculative "
        "starter position only -- size down to 25-50% of your normal max risk. "
        "High risk of continued downside if the flush isn't finished."
    ),
}


# ---------------------------------------------------------------------------
#  BLACK-SCHOLES
# ---------------------------------------------------------------------------

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
    """Score a call option 0-10. Higher = better risk/reward profile."""
    if mid <= 0:
        return 0.0
    ds = max(0, 1 - abs(delta - 0.42) / 0.25) * 3   # sweet spot ~0.42 delta
    ts = max(0, 1 - abs(theta) / mid / 0.04) * 3     # low theta relative to premium
    vs = min(1, volume / 300) * 2                      # volume
    os = min(1, oi / 1_000) * 2                        # open interest
    return round(ds + ts + vs + os, 2)


# ---------------------------------------------------------------------------
#  CORE ANALYSIS
# ---------------------------------------------------------------------------

def analyse_ticker(
    ticker:        str,
    anchor:        Optional[float],
    setup_type:    str  = "FBD_WATCH",
    max_risk:      int  = MAX_RISK,
    swing_min_dte: int  = 7,
    swing_max_dte: int  = 35,
    leap_min_dte:  int  = 180,
    leap_max_dte:  int  = 365,
) -> dict:
    """
    Run full options analysis for one ticker.

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g. "AAPL").
    anchor : float or None
        The price level that was broken (flush low). None = use ATM.
    setup_type : str
        One of the keys in SETUP_TYPES.
    max_risk : int
        Maximum USD risk per trade (default $1,000).
    swing_min_dte / swing_max_dte : int
        DTE range for swing directional call search.
    leap_min_dte / leap_max_dte : int
        DTE range for long-dated LEAP call search (default 180-365).

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
      straddles_df    pd.DataFrame (empty if none found)
      leaps_df        pd.DataFrame (empty if none found)
      recommendation  dict | None
        .type         "CALL" | "LEAP"
        .primary      str    headline trade line
        .detail       str    outlay / B/E
        .greeks       str    delta / theta / IV
        .outlay       float  total USD cost
        .rationale    str    why this structure
        .warning      str | None
        .alt          str | None  alternative trade
    """
    prefer_call, rationale = SETUP_TYPES.get(setup_type, (False, ""))
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
        "fbd_status":    "-",
        "calls_df":      pd.DataFrame(),
        "straddles_df":  pd.DataFrame(),
        "leaps_df":      pd.DataFrame(),
        "recommendation": None,
        "_best_call":    None,
        "_best_leap":    None,
    }

    try:
        # -- price -----------------------------------------------------------
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
                out["fbd_status"] = f"Still below level (${anch:.2f}) -- not yet recovered"
            elif lo_5d > anch:
                out["fbd_status"] = f"5-day low (${lo_5d:.2f}) never touched level -- not a flush"
            else:
                out["fbd_status"] = (
                    f"Flushed below ${anch:.2f}, now {rec_pct:+.1f}% above -- "
                    f"{'confirmed FBD' if rec_pct > 2 else 'early recovery'}"
                )
        else:
            out["fbd_status"] = "ATM reference (no level set)"

        # -- options expiries ------------------------------------------------
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

        swing_exps = [(e, _dte(e)) for e in exps
                      if swing_min_dte <= _dte(e) <= swing_max_dte]
        leap_exps  = [(e, _dte(e)) for e in exps
                      if leap_min_dte <= _dte(e) <= leap_max_dte]

        # -- section 1: swing directional calls ------------------------------
        call_rows = []
        for exp_str, dte in swing_exps:
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

                if mid < 0.01 or iv <= 0:                              continue
                if vol < MIN_VOLUME or oi < MIN_OI:                    continue
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

        # -- section 2: swing straddles (buy call + put, same strike) --------
        strad_rows = []
        for exp_str, dte in swing_exps:
            T = dte / 365
            try:
                chain_pair = _yf_retry(lambda: tk.option_chain(exp_str))
                calls_c = chain_pair.calls
                puts_c  = chain_pair.puts
            except Exception:
                continue

            # Narrow to +/-10% of current price for straddle strikes
            c_near = (
                calls_c[
                    (calls_c.strike >= current * 0.90) &
                    (calls_c.strike <= current * 1.10)
                ]
                .drop_duplicates(subset="strike")
                .set_index("strike")
            )
            p_near = (
                puts_c[
                    (puts_c.strike >= current * 0.90) &
                    (puts_c.strike <= current * 1.10)
                ]
                .drop_duplicates(subset="strike")
                .set_index("strike")
            )

            for k in c_near.index.intersection(p_near.index):
                cr = c_near.loc[k]
                pr = p_near.loc[k]

                c_bid = float(cr.get("bid", 0) or 0)
                c_ask = float(cr.get("ask", 0) or 0)
                p_bid = float(pr.get("bid", 0) or 0)
                p_ask = float(pr.get("ask", 0) or 0)
                c_iv  = float(cr.get("impliedVolatility", 0) or 0)
                p_iv  = float(pr.get("impliedVolatility", 0) or 0)

                _cv = cr.get("volume", 0)
                _co = cr.get("openInterest", 0)
                _pv = pr.get("volume", 0)
                _po = pr.get("openInterest", 0)
                c_vol = int(_cv) if (_cv == _cv and _cv) else 0
                c_oi  = int(_co) if (_co == _co and _co) else 0
                p_vol = int(_pv) if (_pv == _pv and _pv) else 0
                p_oi  = int(_po) if (_po == _po and _po) else 0

                c_mid = (c_bid + c_ask) / 2 if c_bid > 0 and c_ask > 0 else 0
                p_mid = (p_bid + p_ask) / 2 if p_bid > 0 and p_ask > 0 else 0

                if c_mid < 0.01 or p_mid < 0.01 or c_iv <= 0 or p_iv <= 0:
                    continue
                if c_vol < MIN_VOLUME or c_oi < MIN_OI:
                    continue
                if p_vol < MIN_VOLUME or p_oi < MIN_OI:
                    continue
                c_ba = (c_ask - c_bid) / c_mid if c_mid > 0 else 1
                p_ba = (p_ask - p_bid) / p_mid if p_mid > 0 else 1
                if c_ba > MAX_BA_PCT or p_ba > MAX_BA_PCT:
                    continue

                total_ask = round(c_ask + p_ask, 2)
                be_up     = round(k + total_ask, 2)
                be_down   = round(k - total_ask, 2)
                move_pct  = round(total_ask / k * 100, 1)
                avg_iv    = round((c_iv + p_iv) / 2 * 100, 0)
                max_c     = max(1, int(max_risk / (total_ask * 100)))
                outlay    = round(min(max_c * total_ask * 100, max_risk), 0)

                strad_rows.append({
                    "Expiry":      f"{exp_str} ({dte}d)",
                    "Strike":      f"${k:.2f}",
                    "Call Ask":    f"${c_ask:.2f}",
                    "Put Ask":     f"${p_ask:.2f}",
                    "Total":       f"${total_ask:.2f}",
                    "B/E Up":      f"${be_up:.2f}",
                    "B/E Down":    f"${be_down:.2f}",
                    "Move Needed": f"{move_pct:.1f}%",
                    "Avg IV":      f"{avg_iv:.0f}%",
                    "Contracts":   max_c,
                    "Outlay":      f"${outlay:.0f}",
                    "_k": k, "_total": total_ask, "_move": move_pct,
                    "_dist": abs(k - current),
                })

        if strad_rows:
            df_s = pd.DataFrame(strad_rows).sort_values(["_move", "_dist"])
            out["straddles_df"] = (
                df_s.drop(columns=[c for c in df_s.columns if c.startswith("_")])
                    .head(8).reset_index(drop=True)
            )

        # -- section 3: LEAP directional calls --------------------------------
        leap_rows = []
        for exp_str, dte in leap_exps:
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

                if mid < 0.01 or iv <= 0:                              continue
                if vol < MIN_VOLUME or oi < MIN_OI:                    continue
                if bid > 0 and ask > 0 and (ask - bid)/mid > MAX_BA_PCT: continue

                g = bs_greeks(current, k, T, RISK_FREE, iv)
                if not g or not (MIN_DELTA <= g["delta"] <= MAX_DELTA): continue

                max_c  = max(1, int(max_risk / (ask * 100)))
                outlay = round(min(max_c * ask * 100, max_risk), 0)
                sc     = _score_call(g["delta"], g["theta"], mid, vol, oi)
                be     = round(k + ask, 2)

                leap_rows.append({
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
                    "_score": sc, "_ask": ask, "_exp": exp_str,
                    "_dte": dte, "_k": k, "_delta": g["delta"],
                    "_theta": g["theta"], "_iv": iv, "_be": be,
                    "_outlay": outlay,
                })

        if leap_rows:
            df_l = pd.DataFrame(leap_rows).sort_values("_score", ascending=False)
            out["_best_leap"] = df_l.iloc[0].to_dict()
            out["leaps_df"]   = (
                df_l.drop(columns=[c for c in df_l.columns if c.startswith("_")])
                    .head(8).reset_index(drop=True)
            )

        # -- recommendation --------------------------------------------------
        bc = out["_best_call"]
        bl = out["_best_leap"]
        warning = SETUP_WARNINGS.get(setup_type)

        if prefer_call and bc:
            max_c  = max(1, int(max_risk / (bc["_ask"] * 100)))
            outlay = round(min(max_c * bc["_ask"] * 100, max_risk), 0)
            out["recommendation"] = {
                "type":      "CALL",
                "primary":   f"BUY {max_c}x {bc['_exp']}  ${bc['_k']:.2f} Call",
                "detail":    (
                    f"Pay ~${bc['_ask']:.2f}/contract  "
                    f"Total outlay ${outlay:.0f}  "
                    f"Break-even ${bc['_be']:.2f}"
                ),
                "greeks":    (
                    f"Delta {bc['_delta']:.2f}  "
                    f"Theta -${abs(bc['_theta']):.3f}/day  "
                    f"IV {bc['_iv']*100:.0f}%"
                ),
                "outlay":    outlay,
                "rationale": rationale,
                "warning":   warning,
                "alt": (
                    f"Alt LEAP: "
                    f"{max(1, int(max_risk/(bl['_ask']*100)))}x "
                    f"{bl['_exp']}  ${bl['_k']:.2f}C @ ${bl['_ask']:.2f}  "
                    f"Delta {bl['_delta']:.2f}"
                ) if bl else None,
            }

        elif bl:
            max_l  = max(1, int(max_risk / (bl["_ask"] * 100)))
            outlay = round(min(max_l * bl["_ask"] * 100, max_risk), 0)
            out["recommendation"] = {
                "type":      "LEAP",
                "primary":   f"BUY {max_l}x {bl['_exp']}  ${bl['_k']:.2f} Call  [LEAP]",
                "detail":    (
                    f"Pay ~${bl['_ask']:.2f}/contract  "
                    f"Total outlay ${outlay:.0f}  "
                    f"Break-even ${bl['_be']:.2f}"
                ),
                "greeks":    (
                    f"Delta {bl['_delta']:.2f}  "
                    f"Theta -${abs(bl['_theta']):.3f}/day  "
                    f"IV {bl['_iv']*100:.0f}%"
                ),
                "outlay":    outlay,
                "rationale": rationale,
                "warning":   warning,
                "alt": (
                    f"Alt swing call: "
                    f"{max(1, int(max_risk/(bc['_ask']*100)))}x "
                    f"{bc['_exp']}  ${bc['_k']:.2f}C @ ${bc['_ask']:.2f}  "
                    f"Delta {bc['_delta']:.2f}"
                ) if bc else None,
            }

        elif bc:
            max_c  = max(1, int(max_risk / (bc["_ask"] * 100)))
            outlay = round(min(max_c * bc["_ask"] * 100, max_risk), 0)
            out["recommendation"] = {
                "type":      "CALL",
                "primary":   f"BUY {max_c}x {bc['_exp']}  ${bc['_k']:.2f} Call  [no LEAP found]",
                "detail":    (
                    f"Pay ~${bc['_ask']:.2f}  "
                    f"Total ${outlay:.0f}  "
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
