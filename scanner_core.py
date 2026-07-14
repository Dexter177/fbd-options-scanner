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


# CONFIG

MAX_RISK   = 1_000
RISK_FREE  = 0.045
SPREAD_W   = 2

MIN_VOLUME = 20
MIN_OI     = 75
MAX_BA_PCT = 0.30
MIN_DELTA  = 0.18
MAX_DELTA  = 0.70

SETUP_TYPES = {
    "FBD_CONFIRMED": (True,  "Short-dated call -- momentum is confirmed, strike while it's hot"),
    "FBD_ZONE":      (False, "Spread -- flush zone not fully resolved, give it room"),
    "FBD_WATCH":     (False, "Spread -- not triggered yet, wait for the flush then buy the recovery"),
    "SUPPORT_TEST":  (False, "Spread -- support not reconfirmed, cap downside"),
    "LONGER_TERM":   (False, "Spread -- wider DTE, directional thesis not time-sensitive"),
    "KNIFE_CATCH":   (False, "Spread -- starter size only, oversold but not confirmed"),
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
    "FBD_WATCH":   "Enter only if the anchor price is actually tapped. Don't chase.",
}


def bs_greeks(S, K, T, r, iv):
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
        return dict(delta=round(delta, 4), theta=round(theta, 5), vega=round(vega, 4))
    except (ValueError, ZeroDivisionError):
        return None


def _score_call(delta, theta, mid, volume, oi):
    if mid <= 0:
        return 0.0
    ds = max(0, 1 - abs(delta - 0.42) / 0.25) * 3
    ts = max(0, 1 - abs(theta) / mid / 0.04) * 3
    vs = min(1, volume / 300) * 2
    os = min(1, oi / 1_000) * 2
    return round(ds + ts + vs + os, 2)


def analyse_ticker(
    ticker,
    anchor,
    setup_type    = "FBD_WATCH",
    max_risk      = MAX_RISK,
    swing_min_dte = 7,
    swing_max_dte = 35,
    leap_min_dte  = 180,
    leap_max_dte  = 365,
):
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
        "leaps_df":      pd.DataFrame(),
        "recommendation": None,
        "_best_call":    None,
        "_best_leap":    None,
    }

    try:
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

        def _dte(e):
            return (datetime.strptime(e, "%Y-%m-%d").date() - today).days

        swing_exps = [(e, _dte(e)) for e in exps if swing_min_dte <= _dte(e) <= swing_max_dte]
        leap_exps  = [(e, _dte(e)) for e in exps if leap_min_dte  <= _dte(e) <= leap_max_dte]

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

                if mid < 0.01 or iv <= 0: continue
                if vol < MIN_VOLUME or oi < MIN_OI: continue
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

                if mid < 0.01 or iv <= 0: continue
                if vol < MIN_VOLUME or oi < MIN_OI: continue
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

        bc = out["_best_call"]
        bl = out["_best_leap"]
        warning = SETUP_WARNINGS.get(setup_type)

        if prefer_call and bc:
            max_c  = max(1, int(max_risk / (bc["_ask"] * 100)))
            outlay = round(min(max_c * bc["_ask"] * 100, max_risk), 0)
            out["recommendation"] = {
                "type":      "CALL",
                "primary":   f"BUY {max_c}x {bc['_exp']}  ${bc['_k']:.2f} Call",
                "detail":    f"Pay ~${bc['_ask']:.2f}/contract  Total outlay ${outlay:.0f}  Break-even ${bc['_be']:.2f}",
                "greeks":    f"Delta {bc['_delta']:.2f}  Theta -${abs(bc['_theta']):.3f}/day  IV {bc['_iv']*100:.0f}%",
                "outlay":    outlay,
                "rationale": rationale,
                "warning":   warning,
                "alt": (f"Alt LEAP: {max(1, int(max_risk/(bl['_ask']*100)))}x {bl['_exp']}  ${bl['_k']:.2f}C @ ${bl['_ask']:.2f}  Delta {bl['_delta']:.2f}") if bl else None,
            }
        elif bl:
            max_l  = max(1, int(max_risk / (bl["_ask"] * 100)))
            outlay = round(min(max_l * bl["_ask"] * 100, max_risk), 0)
            out["recommendation"] = {
                "type":      "LEAP",
                "primary":   f"BUY {max_l}x {bl['_exp']}  ${bl['_k']:.2f} Call  [LEAP]",
                "detail":    f"Pay ~${bl['_ask']:.2f}/contract  Total outlay ${outlay:.0f}  Break-even ${bl['_be']:.2f}",
                "greeks":    f"Delta {bl['_delta']:.2f}  Theta -${abs(bl['_theta']):.3f}/day  IV {bl['_iv']*100:.0f}%",
                "outlay":    outlay,
                "rationale": rationale,
                "warning":   warning,
                "alt": (f"Alt swing call: {max(1, int(max_risk/(bc['_ask']*100)))}x {bc['_exp']}  ${bc['_k']:.2f}C @ ${bc['_ask']:.2f}  Delta {bc['_delta']:.2f}") if bc else None,
            }
        elif bc:
            max_c  = max(1, int(max_risk / (bc["_ask"] * 100)))
            outlay = round(min(max_c * bc["_ask"] * 100, max_risk), 0)
            out["recommendation"] = {
                "type":      "CALL",
                "primary":   f"BUY {max_c}x {bc['_exp']}  ${bc['_k']:.2f} Call  [no LEAP found]",
                "detail":    f"Pay ~${bc['_ask']:.2f}  Total ${outlay:.0f}  B/E ${bc['_be']:.2f}",
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
