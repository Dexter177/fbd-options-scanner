"""
screener.py  --  FBD Flush Screener (S&P 500)
=============================================
Batch-downloads 6 months of daily price data via yfinance, then applies
the FBD flush criteria to surface stocks making new 26-week lows with
elevated volume and oversold RSI.

Used by app.py -- not intended to be run standalone.

Criteria applied:
  - Price > min_price (default $10)
  - Price >= min_drop_pct % below 6-month high (default 15%)
  - RSI(14) < max_rsi (default 40)
  - Today's volume > min_rel_vol x 20-day average (default 1.5x)
  - New 26-week low within the last flush_days bars (default 3)

All S&P 500 companies have market cap > ~$14B, so no separate
market cap filter is needed when scanning this universe.
"""

import io
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import date

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Run: pip install yfinance")


# ---------------------------------------------------------------------------
#  UNIVERSE
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> tuple:
    """
    Fetch current S&P 500 constituents from Wikipedia.
    Returns (tickers_list, meta_df) where meta has columns:
      ticker, company, sector
    """
    import requests as _requests
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
    }
    resp = _requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})[0]
    # yfinance requires BRK-B not BRK.B
    table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
    tickers = table["Symbol"].tolist()
    meta = table[["Symbol", "Security", "GICS Sector"]].rename(
        columns={"Symbol": "ticker", "Security": "company", "GICS Sector": "sector"}
    )
    return tickers, meta


# ---------------------------------------------------------------------------
#  INDICATORS
# ---------------------------------------------------------------------------

def _calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI via EWM (equivalent to Wilder's smoothing)."""
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, 1e-9)
    return (100 - 100 / (1 + rs)).round(1)


# ---------------------------------------------------------------------------
#  MAIN SCREENER
# ---------------------------------------------------------------------------

def run_screener(
    min_drop_pct: float = 15.0,
    max_rsi:      float = 40.0,
    min_rel_vol:  float = 1.5,
    min_price:    float = 10.0,
    flush_days:   int   = 3,
    progress_cb          = None,
) -> pd.DataFrame:
    """
    Scan S&P 500 for FBD flush candidates.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, company, sector, price, drop_pct, rsi,
                 rel_vol, prior_low, recent_low
        Sorted by drop_pct ascending (deepest flush first).
        Empty DataFrame if no candidates found.
    """
    tickers, meta = get_sp500_tickers()
    n = len(tickers)

    if progress_cb:
        progress_cb(0.02)

    raw = yf.download(
        tickers,
        period="6mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if progress_cb:
        progress_cb(0.30)

    results = []

    for i, ticker in enumerate(tickers):

        if progress_cb and i % 25 == 0:
            progress_cb(0.30 + 0.65 * (i / n))

        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[ticker].dropna(how="all")
            else:
                df = raw.dropna(how="all")

            if len(df) < 30:
                continue

            close  = df["Close"].squeeze().dropna()
            volume = df["Volume"].squeeze().dropna()

            if len(close) < 30:
                continue

            current = float(close.iloc[-1])

            if current < min_price:
                continue

            high_6m  = float(close.max())
            drop_pct = (current - high_6m) / high_6m * 100
            if drop_pct > -min_drop_pct:
                continue

            rsi = float(_calc_rsi(close).iloc[-1])
            if np.isnan(rsi) or rsi > max_rsi:
                continue

            vol_today  = float(volume.iloc[-1])
            avg_vol_20 = float(volume.iloc[-21:-1].mean())
            rel_vol    = vol_today / avg_vol_20 if avg_vol_20 > 0 else 0.0
            if rel_vol < min_rel_vol:
                continue

            history_before = close.iloc[:-flush_days]
            if len(history_before) < 10:
                continue
            prior_low  = float(history_before.min())
            recent_low = float(close.iloc[-flush_days:].min())

            if recent_low >= prior_low:
                continue

            results.append({
                "ticker":     ticker,
                "price":      round(current, 2),
                "drop_pct":   round(drop_pct, 1),
                "rsi":        round(rsi, 1),
                "rel_vol":    round(rel_vol, 2),
                "prior_low":  round(prior_low, 2),
                "recent_low": round(recent_low, 2),
            })

        except Exception:
            continue

    if progress_cb:
        progress_cb(0.97)

    if not results:
        return pd.DataFrame()

    df_out = pd.DataFrame(results)
    df_out = df_out.merge(meta, on="ticker", how="left")
    df_out = df_out.sort_values("drop_pct").reset_index(drop=True)

    ordered = ["ticker", "company", "sector", "price", "drop_pct",
               "rsi", "rel_vol", "prior_low", "recent_low"]
    df_out = df_out[[c for c in ordered if c in df_out.columns]]

    if progress_cb:
        progress_cb(1.0)

    return df_out
