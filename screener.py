"""
screener.py  —  FBD Breakdown Screener  (US Large-/Mid-Cap)
============================================================
Scans US-listed stocks (market cap > $1 B, avg daily volume > 500 K) for
three FBD-relevant breakdown signals:

  1. New 52-week low  — within the last flush_days bars
  2. New 26-week low  — within the last flush_days bars
  3. Key-level break  — 2+ touches within 1% of each other, each bounce
                        >= 3%, price then breaks below that zone within
                        the last flush_days bars

Priority for reporting: 52w Low > 26w Low > Key Level

Additional filters (applied in per-ticker loop):
  - Price > $10
  - 20-day avg volume > 500 K  (confirmed from yfinance data)
  - ATR(14)% > 2%              (sufficient daily range for options)
  - Rel-vol on the breakdown bar > 1.5 x prior 20-day avg
  - Current price within 3% below the broken level  (not extended)
  - Biotech / clinical-stage sector excluded
  - IPOs listed < 6 months ago excluded

Post-screening (applied only to the small candidate list):
  - Listed options chain must exist
  - Combined open interest (nearest 2 expirations) > min_options_oi
  - Earnings within 5 calendar days -> flag  (not excluded)

Universe sources (with automatic fallback):
  Primary : NASDAQ public screener API  (no auth needed)
  Fallback : S&P 1500 (500 + 400 + 600) from Wikipedia
"""

import io
import warnings
import numpy as np
import pandas as pd
import requests as _requests
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Run: pip install yfinance")


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_MIN_AVG_VOLUME  = 500_000
_MIN_MARKET_CAP  = 1_000_000_000

_BIOTECH_TERMS = (
    "biotechnology", "biopharmaceutical", "biopharmaceuticals",
    "clinical stage", "clinical-stage", "drug manufacturers",
    "genomics", "gene therapy",
)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


# ─────────────────────────────────────────────────────────────────────────────
#  UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────

def _is_biotech(sector: str, industry: str) -> bool:
    text = f"{sector} {industry}".lower()
    return any(t in text for t in _BIOTECH_TERMS)


def _parse_market_cap(val) -> float:
    try:
        return float(str(val).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _parse_volume(val) -> float:
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _too_recent_ipo(yr_val) -> bool:
    """Return True if the IPO year is less than ~6 months old."""
    try:
        yr = int(float(str(yr_val)))
    except (ValueError, TypeError):
        return False
    now = datetime.now()
    # Any IPO in the current calendar year could still be < 6 months old
    return yr >= now.year


def get_us_tickers() -> pd.DataFrame:
    """
    Return a DataFrame of US-listed stocks pre-filtered by market cap
    (> $1 B) and average volume (> 500 K).
    Columns: ticker, company, sector, industry.

    Primary source: NASDAQ public screener API.
    Fallback: S&P 1500 via Wikipedia.
    """
    try:
        return _get_nasdaq_universe()
    except Exception as exc:
        print(f"[screener] NASDAQ API unavailable ({exc}); falling back to S&P 1500")
        return _get_sp1500_universe()


def _get_nasdaq_universe() -> pd.DataFrame:
    resp = _requests.get(
        "https://api.nasdaq.com/api/screener/stocks",
        params={"tableonly": "true", "limit": "25000", "download": "true"},
        headers={
            "User-Agent": _UA,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/",
        },
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()["data"]["table"]["rows"]
    df = pd.DataFrame(rows)

    # US only
    if "country" in df.columns:
        df = df[df["country"].str.strip().str.lower() == "united states"]

    # Market cap filter
    df["_mktcap"] = df.get("marketCap", pd.Series(dtype=str)).apply(_parse_market_cap)
    df = df[df["_mktcap"] >= _MIN_MARKET_CAP]

    # Loose volume pre-filter (50% of threshold; 20-day avg confirmed later)
    df["_vol"] = df.get("volume", pd.Series(dtype=str)).apply(_parse_volume)
    df = df[df["_vol"] >= _MIN_AVG_VOLUME * 0.5]

    # Exclude ETFs / funds (no sector)
    df = df[df.get("sector", pd.Series(dtype=str)).astype(str).str.strip() != ""]

    # Exclude biotech / clinical-stage
    df = df[~df.apply(
        lambda r: _is_biotech(str(r.get("sector", "")), str(r.get("industry", ""))),
        axis=1,
    )]

    # Exclude recent IPOs (< ~6 months)
    df = df[~df.get("ipoyear", pd.Series(dtype=str)).apply(_too_recent_ipo)]

    # Normalise symbols  (BRK.B -> BRK-B for yfinance)
    df["symbol"] = (
        df.get("symbol", pd.Series(dtype=str))
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
    )
    # Drop malformed rows
    df = df[df["symbol"].str.match(r"^[A-Z0-9\-]+$", na=False)]

    result = (
        df[["symbol", "name", "sector", "industry"]]
        .rename(columns={"symbol": "ticker", "name": "company"})
        .drop_duplicates("ticker")
        .reset_index(drop=True)
    )
    if len(result) < 50:
        raise ValueError(f"NASDAQ API returned only {len(result)} rows after filtering")
    return result


def _get_sp1500_universe() -> pd.DataFrame:
    """Fallback: S&P 500 + S&P 400 + S&P 600 from Wikipedia."""
    sources = [
        (
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            ["Symbol"],
            ["Security"],
            ["GICS Sector"],
        ),
        (
            "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
            ["Ticker symbol", "Symbol", "Ticker"],
            ["Company", "Security", "Name"],
            ["GICS Sector", "Sector"],
        ),
        (
            "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
            ["Ticker symbol", "Symbol", "Ticker"],
            ["Company", "Security", "Name"],
            ["GICS Sector", "Sector"],
        ),
    ]
    dfs = []
    for url, sym_opts, name_opts, sec_opts in sources:
        try:
            resp   = _requests.get(url, headers={"User-Agent": _UA}, timeout=25)
            tables = pd.read_html(io.StringIO(resp.text))
            for t in tables:
                sym_col  = next((c for c in sym_opts  if c in t.columns), None)
                name_col = next((c for c in name_opts if c in t.columns), None)
                sec_col  = next((c for c in sec_opts  if c in t.columns), None)
                if sym_col is None or len(t) < 10:
                    continue
                sub = pd.DataFrame({
                    "ticker":   t[sym_col].astype(str).str.replace(".", "-", regex=False),
                    "company":  t[name_col].astype(str) if name_col else "",
                    "sector":   t[sec_col].astype(str)  if sec_col  else "",
                    "industry": "",
                })
                dfs.append(sub)
                break  # take first matching table per URL
        except Exception as exc:
            print(f"[screener] Wikipedia {url}: {exc}")

    if not dfs:
        raise RuntimeError("Could not fetch any universe data")

    out = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates("ticker")
        .reset_index(drop=True)
    )
    # Still apply biotech exclusion
    out = out[~out.apply(
        lambda r: _is_biotech(str(r["sector"]), str(r["industry"])), axis=1
    )]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def _calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI via EWM (equivalent to Wilder smoothing)."""
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, 1e-9)
    return (100 - 100 / (1 + rs)).round(1)


def _calc_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ─────────────────────────────────────────────────────────────────────────────
#  KEY LEVEL DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _find_key_levels(
    close:       pd.Series,
    low:         pd.Series,
    touch_pct:   float = 0.01,
    bounce_pct:  float = 0.03,
    min_touches: int   = 2,
    wing:        int   = 5,
) -> list:
    """
    Find key support levels in a historical price series.

    A level qualifies when:
      - At least min_touches local lows lie within touch_pct of each other
      - Each of those lows was followed within 30 bars by a >= bounce_pct
        recovery in the closing price

    Returns list of (level_price, touch_count) tuples.
    """
    lows   = low.values
    closes = close.values
    n      = len(lows)
    if n < 2 * wing + 5:
        return []

    # Collect local minima with a confirmed subsequent bounce
    anchors = []
    for i in range(wing, n - wing):
        window = lows[max(0, i - wing): i + wing + 1]
        if lows[i] != min(window) or lows[i] <= 0:
            continue
        future = closes[i: min(n, i + 30)]
        if len(future) < 2:
            continue
        if (future.max() - lows[i]) / lows[i] >= bounce_pct:
            anchors.append(float(lows[i]))

    if len(anchors) < min_touches:
        return []

    # Greedy cluster: sort ascending, group within touch_pct
    anchors_s = sorted(anchors)
    used      = [False] * len(anchors_s)
    levels    = []

    for i, p_i in enumerate(anchors_s):
        if used[i]:
            continue
        cluster = [p_i]
        used[i] = True
        for j in range(i + 1, len(anchors_s)):
            if used[j]:
                continue
            if abs(anchors_s[j] - p_i) / p_i <= touch_pct:
                cluster.append(anchors_s[j])
                used[j] = True
        if len(cluster) >= min_touches:
            levels.append((float(np.mean(cluster)), len(cluster)))

    return levels


# ─────────────────────────────────────────────────────────────────────────────
#  BREAKDOWN DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _check_breakdown(
    close:      pd.Series,
    low:        pd.Series,
    flush_days: int,
) -> tuple:
    """
    Check for any breakdown signal in the last flush_days bars.

    Priority: 52w Low  >  26w Low  >  Key Level

    Returns (breakdown_type, breakdown_level, recent_low):
      breakdown_type  : '52w Low' | '26w Low' | 'Key Level' | None
      breakdown_level : the prior support level that was breached
      recent_low      : the lowest close in the flush window
    """
    n = len(close)
    if n < flush_days + 15:
        return None, None, None

    flush_cls = close.iloc[-flush_days:]
    hist_cls  = close.iloc[:-flush_days]
    flush_low = low.iloc[-flush_days:]
    hist_low  = low.iloc[:-flush_days]
    recent_lo = float(flush_cls.min())

    # 52-week low
    if len(hist_cls) >= 100:
        hist_52w = hist_cls.iloc[-252:]
        if len(hist_52w) >= 50:
            prior_52w = float(hist_52w.min())
            if recent_lo < prior_52w:
                return "52w Low", prior_52w, recent_lo

    # 26-week low
    if len(hist_cls) >= 20:
        hist_26w = hist_cls.iloc[-130:]
        if len(hist_26w) >= 20:
            prior_26w = float(hist_26w.min())
            if recent_lo < prior_26w:
                return "26w Low", prior_26w, recent_lo

    # Key-level breakdown
    levels = _find_key_levels(hist_cls, hist_low)
    if levels:
        flush_lo_min = float(flush_low.values.min())
        for level_price, _ in levels:
            if flush_lo_min < level_price:
                return "Key Level", level_price, flush_lo_min

    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
#  POST-SCREENING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_options_oi(ticker_obj, min_oi: int = 100) -> tuple:
    """
    Returns (has_options: bool, total_oi: int).
    total_oi = combined open interest across nearest 2 expirations.
    """
    try:
        exps = ticker_obj.options
        if not exps:
            return False, 0
        total_oi = 0
        for exp in list(exps)[:2]:
            chain     = ticker_obj.option_chain(exp)
            total_oi += int(chain.calls["openInterest"].fillna(0).sum())
            total_oi += int(chain.puts["openInterest"].fillna(0).sum())
        return True, total_oi
    except Exception:
        return False, 0


def _earnings_days_out(ticker_obj) -> int:
    """
    Calendar days until the next reported earnings date, or -1 if unknown.
    """
    try:
        cal = ticker_obj.calendar
        if cal is None:
            return -1
        dates = []
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date", [])
            dates = raw if isinstance(raw, (list, tuple)) else [raw]
        elif isinstance(cal, pd.DataFrame):
            for col_name in ("Earnings Date", "earnings_date"):
                if col_name in cal.columns:
                    dates = cal[col_name].dropna().tolist()
                    break
                if hasattr(cal.index, "tolist") and col_name in cal.index.tolist():
                    dates = cal.loc[col_name].dropna().tolist()
                    break
        now = datetime.now()
        future_days = []
        for d in dates:
            try:
                if hasattr(d, "to_pydatetime"):
                    dt = d.to_pydatetime().replace(tzinfo=None)
                elif isinstance(d, datetime):
                    dt = d.replace(tzinfo=None)
                else:
                    dt = datetime.strptime(str(d)[:10], "%Y-%m-%d")
                if dt >= now:
                    future_days.append((dt - now).days)
            except Exception:
                continue
        return min(future_days) if future_days else -1
    except Exception:
        return -1


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN SCREENER
# ─────────────────────────────────────────────────────────────────────────────

def run_screener(
    min_price:        float = 10.0,
    min_rel_vol:      float = 1.5,
    min_atr_pct:      float = 2.0,
    flush_days:       int   = 7,
    max_pct_extended: float = 3.0,
    min_options_oi:   int   = 100,
    progress_cb              = None,
) -> pd.DataFrame:
    """
    Scan US large-/mid-cap stocks for FBD breakdown candidates.

    Parameters
    ----------
    min_price : float
        Minimum closing price ($). Default $10.
    min_rel_vol : float
        Volume on the breakdown bar divided by its prior 20-day average.
        Default 1.5x.
    min_atr_pct : float
        ATR(14) as % of price. Filters out low-volatility stocks where
        options premiums are too thin. Default 2%.
    flush_days : int
        The breakdown signal must have occurred within the last N trading
        bars. Default 7.
    max_pct_extended : float
        Exclude if current price is already more than this % below the
        broken level (setup is too extended to trade). Default 3%.
    min_options_oi : int
        Minimum combined open interest (nearest 2 expirations).
        Default 100 contracts.
    progress_cb : callable or None
        Optional callable(float 0->1) called as the scan progresses.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, company, sector, price, breakdown, bd_level,
                 pct_extended, rel_vol, atr_pct, rsi, options_oi, earnings
        Sorted by signal priority (52w > 26w > Key Level) then pct_extended.
        Returns empty DataFrame if no candidates pass all filters.
    """

    # 1. Universe
    if progress_cb:
        progress_cb(0.02)

    meta    = get_us_tickers()
    tickers = meta["ticker"].tolist()
    n       = len(tickers)

    if progress_cb:
        progress_cb(0.06)

    # 2. Batch-download 1 year of daily OHLCV
    raw = yf.download(
        tickers,
        period      = "1y",
        interval    = "1d",
        group_by    = "ticker",
        auto_adjust = True,
        progress    = False,
        threads     = True,
    )

    if progress_cb:
        progress_cb(0.36)

    # 3. Per-ticker screening loop
    candidates = []

    for i, ticker in enumerate(tickers):
        if progress_cb and i % 100 == 0:
            progress_cb(0.36 + 0.49 * (i / n))

        try:
            # Extract per-ticker slice from the MultiIndex download
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[ticker].dropna(how="all")
            else:
                df = raw.dropna(how="all")

            if len(df) < 40:
                continue

            close  = df["Close"].squeeze().dropna()
            high   = df["High"].squeeze().dropna()
            low    = df["Low"].squeeze().dropna()
            volume = df["Volume"].squeeze().dropna()

            # Align lengths (yfinance occasionally returns mismatched lengths)
            min_len = min(len(close), len(high), len(low), len(volume))
            if min_len < 40:
                continue
            close  = close.iloc[-min_len:]
            high   = high.iloc[-min_len:]
            low    = low.iloc[-min_len:]
            volume = volume.iloc[-min_len:]

            current = float(close.iloc[-1])

            # Price floor
            if current < min_price or np.isnan(current):
                continue

            # 20-day avg volume
            avg_vol_20 = (
                float(volume.iloc[-21:-1].mean())
                if len(volume) > 21
                else float(volume.mean())
            )
            if avg_vol_20 < _MIN_AVG_VOLUME:
                continue

            # ATR %
            atr_series = _calc_atr(high, low, close)
            atr_val    = float(atr_series.iloc[-1])
            if np.isnan(atr_val) or current <= 0:
                continue
            atr_pct = atr_val / current * 100
            if atr_pct < min_atr_pct:
                continue

            # Breakdown signal
            btype, blevel, _blow = _check_breakdown(close, low, flush_days)
            if btype is None:
                continue

            # Distance from broken level
            pct_extended = (blevel - current) / blevel * 100
            if pct_extended > max_pct_extended:
                continue

            # Relative volume on the breakdown bar
            flush_low_slice = low.iloc[-flush_days:]
            flush_offset    = int(flush_low_slice.values.argmin())
            abs_bd_pos      = len(close) - flush_days + flush_offset
            vol_on_bd       = float(volume.iloc[abs_bd_pos])
            avg_start       = max(0, abs_bd_pos - 20)
            avg_vol_bd      = (
                float(volume.iloc[avg_start:abs_bd_pos].mean())
                if abs_bd_pos > 0 else 0.0
            )
            rel_vol = vol_on_bd / avg_vol_bd if avg_vol_bd > 0 else 0.0
            if rel_vol < min_rel_vol:
                continue

            # RSI (informational, not a filter)
            rsi_val = float(_calc_rsi(close).iloc[-1])

            candidates.append({
                "ticker":       ticker,
                "price":        round(current, 2),
                "breakdown":    btype,
                "bd_level":     round(blevel, 2),
                "pct_extended": round(pct_extended, 1),
                "rel_vol":      round(rel_vol, 2),
                "atr_pct":      round(atr_pct, 1),
                "rsi":          round(rsi_val, 1) if not np.isnan(rsi_val) else None,
            })

        except Exception:
            continue

    if progress_cb:
        progress_cb(0.87)

    if not candidates:
        if progress_cb:
            progress_cb(1.0)
        return pd.DataFrame()

    # 4. Post-screening: options OI + earnings flag
    final = []

    for j, cand in enumerate(candidates):
        if progress_cb:
            progress_cb(0.87 + 0.10 * (j / max(len(candidates), 1)))
        try:
            tk = yf.Ticker(cand["ticker"])

            has_opts, total_oi = _get_options_oi(tk, min_oi=min_options_oi)
            if not has_opts or total_oi < min_options_oi:
                continue

            days_out      = _earnings_days_out(tk)
            earnings_flag = f"!! {days_out}d earnings" if 0 <= days_out <= 5 else ""

            row = dict(cand)
            row["options_oi"] = total_oi
            row["earnings"]   = earnings_flag
            final.append(row)

        except Exception:
            continue

    if progress_cb:
        progress_cb(0.98)

    if not final:
        if progress_cb:
            progress_cb(1.0)
        return pd.DataFrame()

    # 5. Build output DataFrame
    df_out = pd.DataFrame(final)
    df_out = df_out.merge(meta, on="ticker", how="left")

    # Sort by signal priority then pct_extended ascending
    rank_map = {"52w Low": 0, "26w Low": 1, "Key Level": 2}
    df_out["_rank"] = df_out["breakdown"].map(rank_map).fillna(3)
    df_out = (
        df_out
        .sort_values(["_rank", "pct_extended"])
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )

    ordered = [
        "ticker", "company", "sector",
        "price", "breakdown", "bd_level", "pct_extended",
        "rel_vol", "atr_pct", "rsi", "options_oi", "earnings",
    ]
    df_out = df_out[[c for c in ordered if c in df_out.columns]]

    if progress_cb:
        progress_cb(1.0)

    return df_out
