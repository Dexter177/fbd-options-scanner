#!/usr/bin/env python3
"""
chart_analyzer.py  —  Stage 2 of the FBD Options Funnel
=========================================================
Detects key support/resistance levels from price history using:
  - Swing highs and lows  (local maxima/minima over a rolling window)
  - Consolidation zones   (price ranging sideways for N+ days)
  - Round numbers         (psychological magnets scaled to stock price)
  - 52-week high/low      (absolute anchors)

Returns a LevelMap that feeds into fbd_stock_scanner.py for
structure-informed strike selection.

Standalone usage:
    python chart_analyzer.py STEP 40.00
    python chart_analyzer.py COIN 142.00
    python chart_analyzer.py ORCL

Imported usage (in fbd_stock_scanner.py):
    from chart_analyzer import get_level_map, print_level_map
    lm = get_level_map("STEP", anchor=40.0)
    r1 = lm.first_resistance_above(lm.current)

Requirements: pip install yfinance numpy pandas
"""

import math
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise ImportError(f"Missing dependency: {exc}. "
                      "Run: pip install yfinance numpy pandas") from exc

# ═══════════════════════════════════════════════════════════════════════════
#  TUNING PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

SWING_WINDOW      = 5      # bars either side for swing detection
CLUSTER_TOL       = 0.016  # merge levels within 1.6% of each other
CONSOLIDATION_N   = 3      # days in range to flag as consolidation zone
CONSOLIDATION_RNG = 0.022  # ±2.2% range = sideways
LOOKBACK_DEFAULT  = 180    # days of daily history to fetch
MAX_LEVELS_SHOWN  = 6      # per side in the pretty-print

# ═══════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Level:
    price:              float
    touches:            int
    tags:               List[str]
    strength:           float   # 1.0 – 10.0
    last_touch_days_ago: int
    side:               str     # "support" | "resistance" | "pivot"

    @property
    def stars(self) -> str:
        n = min(5, max(1, round(self.strength / 2)))
        return "★" * n + "☆" * (5 - n)

    @property
    def tag_str(self) -> str:
        return ", ".join(self.tags)

    def __repr__(self) -> str:
        return (f"${self.price:.2f}  {self.stars}  [{self.tag_str}]  "
                f"{self.touches}x  {self.last_touch_days_ago}d ago  ({self.side})")


@dataclass
class LevelMap:
    ticker:       str
    current:      float
    anchor:       Optional[float]
    levels:       List[Level]   # sorted ascending by price
    lookback_days: int

    # ── helpers ──────────────────────────────────────────────────────────
    def resistances(self) -> List[Level]:
        """All levels above current price."""
        return [l for l in self.levels if l.price > self.current * 1.005]

    def supports(self) -> List[Level]:
        """All levels below anchor (or current if no anchor)."""
        ref = self.anchor or self.current
        return [l for l in self.levels if l.price < ref * 0.995]

    def first_resistance_above(self, price: float) -> Optional[Level]:
        above = [l for l in self.levels if l.price > price * 1.005]
        return above[0] if above else None

    def second_resistance_above(self, price: float) -> Optional[Level]:
        above = [l for l in self.levels if l.price > price * 1.005]
        return above[1] if len(above) >= 2 else None

    def nearest_support_below(self, price: float) -> Optional[Level]:
        below = [l for l in self.levels if l.price < price * 0.995]
        return below[-1] if below else None

    def snap_to_strike(self, target_price: float,
                       available_strikes: List[float]) -> Optional[float]:
        """
        Find the available strike closest to target_price from above.
        Used to snap a resistance level onto a real option strike.
        """
        candidates = [k for k in available_strikes if k >= target_price * 0.97]
        if not candidates:
            return None
        return min(candidates, key=lambda k: abs(k - target_price))


# ═══════════════════════════════════════════════════════════════════════════
#  ROUND-NUMBER GRID
# ═══════════════════════════════════════════════════════════════════════════

def round_number_grid(current: float) -> List[float]:
    """
    Generate psychologically meaningful round-number levels near current price.
    Step sizes are scaled to the stock's price tier.
    """
    if current < 5:
        steps = [0.50, 1.00, 2.50]
    elif current < 15:
        steps = [0.50, 1.00, 2.50, 5.00]
    elif current < 50:
        steps = [1.00, 2.50, 5.00, 10.00]
    elif current < 150:
        steps = [5.00, 10.00, 25.00, 50.00]
    elif current < 500:
        steps = [10.00, 25.00, 50.00, 100.00]
    else:
        steps = [25.00, 50.00, 100.00, 250.00]

    lo, hi = current * 0.60, current * 1.50
    levels: set = set()
    for step in steps:
        n0 = math.floor(lo / step)
        n1 = math.ceil(hi / step)
        for n in range(n0, n1 + 1):
            v = round(n * step, 2)
            if lo <= v <= hi:
                levels.add(v)
    return sorted(levels)


# ═══════════════════════════════════════════════════════════════════════════
#  SWING HIGH / LOW DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def find_swing_levels(df: pd.DataFrame,
                      window: int = SWING_WINDOW
                      ) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Identify swing highs and lows.
    A swing high at bar i: df.High[i] is >= all bars in [i-window, i+window].
    A swing low  at bar i: df.Low[i]  is <= all bars in [i-window, i+window].
    Returns two lists of (price, timestamp) tuples.
    """
    highs  = df["High"].values
    lows   = df["Low"].values
    dates  = df.index.tolist()
    n      = len(highs)

    s_highs: List[Tuple] = []
    s_lows:  List[Tuple] = []

    for i in range(window, n - window):
        lo_i = max(0, i - window)
        hi_i = i + window + 1
        if highs[i] >= max(highs[lo_i:hi_i]):
            s_highs.append((float(highs[i]), dates[i]))
        if lows[i] <= min(lows[lo_i:hi_i]):
            s_lows.append((float(lows[i]), dates[i]))

    return s_highs, s_lows


# ═══════════════════════════════════════════════════════════════════════════
#  CONSOLIDATION ZONE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def find_consolidation_levels(df: pd.DataFrame,
                               n_days: int = CONSOLIDATION_N,
                               rng: float = CONSOLIDATION_RNG
                               ) -> List[Tuple]:
    """
    Detect multi-day sideways zones where close stayed within ±rng%.
    Returns list of (midpoint_price, midpoint_timestamp) tuples.
    """
    closes = df["Close"].values
    dates  = df.index.tolist()
    levels: List[Tuple] = []
    i = 0
    while i < len(closes) - n_days:
        window = closes[i: i + n_days]
        mid    = float(np.mean(window))
        if mid == 0:
            i += 1
            continue
        spread = (float(max(window)) - float(min(window))) / mid
        if spread <= rng:
            mid_idx = i + n_days // 2
            levels.append((mid, dates[mid_idx]))
            i += n_days          # skip ahead — don't double-count same zone
        else:
            i += 1
    return levels


# ═══════════════════════════════════════════════════════════════════════════
#  CLUSTER AND SCORE
# ═══════════════════════════════════════════════════════════════════════════

def cluster_and_score(raw: List[Tuple],
                      today: date,
                      round_set: set,
                      current: float,
                      tolerance: float = CLUSTER_TOL) -> List[Level]:
    """
    raw: list of (price, date_or_None, tag_string) tuples.
    Groups prices within tolerance% of each other into clusters,
    then scores each cluster by touches × recency × round-number bonus.
    Returns list of Level objects sorted by price.
    """
    if not raw:
        return []

    raw_sorted = sorted(raw, key=lambda x: x[0])

    # ── group into clusters ──────────────────────────────────────────────
    clusters: List[List[Tuple]] = []
    current_cluster = [raw_sorted[0]]

    for item in raw_sorted[1:]:
        ref = current_cluster[0][0]
        if ref > 0 and abs(item[0] - ref) / ref <= tolerance:
            current_cluster.append(item)
        else:
            clusters.append(current_cluster)
            current_cluster = [item]
    clusters.append(current_cluster)

    # ── score each cluster ───────────────────────────────────────────────
    levels: List[Level] = []
    for cluster in clusters:
        prices  = [x[0] for x in cluster]
        dates_c = [x[1] for x in cluster if x[1] is not None]
        tags    = sorted(set(x[2] for x in cluster))

        price   = float(np.mean(prices))
        touches = len(cluster)

        # Recency: days since the most recent touch
        if dates_c:
            most_recent = max(
                (d.date() if hasattr(d, "date") else d) for d in dates_c
            )
            days_ago = max(0, (today - most_recent).days)
        else:
            days_ago = lookback_sentinel  # round numbers have no date

        # Round-number bonus
        is_round = any(abs(price - rn) / max(price, 1) < 0.008 for rn in round_set)
        if is_round and "round" not in tags:
            tags.append("round")

        # Strength formula: base from touches × recency weight + round bonus
        recency_w = max(0.15, 1.0 - days_ago / 220.0)
        strength  = min(10.0, touches * 1.6 * recency_w + (1.8 if is_round else 0.0))
        strength  = round(strength, 1)

        # Side relative to current price
        if price > current * 1.01:
            side = "resistance"
        elif price < current * 0.99:
            side = "support"
        else:
            side = "pivot"

        levels.append(Level(
            price=round(price, 2),
            touches=touches,
            tags=tags,
            strength=strength,
            last_touch_days_ago=days_ago if days_ago < 9000 else 999,
            side=side,
        ))

    return sorted(levels, key=lambda l: l.price)


# Sentinel for levels without a real date (round numbers, anchor)
lookback_sentinel = 9999


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN API
# ═══════════════════════════════════════════════════════════════════════════

def get_level_map(ticker: str,
                  anchor: Optional[float] = None,
                  current: Optional[float] = None,
                  lookback_days: int = LOOKBACK_DEFAULT) -> LevelMap:
    """
    Fetch price history and return a LevelMap.

    ticker       : stock symbol
    anchor       : FBD anchor price (e.g., 40.0 for STEP)
    current      : current price (fetched if not provided)
    lookback_days: days of daily OHLC history to analyse
    """
    today = date.today()

    tk = yf.Ticker(ticker)
    df = tk.history(period=f"{lookback_days}d", interval="1d")
    if df.empty:
        raise ValueError(f"No price data returned for {ticker}")

    if current is None:
        current = float(df["Close"].iloc[-1])

    # ── raw level sources ────────────────────────────────────────────────
    s_highs, s_lows = find_swing_levels(df)
    consol          = find_consolidation_levels(df)
    rn_list         = round_number_grid(current)
    rn_set          = set(rn_list)

    # 52-week anchors
    hi_52 = float(df["High"].max())
    lo_52 = float(df["Low"].min())
    hi_dt = df["High"].idxmax()
    lo_dt = df["Low"].idxmin()

    raw: List[Tuple] = []

    for price, dt in s_highs:
        raw.append((price, dt, "swing-high"))
    for price, dt in s_lows:
        raw.append((price, dt, "swing-low"))
    for price, dt in consol:
        raw.append((price, dt, "consolidation"))

    raw.append((hi_52, hi_dt, "52w-high"))
    raw.append((lo_52, lo_dt, "52w-low"))

    for rn in rn_list:
        raw.append((rn, None, "round"))

    if anchor is not None:
        raw.append((anchor, None, "fbd-anchor"))

    # ── cluster, score, filter ───────────────────────────────────────────
    levels = cluster_and_score(raw, today, rn_set, current)

    # Keep only levels within 65 %–150 % of current price
    lo_filter = current * 0.65
    hi_filter = current * 1.50
    levels = [l for l in levels if lo_filter <= l.price <= hi_filter]

    return LevelMap(
        ticker=ticker,
        current=round(current, 4),
        anchor=anchor,
        levels=levels,
        lookback_days=lookback_days,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def print_level_map(lm: LevelMap, n: int = MAX_LEVELS_SHOWN) -> None:
    """Pretty-print the level map inline (used by the scanner and CLI)."""
    W = 72
    anch_str = f"  |  Anchor: ${lm.anchor:.2f}" if lm.anchor else ""
    print(f"\n  {'─'*W}")
    print(f"  LEVEL MAP  {lm.ticker}  |  Current: ${lm.current:.2f}{anch_str}"
          f"  |  {lm.lookback_days}d lookback")
    print(f"  {'─'*W}")

    res = lm.resistances()
    sup = lm.supports()

    # Resistance table
    print(f"\n  RESISTANCE ABOVE  (nearest {min(n, len(res))} of {len(res)})")
    hdr = f"  {'Price':>8}  {'Strength':8}  {'Tags':<28}  {'Touches':>7}  Age"
    print(hdr)
    print(f"  {'─'*65}")
    for lv in res[:n]:
        tstr = lv.tag_str[:26]
        age  = f"{lv.last_touch_days_ago}d" if lv.last_touch_days_ago < 900 else "—"
        print(f"  ${lv.price:>7.2f}  {lv.stars}  {tstr:<28}  {lv.touches:>7}x  {age}")

    if not res:
        print(f"  (none above ${lm.current:.2f})")

    # Support table
    print(f"\n  SUPPORT BELOW  (nearest {min(n, len(sup))} of {len(sup)})")
    print(hdr)
    print(f"  {'─'*65}")
    for lv in reversed(sup[-n:]):
        tstr = lv.tag_str[:26]
        age  = f"{lv.last_touch_days_ago}d" if lv.last_touch_days_ago < 900 else "—"
        print(f"  ${lv.price:>7.2f}  {lv.stars}  {tstr:<28}  {lv.touches:>7}x  {age}")

    if not sup:
        print(f"  (none below ${lm.anchor or lm.current:.2f})")

    # Trade structure callout
    r1 = lm.first_resistance_above(lm.current)
    r2 = lm.second_resistance_above(lm.current)
    s1 = lm.nearest_support_below(lm.anchor or lm.current)

    print(f"\n  STRUCTURE-INFORMED TARGETS")
    print(f"  {'─'*65}")
    if r1:
        mv1 = round((r1.price - lm.current) / lm.current * 100, 1)
        print(f"  R1 (first wall)  : ${r1.price:.2f}  ({mv1:+.1f}%)  {r1.stars}")
        print(f"     → Short-dated call target.  Spread sell leg: snap to nearest strike at ${r1.price:.2f}")
    if r2:
        mv2 = round((r2.price - lm.current) / lm.current * 100, 1)
        print(f"  R2 (second wall) : ${r2.price:.2f}  ({mv2:+.1f}%)  {r2.stars}")
        print(f"     → Wider ~45d spread: buy near anchor, sell near ${r2.price:.2f}")
    if s1:
        mv_s = round((s1.price - (lm.anchor or lm.current)) / (lm.anchor or lm.current) * 100, 1)
        print(f"  Key floor        : ${s1.price:.2f}  ({mv_s:+.1f}%)  {s1.stars}")
        print(f"     → Trade fails if price breaks ${s1.price:.2f} with conviction")

    print()


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python chart_analyzer.py <TICKER> [anchor_price] [lookback_days]")
        print("Examples:")
        print("  python chart_analyzer.py STEP 40.00")
        print("  python chart_analyzer.py COIN 142.00 250")
        print("  python chart_analyzer.py ORCL")
        sys.exit(0)

    _ticker  = sys.argv[1].upper()
    _anchor  = float(sys.argv[2]) if len(sys.argv) >= 3 else None
    _lookback = int(sys.argv[3]) if len(sys.argv) >= 4 else LOOKBACK_DEFAULT

    print(f"\nAnalyzing {_ticker} ({_lookback}d lookback)...")
    _lm = get_level_map(_ticker, anchor=_anchor, lookback_days=_lookback)
    print_level_map(_lm)
