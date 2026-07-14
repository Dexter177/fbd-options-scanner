"""
app.py  —  FBD Options Scanner (Streamlit)
==========================================
Three tabs:
  📡 Breakdown Scanner      — scan US large/mid-cap stocks for breakdown candidates
  🎯 Fake Breakdown Scanner  — stocks that broke down and have since reclaimed
  🔍 Options Analyser        — full swing call / LEAP analysis for any ticker
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import date, datetime

import streamlit as st
import pandas as pd

from screener     import run_screener, run_fbd_screener
from scanner_core import analyse_ticker, SETUP_TYPES, SETUP_LABELS, MAX_RISK


@st.cache_data(ttl=300, show_spinner=False)
def _cached_analyse(ticker, anchor, setup_type, max_risk,
                    swing_min_dte, swing_max_dte, leap_min_dte, leap_max_dte):
    """Cache options-chain results for 5 min to avoid Yahoo rate-limits."""
    return analyse_ticker(ticker, anchor, setup_type, max_risk,
                          swing_min_dte, swing_max_dte, leap_min_dte, leap_max_dte)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FBD Options Scanner",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    div[data-testid="metric-container"] { padding: 8px 12px; }
    .rec-box {
        background: #161b22;
        border: 1px solid #00d395;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .rec-box h3 { margin: 0 0 4px 0; color: #00d395; }
    .rec-primary { font-size: 1.1em; font-weight: bold; margin: 6px 0; }
    .rec-detail  { color: #c9d1d9; margin: 4px 0; }
    .rec-greeks  { color: #8b949e; font-size: 0.9em; margin-top: 8px; }
    .rec-alt     { color: #8b949e; font-size: 0.85em; font-style: italic; margin-top: 6px; }
    .rec-warning { color: #f0883e; font-size: 0.9em; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "scan_results":     None,   # pd.DataFrame | None  (breakdown scanner)
        "scan_params":      None,
        "scan_time":        None,
        "fbd_scan_results": None,   # pd.DataFrame | None  (FBD scanner)
        "fbd_scan_params":  None,
        "fbd_scan_time":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("📡 FBD Options Scanner")
st.caption(
    f"US large/mid-cap breakdown & fake-breakdown screener · Black-Scholes options analyser · "
    f"$1,000 max risk · {date.today().strftime('%A %d %B %Y')}"
)

tab_breakdown, tab_fbd, tab_analyse = st.tabs(
    ["📡  Breakdown Scanner", "🎯  Fake Breakdown Scanner", "🔍  Options Analyser"]
)


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _scan_filter_ui(key_prefix: str) -> dict:
    """Render filter controls; returns dict of scan params."""
    with st.expander("⚙️ Filter settings", expanded=True):
        fc1, fc2 = st.columns(2)
        with fc1:
            flush_days = st.slider(
                "Flush window (days)", 1, 21, 7, step=1,
                key=f"{key_prefix}_flush",
                help="Breakdown signal must have occurred within this many trading bars"
            )
            min_atr = st.slider(
                "Min ATR(14)%", 1.0, 5.0, 2.0, step=0.5,
                key=f"{key_prefix}_atr",
                help="ATR as % of price — filters out low-volatility stocks where options premiums are too thin"
            )
        with fc2:
            min_rvol = st.slider(
                "Min relative volume", 1.0, 3.0, 1.5, step=0.1,
                key=f"{key_prefix}_rvol",
                help="Volume on the breakdown bar ÷ prior 20-day avg. >1.5 confirms real selling pressure"
            )
            min_price = st.number_input(
                "Min price ($)", value=10.0, step=5.0, format="%.0f",
                key=f"{key_prefix}_price"
            )
        with st.expander("🔧 Advanced", expanded=False):
            min_oi = st.number_input(
                "Min options OI (contracts)", value=100, step=50, min_value=0,
                key=f"{key_prefix}_oi",
                help="Combined open interest across the nearest 2 expirations"
            )
        st.caption(
            "Universe: US-listed stocks, market cap > $1B and avg daily volume > 500K "
            "(~1,500–2,000 names). Biotech/clinical-stage and recent IPOs excluded. "
            "Options liquidity confirmed post-screening."
        )
    return dict(
        flush_days    = flush_days,
        min_atr_pct   = min_atr,
        min_rel_vol   = min_rvol,
        min_price     = min_price,
        min_options_oi= int(min_oi),
    )


def _make_progress_cb(prog_widget):
    def _cb(pct: float):
        pct = min(float(pct), 0.99)
        if pct < 0.06:
            txt = "Fetching US stock universe (~1,500–2,000 stocks)…"
        elif pct < 0.36:
            txt = "Downloading 1 year of daily price data…"
        elif pct < 0.87:
            txt = f"Scanning breakdown signals… {int(pct * 100)}%"
        elif pct < 0.98:
            txt = "Checking options liquidity for candidates…"
        else:
            txt = "Finalising results…"
        prog_widget.progress(pct, text=txt)
    return _cb


_SCAN_COL_CONFIG = {
    "Price":      st.column_config.NumberColumn(format="$%.2f"),
    "BD Level":   st.column_config.NumberColumn(format="$%.2f"),
    "Rel Vol":    st.column_config.NumberColumn(format="%.2fx"),
    "ATR%":       st.column_config.NumberColumn(format="%.1f%%"),
    "RSI":        st.column_config.NumberColumn(format="%.0f"),
    "Options OI": st.column_config.NumberColumn(format="%d"),
}


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — BREAKDOWN SCANNER
# ══════════════════════════════════════════════════════════════════════════════

with tab_breakdown:

    st.subheader("Breakdown Scanner — US Large/Mid-Cap")
    st.caption(
        "Finds US large/mid-cap stocks breaking down from 52-week lows, "
        "26-week lows, or key support levels with elevated volume. "
        "Candidates still below the broken level — watch for a reclaim, "
        "then open the Options Analyser to size the trade."
    )

    bd_params = _scan_filter_ui("bd")
    bd_run = st.button("🔍  Run Scan", type="primary", use_container_width=True, key="bd_run")

    if bd_run:
        prog = st.progress(0, text="Fetching US stock universe…")
        try:
            df_bd = run_screener(**bd_params, progress_cb=_make_progress_cb(prog))
            st.session_state.scan_results = df_bd
            st.session_state.scan_params  = bd_params
            st.session_state.scan_time    = datetime.now()
        except Exception as e:
            st.error(f"Scan failed: {e}")
            df_bd = pd.DataFrame()
        prog.empty()

    df_bd = st.session_state.scan_results

    if df_bd is not None:
        if st.session_state.scan_time:
            st.caption(
                f"Last scanned: {st.session_state.scan_time.strftime('%H:%M:%S')} — re-run to refresh"
            )
        if df_bd.empty:
            st.info(
                "No stocks passed all filters today. "
                "Try widening the Flush window, reducing Min ATR%, "
                "lowering the Min relative volume, or reducing Min options OI."
            )
        else:
            st.success(f"**{len(df_bd)} candidates** passed all filters.")
            disp = df_bd.rename(columns={
                "ticker":       "Ticker",
                "company":      "Company",
                "sector":       "Sector",
                "price":        "Price",
                "breakdown":    "Signal",
                "bd_level":     "BD Level",
                "pct_extended": "% Below",
                "rel_vol":      "Rel Vol",
                "atr_pct":      "ATR%",
                "rsi":          "RSI",
                "options_oi":   "Options OI",
                "earnings":     "Earnings",
            })
            col_cfg = dict(_SCAN_COL_CONFIG)
            col_cfg["% Below"] = st.column_config.NumberColumn(
                format="%.1f%%", help="How far current price is below the broken level"
            )
            st.dataframe(disp, use_container_width=True, hide_index=True, column_config=col_cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — FAKE BREAKDOWN SCANNER
# ══════════════════════════════════════════════════════════════════════════════

with tab_fbd:

    st.subheader("Fake Breakdown Scanner — US Large/Mid-Cap")
    st.caption(
        "Finds US large/mid-cap stocks that broke BELOW a key level (52-week low, "
        "26-week low, or key support) within the flush window and have since "
        "**reclaimed back above** that level — the classic Failed Breakdown (FBD) setup. "
        "Use the Options Analyser tab to size a call or LEAP on a candidate."
    )

    fbd_params = _scan_filter_ui("fbd")
    fbd_run = st.button("🔍  Run Scan", type="primary", use_container_width=True, key="fbd_run")

    if fbd_run:
        prog = st.progress(0, text="Fetching US stock universe…")
        try:
            df_fbd = run_fbd_screener(**fbd_params, progress_cb=_make_progress_cb(prog))
            st.session_state.fbd_scan_results = df_fbd
            st.session_state.fbd_scan_params  = fbd_params
            st.session_state.fbd_scan_time    = datetime.now()
        except Exception as e:
            st.error(f"Scan failed: {e}")
            df_fbd = pd.DataFrame()
        prog.empty()

    df_fbd = st.session_state.fbd_scan_results

    if df_fbd is not None:
        if st.session_state.fbd_scan_time:
            st.caption(
                f"Last scanned: {st.session_state.fbd_scan_time.strftime('%H:%M:%S')} — re-run to refresh"
            )
        if df_fbd.empty:
            st.info(
                "No FBD setups found today. "
                "Try widening the Flush window, reducing Min ATR%, "
                "lowering the Min relative volume, or reducing Min options OI."
            )
        else:
            st.success(f"**{len(df_fbd)} FBD setups** passed all filters.")
            disp = df_fbd.rename(columns={
                "ticker":    "Ticker",
                "company":   "Company",
                "sector":    "Sector",
                "price":     "Price",
                "breakdown": "Signal",
                "bd_level":  "BD Level",
                "pct_above": "% Above",
                "rel_vol":   "Rel Vol",
                "atr_pct":   "ATR%",
                "rsi":       "RSI",
                "options_oi":"Options OI",
                "earnings":  "Earnings",
            })
            col_cfg = dict(_SCAN_COL_CONFIG)
            col_cfg["% Above"] = st.column_config.NumberColumn(
                format="%.1f%%", help="How far current price is above the reclaimed level"
            )
            st.dataframe(disp, use_container_width=True, hide_index=True, column_config=col_cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — OPTIONS ANALYSER
# ══════════════════════════════════════════════════════════════════════════════

with tab_analyse:

    st.subheader("Options Analyser")
    st.caption(
        "Enter any ticker and level to get a swing call or LEAP "
        "recommendation sized to your max risk."
    )

    # ── inputs ────────────────────────────────────────────────────────────
    ac1, ac2, ac3 = st.columns([2, 2, 3])
    with ac1:
        a_ticker = st.text_input("Ticker", value="", key="a_ticker").upper().strip()
    with ac2:
        a_anchor = st.number_input(
            "Level ($)", value=0.0, step=0.5, format="%.2f", key="a_anchor",
            help="The level that was broken (flush low). Leave 0 to use ATM."
        )
    with ac3:
        type_keys = list(SETUP_TYPES.keys())
        a_setup = st.selectbox(
            "Setup type", type_keys, index=2,
            format_func=lambda k: SETUP_LABELS[k],
            key="a_setup"
        )

    with st.expander("⚙️ Advanced options", expanded=False):
        adv1, adv2, adv3 = st.columns(3)
        with adv1:
            a_risk = st.number_input("Max risk ($)", value=MAX_RISK, step=100, key="a_risk")
        with adv2:
            a_swing_min = st.number_input("Swing DTE min", value=7,   step=1,  key="a_smin")
            a_swing_max = st.number_input("Swing DTE max", value=35,  step=1,  key="a_smax")
        with adv3:
            a_leap_min  = st.number_input("LEAP DTE min",  value=180, step=30, key="a_lmin")
            a_leap_max  = st.number_input("LEAP DTE max",  value=365, step=30, key="a_lmax")

    analyse_btn = st.button("⚡  Analyse", type="primary", use_container_width=True, key="a_btn")

    if analyse_btn and a_ticker:
        with st.spinner(f"Fetching options chain for {a_ticker}…"):
            result = _cached_analyse(
                ticker        = a_ticker,
                anchor        = a_anchor if a_anchor > 0 else None,
                setup_type    = a_setup,
                max_risk      = int(a_risk),
                swing_min_dte = int(a_swing_min),
                swing_max_dte = int(a_swing_max),
                leap_min_dte  = int(a_leap_min),
                leap_max_dte  = int(a_leap_max),
            )

        if result["status"] == "error":
            if result["error"] == "rate_limited":
                st.warning(
                    "⏳ Yahoo Finance is rate-limiting this server — "
                    "wait **30 seconds** then click Analyse again. "
                    "This usually happens right after a full screener scan."
                )
            else:
                st.error(f"❌ {result['error']}")

        else:
            # ── price summary ─────────────────────────────────────────────
            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Current Price",       f"${result['current_price']:.2f}")
            pm2.metric("Level",               f"${result['anchor']:.2f}")
            pm3.metric("5-day Low",           f"${result['lo_5d']:.2f}")
            if result["recovery_pct"] is not None:
                pm4.metric("Recovery from level", f"{result['recovery_pct']:+.1f}%")

            fbd_color = "🟢" if "✅" in result["fbd_status"] else "🔴" if "⚠️" in result["fbd_status"] else "⚪"
            st.info(f"{fbd_color}  **FBD Status:** {result['fbd_status']}")

            # ── recommendation ────────────────────────────────────────────
            rec = result.get("recommendation")
            if rec:
                rec_type = rec["type"]
                if rec_type == "CALL":
                    badge_color, badge_label = "#00d395", "SWING CALL"
                elif rec_type == "LEAP":
                    badge_color, badge_label = "#f0a500", "LEAP CALL"
                else:
                    badge_color, badge_label = "#58a6ff", rec_type
                st.markdown(
                    f"""<div class="rec-box">
                        <h3>⭐ Recommendation — <span style="color:{badge_color}">{badge_label}</span></h3>
                        <div class="rec-detail" style="color:#8b949e">{rec['rationale']}</div>
                        <div class="rec-primary">{rec['primary']}</div>
                        <div class="rec-detail">{rec['detail']}</div>
                        <div class="rec-greeks">{rec['greeks']}</div>
                        {"<div class='rec-alt'>Alternative: " + rec['alt'] + "</div>" if rec.get('alt') else ""}
                        {"<div class='rec-warning'>" + rec['warning'] + "</div>" if rec.get('warning') else ""}
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("No liquid options found matching the filter criteria.")

            # ── swing calls table ─────────────────────────────────────────
            if not result["calls_df"].empty:
                st.markdown("---")
                st.subheader(f"Swing Calls  ({int(a_swing_min)}–{int(a_swing_max)} DTE)")
                st.caption("Sorted by score (best risk/reward first)")
                st.dataframe(result["calls_df"], use_container_width=True, hide_index=True)
            else:
                st.info(f"No liquid calls found in {int(a_swing_min)}–{int(a_swing_max)} DTE window.")

            # ── LEAP calls table ──────────────────────────────────────────
            if not result["leaps_df"].empty:
                st.markdown("---")
                st.subheader(f"LEAP Calls  ({int(a_leap_min)}–{int(a_leap_max)} DTE)")
                st.caption("Long-dated directional calls sorted by score")
                st.dataframe(result["leaps_df"], use_container_width=True, hide_index=True)
            else:
                st.info(f"No LEAP calls found in {int(a_leap_min)}–{int(a_leap_max)} DTE window.")

            st.markdown("---")
            st.caption(
                "⚠️ Not financial advice. Verify all prices with your broker before trading. "
                "Greeks are Black-Scholes estimates from yfinance implied volatility."
            )
