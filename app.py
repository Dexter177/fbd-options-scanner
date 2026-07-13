"""
app.py  —  FBD Options Scanner (Streamlit)
==========================================
Shared web app for Dexter (UK) and Danny (US).
Hosted free on Streamlit Community Cloud.

Three tabs:
  📡 Screener        — scan S&P 500 for FBD flush candidates
  🔍 Options Analyser — full call/spread analysis for any ticker
  📋 Watchlist        — shared list with export/import CSV
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import date, datetime
import io

import streamlit as st
import pandas as pd
import yfinance as yf

from screener    import run_screener
from scanner_core import analyse_ticker, SETUP_TYPES, SETUP_LABELS, MAX_RISK


@st.cache_data(ttl=300, show_spinner=False)
def _cached_analyse(ticker, anchor, setup_type, max_risk,
                    short_min_dte, short_max_dte, spread_dte_target):
    """Cache options-chain results for 5 min to avoid Yahoo rate-limits."""
    return analyse_ticker(ticker, anchor, setup_type, max_risk,
                          short_min_dte, short_max_dte, spread_dte_target)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FBD Options Scanner",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# custom CSS tweaks
st.markdown("""
<style>
    /* tighten metric padding */
    div[data-testid="metric-container"] { padding: 8px 12px; }
    /* recommendation box */
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
        "watchlist":    [],     # list of dicts
        "scan_results": None,   # pd.DataFrame | None
        "scan_params":  None,   # dict of params used for last scan
        "scan_time":    None,   # datetime
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
    f"S&P 500 flush screener · Black-Scholes options analyser · "
    f"$1,000 max risk · {date.today().strftime('%A %d %B %Y')}"
)

tab_screen, tab_analyse, tab_watch = st.tabs(
    ["📡  Screener", "🔍  Options Analyser", "📋  Watchlist"]
)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — SCREENER
# ══════════════════════════════════════════════════════════════════════════════

with tab_screen:

    st.subheader("FBD Flush Screener — S&P 500")
    st.caption(
        "Finds stocks making new 26-week lows with elevated volume and "
        "oversold RSI. These are FBD watch candidates — add them to the "
        "Watchlist and set a TradingView price alert at the recovery level."
    )

    # ── filter controls ───────────────────────────────────────────────────
    with st.expander("⚙️ Filter settings", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            f_min_drop = st.slider(
                "Min drop from 6-month high (%)", 10, 50, 15, step=5,
                help="Stock must be at least this far below its 6-month peak"
            )
            f_min_price = st.number_input("Min price ($)", value=10.0, step=5.0, format="%.0f")
        with fc2:
            f_max_rsi = st.slider(
                "Max RSI(14)", 20, 55, 40, step=5,
                help="Lower = deeper oversold. 40 catches most meaningful flushes"
            )
            f_min_rvol = st.slider(
                "Min relative volume", 1.0, 3.0, 1.5, step=0.1,
                help="Today's volume ÷ 20-day average. >1.5 confirms real selling pressure"
            )
        with fc3:
            f_flush_days = st.slider(
                "Flush window (days)", 1, 7, 3, step=1,
                help="New 26-week low must have occurred within this many bars"
            )
            st.markdown("")
            st.markdown(
                "ℹ️ All S&P 500 companies have market cap > $14B, "
                "so options liquidity is guaranteed."
            )

    run_btn = st.button("🔍  Run Scan", type="primary", use_container_width=True)

    # ── run scan ──────────────────────────────────────────────────────────
    if run_btn:
        current_params = dict(
            min_drop_pct=f_min_drop,
            max_rsi=f_max_rsi,
            min_rel_vol=f_min_rvol,
            min_price=f_min_price,
            flush_days=f_flush_days,
        )

        prog = st.progress(0, text="Fetching S&P 500 constituent list…")

        def _update_prog(pct: float):
            pct = min(float(pct), 0.99)
            if pct < 0.10:
                txt = "Downloading 6 months of price data…"
            elif pct < 0.50:
                txt = f"Scanning tickers… {int(pct*100)}%"
            else:
                txt = f"Applying FBD filters… {int(pct*100)}%"
            prog.progress(pct, text=txt)

        try:
            df_scan = run_screener(**current_params, progress_cb=_update_prog)
            st.session_state.scan_results = df_scan
            st.session_state.scan_params  = current_params
            st.session_state.scan_time    = datetime.now()
        except Exception as e:
            st.error(f"Scan failed: {e}")
            df_scan = pd.DataFrame()

        prog.empty()

    # ── results ───────────────────────────────────────────────────────────
    df_scan = st.session_state.scan_results

    if df_scan is not None:
        if st.session_state.scan_time:
            st.caption(
                f"Last scanned: {st.session_state.scan_time.strftime('%H:%M:%S')} — "
                f"re-run to refresh"
            )

        if df_scan.empty:
            st.info(
                "No stocks passed all filters today. "
                "Try relaxing Max RSI, reducing the Min drop %, or widening the flush window."
            )
        else:
            st.success(f"**{len(df_scan)} candidates** passed all filters.")

            # friendly column names for display
            disp = df_scan.copy()
            disp = disp.rename(columns={
                "ticker":     "Ticker",
                "company":    "Company",
                "sector":     "Sector",
                "price":      "Price",
                "drop_pct":   "% from High",
                "rsi":        "RSI",
                "rel_vol":    "Rel Volume",
                "prior_low":  "Prior 26w Low",
                "recent_low": "Flush Low",
            })

            st.dataframe(
                disp,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Price":         st.column_config.NumberColumn(format="$%.2f"),
                    "% from High":   st.column_config.NumberColumn(format="%.1f%%"),
                    "Rel Volume":    st.column_config.NumberColumn(format="%.2fx"),
                    "Prior 26w Low": st.column_config.NumberColumn(format="$%.2f"),
                    "Flush Low":     st.column_config.NumberColumn(format="$%.2f"),
                },
            )

            # ── add to watchlist ──────────────────────────────────────────
            st.markdown("---")
            st.subheader("Add to Watchlist")

            wl_col1, wl_col2, wl_col3 = st.columns([2, 2, 3])
            with wl_col1:
                sel_ticker = st.selectbox("Ticker", df_scan["ticker"].tolist(), key="s_sel_t")
            with wl_col2:
                match_row = df_scan[df_scan["ticker"] == sel_ticker]
                default_anchor = float(match_row["recent_low"].iloc[0]) if not match_row.empty else 0.0
                sel_anchor = st.number_input(
                    "Anchor (flush low $)", value=default_anchor,
                    step=0.5, format="%.2f", key="s_anchor"
                )
            with wl_col3:
                sel_type = st.selectbox(
                    "Setup type", list(SETUP_TYPES.keys()),
                    index=2,   # FBD_WATCH default
                    help="Upgrade to FBD_CONFIRMED once price reclaims the anchor",
                    key="s_setup_type"
                )

            sel_notes = st.text_input(
                "Notes (optional)",
                placeholder="e.g. broke 200MA on heavy vol, watching $42 reclaim",
                key="s_notes"
            )

            if st.button("➕  Add to Watchlist", key="s_add_btn"):
                existing = {w["ticker"] for w in st.session_state.watchlist}
                if sel_ticker in existing:
                    st.warning(f"{sel_ticker} is already in the watchlist.")
                else:
                    row = match_row.iloc[0] if not match_row.empty else {}
                    st.session_state.watchlist.append({
                        "ticker":       sel_ticker,
                        "company":      row.get("company", "") if hasattr(row, "get") else "",
                        "sector":       row.get("sector", "")  if hasattr(row, "get") else "",
                        "anchor":       sel_anchor,
                        "setup_type":   sel_type,
                        "notes":        sel_notes,
                        "added":        date.today().isoformat(),
                        "price_at_add": row.get("price", 0.0) if hasattr(row, "get") else 0.0,
                    })
                    st.success(f"✅ {sel_ticker} added to watchlist.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — OPTIONS ANALYSER
# ══════════════════════════════════════════════════════════════════════════════

with tab_analyse:

    st.subheader("Options Analyser")
    st.caption(
        "Enter any ticker and anchor price to get a call vs. debit spread "
        "recommendation sized to your max risk."
    )

    # ── inputs ────────────────────────────────────────────────────────────
    wl_tickers = [w["ticker"] for w in st.session_state.watchlist]
    wl_map     = {w["ticker"]: w for w in st.session_state.watchlist}

    ac1, ac2, ac3 = st.columns([2, 2, 3])
    with ac1:
        default_t = wl_tickers[0] if wl_tickers else "ORCL"
        a_ticker  = st.text_input("Ticker", value=default_t, key="a_ticker").upper().strip()
    with ac2:
        wl_entry      = wl_map.get(a_ticker)
        default_anch  = float(wl_entry["anchor"]) if wl_entry else 0.0
        a_anchor      = st.number_input(
            "Anchor price ($)", value=default_anch,
            step=0.5, format="%.2f", key="a_anchor",
            help="The level that was broken (flush low). Leave 0 to use ATM."
        )
    with ac3:
        type_keys = list(SETUP_TYPES.keys())
        default_ti = type_keys.index(wl_entry["setup_type"]) if wl_entry and wl_entry["setup_type"] in type_keys else 2
        a_setup = st.selectbox(
            "Setup type", type_keys,
            index=default_ti,
            format_func=lambda k: SETUP_LABELS[k],
            key="a_setup"
        )

    with st.expander("⚙️ Advanced options", expanded=False):
        adv1, adv2, adv3 = st.columns(3)
        with adv1:
            a_risk = st.number_input("Max risk ($)", value=MAX_RISK, step=100, key="a_risk")
        with adv2:
            a_short_min = st.number_input("Short DTE min", value=7,  step=1, key="a_smin")
            a_short_max = st.number_input("Short DTE max", value=35, step=1, key="a_smax")
        with adv3:
            a_spread_dte = st.number_input("Spread DTE target", value=60, step=5, key="a_spdte")

    analyse_btn = st.button("⚡  Analyse", type="primary", use_container_width=True, key="a_btn")

    if analyse_btn and a_ticker:
        with st.spinner(f"Fetching options chain for {a_ticker}…"):
            result = _cached_analyse(
                ticker            = a_ticker,
                anchor            = a_anchor if a_anchor > 0 else None,
                setup_type        = a_setup,
                max_risk          = int(a_risk),
                short_min_dte     = int(a_short_min),
                short_max_dte     = int(a_short_max),
                spread_dte_target = int(a_spread_dte),
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
            pm1.metric("Current Price",    f"${result['current_price']:.2f}")
            pm2.metric("Anchor",           f"${result['anchor']:.2f}")
            pm3.metric("5-day Low",        f"${result['lo_5d']:.2f}")
            if result["recovery_pct"] is not None:
                pm4.metric("Recovery from anchor", f"{result['recovery_pct']:+.1f}%")

            fbd_color = "🟢" if "✅" in result["fbd_status"] else "🔴" if "⚠️" in result["fbd_status"] else "⚪"
            st.info(f"{fbd_color}  **FBD Status:** {result['fbd_status']}")

            # ── recommendation ────────────────────────────────────────────
            rec = result.get("recommendation")
            if rec:
                badge_color = "#00d395" if rec["type"] == "CALL" else "#58a6ff"
                badge_label = "CALL" if rec["type"] == "CALL" else "SPREAD"
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

            # ── calls table ───────────────────────────────────────────────
            if not result["calls_df"].empty:
                st.markdown("---")
                st.subheader(f"Directional Calls  ({int(a_short_min)}–{int(a_short_max)} DTE)")
                st.caption("Sorted by score (best risk/reward first)")
                st.dataframe(
                    result["calls_df"],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(f"No liquid calls found in {int(a_short_min)}–{int(a_short_max)} DTE window.")

            # ── spreads table ─────────────────────────────────────────────
            if not result["spreads_df"].empty:
                st.markdown("---")
                st.subheader(f"Debit Spreads  (~{int(a_spread_dte)} DTE)")
                st.caption("Sorted by R:R (best risk-reward first)")
                st.dataframe(
                    result["spreads_df"],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(f"No spread candidates near {int(a_spread_dte)} DTE.")

            st.markdown("---")
            st.caption("⚠️ Not financial advice. Verify all prices with your broker before trading. Greeks are Black-Scholes estimates from yfinance implied volatility.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — WATCHLIST
# ══════════════════════════════════════════════════════════════════════════════

with tab_watch:

    st.subheader("📋 Watchlist")
    st.caption(
        "Share between sessions via the Export/Import buttons. "
        "Dexter exports → sends CSV to Danny via WhatsApp/email → Danny imports. "
        "Both see the same list."
    )

    # ── import / export row ───────────────────────────────────────────────
    iex1, iex2, iex3 = st.columns([3, 3, 2])

    with iex1:
        uploaded = st.file_uploader(
            "📥  Import watchlist CSV", type="csv",
            help="Upload a CSV previously exported by you or Danny",
            key="wl_upload"
        )
        if uploaded:
            try:
                imported_df  = pd.read_csv(uploaded)
                existing_tks = {w["ticker"] for w in st.session_state.watchlist}
                added = 0
                for entry in imported_df.to_dict("records"):
                    if str(entry.get("ticker", "")).upper() not in existing_tks:
                        entry["ticker"] = str(entry["ticker"]).upper()
                        st.session_state.watchlist.append(entry)
                        existing_tks.add(entry["ticker"])
                        added += 1
                if added:
                    st.success(f"Imported {added} new entr{'y' if added==1 else 'ies'}.")
                else:
                    st.info("All tickers already in watchlist — nothing new imported.")
            except Exception as e:
                st.error(f"Import failed: {e}")

    with iex2:
        if st.session_state.watchlist:
            wl_df   = pd.DataFrame(st.session_state.watchlist)
            csv_out = wl_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label    = "📤  Export watchlist CSV",
                data     = csv_out,
                file_name= f"fbd_watchlist_{date.today().isoformat()}.csv",
                mime     = "text/csv",
                help     = "Send this file to Danny (or import on another device)",
                key      = "wl_export_btn",
            )

    with iex3:
        if st.session_state.watchlist and st.button("🔄  Refresh prices", key="wl_refresh"):
            st.rerun()

    # ── manual add ────────────────────────────────────────────────────────
    with st.expander("➕  Add ticker manually"):
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            m_ticker = st.text_input("Ticker", key="m_t").upper().strip()
        with mc2:
            m_anchor = st.number_input("Anchor ($)", value=0.0, step=0.5, format="%.2f", key="m_a")
        with mc3:
            m_setup = st.selectbox(
                "Setup type", list(SETUP_TYPES.keys()),
                format_func=lambda k: SETUP_LABELS[k],
                key="m_st"
            )
        m_notes = st.text_input("Notes", key="m_n")

        if st.button("Add", key="m_add"):
            if not m_ticker:
                st.warning("Enter a ticker.")
            elif m_ticker in {w["ticker"] for w in st.session_state.watchlist}:
                st.warning(f"{m_ticker} already in watchlist.")
            else:
                st.session_state.watchlist.append({
                    "ticker":       m_ticker,
                    "company":      "",
                    "sector":       "",
                    "anchor":       m_anchor,
                    "setup_type":   m_setup,
                    "notes":        m_notes,
                    "added":        date.today().isoformat(),
                    "price_at_add": 0.0,
                })
                st.success(f"✅ {m_ticker} added.")
                st.rerun()

    # ── watchlist table ───────────────────────────────────────────────────
    st.markdown("---")

    if not st.session_state.watchlist:
        st.info("Watchlist is empty. Add tickers from the Screener tab or use the manual form above.")
    else:
        # fetch live prices for all tickers at once
        all_tks = [w["ticker"] for w in st.session_state.watchlist]
        live_px: dict = {}
        try:
            batch = yf.Tickers(" ".join(all_tks))
            for t in all_tks:
                try:
                    live_px[t] = round(float(batch.tickers[t].fast_info.last_price), 2)
                except Exception:
                    live_px[t] = None
        except Exception:
            live_px = {t: None for t in all_tks}

        rows = []
        for i, w in enumerate(st.session_state.watchlist):
            t      = w["ticker"]
            lp     = live_px.get(t)
            anchor = float(w.get("anchor", 0) or 0)
            pct_vs = (
                round((lp - anchor) / anchor * 100, 1)
                if lp and anchor > 0 else None
            )
            # status emoji
            if pct_vs is None:
                status = "—"
            elif pct_vs >= 3:
                status = "✅ Above anchor"
            elif pct_vs >= 0:
                status = "🟡 Just reclaimed"
            else:
                status = "🔴 Below anchor"

            rows.append({
                "Ticker":       t,
                "Company":      w.get("company", ""),
                "Setup":        SETUP_LABELS.get(w.get("setup_type", ""), w.get("setup_type", "")),
                "Anchor":       anchor if anchor > 0 else None,
                "Current":      lp,
                "% vs Anchor":  pct_vs,
                "Status":       status,
                "Added":        w.get("added", ""),
                "Notes":        w.get("notes", ""),
            })

        wl_display = pd.DataFrame(rows)

        st.dataframe(
            wl_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Anchor":      st.column_config.NumberColumn(format="$%.2f"),
                "Current":     st.column_config.NumberColumn(format="$%.2f"),
                "% vs Anchor": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

        # ── remove + analyse shortcuts ────────────────────────────────────
        rm1, rm2, rm3 = st.columns([3, 2, 2])
        with rm1:
            rm_ticker = st.selectbox(
                "Select ticker to remove",
                [w["ticker"] for w in st.session_state.watchlist],
                key="rm_sel"
            )
        with rm2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑  Remove", key="rm_btn"):
                st.session_state.watchlist = [
                    w for w in st.session_state.watchlist
                    if w["ticker"] != rm_ticker
                ]
                st.rerun()
        with rm3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Switch to Options Analyser tab to analyse any ticker in detail.")
