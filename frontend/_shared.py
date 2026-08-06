"""Shared Streamlit helpers: path bootstrap, sidebar controls, session-cached analysis.

Imported first by every page so the project packages resolve and the sidebar/state exist.
"""
from __future__ import annotations

import pathlib
import sys

# ── Path bootstrap (so `orchestration`, `visualization`, … import from any page) ──
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from config import universe  # noqa: E402
from config.settings import settings  # noqa: E402
from data_ingestion.markets import normalize_ticker  # noqa: E402
from orchestration.pipeline import analyze  # noqa: E402
from orchestration.schemas import DISCLAIMER  # noqa: E402

_ACTION_COLOR = {"Buy": "#16a34a", "Hold": "#d97706", "Sell": "#dc2626"}
_REGION_LABEL = {"US": "🇺🇸 US", "INDIA": "🇮🇳 India"}


def ensure_state() -> None:
    st.session_state.setdefault("results", {})
    st.session_state.setdefault("ticker", "AAPL")
    st.session_state.setdefault("use_llm", True)
    st.session_state.setdefault("region", settings.default_region)
    st.session_state.setdefault("explore_index", None)


def _run(ticker: str, use_llm: bool) -> None:
    from database.db import save_run

    with st.status(f"Analyzing {ticker}…", expanded=True) as status:
        result = analyze(ticker, use_llm=use_llm, progress=lambda m: status.write(m))
        if result.errors and result.forecast is None:
            status.update(label=f"Could not analyze {ticker}", state="error")
        else:
            status.update(label=f"Analysis complete · {ticker}", state="complete")
    st.session_state["results"][(ticker, use_llm)] = result
    if not result.errors:
        save_run(result)


def sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📈 StockSense AI")
        st.caption("Time-series + sentiment + agentic reasoning")

        region = st.radio("Market", ["US", "INDIA"], horizontal=True,
                          index=["US", "INDIA"].index(st.session_state.get("region", "US")),
                          format_func=lambda r: _REGION_LABEL[r])
        st.session_state["region"] = region

        ticker = st.text_input("Stock ticker", value=st.session_state["ticker"],
                               placeholder="AAPL, MSFT · RELIANCE, TCS").strip().upper()
        use_llm = st.toggle("Use Gemini reasoning", value=st.session_state["use_llm"],
                            help="Off → deterministic rule-based recommendation")
        if not settings.has_gemini:
            st.warning("No Gemini key configured — rule-based reasoning will be used.")
        if st.button("🚀 Run analysis", type="primary", width="stretch"):
            st.session_state["use_llm"] = use_llm
            if ticker:
                norm = normalize_ticker(ticker, region)
                st.session_state["ticker"] = norm
                _run(norm, use_llm)
            else:
                st.error("Please enter a ticker.")

        from database.db import list_watch

        wl = list_watch()
        with st.expander(f"⭐ Watchlist ({len(wl)})"):
            if not wl:
                st.caption("No saved stocks yet.")
            for w in wl:
                if st.button(w["ticker"], key=f"wl_{w['ticker']}", width="stretch"):
                    st.session_state["region"] = w["region"]
                    analyze_ticker(w["ticker"])
                    st.rerun()

        st.divider()
        st.caption("⚠️ " + DISCLAIMER)


def setup(title: str, icon: str = "📈") -> None:
    st.set_page_config(page_title=f"StockSense · {title}", page_icon=icon, layout="wide")
    import _style  # frontend/ is on sys.path (bootstrapped above)

    _style.inject()
    ensure_state()
    sidebar()


def current_result():
    key = (st.session_state.get("ticker"), st.session_state.get("use_llm", True))
    return st.session_state.get("results", {}).get(key)


def require_result():
    """Return the current AnalysisResult, or render guidance and return None."""
    r = current_result()
    if r is None:
        st.info("👈 Enter a ticker and click **Run analysis** to begin.")
        return None
    if r.forecast is None:
        st.error(" · ".join(r.errors) or "Analysis could not be completed.")
        return None
    return r


def action_badge(action: str, confidence: float) -> None:
    color = _ACTION_COLOR.get(action, "#6b7280")
    st.markdown(
        f"<div style='background:{color};color:white;padding:14px 20px;border-radius:12px;"
        f"font-size:1.5rem;font-weight:700;text-align:center'>"
        f"{action.upper()} &nbsp;·&nbsp; {confidence*100:.0f}% confidence</div>",
        unsafe_allow_html=True,
    )


def analyze_ticker(ticker: str) -> None:
    """Normalize (by current region), run analysis, store result — used by Explore/tiles."""
    region = st.session_state.get("region", "US")
    norm = normalize_ticker(ticker, region)
    st.session_state["ticker"] = norm
    _run(norm, st.session_state.get("use_llm", True))


def explore() -> None:
    """Landing view: browse indices for the selected market, search, and recent runs."""
    region = st.session_state.get("region", "US")
    st.markdown(
        f"<div class='ss-hero'><h3>🧭 Explore {_REGION_LABEL.get(region, region)} markets</h3>"
        f"<span class='ss-muted'>Pick an index category below, choose a stock, or search a ticker "
        f"in the sidebar. Forecast · sentiment · risk · a grounded call.</span></div>",
        unsafe_allow_html=True,
    )

    infos = universe.list_indices(region)
    ncol = 4
    for i in range(0, len(infos), ncol):
        cols = st.columns(ncol)
        for col, info in zip(cols, infos[i:i + ncol]):
            suffix = "" if info.full else " · subset"
            if col.button(f"{info.name}\n\n{info.count} stocks{suffix}",
                          key=f"idx_{region}_{info.key}", width="stretch"):
                st.session_state["explore_index"] = info.key

    key = st.session_state.get("explore_index")
    if key and key in universe.index_keys(region):
        info = universe.get_index(region, key)
        st.divider()
        st.markdown(f"**{info.name}** — {info.count} constituents"
                    + ("" if info.full else " *(curated subset — refresh for the full list)*"))
        c1, c2 = st.columns([3, 1])
        pick = c1.selectbox("Choose a stock", info.tickers, key=f"pick_{region}_{key}")
        if c2.button("Analyze ▶", type="primary", width="stretch"):
            analyze_ticker(pick)
            st.rerun()

    from database.db import recent_runs
    rows = recent_runs(8)
    if rows:
        import pandas as pd

        st.divider()
        st.markdown("#### 🕘 Recent analyses")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def watch_button(ticker: str, region: str) -> None:
    """Toggle a ticker in the watchlist."""
    from database.db import add_watch, is_watched, remove_watch

    if is_watched(ticker):
        if st.button("★ Remove from watchlist", key="wl_toggle", width="stretch"):
            remove_watch(ticker)
            st.rerun()
    else:
        if st.button("☆ Add to watchlist", key="wl_toggle", width="stretch"):
            add_watch(ticker, region)
            st.rerun()


def report_downloads(result) -> None:
    """Markdown + HTML download buttons for an analysis result."""
    from report.export import to_html, to_markdown

    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Markdown report", to_markdown(result),
                       file_name=f"{result.ticker}_stocksense.md", mime="text/markdown",
                       width="stretch")
    c2.download_button("⬇️ HTML report", to_html(result),
                       file_name=f"{result.ticker}_stocksense.html", mime="text/html",
                       width="stretch")
