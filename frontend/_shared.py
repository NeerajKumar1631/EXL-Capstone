"""Shared Streamlit helpers: path bootstrap, chrome, sidebar, session-cached analysis.

`boot()` runs once per rerun from `app.py` (the navigation entry point); the views under
`views/` render content only and never touch page config.
"""
from __future__ import annotations

import pathlib
import sys

# ── Path bootstrap (so `orchestration`, `visualization`, … import from any view) ──
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

import _style  # noqa: E402  (frontend/ is on sys.path via app.py)
import _warmup  # noqa: E402
from config import universe  # noqa: E402
from config.settings import settings  # noqa: E402
from data_ingestion.markets import normalize_ticker, search_symbols  # noqa: E402
from orchestration.pipeline import analyze  # noqa: E402
from orchestration.schemas import DISCLAIMER  # noqa: E402

_ACTION_COLOR = {"Buy": "#16a34a", "Hold": "#d97706", "Sell": "#dc2626"}
_REGION_LABEL = {"US": "United States", "INDIA": "India"}


def ensure_state() -> None:
    st.session_state.setdefault("results", {})
    st.session_state.setdefault("ticker", "AAPL")
    st.session_state.setdefault("use_llm", True)
    st.session_state.setdefault("region", settings.default_region)
    st.session_state.setdefault("explore_index", None)


def boot() -> None:
    """Page config + stylesheet + background model warm-up + session defaults."""
    st.set_page_config(
        page_title="StockSense",
        page_icon=":material/trending_up:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _warmup.start()  # load FinBERT/MiniLM off the first analysis' critical path
    _style.inject()
    ensure_state()


def page_header(title: str, subtitle: str = "") -> None:
    """Consistent title + one-line explanation at the top of every view."""
    st.title(title)
    if subtitle:
        st.markdown(f"<div class='ss-page-sub'>{subtitle}</div>", unsafe_allow_html=True)


# ── Analysis plumbing ────────────────────────────────────────────────
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


def analyze_ticker(ticker: str) -> None:
    """Normalize (by current region), run analysis, store result."""
    region = st.session_state.get("region", "US")
    norm = normalize_ticker(ticker, region)
    st.session_state["ticker"] = norm
    _run(norm, st.session_state.get("use_llm", True))


def current_result():
    key = (st.session_state.get("ticker"), st.session_state.get("use_llm", True))
    return st.session_state.get("results", {}).get(key)


def sidebar() -> None:
    with st.sidebar:
        st.markdown(_style.BRAND, unsafe_allow_html=True)

        region = st.radio("Market", ["US", "INDIA"], horizontal=True,
                          index=["US", "INDIA"].index(st.session_state.get("region", "US")),
                          format_func=lambda r: _REGION_LABEL[r])
        st.session_state["region"] = region

        ticker = ticker_input("Search stock", key="sidebar_ticker")
        use_llm = st.toggle("Gemini reasoning", value=st.session_state["use_llm"],
                            help="Off uses a deterministic rule-based recommendation")
        if not settings.has_gemini:
            st.caption("No Gemini key configured — reasoning will be rule-based.")
        if st.button("Run analysis", type="primary", width="stretch"):
            st.session_state["use_llm"] = use_llm
            if ticker:
                # `ticker_input` already resolved and qualified the symbol.
                st.session_state["ticker"] = ticker
                _run(ticker, use_llm)
            else:
                st.error("Search for a company or enter a symbol first.")

        from database.db import list_watch

        wl = list_watch()
        with st.expander(f"Watchlist ({len(wl)})"):
            if not wl:
                st.caption("No saved stocks yet.")
            for w in wl:
                if st.button(w["ticker"], key=f"wl_{w['ticker']}", width="stretch"):
                    st.session_state["region"] = w["region"]
                    analyze_ticker(w["ticker"])
                    st.rerun()

        st.divider()
        st.caption(DISCLAIMER)


def ticker_input(label: str, key: str) -> str:
    """Search by ticker **or company name**; returns a ready-to-analyze symbol.

    A symbol we already know is used as typed. Anything else goes to Yahoo's symbol
    search, so "apple" finds AAPL and "tata" finds TCS.NS — the user does not have to
    know the ticker. Picked results are already exchange-qualified, so they are returned
    verbatim rather than being re-normalized (which would wrongly turn NYSE-listed INFY
    into INFY.NS while the India market is selected).
    """
    region = st.session_state.get("region", "US")
    st.session_state.setdefault(key, st.session_state.get("ticker", ""))
    typed = st.text_input(
        label, key=key,
        placeholder="AAPL or Apple · TCS or Tata",
        # Streamlit only reruns a text box on Enter/blur, so say so — otherwise typing a
        # company name and waiting looks like the search is broken.
        help="Type a symbol or a company name, then press Enter.",
    ).strip()
    if not typed:
        return ""

    upper = typed.upper()
    known = universe.searchable(region)
    if upper in known or normalize_ticker(upper, region) in known:
        return normalize_ticker(upper, region)

    hits = search_symbols(typed, region)
    if not hits:
        # Offline, or nothing matched — try it as typed rather than blocking the user.
        return normalize_ticker(upper, region)

    labels = {h.symbol: h.label for h in hits}
    return st.selectbox("Matches", list(labels), format_func=lambda s: labels[s],
                        key=f"{key}_match")


# ── Result access & empty states ─────────────────────────────────────
def empty_state(what: str) -> None:
    """Explain what a view will show, instead of leaving it blank."""
    st.markdown(
        f"<div class='ss-empty'><h3>Nothing to show yet</h3>"
        f"<p>Run an analysis from the sidebar and this page will show {what}.</p></div>",
        unsafe_allow_html=True,
    )


def require_result(what: str):
    """Return the current AnalysisResult, or render a proper empty/error state."""
    r = current_result()
    if r is None:
        empty_state(what)
        return None
    if r.forecast is None:
        show_problems(r)
        empty_state(what)
        return None
    show_problems(r)
    return r


def show_problems(r) -> None:
    """Plain-English problems up front; the technical text stays in a details expander."""
    for msg in r.errors:
        st.error(msg)
    for msg in r.warnings:
        st.warning(msg)
    if r.details:
        with st.expander("Technical details"):
            for d in r.details:
                st.code(d, language=None)


def action_badge(action: str, confidence: float) -> None:
    color = _ACTION_COLOR.get(action, "#6b7280")
    st.markdown(
        f"<div class='ss-verdict' style='background:{color}'>"
        f"<span class='ss-action'>{action.upper()}</span>"
        f"<span class='ss-conf'>{confidence*100:.0f}% confidence</span></div>",
        unsafe_allow_html=True,
    )


def tone_dot(label: str | None) -> str:
    """Inline coloured dot + word for a sentiment label (no emoji)."""
    lab = (label or "n/a").lower()
    cls = lab if lab in ("positive", "negative", "neutral") else "neutral"
    return f"<span class='ss-tone'><span class='ss-dot ss-dot-{cls}'></span>{lab.title()}</span>"


def watch_button(ticker: str, region: str) -> None:
    from database.db import add_watch, is_watched, remove_watch

    if is_watched(ticker):
        if st.button("Remove from watchlist", key="wl_toggle", width="stretch"):
            remove_watch(ticker)
            st.rerun()
    else:
        if st.button("Add to watchlist", key="wl_toggle", width="stretch"):
            add_watch(ticker, region)
            st.rerun()


def report_downloads(result) -> None:
    """Markdown / HTML / PDF downloads for an analysis result."""
    from report.export import to_html, to_markdown, to_pdf

    pdf = to_pdf(result)
    cols = st.columns(3 if pdf else 2)
    cols[0].download_button("Markdown report", to_markdown(result),
                            file_name=f"{result.ticker}_stocksense.md",
                            mime="text/markdown", width="stretch")
    cols[1].download_button("HTML report", to_html(result),
                            file_name=f"{result.ticker}_stocksense.html",
                            mime="text/html", width="stretch")
    if pdf:
        cols[2].download_button("PDF report", pdf,
                                file_name=f"{result.ticker}_stocksense.pdf",
                                mime="application/pdf", width="stretch")


def explore() -> None:
    """Landing view: browse indices for the selected market, then pick a stock."""
    region = st.session_state.get("region", "US")
    st.markdown(
        f"<div class='ss-empty'><h3>Explore {_REGION_LABEL.get(region, region)} markets</h3>"
        f"<p>Pick an index below and choose a stock, or enter a symbol in the sidebar. "
        f"You will get a forecast, news sentiment, risk profile and a grounded call.</p></div>",
        unsafe_allow_html=True,
    )
    st.write("")

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
                    + ("" if info.full else " *(curated subset)*"))
        c1, c2 = st.columns([3, 1])
        pick = c1.selectbox("Choose a stock", info.tickers, key=f"pick_{region}_{key}")
        if c2.button("Analyze", type="primary", width="stretch"):
            analyze_ticker(pick)
            st.rerun()

    from database.db import recent_runs

    rows = recent_runs(8)
    if rows:
        import pandas as pd

        st.divider()
        st.markdown("#### Recent analyses")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
