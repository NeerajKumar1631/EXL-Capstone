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

# (light, dark) gradient stops + glow per verdict — must stay high-contrast under white text
_ACTION_STYLE = {
    "Buy": ("#22c55e", "#15803d", "rgba(34,197,94,.35)"),
    "Hold": ("#f59e0b", "#b45309", "rgba(245,158,11,.35)"),
    "Sell": ("#ef4444", "#b91c1c", "rgba(239,68,68,.35)"),
}
_ACTION_FALLBACK = ("#94a3b8", "#64748b", "rgba(148,163,184,.35)")
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


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    """Consistent header: eyebrow (nav-section label) + title + one-line explanation.

    Keeps `st.title` underneath so the AppTest sweep's title assertions stay meaningful.
    """
    if eyebrow:
        st.markdown(f"<div class='ss-eyebrow'>{eyebrow}</div>", unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f"<div class='ss-page-sub'>{subtitle}</div>", unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    """Section header with an accent bar — replaces bare st.subheader on the main views."""
    cap = f"<span class='ss-section-cap'>{caption}</span>" if caption else ""
    st.markdown(
        f"<div class='ss-section'><span class='ss-section-bar'></span>"
        f"<span class='ss-section-title'>{title}</span>{cap}</div>",
        unsafe_allow_html=True,
    )


def kpi_row(items: list[dict]) -> None:
    """A row of stat cards in a self-aligning grid.

    Each item: {label, value, delta?, tone? ('pos'|'neg'|'muted'), sub?}. One HTML grid
    instead of st.columns, so cards are equal-height and wrap cleanly on narrow screens.
    Tone defaults from the delta's leading sign when not given.
    """
    import html as _html

    cards = []
    for it in items:
        delta = it.get("delta")
        tone = it.get("tone")
        if delta and not tone:
            tone = "pos" if str(delta).startswith("+") else "neg" if str(delta).startswith("−") or str(delta).startswith("-") else "muted"
        delta_html = (f"<div class='ss-kpi-delta {tone or 'muted'}'>{_html.escape(str(delta))}</div>"
                      if delta else "")
        sub_html = (f"<div class='ss-kpi-sub'>{_html.escape(str(it['sub']))}</div>"
                    if it.get("sub") else "")
        cards.append(
            f"<div class='ss-kpi'><div class='ss-kpi-label'>{_html.escape(str(it['label']))}</div>"
            f"<div class='ss-kpi-value'>{_html.escape(str(it['value']))}</div>{delta_html}{sub_html}</div>"
        )
    st.markdown(f"<div class='ss-kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def hero(eyebrow: str, title: str, sub: str, chips: list[str] | None = None) -> None:
    """Gradient hero panel with optional glass chips (landing views)."""
    import html as _html

    chip_html = ""
    if chips:
        chip_html = ("<div class='ss-hero-chips'>"
                     + "".join(f"<span class='ss-hero-chip'>{_html.escape(c)}</span>" for c in chips)
                     + "</div>")
    st.markdown(
        f"<div class='ss-hero'><div class='ss-hero-eyebrow'>{_html.escape(eyebrow)}</div>"
        f"<div class='ss-hero-title'>{_html.escape(title)}</div>"
        f"<div class='ss-hero-sub'>{_html.escape(sub)}</div>{chip_html}</div>",
        unsafe_allow_html=True,
    )


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
        # Streamlit only reruns a text box on Enter/blur, so the placeholder says so —
        # the built-in "Press Enter to apply" overlay is hidden (it collided with this text).
        placeholder="AAPL, Apple, Tata… then Enter",
        help="Type a symbol or a company name, then press Enter to search.",
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
def empty_state(what: str, action: str = "Run an analysis from the sidebar") -> None:
    """Explain what a view will show, instead of leaving it blank.

    `action` is the step that fills the page — pages with their own controls (Screener,
    Compare) pass their own instead of the sidebar default.
    """
    st.markdown(
        f"<div class='ss-empty'><h3>Nothing to show yet</h3>"
        f"<p>{action}, and this page will show {what}.</p></div>",
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
    light, dark, glow = _ACTION_STYLE.get(action, _ACTION_FALLBACK)
    st.markdown(
        f"<div class='ss-verdict' style='background:linear-gradient(135deg,{light},{dark});"
        f"box-shadow:0 8px 24px {glow}'>"
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
    infos = universe.list_indices(region)
    total_stocks = sum(i.count for i in infos)
    hero(
        "Explore markets",
        f"{_REGION_LABEL.get(region, region)} — pick an index, get a grounded call",
        "Choose a stock below or search by company name in the sidebar. Every analysis "
        "combines an ML forecast, news sentiment and a risk profile into one recommendation "
        "— with the model graded honestly against a naive baseline.",
        chips=[f"{len(infos)} indices", f"{total_stocks} stocks",
               "Forecast + sentiment + risk", "Not financial advice"],
    )
    st.write("")
    ncol = 4
    for i in range(0, len(infos), ncol):
        cols = st.columns(ncol)
        for col, info in zip(cols, infos[i:i + ncol]):
            suffix = "" if info.full else " · curated subset"
            # Drop parentheticals like "(large-cap subset)" from the tile face — the count
            # line already says "curated subset", and long names force ugly wrapping.
            name = info.name.split(" (")[0]
            # Markdown in the label: the tile CSS (`st-key-idx_*`) styles the two paragraphs
            # separately — bold display name on top, muted count line below.
            if col.button(f"**{name}**\n\n{info.count} stocks{suffix}",
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
        section("Recent analyses", "most recent first")
        st.dataframe(
            pd.DataFrame(rows), width="stretch", hide_index=True,
            column_config={
                "when": st.column_config.TextColumn("When", width="medium"),
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "company": st.column_config.TextColumn("Company"),
                "action": st.column_config.TextColumn("Verdict", width="small"),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", format="percent", min_value=0.0, max_value=1.0),
                "last_close": st.column_config.NumberColumn("Last close", format="%.2f"),
                "next_day": st.column_config.NumberColumn("Predicted next day", format="%.2f"),
                "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "beats_baseline": st.column_config.CheckboxColumn(
                    "Beat baseline", help="Did the model beat a naive 'tomorrow = today' forecast?"),
            },
        )
