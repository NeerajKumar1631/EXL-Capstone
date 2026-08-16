"""Screener — rank a whole index by a fast, LLM-free composite score."""
import _shared
import streamlit as st

from config import universe

_shared.page_header("Index Screener",
                    "Rank a whole index on momentum, trend and calmness. No LLM, so it is fast.",
                    eyebrow="Discover")

region = st.session_state.get("region", "US")
indices = universe.list_indices(region)
labels = {i.key: f"{i.name} ({i.count})" for i in indices}

c1, c2 = st.columns([3, 1])
index_key = c1.selectbox("Index", [i.key for i in indices], format_func=lambda k: labels[k])
run = c2.button("Run screener", type="primary", width="stretch")

if run:
    from screener.screener import screen

    with st.status("Scoring constituents…", expanded=True) as status:
        lb = screen(region, index_key, progress=lambda m: status.write(m))
        status.update(label=f"Scored {lb.scored} of {lb.requested}", state="complete")
    st.session_state["leaderboard"] = lb

lb = st.session_state.get("leaderboard")
if lb and lb.region == region:
    import pandas as pd

    cov = f"Scored {lb.scored} of {lb.requested}"
    if lb.failed:
        cov += f" · {lb.failed} unavailable"
    if lb.capped:
        cov += f" · showing the first {lb.requested} of {lb.total_constituents}"
    _shared.section("Leaderboard", cov)

    rows = [{
        "Rank": i + 1, "Ticker": c.ticker, "Composite": c.composite,
        "Momentum": c.momentum, "Trend": c.trend, "Calmness": c.low_vol,
        "3-month": c.ret_3m, "RSI": c.rsi, "Volatility": c.annual_volatility,
        "Close": c.last_close,
    } for i, c in enumerate(lb.cards)]

    st.dataframe(
        pd.DataFrame(rows), width="stretch", hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Composite": st.column_config.ProgressColumn(
                format="%.1f", min_value=0.0, max_value=100.0,
                help="0.45 momentum + 0.30 trend + 0.25 calmness"),
            "Momentum": st.column_config.NumberColumn(format="%.1f"),
            "Trend": st.column_config.NumberColumn(format="%.1f"),
            "Calmness": st.column_config.NumberColumn(format="%.1f", help="Higher means less volatile"),
            "3-month": st.column_config.NumberColumn(format="percent"),
            "RSI": st.column_config.NumberColumn(format="%.0f"),
            "Volatility": st.column_config.NumberColumn(format="percent", help="Annualized"),
            "Close": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    if lb.cards:
        st.divider()
        p1, p2 = st.columns([3, 1])
        pick = p1.selectbox("Analyze one of these", [c.ticker for c in lb.cards])
        if p2.button("Full analysis", type="primary", width="stretch"):
            _shared.analyze_ticker(pick)
            st.success(f"Analyzed {pick}. Open **Dashboard** to see the full report.")
else:
    _shared.empty_state("a ranked leaderboard of every stock in that index — "
                        "momentum, trend and calmness scored out of 100",
                        action="Choose an index above and run the screener")
