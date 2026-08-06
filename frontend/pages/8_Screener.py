import _shared
import streamlit as st

_shared.setup("Screener", "🏆")
st.title("🏆 Index Screener")

from config import universe

region = st.session_state.get("region", "US")
st.caption(f"Rank the constituents of an index by a fast momentum + trend + low-volatility score. "
           f"Market: {region}")

indices = universe.list_indices(region)
labels = {i.key: f"{i.name} ({i.count})" for i in indices}
index_key = st.selectbox("Index", [i.key for i in indices], format_func=lambda k: labels[k])

if st.button("Run screener ▶", type="primary"):
    from screener.screener import screen

    with st.status("Scoring constituents…", expanded=True) as status:
        lb = screen(region, index_key, progress=lambda m: status.write(m))
        status.update(label=f"Scored {lb.scored}/{lb.requested}", state="complete")
    st.session_state["leaderboard"] = lb

lb = st.session_state.get("leaderboard")
if lb and lb.region == region:
    import pandas as pd

    cov = f"Scored {lb.scored}/{lb.requested}"
    if lb.failed:
        cov += f" · {lb.failed} unavailable"
    if lb.capped:
        cov += f" · showing top {lb.requested} of {lb.total_constituents} (capped)"
    st.caption(cov)

    rows = [{
        "Rank": i + 1, "Ticker": c.ticker, "Composite": c.composite,
        "Momentum": c.momentum, "Trend": c.trend, "Low-vol": c.low_vol,
        "3m %": round(c.ret_3m * 100, 1), "RSI": c.rsi,
        "Vol %": round(c.annual_volatility * 100, 1), "Close": c.last_close,
    } for i, c in enumerate(lb.cards)]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if lb.cards:
        pick = st.selectbox("Analyze a stock from the leaderboard", [c.ticker for c in lb.cards])
        if st.button(f"Full analysis of {pick} ▶", type="primary"):
            _shared.analyze_ticker(pick)
            st.success(f"Analyzed {pick}. Open the **Dashboard** page to view the full report.")
