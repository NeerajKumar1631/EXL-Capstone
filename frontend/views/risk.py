"""Risk — volatility, drawdown, beta, value at risk, and past shocks."""
import _shared
import streamlit as st

_shared.page_header("Risk & History",
                    "How volatile this stock is, and how badly it has fallen before.")

r = _shared.require_result("volatility, drawdown, beta, value at risk and past shocks")
if r and r.risk:
    import pandas as pd

    from visualization import charts

    rk = r.risk
    st.caption(f"{r.company_name} ({r.ticker})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualized volatility", f"{rk.annual_volatility*100:.1f}%")
    c2.metric("Max drawdown", f"{rk.max_drawdown*100:.1f}%",
              help=f"Peak {rk.drawdown_peak} to trough {rk.drawdown_trough}")
    c3.metric("Beta", "—" if rk.beta is None else f"{rk.beta:.2f}",
              help=f"Against {rk.benchmark}" if rk.benchmark else None)
    c4.metric("Sharpe-like", "—" if rk.sharpe_like is None else f"{rk.sharpe_like:.2f}",
              help="Annualized mean divided by volatility, risk-free rate assumed 0")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("1-day VaR (95%)", f"{rk.var_95*100:.2f}%",
              help="On 95% of days the loss was no worse than this")
    c6.metric("1-day VaR (99%)", f"{rk.var_99*100:.2f}%")
    c7.metric("52-week range", f"${rk.week52_low:,.0f} – ${rk.week52_high:,.0f}")
    c8.metric("Position in range", f"{rk.price_position_52w*100:.0f}%",
              help="0% is the 52-week low, 100% is the high")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.drawdown_chart(rk), width="stretch")
    with right:
        st.plotly_chart(charts.rolling_vol_chart(rk), width="stretch")

    st.subheader("Biggest single-day moves")
    move_cfg = {"pct": st.column_config.NumberColumn("Move", format="percent"),
                "date": st.column_config.TextColumn("Date")}
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"{_shared.tone_dot('positive')} **Largest gains**", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"date": m.date, "pct": m.pct / 100.0} for m in rk.biggest_up]),
                     width="stretch", hide_index=True, column_config=move_cfg)
    with d2:
        st.markdown(f"{_shared.tone_dot('negative')} **Largest drops**", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"date": m.date, "pct": m.pct / 100.0} for m in rk.biggest_down]),
                     width="stretch", hide_index=True, column_config=move_cfg)

    for n in rk.notes:
        st.caption(n)
elif r:
    st.info("Risk metrics were not available for this analysis.")
