import _shared
import streamlit as st

_shared.setup("Risk & History", "🛡️")
st.title("🛡️ Risk & History")

r = _shared.require_result()
if r and r.risk:
    from visualization import charts

    rk = r.risk
    st.caption(f"{r.company_name} ({r.ticker}) — risk profile and how it behaved in the past.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualized volatility", f"{rk.annual_volatility*100:.1f}%")
    c2.metric("Max drawdown", f"{rk.max_drawdown*100:.1f}%",
              help=f"Peak {rk.drawdown_peak} → trough {rk.drawdown_trough}")
    c3.metric("Beta", "—" if rk.beta is None else f"{rk.beta:.2f}",
              help=f"vs {rk.benchmark}" if rk.benchmark else None)
    c4.metric("Sharpe-like", "—" if rk.sharpe_like is None else f"{rk.sharpe_like:.2f}",
              help="annualized mean/vol, rf=0")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("1-day VaR (95%)", f"{rk.var_95*100:.2f}%")
    c6.metric("1-day VaR (99%)", f"{rk.var_99*100:.2f}%")
    c7.metric("52-week range", f"${rk.week52_low:,.0f}–${rk.week52_high:,.0f}")
    c8.metric("Position in 52w", f"{rk.price_position_52w*100:.0f}%",
              help="0% = at 52-week low, 100% = at high")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.drawdown_chart(rk), width="stretch")
    with right:
        st.plotly_chart(charts.rolling_vol_chart(rk), width="stretch")

    st.subheader("Biggest single-day moves (history)")
    import pandas as pd

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Largest gains** 🟢")
        st.dataframe(pd.DataFrame([m.model_dump() for m in rk.biggest_up]),
                     width="stretch", hide_index=True)
    with d2:
        st.markdown("**Largest drops** 🔴")
        st.dataframe(pd.DataFrame([m.model_dump() for m in rk.biggest_down]),
                     width="stretch", hide_index=True)

    for n in rk.notes:
        st.caption("• " + n)
elif r:
    st.info("Risk metrics were unavailable for this analysis.")
