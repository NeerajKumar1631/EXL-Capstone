"""Risk — volatility, drawdown, beta, value at risk, and past shocks."""
import _shared
import streamlit as st

_shared.page_header("Risk & History",
                    "How volatile this stock is, and how badly it has fallen before.",
                    eyebrow="Decision")

r = _shared.require_result("volatility, drawdown, beta, value at risk and past shocks")
if r and r.risk:
    import pandas as pd

    from visualization import charts

    rk = r.risk
    st.caption(f"{r.company_name} ({r.ticker})")

    _shared.kpi_row([
        {"label": "Annualized volatility", "value": f"{rk.annual_volatility*100:.1f}%"},
        {"label": "Max drawdown", "value": f"{rk.max_drawdown*100:.1f}%", "tone": "neg",
         "sub": f"peak {rk.drawdown_peak} → trough {rk.drawdown_trough}"},
        {"label": "Beta", "value": "—" if rk.beta is None else f"{rk.beta:.2f}",
         "sub": f"against {rk.benchmark}" if rk.benchmark else None},
        {"label": "Sharpe-like",
         "value": "—" if rk.sharpe_like is None else f"{rk.sharpe_like:.2f}",
         "sub": "annualized mean / volatility, rf = 0"},
    ])
    _shared.kpi_row([
        {"label": "1-day VaR (95%)", "value": f"{rk.var_95*100:.2f}%",
         "sub": "on 95% of days the loss was no worse than this"},
        {"label": "1-day VaR (99%)", "value": f"{rk.var_99*100:.2f}%"},
        {"label": "52-week range",
         "value": f"${rk.week52_low:,.0f} – ${rk.week52_high:,.0f}"},
        {"label": "Position in range", "value": f"{rk.price_position_52w*100:.0f}%",
         "sub": "0% is the 52-week low, 100% is the high"},
    ])

    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.drawdown_chart(rk), width="stretch")
    with right:
        st.plotly_chart(charts.rolling_vol_chart(rk), width="stretch")

    _shared.section("Biggest single-day moves", "history")
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
