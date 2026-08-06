import _shared
import streamlit as st

_shared.setup("Technical", "📊")
st.title("📊 Technical Indicators")

r = _shared.require_result()
if r:
    from visualization import charts

    st.caption(f"{r.company_name} ({r.ticker}) — candlesticks with SMA/EMA/Bollinger, RSI and MACD.")
    st.plotly_chart(charts.technical_chart(r.prices), width="stretch")
    with st.expander("What these indicators mean"):
        st.markdown(
            "- **SMA/EMA** — trend direction; price above rising averages is bullish.\n"
            "- **Bollinger Bands** — volatility envelope; touches of the bands can signal stretch.\n"
            "- **RSI (14)** — momentum; >70 overbought, <30 oversold.\n"
            "- **MACD** — momentum crossovers; MACD above signal is bullish."
        )
