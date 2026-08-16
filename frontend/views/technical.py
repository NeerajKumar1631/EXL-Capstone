"""Technical — candlesticks with moving averages, Bollinger bands, RSI and MACD."""
import _shared
import streamlit as st

_shared.page_header("Technical Indicators",
                    "Price action with the indicators that feed the forecast model.")

r = _shared.require_result("candlesticks with moving averages, Bollinger bands, RSI and MACD")
if r:
    from visualization import charts

    st.caption(f"{r.company_name} ({r.ticker})")
    st.plotly_chart(charts.technical_chart(r.prices), width="stretch")

    with st.expander("What these indicators mean"):
        st.markdown(
            "- **SMA / EMA** — trend direction. Price above a rising average is bullish.\n"
            "- **Bollinger Bands** — a volatility envelope. Touching a band suggests the move "
            "is stretched.\n"
            "- **RSI (14)** — momentum. Above 70 is overbought, below 30 is oversold.\n"
            "- **MACD** — momentum crossovers. MACD above its signal line is bullish."
        )
