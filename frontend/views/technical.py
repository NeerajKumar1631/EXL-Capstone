"""Technical — candlesticks with moving averages, Bollinger bands, RSI and MACD."""
import _shared
import streamlit as st

_shared.page_header("Technical Indicators",
                    "Price action with the indicators that feed the forecast model.",
                    eyebrow="Analysis")

r = _shared.require_result("candlesticks with moving averages, Bollinger bands, RSI and MACD")
if r:
    import math

    from technical_analysis.indicators import compute_indicators
    from visualization import charts

    st.caption(f"{r.company_name} ({r.ticker})")

    # Latest readings, from the same indicator table the chart plots.
    ind = compute_indicators(r.prices).iloc[-1]
    close = float(r.prices["Close"].iloc[-1])
    cards = [{"label": "Last close", "value": f"${close:,.2f}"}]

    rsi = float(ind.get("rsi_14", float("nan")))
    if not math.isnan(rsi):
        state, tone = (("Overbought", "neg") if rsi > 70
                       else ("Oversold", "neg") if rsi < 30 else ("Neutral", "muted"))
        cards.append({"label": "RSI (14)", "value": f"{rsi:.0f}", "delta": state, "tone": tone,
                      "sub": "above 70 overbought · below 30 oversold"})

    sma50 = float(ind.get("sma_50", float("nan")))
    if not math.isnan(sma50) and sma50:
        vs = close / sma50 - 1.0
        cards.append({"label": "Price vs 50-day average", "value": f"{vs*100:+.1f}%",
                      "tone": "pos" if vs > 0 else "neg",
                      "delta": "above trend" if vs > 0 else "below trend"})

    macd_diff = float(ind.get("macd_diff", float("nan")))
    if not math.isnan(macd_diff):
        cards.append({"label": "MACD", "value": "Bullish" if macd_diff > 0 else "Bearish",
                      "delta": f"{macd_diff:+.2f}", "tone": "pos" if macd_diff > 0 else "neg",
                      "sub": "MACD line minus its signal line"})

    _shared.kpi_row(cards)

    _shared.section("Price & indicators", "candlesticks · SMA/EMA · Bollinger · RSI · MACD")
    st.plotly_chart(charts.technical_chart(r.prices), width="stretch")

    with st.expander("What these indicators mean"):
        st.markdown(
            "- **SMA / EMA** — trend direction. Price above a rising average is bullish.\n"
            "- **Bollinger Bands** — a volatility envelope. Touching a band suggests the move "
            "is stretched.\n"
            "- **RSI (14)** — momentum. Above 70 is overbought, below 30 is oversold.\n"
            "- **MACD** — momentum crossovers. MACD above its signal line is bullish."
        )
