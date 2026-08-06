import _shared
import streamlit as st

_shared.setup("Compare", "⚖️")
st.title("⚖️ Compare Stocks")
st.caption("Put 2–3 stocks side-by-side across forecast, sentiment, and risk (rebased overlay).")

default = st.session_state.get("ticker", "AAPL")
tickers_str = st.text_input("Tickers (comma-separated, up to 3)", value=f"{default}, MSFT")
use_llm = st.toggle("Use Gemini reasoning (slower)", value=False)

if st.button("Compare ▶", type="primary"):
    from compare.compare import compare

    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    with st.status("Analyzing tickers…", expanded=True) as status:
        cmp = compare(tickers, use_llm=use_llm)
        status.update(label="Comparison ready", state="complete")
    st.session_state["comparison"] = cmp

cmp = st.session_state.get("comparison")
if cmp:
    import pandas as pd

    from visualization import charts

    for n in cmp.notes:
        st.info(n)
    for bad in [i for i in cmp.items if not i.ok]:
        st.warning(f"{bad.ticker}: {bad.error}")

    ok = [i for i in cmp.items if i.ok]
    if len(ok) >= 1:
        st.plotly_chart(charts.compare_prices_chart(ok), width="stretch")

        rows = [{
            "Ticker": i.ticker, "Company": i.company, "Verdict": i.action or "—",
            "Confidence": f"{i.confidence*100:.0f}%", "Last": round(i.last_close, 2),
            "Next-day %": round(i.next_day_return * 100, 2),
            "Dir. acc %": round(i.directional_accuracy * 100, 0),
            "Beats base": "✅" if i.beats_baseline else "—",
            "Sentiment": i.sentiment_label, "Vol %": round(i.annual_volatility * 100, 1),
            "Max DD %": round(i.max_drawdown * 100, 1),
            "Beta": "—" if i.beta is None else round(i.beta, 2),
        } for i in ok]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("⚠️ Educational comparison only — not financial advice.")
