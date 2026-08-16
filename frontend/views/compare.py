"""Compare — two or three stocks side by side, with a rebased price overlay."""
import _shared
import streamlit as st

_shared.page_header("Compare Stocks",
                    "Up to three stocks side by side on forecast, sentiment and risk.")

default = st.session_state.get("ticker", "AAPL")
c1, c2 = st.columns([3, 1])
tickers_str = c1.text_input("Symbols, comma separated (up to 3)", value=f"{default}, MSFT")
use_llm = c2.toggle("Gemini reasoning", value=False, help="Slower, and uses API quota")

if st.button("Compare", type="primary"):
    from compare.compare import compare

    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    with st.status("Analyzing…", expanded=True) as status:
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
        st.warning(f"{bad.ticker} could not be analyzed: {bad.error}")

    ok = [i for i in cmp.items if i.ok]
    if ok:
        st.subheader("Price, rebased to 100")
        st.caption("Every stock starts at 100, so the lines show relative performance.")
        st.plotly_chart(charts.compare_prices_chart(ok), width="stretch")

        st.subheader("Side by side")
        rows = [{
            "Ticker": i.ticker, "Company": i.company, "Verdict": i.action or "—",
            "Confidence": i.confidence, "Last close": i.last_close,
            "Next-day": i.next_day_return, "Directional accuracy": i.directional_accuracy,
            "Beats baseline": i.beats_baseline, "Sentiment": i.sentiment_label.title(),
            "Volatility": i.annual_volatility, "Max drawdown": i.max_drawdown,
            "Beta": i.beta,
        } for i in ok]

        st.dataframe(
            pd.DataFrame(rows), width="stretch", hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    format="percent", min_value=0.0, max_value=1.0),
                "Last close": st.column_config.NumberColumn(format="%.2f"),
                "Next-day": st.column_config.NumberColumn(format="percent"),
                "Directional accuracy": st.column_config.NumberColumn(format="percent"),
                "Beats baseline": st.column_config.CheckboxColumn(
                    help="Does the model beat a naive 'tomorrow = today' forecast?"),
                "Volatility": st.column_config.NumberColumn(format="percent", help="Annualized"),
                "Max drawdown": st.column_config.NumberColumn(format="percent"),
                "Beta": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    st.caption("Educational comparison only — not financial advice.")
else:
    st.info("Enter two or three symbols above and run a comparison.")
