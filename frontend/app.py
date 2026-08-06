"""StockSense AI — Dashboard (entry point).

Run from the project root:  streamlit run frontend/app.py
"""
import _shared
import streamlit as st

_shared.setup("Dashboard", "📊")

st.title("📊 Stock Prediction Dashboard")
st.caption("AI-powered insights for your investments — forecast + news sentiment + a grounded call.")

r = _shared.current_result()
if r is None or r.forecast is None:
    if r is not None and r.errors:
        st.error(" · ".join(r.errors))
    _shared.explore()
else:
    from visualization import charts

    for w in r.warnings:
        st.warning(w)

    fc, reco, news = r.forecast, r.recommendation, r.news
    nd = fc.ensemble.next_day

    st.subheader(f"{r.company_name} ({r.ticker})")
    if reco:
        _shared.action_badge(reco.action, reco.confidence)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close", f"${fc.last_close:,.2f}")
    c2.metric("Next-day (ensemble)", f"${nd.predicted_price:,.2f}", f"{nd.predicted_return*100:+.2f}%")
    c3.metric("Directional accuracy", f"{fc.ensemble.metrics.directional_accuracy*100:.0f}%")
    c4.metric("News sentiment", news.sentiment.label.title(), f"{news.sentiment.weighted_score:+.2f}")

    if not fc.beats_baseline:
        st.warning(
            "⚠️ The price model does **not** beat a naive 'tomorrow = today' baseline on the recent "
            "holdout — daily direction is near-random, so treat the point forecast as low-confidence "
            "and lean on the news/fundamentals."
        )

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(charts.price_and_forecast(r.prices, fc), width="stretch")
    with right:
        st.plotly_chart(charts.sentiment_gauge(news.sentiment), width="stretch")

    if reco:
        st.subheader("💡 Why buy / why not")
        st.write(reco.thesis)

    st.subheader("📰 Latest headlines")
    for a in news.top_articles[:4]:
        tone = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(a.sentiment_label, "⚪")
        st.markdown(f"{tone} [{a.title}]({a.url}) — *{a.source}*")

    st.divider()
    cta1, cta2 = st.columns([1, 2])
    with cta1:
        _shared.watch_button(r.ticker, st.session_state.get("region", "US"))
    with cta2:
        _shared.report_downloads(r)

    st.caption("Use the sidebar pages for Forecast, Technical, News, Sentiment, Risk, Screener, Compare & Ask.")
