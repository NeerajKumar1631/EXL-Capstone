"""Dashboard — the headline view: verdict, key numbers, price chart, thesis, headlines."""
import _shared
import streamlit as st

r = _shared.current_result()

if r is None or r.forecast is None:
    _shared.page_header("Stock Analysis Dashboard",
                        "Forecast, news sentiment and risk, combined into one grounded call.")
    if r is not None:
        _shared.show_problems(r)
    _shared.explore()
else:
    from visualization import charts

    _shared.show_problems(r)

    fc, reco, news = r.forecast, r.recommendation, r.news
    nd = fc.ensemble.next_day

    _shared.page_header(f"{r.company_name}",
                        f"{r.ticker} · analysis as of {r.as_of.strftime('%d %b %Y, %H:%M')}")

    if reco:
        _shared.action_badge(reco.action, reco.confidence)
        st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close", f"${fc.last_close:,.2f}")
    c2.metric("Next-day (ensemble)", f"${nd.predicted_price:,.2f}", f"{nd.predicted_return*100:+.2f}%")
    c3.metric("Directional accuracy", f"{fc.ensemble.metrics.directional_accuracy*100:.0f}%")
    c4.metric("News sentiment", news.sentiment.label.title(), f"{news.sentiment.weighted_score:+.2f}")

    if not fc.beats_baseline:
        st.warning(
            "The price model does **not** beat a naive 'tomorrow = today' baseline on the recent "
            "holdout. Daily direction is close to random, so treat the point forecast as "
            "low-confidence and lean on the news and fundamentals."
        )

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(charts.price_and_forecast(r.prices, fc), width="stretch")
    with right:
        st.plotly_chart(charts.sentiment_gauge(news.sentiment), width="stretch")

    if reco:
        st.subheader("Why buy, why not")
        st.write(reco.thesis)

    st.subheader("Latest headlines")
    for a in news.top_articles[:4]:
        st.markdown(f"{_shared.tone_dot(a.sentiment_label)} &nbsp; [{a.title}]({a.url}) "
                    f"<span class='ss-muted'>— {a.source}</span>", unsafe_allow_html=True)

    st.divider()
    cta1, cta2 = st.columns([1, 2])
    with cta1:
        _shared.watch_button(r.ticker, st.session_state.get("region", "US"))
    with cta2:
        _shared.report_downloads(r)
