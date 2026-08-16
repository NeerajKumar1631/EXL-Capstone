"""Dashboard — the headline view: verdict, key numbers, price chart, thesis, headlines."""
import _shared
import streamlit as st

r = _shared.current_result()

if r is None or r.forecast is None:
    _shared.page_header("Stock Analysis Dashboard",
                        "Forecast, news sentiment and risk, combined into one grounded call.",
                        eyebrow="Analysis")
    if r is not None:
        _shared.show_problems(r)
    _shared.explore()
else:
    from visualization import charts

    _shared.show_problems(r)

    fc, reco, news = r.forecast, r.recommendation, r.news
    nd = fc.ensemble.next_day

    _shared.page_header(f"{r.company_name}",
                        f"{r.ticker} · analysis as of {r.as_of.strftime('%d %b %Y, %H:%M')}",
                        eyebrow="Analysis")

    if reco:
        _shared.action_badge(reco.action, reco.confidence)
        st.write("")

    move = nd.predicted_return * 100
    sent = news.sentiment
    _shared.kpi_row([
        {"label": "Last close", "value": f"${fc.last_close:,.2f}"},
        {"label": "Next-day (ensemble)", "value": f"${nd.predicted_price:,.2f}",
         "delta": f"{move:+.2f}%",
         "sub": (f"80% range ${nd.lower:,.2f} – ${nd.upper:,.2f}"
                 if nd.lower is not None and nd.upper is not None else None)},
        {"label": "Directional accuracy",
         "value": f"{fc.ensemble.metrics.directional_accuracy*100:.0f}%",
         "delta": "beats baseline" if fc.beats_baseline else "≈ coin-flip",
         "tone": "pos" if fc.beats_baseline else "muted"},
        {"label": "News sentiment", "value": sent.label.title(),
         "delta": f"{sent.weighted_score:+.2f}",
         "tone": ("pos" if sent.label == "positive"
                  else "neg" if sent.label == "negative" else "muted"),
         "sub": f"{sent.n_articles} articles, credibility-weighted"},
    ])

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
        _shared.section("Why buy, why not")
        with st.container(border=True):
            st.write(reco.thesis)

    _shared.section("Latest headlines")
    for a in news.top_articles[:4]:
        st.markdown(f"{_shared.tone_dot(a.sentiment_label)} &nbsp; [{a.title}]({a.url}) "
                    f"<span class='ss-muted'>— {a.source}</span>", unsafe_allow_html=True)

    st.divider()
    cta1, cta2 = st.columns([1, 2])
    with cta1:
        _shared.watch_button(r.ticker, st.session_state.get("region", "US"))
    with cta2:
        _shared.report_downloads(r)
