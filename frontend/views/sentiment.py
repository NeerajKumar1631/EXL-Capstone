"""Sentiment — FinBERT scores per article, weighted by source credibility."""
import _shared
import streamlit as st

_shared.page_header("News Sentiment",
                    "Each article scored by FinBERT, then weighted by how reliable the source is.")

r = _shared.require_result("a sentiment score for every article, and the overall reading")
if r:
    import pandas as pd

    from visualization import charts

    s = r.news.sentiment
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(charts.sentiment_gauge(s), width="stretch")
    with right:
        st.metric("Overall", s.label.title(), f"{s.weighted_score:+.3f}")
        st.write(f"Across **{s.n_articles}** articles, weighted by source credibility:")
        st.markdown(
            f"{_shared.tone_dot('positive')} {s.n_positive} &nbsp;&nbsp; "
            f"{_shared.tone_dot('negative')} {s.n_negative} &nbsp;&nbsp; "
            f"{_shared.tone_dot('neutral')} {s.n_neutral}",
            unsafe_allow_html=True,
        )

    st.subheader("Per-article scores")
    rows = [{
        "Article": a.title[:70],
        "Source": a.source,
        "Sentiment": (a.sentiment_label or "n/a").title(),
        "Confidence": a.sentiment_confidence,
        "Signed score": a.sentiment_score,
        "Credibility": a.credibility,
    } for a in r.news.top_articles]

    if rows:
        st.dataframe(
            pd.DataFrame(rows), width="stretch", hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    format="percent", min_value=0.0, max_value=1.0,
                    help="How sure FinBERT is of the label"),
                "Signed score": st.column_config.NumberColumn(
                    format="%+.2f", help="+confidence positive, −confidence negative, 0 neutral"),
                "Credibility": st.column_config.ProgressColumn(
                    format="%.2f", min_value=0.0, max_value=1.0,
                    help="Editorial reliability prior for this source"),
            },
        )
    st.caption("The overall score is the average signed score, weighted by source credibility.")
