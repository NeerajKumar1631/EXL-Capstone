"""Sentiment — FinBERT scores per article, weighted by source credibility."""
import _shared
import streamlit as st

_shared.page_header("News Sentiment",
                    "Each article scored by FinBERT, then weighted by how reliable the source is.",
                    eyebrow="News")

r = _shared.require_result("a sentiment score for every article, and the overall reading")
if r:
    import pandas as pd

    from visualization import charts

    s = r.news.sentiment
    tone = ("pos" if s.label == "positive" else "neg" if s.label == "negative" else "muted")
    _shared.kpi_row([
        {"label": "Overall tone", "value": s.label.title(),
         "delta": f"{s.weighted_score:+.3f}", "tone": tone,
         "sub": "credibility-weighted mean of signed scores"},
        {"label": "Articles scored", "value": str(s.n_articles),
         "sub": "after de-duplication and relevance ranking"},
        {"label": "Mix", "value": f"{s.n_positive} · {s.n_negative} · {s.n_neutral}",
         "sub": "positive · negative · neutral"},
    ])

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(charts.sentiment_gauge(s), width="stretch")
    with right:
        with st.container(border=True):
            st.markdown("**How to read this**")
            st.markdown(
                f"- {_shared.tone_dot('positive')} above **+0.05** reads as positive\n"
                f"- {_shared.tone_dot('neutral')} between −0.05 and +0.05 is neutral\n"
                f"- {_shared.tone_dot('negative')} below **−0.05** reads as negative",
                unsafe_allow_html=True,
            )
            st.caption("Each article's signed score is +confidence when FinBERT says positive, "
                       "−confidence when negative, 0 when neutral. Reliable sources (Reuters, "
                       "Bloomberg…) count for more than blogs.")

    _shared.section("Per-article scores")
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
