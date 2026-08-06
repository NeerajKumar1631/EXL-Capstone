import _shared
import streamlit as st

_shared.setup("Sentiment", "🧠")
st.title("🧠 News Sentiment (FinBERT)")

r = _shared.require_result()
if r:
    import pandas as pd

    from visualization import charts

    s = r.news.sentiment
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(charts.sentiment_gauge(s), width="stretch")
    with right:
        st.metric("Overall", s.label.title(), f"{s.weighted_score:+.3f}")
        st.write(f"Across **{s.n_articles}** articles (credibility-weighted):")
        st.write(f"🟢 {s.n_positive} positive · 🔴 {s.n_negative} negative · ⚪ {s.n_neutral} neutral")

    st.subheader("Per-article sentiment")
    rows = [{
        "Article": a.title[:70],
        "Source": a.source,
        "Sentiment": (a.sentiment_label or "n/a").title(),
        "Confidence": round(a.sentiment_confidence, 2),
        "Signed score": round(a.sentiment_score, 2),
        "Credibility": round(a.credibility, 2),
    } for a in r.news.top_articles]
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Signed score = +confidence (positive), −confidence (negative), 0 (neutral). "
               "The overall score weights each article by source credibility.")
