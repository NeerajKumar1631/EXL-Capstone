import _shared
import streamlit as st

_shared.setup("News", "📰")
st.title("📰 News")

r = _shared.require_result()
if r:
    news = r.news
    st.subheader("AI summary")
    st.write(news.summary)
    st.caption(f"{news.n_collected} collected → {news.n_after_dedup} after dedup → "
               f"{len(news.top_articles)} ranked by relevance")

    tone = {"positive": "🟢 Positive", "negative": "🔴 Negative", "neutral": "⚪ Neutral"}
    for a in news.top_articles:
        with st.container(border=True):
            top = st.columns([4, 1])
            when = a.published_at.strftime("%Y-%m-%d") if a.published_at else "n/a"
            top[0].markdown(f"**[{a.title}]({a.url})**  \n*{a.source} · {when}*")
            top[1].markdown(f"{tone.get(a.sentiment_label, '⚪ n/a')}  \n"
                            f"rel {a.relevance_score:.2f}")
            if a.snippet:
                st.caption(a.snippet[:280] + ("…" if len(a.snippet) > 280 else ""))
