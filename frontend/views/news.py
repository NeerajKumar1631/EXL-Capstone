"""News — the generated summary plus the ranked articles behind it."""
import _shared
import streamlit as st

_shared.page_header("News", "What was published recently, and what it adds up to.",
                    eyebrow="News")

r = _shared.require_result("a news summary and the articles it was built from")
if r:
    news = r.news

    _shared.section("Summary")
    st.write(news.summary)
    st.caption(f"{news.n_collected} collected · {news.n_after_dedup} after removing duplicates · "
               f"{len(news.top_articles)} ranked by relevance")

    _shared.section("Articles", "ranked by relevance")
    for a in news.top_articles:
        with st.container(border=True):
            head, meta = st.columns([4, 1])
            when = a.published_at.strftime("%d %b %Y") if a.published_at else "date unknown"
            head.markdown(f"**[{a.title}]({a.url})**  \n"
                          f"<span class='ss-muted'>{a.source} · {when}</span>",
                          unsafe_allow_html=True)
            meta.markdown(f"{_shared.tone_dot(a.sentiment_label)}  \n"
                          f"<span class='ss-muted'>relevance {a.relevance_score:.2f}</span>",
                          unsafe_allow_html=True)
            if a.snippet:
                st.caption(a.snippet[:280] + ("…" if len(a.snippet) > 280 else ""))
