"""Recommendation — the verdict, the reasoning, and the evidence it was built from."""
import _shared
import streamlit as st

_shared.page_header("Recommendation",
                    "Every factor below is tied to a computed number or a cited article.")

r = _shared.require_result("a Buy / Hold / Sell call with the reasoning behind it")
if r and r.recommendation:
    reco = r.recommendation
    _shared.action_badge(reco.action, reco.confidence)
    st.write("")

    st.subheader("Why buy, why not")
    st.write(reco.thesis)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Positive factors")
        for f in reco.positive_factors or ["—"]:
            st.markdown(f"- {f}")
        st.markdown("#### Opportunities")
        for f in reco.opportunities or ["—"]:
            st.markdown(f"- {f}")
    with c2:
        st.markdown("#### Negative factors")
        for f in reco.negative_factors or ["—"]:
            st.markdown(f"- {f}")
        st.markdown("#### Risks")
        for f in reco.risks or ["—"]:
            st.markdown(f"- {f}")

    st.subheader("Sources")
    for a in r.news.top_articles[:6]:
        st.markdown(f"{_shared.tone_dot(a.sentiment_label)} &nbsp; [{a.title}]({a.url}) "
                    f"<span class='ss-muted'>— {a.source}</span>", unsafe_allow_html=True)

    st.divider()
    _shared.report_downloads(r)
    st.error(reco.disclaimer)
elif r:
    st.info("No recommendation was produced for this analysis.")
