"""Recommendation — the verdict, the reasoning, and the evidence it was built from."""
import _shared
import streamlit as st

_shared.page_header("Recommendation",
                    "Every factor below is tied to a computed number or a cited article.",
                    eyebrow="Decision")

r = _shared.require_result("a Buy / Hold / Sell call with the reasoning behind it")
if r and r.recommendation:
    reco = r.recommendation
    _shared.action_badge(reco.action, reco.confidence)
    st.write("")

    _shared.section("Why buy, why not")
    with st.container(border=True):
        st.write(reco.thesis)

    def _factor_card(col, dot: str, heading: str, items: list[str]) -> None:
        with col, st.container(border=True):
            st.markdown(f"{_shared.tone_dot(dot)} **{heading}**", unsafe_allow_html=True)
            for f in items or ["—"]:
                st.markdown(f"- {f}")

    _shared.section("The case, both ways")
    c1, c2 = st.columns(2)
    _factor_card(c1, "positive", "Positive factors", reco.positive_factors)
    _factor_card(c2, "negative", "Negative factors", reco.negative_factors)
    c3, c4 = st.columns(2)
    _factor_card(c3, "positive", "Opportunities", reco.opportunities)
    _factor_card(c4, "negative", "Risks", reco.risks)

    _shared.section("Sources", "the evidence the call cites")
    for a in r.news.top_articles[:6]:
        st.markdown(f"{_shared.tone_dot(a.sentiment_label)} &nbsp; [{a.title}]({a.url}) "
                    f"<span class='ss-muted'>— {a.source}</span>", unsafe_allow_html=True)

    st.divider()
    _shared.report_downloads(r)
    st.error(reco.disclaimer)
elif r:
    st.info("No recommendation was produced for this analysis.")
