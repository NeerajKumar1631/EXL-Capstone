import _shared
import streamlit as st

_shared.setup("Recommendation", "💡")
st.title("💡 Recommendation")

r = _shared.require_result()
if r and r.recommendation:
    reco = r.recommendation
    _shared.action_badge(reco.action, reco.confidence)
    st.write("")

    st.subheader("Why buy / why not")
    st.write(reco.thesis)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ✅ Positive factors")
        for f in reco.positive_factors or ["—"]:
            st.markdown(f"- {f}")
        st.markdown("#### 🚀 Opportunities")
        for f in reco.opportunities or ["—"]:
            st.markdown(f"- {f}")
    with c2:
        st.markdown("#### ⚠️ Negative factors")
        for f in reco.negative_factors or ["—"]:
            st.markdown(f"- {f}")
        st.markdown("#### 🛑 Risks")
        for f in reco.risks or ["—"]:
            st.markdown(f"- {f}")

    st.subheader("🔗 Evidence (sources)")
    for a in r.news.top_articles[:6]:
        st.markdown(f"- [{a.title}]({a.url}) — *{a.source}* ({a.sentiment_label or 'n/a'})")

    st.divider()
    _shared.report_downloads(r)
    st.error("⚠️ " + reco.disclaimer)
elif r:
    st.info("No recommendation was produced (the forecast may have been unavailable).")
