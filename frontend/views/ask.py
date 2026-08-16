"""Ask — a conversational analyst that calls the real analysis tools."""
import _shared
import streamlit as st

_shared.page_header("Ask the Analyst",
                    "Ask about any stock, compare two, or screen an index. "
                    "Answers are grounded in real data, and this is not financial advice.",
                    eyebrow="More")

st.session_state.setdefault("chat_history", [])

# First visit: offer runnable starter questions instead of an empty void.
seed = None
if not st.session_state["chat_history"]:
    with st.container(border=True):
        st.markdown("**Try one of these**")
        c1, c2, c3 = st.columns(3)
        if c1.button("How risky is Tesla?", width="stretch"):
            seed = "How risky is Tesla?"
        if c2.button("Compare AAPL and MSFT", width="stretch"):
            seed = "Compare AAPL and MSFT"
        if c3.button("Top Nifty 50 stocks", width="stretch"):
            seed = "Top Nifty 50 stocks"

for role, content in st.session_state["chat_history"]:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("How risky is Tesla?  ·  Compare AAPL and MSFT  ·  Top Nifty 50 stocks")
prompt = prompt or seed
if prompt:
    st.session_state["chat_history"].append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        from chat.agent import get_chat_agent

        agent = get_chat_agent()
        with st.spinner("Looking it up…"):
            answer, tools = agent.ask(prompt, st.session_state["chat_history"][:-1])
        st.markdown(answer)
        if tools:
            st.caption("Tools used: " + ", ".join(tools))
        if not agent.available:
            st.caption("Gemini is unavailable, so this used rule-based routing.")
    st.session_state["chat_history"].append(("assistant", answer))
