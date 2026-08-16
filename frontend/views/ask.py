"""Ask — a conversational analyst that calls the real analysis tools."""
import _shared
import streamlit as st

_shared.page_header("Ask the Analyst",
                    "Ask about any stock, compare two, or screen an index. "
                    "Answers are grounded in real data, and this is not financial advice.")

st.session_state.setdefault("chat_history", [])

for role, content in st.session_state["chat_history"]:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("How risky is Tesla?  ·  Compare AAPL and MSFT  ·  Top Nifty 50 stocks")
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
