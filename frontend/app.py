"""StockSense AI — application entry point.

Owns page config, the stylesheet, the sidebar, and the grouped navigation. The views
under `views/` render content only; they never touch page config.

Run from the project root:  ./run.sh   (or: streamlit run frontend/app.py)
"""
import _shared
import streamlit as st

_shared.boot()

_NAV = {
    "Analysis": [
        st.Page("views/dashboard.py", title="Dashboard",
                icon=":material/dashboard:", default=True),
        st.Page("views/forecast.py", title="Forecast", icon=":material/insights:"),
        st.Page("views/technical.py", title="Technical", icon=":material/candlestick_chart:"),
    ],
    "News": [
        st.Page("views/news.py", title="News", icon=":material/newspaper:"),
        st.Page("views/sentiment.py", title="Sentiment", icon=":material/psychology:"),
    ],
    "Decision": [
        st.Page("views/recommendation.py", title="Recommendation", icon=":material/lightbulb:"),
        st.Page("views/risk.py", title="Risk", icon=":material/shield:"),
    ],
    "Discover": [
        st.Page("views/screener.py", title="Screener", icon=":material/leaderboard:"),
        st.Page("views/compare.py", title="Compare", icon=":material/balance:"),
        st.Page("views/watchlist.py", title="Watchlist", icon=":material/star:"),
    ],
    "More": [
        st.Page("views/ask.py", title="Ask", icon=":material/forum:"),
        st.Page("views/track_record.py", title="Track Record", icon=":material/target:"),
        st.Page("views/history.py", title="History", icon=":material/history:"),
    ],
}

page = st.navigation(_NAV)
_shared.sidebar()
page.run()
