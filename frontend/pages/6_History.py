import _shared
import streamlit as st

_shared.setup("History", "🗂")
st.title("🗂 Analysis History")

from database.db import recent_runs

rows = recent_runs(50)
if not rows:
    st.info("No past analyses yet. Run one from the sidebar.")
else:
    import pandas as pd

    st.caption("Previously analyzed tickers (persisted to SQLite).")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
