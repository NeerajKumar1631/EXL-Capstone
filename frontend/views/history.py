"""History — every analysis this installation has produced, newest first."""
import _shared
import streamlit as st

from database.db import recent_runs

_shared.page_header("Analysis History", "Every analysis run on this machine, saved to SQLite.")

rows = recent_runs(50)
if not rows:
    _shared.empty_state("a record of every analysis you run")
else:
    import pandas as pd

    df = pd.DataFrame(rows)
    for col in ("confidence",):
        if col in df:
            df[col] = df[col].astype(float)

    st.dataframe(
        df, width="stretch", hide_index=True,
        column_config={
            "created_at": st.column_config.DatetimeColumn("When", format="DD MMM YYYY, HH:mm"),
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "company": st.column_config.TextColumn("Company"),
            "action": st.column_config.TextColumn("Verdict", width="small"),
            "confidence": st.column_config.ProgressColumn(
                "Confidence", format="percent", min_value=0.0, max_value=1.0),
            "last_close": st.column_config.NumberColumn("Last close", format="%.2f"),
            "next_day_price": st.column_config.NumberColumn("Predicted next day", format="%.2f"),
            "beats_baseline": st.column_config.CheckboxColumn(
                "Beat baseline", help="Did the model beat a naive 'tomorrow = today' forecast?"),
            "best_model": st.column_config.TextColumn("Best model"),
            "sentiment_label": st.column_config.TextColumn("Sentiment"),
            "sentiment_score": st.column_config.NumberColumn("Score", format="%+.2f"),
            "thesis": st.column_config.TextColumn("Thesis", width="large"),
        },
    )
    st.caption(f"{len(rows)} most recent analyses.")
