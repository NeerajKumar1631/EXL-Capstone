"""Watchlist — saved stocks with their latest verdict at a glance."""
import _shared
import streamlit as st

from database.db import latest_run_by_ticker, list_watch, remove_watch

_shared.page_header("Watchlist", "Your saved stocks and the last call made on each.")

saved = list_watch()

if not saved:
    st.markdown(
        "<div class='ss-empty'><h3>No saved stocks yet</h3>"
        "<p>Run an analysis, then use <b>Add to watchlist</b> on the Dashboard. "
        "Saved stocks appear here with their most recent verdict.</p></div>",
        unsafe_allow_html=True,
    )
else:
    latest = latest_run_by_ticker()
    st.caption(f"{len(saved)} saved · verdicts come from the most recent analysis of each stock.")

    for i in range(0, len(saved), 2):
        cols = st.columns(2)
        for col, entry in zip(cols, saved[i:i + 2]):
            ticker = entry["ticker"]
            run = latest.get(ticker)
            with col.container(border=True):
                head, act = st.columns([3, 2])
                head.markdown(f"**{ticker}**  \n"
                              f"<span class='ss-muted'>{(run or {}).get('company') or entry['region']}</span>",
                              unsafe_allow_html=True)
                if run:
                    move = run["next_day_price"] / run["last_close"] - 1.0 if run["last_close"] else 0.0
                    act.metric(run["action"], f"{run['confidence']*100:.0f}%",
                               f"{move*100:+.2f}% next day")
                else:
                    act.markdown("<span class='ss-muted'>Not analyzed yet</span>",
                                 unsafe_allow_html=True)

                if run:
                    st.caption(f"Last analyzed {run['created_at'].strftime('%d %b %Y, %H:%M')} · "
                               f"last close {run['last_close']:,.2f} · "
                               f"{'beat' if run['beats_baseline'] else 'did not beat'} the naive baseline")

                b1, b2 = st.columns(2)
                if b1.button("Analyze", key=f"wl_run_{ticker}", width="stretch"):
                    st.session_state["region"] = entry["region"]
                    _shared.analyze_ticker(ticker)
                    st.rerun()
                if b2.button("Remove", key=f"wl_del_{ticker}", width="stretch"):
                    remove_watch(ticker)
                    st.rerun()

    st.caption("Not financial advice.")
