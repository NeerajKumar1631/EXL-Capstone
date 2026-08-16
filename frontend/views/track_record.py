"""Track record — how the app's own past predictions actually turned out."""
import _shared
import streamlit as st

from analytics.track_record import evaluate

_shared.page_header("Track Record",
                    "Every saved prediction, checked against what the price actually did.")

with st.spinner("Checking past predictions against real prices…"):
    rec = evaluate()

if not rec.total_runs:
    _shared.empty_state("a running tally of how accurate its own predictions have been")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predictions scored", rec.n_graded)
    c2.metric("Direction correct", rec.n_correct)
    c3.metric("Hit rate", "—" if rec.hit_rate is None else f"{rec.hit_rate*100:.0f}%",
              help="A coin-flip is 50%. Daily direction is close to random.")
    c4.metric("Avg price error", "—" if rec.mean_abs_pct_error is None
              else f"{rec.mean_abs_pct_error*100:.2f}%",
              help="Mean absolute difference between the predicted and actual next-day close")

    if rec.n_graded == 0 and rec.pending:
        st.info(
            f"**Nothing can be scored yet.** All {rec.pending} saved prediction(s) were made "
            f"today, and a next-day forecast can only be graded once that day has closed. "
            f"Check back after the next trading session — the tally builds itself from here."
        )

    if rec.hit_rate is not None:
        if rec.n_graded < 20:
            st.warning(
                f"Only **{rec.n_graded}** prediction(s) have been scored. That is far too small "
                f"a sample to say anything about skill — a 100% or 0% hit rate here is noise. "
                f"Keep running analyses and this becomes meaningful."
            )
        elif rec.hit_rate > 0.55:
            st.success(f"Direction called correctly {rec.hit_rate*100:.0f}% of the time across "
                       f"{rec.n_graded} predictions — better than a coin-flip on this sample.")
        elif rec.hit_rate < 0.45:
            st.error(f"Direction called correctly only {rec.hit_rate*100:.0f}% of the time across "
                     f"{rec.n_graded} predictions — worse than a coin-flip on this sample.")
        else:
            st.info(f"Direction called correctly {rec.hit_rate*100:.0f}% of the time across "
                    f"{rec.n_graded} predictions — indistinguishable from a coin-flip, which is "
                    f"what the holdout metrics predict for daily moves.")

    for n in rec.notes:
        st.caption(n)

    if rec.graded:
        import pandas as pd

        st.subheader("Every scored prediction")
        rows = [{
            "When": g.when, "Ticker": g.ticker, "Verdict": g.action,
            "Predicted": g.predicted_price, "Actual": g.actual_price,
            "Predicted move": g.predicted_return, "Actual move": g.actual_return,
            "Direction right": g.direction_correct, "Price error": g.abs_pct_error,
        } for g in rec.graded]

        st.dataframe(
            pd.DataFrame(rows), width="stretch", hide_index=True,
            column_config={
                "Predicted": st.column_config.NumberColumn(format="%.2f"),
                "Actual": st.column_config.NumberColumn(format="%.2f"),
                "Predicted move": st.column_config.NumberColumn(format="percent"),
                "Actual move": st.column_config.NumberColumn(format="percent"),
                "Direction right": st.column_config.CheckboxColumn(),
                "Price error": st.column_config.NumberColumn(
                    format="percent", help="How far the predicted close was from the real one"),
            },
        )

    st.caption("This measures live predictions, unlike the Forecast page, which measures a "
               "held-out backtest. Not financial advice.")
