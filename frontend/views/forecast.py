"""Forecast — per-model metrics against the naive baseline, backtest, horizons."""
import _shared
import streamlit as st

_shared.page_header("Forecast",
                    "Every model graded on a held-out window, against a naive "
                    "'tomorrow = today' baseline.")

r = _shared.require_result("the model comparison, backtest and multi-horizon forecast")
if r:
    import pandas as pd

    from visualization import charts

    fc = r.forecast
    ens = fc.ensemble

    if fc.beats_baseline:
        st.success(f"The ensemble beats the naive baseline on the holdout "
                   f"(skill {ens.metrics.skill_vs_baseline:+.3f}).")
    else:
        st.warning("The ensemble does **not** beat the naive baseline. Daily direction is "
                   "essentially a coin-flip, so the horizon prices below are indicative only.")

    st.subheader("Model comparison")
    st.caption("Evaluated on a held-out window with no look-ahead.")
    rows = []
    for m in [*fc.models, ens]:
        mm = m.metrics
        rows.append({
            "Model": m.name, "Weight": m.weight, "RMSE": mm.rmse, "MAE": mm.mae,
            "MAPE": mm.mape / 100.0, "R²": mm.r2,
            "Directional accuracy": mm.directional_accuracy,
            "Skill vs baseline": mm.skill_vs_baseline,
        })
    bm = fc.baseline_metrics
    rows.append({"Model": "Naive baseline", "Weight": None, "RMSE": bm.rmse, "MAE": bm.mae,
                 "MAPE": bm.mape / 100.0, "R²": bm.r2,
                 "Directional accuracy": bm.directional_accuracy, "Skill vs baseline": 0.0})

    st.dataframe(
        pd.DataFrame(rows), width="stretch", hide_index=True,
        column_config={
            "Weight": st.column_config.NumberColumn(format="%.2f", help="Share of the ensemble"),
            "RMSE": st.column_config.NumberColumn(format="%.5f", help="On next-day log returns"),
            "MAE": st.column_config.NumberColumn(format="%.5f"),
            "MAPE": st.column_config.NumberColumn(format="percent", help="On reconstructed prices"),
            "R²": st.column_config.NumberColumn(format="%.3f"),
            "Directional accuracy": st.column_config.ProgressColumn(
                format="percent", min_value=0.0, max_value=1.0,
                help="How often the sign of the move is right. 50% is a coin-flip."),
            "Skill vs baseline": st.column_config.NumberColumn(
                format="%.3f", help="Above 0 beats the naive baseline"),
        },
    )
    st.caption(f"Best model by holdout RMSE: **{fc.best_model}**.")

    st.subheader("Actual vs predicted")
    st.plotly_chart(charts.backtest_actual_vs_pred(fc), width="stretch")

    st.subheader("Multi-horizon forecast (ensemble)")
    if fc.interval_level:
        st.caption(f"Each figure carries an {fc.interval_level:.0%} prediction range, not just a "
                   f"point estimate.")
    cols = st.columns(len(ens.horizons))
    for col, h in zip(cols, ens.horizons):
        col.metric(f"{h.horizon} ({h.horizon_days}d)", f"${h.predicted_price:,.2f}",
                   f"{h.predicted_return*100:+.2f}%")
        if h.lower is not None and h.upper is not None:
            col.caption(f"{fc.interval_level:.0%} range  \n${h.lower:,.2f} – ${h.upper:,.2f}")

    if fc.interval_level and fc.interval_coverage is not None:
        delta = fc.interval_coverage - fc.interval_level
        verdict = ("about right" if abs(delta) <= 0.10
                   else "wider than needed" if delta > 0 else "too narrow")
        st.caption(
            f"Range calibrated by split conformal prediction on {fc.interval_n_calibration} "
            f"held-out days. On days **not** used to set the width, the real price landed inside "
            f"it **{fc.interval_coverage:.0%}** of the time against a {fc.interval_level:.0%} "
            f"target — {verdict}."
        )

    # ── Would following it have made money? ──────────────────
    if fc.strategy:
        s = fc.strategy
        st.subheader("Would following it have made money?")
        st.caption("Go long whenever the model predicts a rise, otherwise hold cash — simulated "
                   "over the held-out window and compared with simply owning the stock.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Following the forecast", f"{s.strategy_return*100:+.2f}%")
        m2.metric("Buy and hold", f"{s.buy_hold_return*100:+.2f}%")
        m3.metric("Difference", f"{s.excess_return*100:+.2f}%",
                  delta=f"{s.excess_return*100:+.2f}%")
        m4.metric("Trades", s.trades,
                  help=f"In the market {s.days_in_market} of {s.n_days} days · "
                       f"{s.cost_bps:.0f} bps charged per position change")

        if s.beat_buy_and_hold:
            st.success(f"On this window the signal beat buy-and-hold by "
                       f"{s.excess_return*100:.2f} percentage points — after costs.")
        else:
            st.warning(f"On this window the signal **lost** to buy-and-hold by "
                       f"{abs(s.excess_return)*100:.2f} percentage points — after costs.")

        for n in s.notes:
            st.caption(n)

    if fc.feature_importance:
        st.subheader("What the models leaned on")
        st.plotly_chart(charts.feature_importance_bar(fc), width="stretch")

    for n in fc.notes:
        st.caption(n)
