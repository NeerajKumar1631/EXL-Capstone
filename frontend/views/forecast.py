"""Forecast — per-model metrics against the naive baseline, backtest, horizons, strategy."""
import _shared
import streamlit as st

_shared.page_header("Forecast",
                    "Every model graded on a held-out window, against a naive "
                    "'tomorrow = today' baseline.",
                    eyebrow="Analysis")

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

    _shared.section("Model comparison", "held-out window, no look-ahead")
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

    _shared.section("Actual vs predicted", "holdout backtest")
    st.plotly_chart(charts.backtest_actual_vs_pred(fc), width="stretch")

    _shared.section("Multi-horizon forecast",
                    f"ensemble · each with an {fc.interval_level:.0%} prediction range"
                    if fc.interval_level else "ensemble")
    _shared.kpi_row([
        {"label": f"{h.horizon} · {h.horizon_days} trading day{'s' if h.horizon_days > 1 else ''}",
         "value": f"${h.predicted_price:,.2f}",
         "delta": f"{h.predicted_return*100:+.2f}%",
         "sub": (f"{fc.interval_level:.0%} range ${h.lower:,.2f} – ${h.upper:,.2f}"
                 if h.lower is not None and h.upper is not None else None)}
        for h in ens.horizons
    ])

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
        _shared.section("Would following it have made money?", "after transaction costs")
        st.caption("Go long whenever the model predicts a rise, otherwise hold cash — simulated "
                   "over the held-out window and compared with simply owning the stock.")

        _shared.kpi_row([
            {"label": "Following the forecast", "value": f"{s.strategy_return*100:+.2f}%",
             "delta": "beat buy & hold" if s.beat_buy_and_hold else "lost to buy & hold",
             "tone": "pos" if s.beat_buy_and_hold else "neg",
             "sub": f"in the market {s.days_in_market} of {s.n_days} days"},
            {"label": "Buy and hold", "value": f"{s.buy_hold_return*100:+.2f}%"},
            {"label": "Difference", "value": f"{s.excess_return*100:+.2f} pp",
             "tone": "pos" if s.excess_return > 0 else "neg"},
            {"label": "Trades", "value": str(s.trades),
             "sub": f"{s.cost_bps:.0f} bps charged per position change"},
        ])

        if s.beat_buy_and_hold:
            st.success(f"On this window the signal beat buy-and-hold by "
                       f"{s.excess_return*100:.2f} percentage points — after costs.")
        else:
            st.warning(f"On this window the signal **lost** to buy-and-hold by "
                       f"{abs(s.excess_return)*100:.2f} percentage points — after costs.")

        for n in s.notes:
            st.caption(n)

    if fc.feature_importance:
        _shared.section("What the models leaned on")
        st.plotly_chart(charts.feature_importance_bar(fc), width="stretch")

    for n in fc.notes:
        st.caption(n)
