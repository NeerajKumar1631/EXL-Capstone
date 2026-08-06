import _shared
import streamlit as st

_shared.setup("Forecast", "📈")
st.title("📈 Forecast")

r = _shared.require_result()
if r:
    import pandas as pd

    from visualization import charts

    fc = r.forecast
    ens = fc.ensemble

    if fc.beats_baseline:
        st.success(f"✅ The ensemble beats the naive 'tomorrow=today' baseline on the holdout "
                   f"(skill {ens.metrics.skill_vs_baseline:+.3f}).")
    else:
        st.warning("⚠️ The ensemble does **not** beat the naive baseline. Daily direction is "
                   "essentially a coin-flip — the multi-horizon prices below are indicative only.")

    st.subheader("Model comparison (evaluated on a held-out window, no look-ahead)")
    rows = []
    for m in [*fc.models, ens]:
        mm = m.metrics
        rows.append({
            "Model": m.name,
            "Weight": round(m.weight, 2),
            "RMSE (ret)": round(mm.rmse, 5),
            "MAE (ret)": round(mm.mae, 5),
            "MAPE (price) %": round(mm.mape, 2),
            "R²": round(mm.r2, 3),
            "Dir. acc %": round(mm.directional_accuracy * 100, 1),
            "Skill vs base": round(mm.skill_vs_baseline, 3),
        })
    bm = fc.baseline_metrics
    rows.append({"Model": "Naive baseline", "Weight": None, "RMSE (ret)": round(bm.rmse, 5),
                 "MAE (ret)": round(bm.mae, 5), "MAPE (price) %": round(bm.mape, 2), "R²": round(bm.r2, 3),
                 "Dir. acc %": round(bm.directional_accuracy * 100, 1), "Skill vs base": 0.0})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(f"Best model by holdout RMSE: **{fc.best_model}**. Metrics are on next-day log returns; "
               f"MAPE is on reconstructed prices for comparability.")

    st.subheader("Actual vs predicted (holdout backtest)")
    st.plotly_chart(charts.backtest_actual_vs_pred(fc), width="stretch")

    st.subheader("Multi-horizon forecast (ensemble)")
    cols = st.columns(len(ens.horizons))
    for col, h in zip(cols, ens.horizons):
        col.metric(f"{h.horizon} → {h.horizon_days}d", f"${h.predicted_price:,.2f}",
                   f"{h.predicted_return*100:+.2f}%")

    if fc.feature_importance:
        st.plotly_chart(charts.feature_importance_bar(fc), width="stretch")

    for n in fc.notes:
        st.caption("• " + n)
