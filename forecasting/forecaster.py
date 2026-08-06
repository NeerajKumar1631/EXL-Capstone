"""Forecasting orchestrator.

Pipeline (no look-ahead anywhere):
1. Engineer features; target = next-day log return.
2. Time-ordered split: train | test-holdout (last `test_size`).
3. Ensemble weights from out-of-sample validation RMSE (TimeSeriesSplit CV for GBMs;
   walk-forward for ARIMA).
4. Fit on train, predict the holdout → per-model + ensemble metrics vs. the naive baseline.
5. Refit on all data → multi-horizon (1d/1w/1m) future forecast.

Returns a fully-populated `ForecastResult`.
"""
from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from config.logging_config import get_logger
from config.settings import settings
from forecasting import arima_model
from forecasting.baseline import baseline_return_predictions, up_fraction
from forecasting.ensemble import inverse_rmse_weights, weighted_average, weighted_scalar
from forecasting.evaluate import make_metrics
from forecasting.models import build_gbm_models
from orchestration.schemas import ForecastResult, HorizonForecast, ModelForecast
from technical_analysis.features import TARGET, build_features, feature_columns

logger = get_logger("forecast")

HORIZONS: list[tuple[str, int]] = [("1d", 1), ("1w", 5), ("1m", 21)]


class ForecastError(RuntimeError):
    """Raised when a forecast cannot be produced (e.g. insufficient history)."""


def _cv_rmse(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> float:
    """Mean out-of-sample RMSE across expanding time-series folds."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    errs: list[float] = []
    for tr_idx, va_idx in tscv.split(X):
        m = model.fresh().fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = m.predict(X.iloc[va_idx])
        errs.append(float(np.sqrt(mean_squared_error(y.iloc[va_idx], pred))))
    return float(np.mean(errs)) if errs else float("inf")


def _horizon_targets(close: pd.Series, h: int) -> pd.Series:
    """Cumulative log-return over the next `h` days."""
    return np.log(close.shift(-h) / close)


def run_forecast(prices: pd.DataFrame, ticker: str) -> ForecastResult:
    feats = build_features(prices)
    cols = feature_columns(feats)
    close = prices["Close"]

    labelled = feats.dropna(subset=cols + [TARGET])
    if len(labelled) < settings.min_history_rows:
        raise ForecastError(
            f"insufficient history for {ticker}: {len(labelled)} usable rows "
            f"(need ≥{settings.min_history_rows})."
        )

    X, y, dates = labelled[cols], labelled[TARGET], labelled.index
    n = len(X)
    test_size = max(10, min(settings.forecast_test_size, n // 5))
    test_start = n - test_size

    X_tr, y_tr = X.iloc[:test_start], y.iloc[:test_start]
    X_te, y_te = X.iloc[test_start:], y.iloc[test_start:]
    prev_close_te = close.loc[dates[test_start:]].to_numpy(dtype=float)
    y_te_arr = y_te.to_numpy(dtype=float)

    # inference row: latest bar with complete features (target may be NaN there)
    feat_valid = feats.dropna(subset=cols)
    X_infer = feat_valid[cols].iloc[[-1]]
    last_close = float(close.loc[feat_valid.index[-1]])

    # ── Baseline (naive persistence) ─────────────────────────────
    base_pred = baseline_return_predictions(len(y_te_arr))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_te_arr, base_pred)))
    baseline_metrics = make_metrics(
        y_te_arr, base_pred, prev_close_te, baseline_rmse,
        dir_acc_override=up_fraction(y_te_arr),
    )

    val_rmse: dict[str, float] = {}
    test_preds: dict[str, np.ndarray] = {}
    metrics: dict[str, object] = {}
    horizon_ret: dict[str, dict[int, float]] = {}
    importances: dict[str, dict[str, float]] = {}
    notes: list[str] = []

    # ── Gradient-boosted models ──────────────────────────────────
    for m in build_gbm_models():
        try:
            val_rmse[m.name] = _cv_rmse(m, X_tr, y_tr)
            fitted = m.fresh().fit(X_tr, y_tr)
            tpred = fitted.predict(X_te)
            test_preds[m.name] = tpred
            metrics[m.name] = make_metrics(y_te_arr, tpred, prev_close_te, baseline_rmse)
            importances[m.name] = fitted.importances(cols)
            # horizons: refit on all labelled-for-h, predict inference row
            hr: dict[int, float] = {}
            for _, h in HORIZONS:
                tgt = _horizon_targets(close, h)
                lab = feats.assign(_t=tgt).dropna(subset=cols + ["_t"])
                mh = m.fresh().fit(lab[cols], lab["_t"])
                hr[h] = float(mh.predict(X_infer)[0])
            horizon_ret[m.name] = hr
        except Exception as exc:  # noqa: BLE001
            logger.warning("model %s failed, skipping: %s", m.name, exc)

    # ── ARIMA (best-effort statistical baseline) ─────────────────
    try:
        ret_full = feats["log_ret"].dropna()
        # validation walk-forward on the last test_size of the training window
        val_start = test_start - test_size
        if val_start > settings.min_history_rows // 2:
            av = arima_model.walk_forward(y, val_start)[:test_size]
            val_rmse["ARIMA"] = float(np.sqrt(mean_squared_error(y.iloc[val_start:test_start], av)))
        else:
            val_rmse["ARIMA"] = baseline_rmse  # weak prior if too little data
        at = arima_model.walk_forward(y, test_start)
        test_preds["ARIMA"] = at
        metrics["ARIMA"] = make_metrics(y_te_arr, at, prev_close_te, baseline_rmse)
        horizon_ret["ARIMA"] = arima_model.horizon_cumulative_returns(ret_full, [h for _, h in HORIZONS])
    except Exception as exc:  # noqa: BLE001
        logger.warning("ARIMA failed, skipping: %s", exc)
        notes.append("ARIMA was unavailable for this series and excluded from the ensemble.")

    if not test_preds:
        raise ForecastError(f"all models failed for {ticker}")

    # ── Ensemble ─────────────────────────────────────────────────
    weights = inverse_rmse_weights({k: v for k, v in val_rmse.items() if k in test_preds})
    ens_test = weighted_average(test_preds, weights)
    ens_metrics = make_metrics(y_te_arr, ens_test, prev_close_te, baseline_rmse)
    ens_horizon = {
        h: weighted_scalar({k: horizon_ret[k][h] for k in horizon_ret if h in horizon_ret[k]}, weights)
        for _, h in HORIZONS
    }

    def _horizons(hr: dict[int, float]) -> list[HorizonForecast]:
        out = []
        for label, h in HORIZONS:
            r = float(hr.get(h, 0.0))
            out.append(HorizonForecast(
                horizon=label, horizon_days=h,
                predicted_return=r, predicted_price=last_close * float(np.exp(r)),
            ))
        return out

    model_forecasts = [
        ModelForecast(name=name, weight=float(weights.get(name, 0.0)),
                      metrics=metrics[name], horizons=_horizons(horizon_ret.get(name, {})))
        for name in test_preds
    ]
    ensemble = ModelForecast(name="Ensemble", weight=1.0, metrics=ens_metrics,
                             horizons=_horizons(ens_horizon))

    # best model by holdout RMSE (ensemble included)
    candidates = {name: metrics[name].rmse for name in test_preds}
    candidates["Ensemble"] = ens_metrics.rmse
    best_model = min(candidates, key=candidates.get)
    beats_baseline = ens_metrics.skill_vs_baseline > 0

    # aggregated feature importance (weighted across GBMs)
    agg_imp: dict[str, float] = {}
    for name, imp in importances.items():
        w = weights.get(name, 0.0)
        for feat, val in imp.items():
            agg_imp[feat] = agg_imp.get(feat, 0.0) + w * val
    if agg_imp:
        tot = sum(agg_imp.values()) or 1.0
        agg_imp = {k: v / tot for k, v in sorted(agg_imp.items(), key=lambda kv: -kv[1])}

    # backtest arrays for charting
    actual_price = prev_close_te * np.exp(y_te_arr)
    pred_price = prev_close_te * np.exp(ens_test)

    if not beats_baseline:
        notes.append(
            "The ensemble did not beat the naive 'tomorrow=today' baseline on this window — "
            "treat the point forecast with caution."
        )

    return ForecastResult(
        ticker=ticker,
        last_close=last_close,
        as_of=datetime.now(),
        models=model_forecasts,
        ensemble=ensemble,
        baseline_metrics=baseline_metrics,
        beats_baseline=beats_baseline,
        best_model=best_model,
        backtest_dates=[d.strftime("%Y-%m-%d") for d in dates[test_start:]],
        backtest_actual=[float(v) for v in actual_price],
        backtest_pred=[float(v) for v in pred_price],
        feature_importance={k: round(v, 4) for k, v in list(agg_imp.items())[:12]},
        notes=notes,
    )
