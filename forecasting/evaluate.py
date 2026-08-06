"""Forecast evaluation metrics.

We predict next-day (or h-day) LOG RETURNS. Metrics live in two spaces:
- return space: RMSE, MAE, R², directional accuracy, and skill vs. a naive baseline
- price space: MAPE, reconstructed as prev_close * exp(pred_return), so it's directly
  comparable to price-forecasting literature.
"""
from __future__ import annotations

import numpy as np

from orchestration.schemas import ModelMetrics

_EPS = 1e-9


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with the correct sign (up/down)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.abs(y_true) > _EPS
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))


def price_mape(prev_close: np.ndarray, y_true_ret: np.ndarray, y_pred_ret: np.ndarray) -> float:
    """MAPE (%) on reconstructed prices."""
    prev_close = np.asarray(prev_close, dtype=float)
    actual_price = prev_close * np.exp(np.asarray(y_true_ret, dtype=float))
    pred_price = prev_close * np.exp(np.asarray(y_pred_ret, dtype=float))
    denom = np.where(np.abs(actual_price) < _EPS, _EPS, actual_price)
    return float(np.mean(np.abs(pred_price - actual_price) / np.abs(denom)) * 100.0)


def make_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prev_close: np.ndarray,
    baseline_rmse: float,
    dir_acc_override: float | None = None,
) -> ModelMetrics:
    """Build a ModelMetrics from return-space predictions."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = 0.0
    mape = price_mape(prev_close, y_true, y_pred)
    diracc = dir_acc_override if dir_acc_override is not None else directional_accuracy(y_true, y_pred)
    skill = float((baseline_rmse - rmse) / baseline_rmse) if baseline_rmse > _EPS else 0.0

    return ModelMetrics(
        rmse=rmse, mae=mae, mape=mape, r2=r2,
        directional_accuracy=float(diracc), skill_vs_baseline=skill,
    )
