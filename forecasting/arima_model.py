"""ARIMA on the log-return series (statistical baseline).

Uses SARIMAX with efficient one-step walk-forward via `.append(refit=False)` so we get
genuine out-of-sample predictions without a slow refit per step. Everything is wrapped
so a convergence failure degrades gracefully (the caller simply drops ARIMA).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from config.logging_config import get_logger

logger = get_logger("forecast.arima")

DEFAULT_ORDER = (2, 0, 1)  # returns are ~stationary, so d=0


def _fit(series: pd.Series, order=DEFAULT_ORDER):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series.astype(float), order=order, trend="c",
            enforce_stationarity=False, enforce_invertibility=False,
        )
        return model.fit(disp=False)


def walk_forward(returns: pd.Series, start_pos: int, order=DEFAULT_ORDER) -> np.ndarray:
    """One-step-ahead predictions for positions [start_pos:], each using only prior data."""
    res = _fit(returns.iloc[:start_pos], order)
    preds: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(start_pos, len(returns)):
            preds.append(float(np.asarray(res.forecast(steps=1))[0]))
            res = res.append([float(returns.iloc[i])], refit=False)
    return np.array(preds, dtype=float)


def horizon_cumulative_returns(returns: pd.Series, horizons_days: list[int], order=DEFAULT_ORDER) -> dict[int, float]:
    """Forecast cumulative log-return over each horizon from the end of the series."""
    res = _fit(returns, order)
    max_h = max(horizons_days)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = np.asarray(res.forecast(steps=max_h), dtype=float)
    return {h: float(np.sum(fc[:h])) for h in horizons_days}
