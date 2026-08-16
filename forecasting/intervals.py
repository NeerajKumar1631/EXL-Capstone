"""Prediction intervals via split conformal prediction.

A point forecast with no range is exactly the false precision this project claims to avoid,
and `HorizonForecast.lower`/`upper` existed unused from the start. This fills them.

Why conformal rather than a model's own uncertainty estimate:
- It is **distribution-free** — no assumption that residuals are Gaussian, which daily
  returns violate badly (fat tails).
- It wraps whatever model produced the point forecast, so the ensemble is covered too.
- Its coverage guarantee is checkable, and we check it.

Method (split conformal): take the absolute residuals of the ensemble on data it never
trained on, and let `q` be their (1-alpha) quantile. The interval is `prediction ± q`.

Honesty rules applied here:
- Calibration and coverage measurement use **disjoint** slices of the holdout, so the
  reported coverage is not the same data that produced the interval width.
- If the holdout is too small to split, we still calibrate but report coverage as unknown
  rather than quoting a number that is true by construction.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from config.logging_config import get_logger

logger = get_logger("forecast.intervals")

DEFAULT_LEVEL = 0.80          # 80% interval: wide enough to be honest, narrow enough to read
_MIN_CALIBRATION = 8          # below this the quantile is meaningless
_MIN_TO_SPLIT = 20            # below this we cannot spare points to measure coverage


def conformal_quantile(residuals: np.ndarray, level: float = DEFAULT_LEVEL) -> Optional[float]:
    """The (1-alpha) conformal quantile of absolute residuals, or None if too few points.

    Uses the finite-sample correction ceil((n+1)(1-alpha))/n, which is what gives split
    conformal its coverage guarantee for small samples.
    """
    r = np.abs(np.asarray(residuals, dtype=float))
    r = r[np.isfinite(r)]
    n = len(r)
    if n < _MIN_CALIBRATION:
        return None
    rank = math.ceil((n + 1) * level)
    if rank > n:            # too few points to reach this level at all
        return float(np.max(r))
    return float(np.sort(r)[rank - 1])


def calibrate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    level: float = DEFAULT_LEVEL,
) -> tuple[Optional[float], Optional[float], int]:
    """Return (quantile, measured_coverage, n_calibration).

    The holdout is split: the earlier portion sets the interval width, the later portion
    measures how often the truth actually landed inside it. `measured_coverage` is None when
    the sample is too small to spare points for an honest check.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    n = len(residuals)

    if n < _MIN_TO_SPLIT:
        q = conformal_quantile(residuals, level)
        if q is None:
            logger.info("too few holdout points (%d) for a prediction interval", n)
        return q, None, n

    cut = int(n * 0.6)
    q = conformal_quantile(residuals[:cut], level)
    if q is None:
        return None, None, cut

    held_back = residuals[cut:]
    coverage = float(np.mean(np.abs(held_back) <= q))
    return q, coverage, cut


def scale_for_horizon(q: float, horizon_days: int) -> float:
    """Widen a 1-day interval to `horizon_days` under a random-walk assumption (sqrt of time).

    Calibration data only exists for the next-day forecast, so longer horizons are scaled
    rather than measured. This is a modelling assumption, not an empirical result — callers
    should say so to the user.
    """
    return q * math.sqrt(max(1, horizon_days))
