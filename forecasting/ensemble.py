"""Weighted ensemble: weights proportional to inverse validation RMSE.

Models that forecast better on out-of-sample validation data get more weight.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def inverse_rmse_weights(val_rmse: dict[str, float]) -> dict[str, float]:
    """Normalized weights ∝ 1 / validation RMSE (eps-guarded)."""
    if not val_rmse:
        return {}
    raw = {name: 1.0 / max(rmse, _EPS) for name, rmse in val_rmse.items()}
    total = sum(raw.values())
    if total <= _EPS:
        # degenerate — fall back to equal weights
        n = len(raw)
        return {name: 1.0 / n for name in raw}
    return {name: w / total for name, w in raw.items()}


def weighted_average(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Combine per-model prediction arrays using the given weights."""
    names = [n for n in preds if n in weights]
    if not names:
        raise ValueError("no overlapping models between preds and weights")
    stacked = np.vstack([np.asarray(preds[n], dtype=float) for n in names])
    w = np.array([weights[n] for n in names], dtype=float)
    w = w / w.sum()
    return np.average(stacked, axis=0, weights=w)


def weighted_scalar(values: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted combination of per-model scalars (e.g. horizon return predictions)."""
    names = [n for n in values if n in weights]
    if not names:
        return 0.0
    w = np.array([weights[n] for n in names], dtype=float)
    v = np.array([values[n] for n in names], dtype=float)
    return float(np.average(v, weights=w / w.sum()))
