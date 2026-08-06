"""Naive persistence baseline: 'tomorrow = today' (predicted next-day return = 0).

Every model is judged against this. A model that can't beat it demonstrates no skill.
"""
from __future__ import annotations

import numpy as np


def baseline_return_predictions(n: int) -> np.ndarray:
    """Predicted returns for the naive baseline: all zeros (price unchanged)."""
    return np.zeros(n, dtype=float)


def up_fraction(y_true: np.ndarray) -> float:
    """Fraction of up-days — the directional accuracy of an 'always up' naive call."""
    y_true = np.asarray(y_true, dtype=float)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true > 0))
