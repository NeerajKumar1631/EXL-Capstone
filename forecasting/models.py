"""Gradient-boosted regressors (XGBoost, LightGBM, CatBoost) behind one interface.

Each model predicts a (log) return from the engineered feature matrix. The uniform
`RegressorModel` wrapper gives them a common fit/predict/importance API and a `fresh()`
factory used for cross-validation.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd


class RegressorModel:
    """Uniform wrapper around a scikit-learn-style regressor."""

    def __init__(self, name: str, factory: Callable[[], object]) -> None:
        self.name = name
        self._factory = factory
        self.estimator = factory()

    def fresh(self) -> "RegressorModel":
        return RegressorModel(self.name, self._factory)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegressorModel":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=float)

    def importances(self, columns: list[str]) -> dict[str, float]:
        imp = getattr(self.estimator, "feature_importances_", None)
        if imp is None:
            return {}
        arr = np.asarray(imp, dtype=float)
        if arr.sum() > 0:
            arr = arr / arr.sum()
        return {c: float(v) for c, v in zip(columns, arr)}


def _xgb() -> object:
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        n_jobs=-1, random_state=42, verbosity=0,
    )


def _lgbm() -> object:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=300, max_depth=4, num_leaves=15, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        n_jobs=-1, random_state=42, verbose=-1,
    )


def _catboost() -> object:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=3.0,
        random_seed=42, verbose=0, allow_writing_files=False,
    )


def build_gbm_models() -> list[RegressorModel]:
    """The gradient-boosted model roster."""
    return [
        RegressorModel("XGBoost", _xgb),
        RegressorModel("LightGBM", _lgbm),
        RegressorModel("CatBoost", _catboost),
    ]
