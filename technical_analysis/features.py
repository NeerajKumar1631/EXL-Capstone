"""Feature engineering for forecasting.

Builds a model-ready matrix of (mostly) stationary features from OHLCV + technical
indicators, plus the supervised target = next-day log return.

Design choices for honesty/generalization:
- Model the LOG RETURN, not the price level (price levels are non-stationary and give
  deceptively good metrics).
- Convert price-unit indicators (SMA/EMA/MACD/ATR) to ratios so features are ~stationary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from technical_analysis.indicators import compute_indicators

LAGS = (1, 2, 3, 5, 10)
ROLL_WINDOWS = (5, 10, 20)
TARGET = "target"


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with engineered features, the raw indicators, and `target`.

    The last row has a NaN target (its next-day return is unknown) and is used for
    inference. Callers should drop NaN-target rows for training.
    """
    df = prices.copy()
    close = df["Close"]

    # Core return series
    df["log_ret"] = np.log(close / close.shift(1))

    # Lagged returns
    for lag in LAGS:
        df[f"ret_lag_{lag}"] = df["log_ret"].shift(lag)

    # Rolling return statistics
    for w in ROLL_WINDOWS:
        df[f"ret_mean_{w}"] = df["log_ret"].rolling(w).mean()
        df[f"ret_std_{w}"] = df["log_ret"].rolling(w).std()

    # Realized volatility (annualized)
    df["vol_20"] = df["log_ret"].rolling(20).std() * np.sqrt(252)

    # Volume features
    with np.errstate(divide="ignore"):
        df["vol_chg"] = np.log(df["Volume"] / df["Volume"].shift(1))
    df["vol_chg"] = df["vol_chg"].replace([np.inf, -np.inf], np.nan)
    df["vol_ratio_20"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # Intraday range
    df["hl_range"] = (df["High"] - df["Low"]) / close
    df["co_range"] = (close - df["Open"]) / df["Open"]

    # Technical indicators (raw) — kept for charts
    ind = compute_indicators(df)
    df = df.join(ind)

    # Stationary indicator-derived features
    df["close_sma20"] = close / ind["sma_20"]
    df["close_sma50"] = close / ind["sma_50"]
    df["close_ema12"] = close / ind["ema_12"]
    df["ema_ratio"] = ind["ema_12"] / ind["ema_26"]
    df["macd_diff_rel"] = ind["macd_diff"] / close
    df["rsi_norm"] = ind["rsi_14"] / 100.0
    df["bb_pctb"] = ind["bb_pctb"]
    df["bb_width_rel"] = ind["bb_width"] / 100.0
    df["atr_rel"] = ind["atr_14"] / close

    # Supervised target: NEXT-day log return
    df[TARGET] = df["log_ret"].shift(-1)

    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Names of the columns used as model inputs (stationary features only)."""
    cols = [c for c in df.columns if c.startswith(("ret_lag_", "ret_mean_", "ret_std_"))]
    cols += [
        "vol_20", "vol_chg", "vol_ratio_20", "hl_range", "co_range",
        "close_sma20", "close_sma50", "close_ema12", "ema_ratio",
        "macd_diff_rel", "rsi_norm", "bb_pctb", "bb_width_rel", "atr_rel",
    ]
    return [c for c in cols if c in df.columns]


def training_frame(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """Prepare (X_train, y_train, features_full, feature_cols).

    - `features_full` is the full engineered frame (indicators retained for charts).
    - X_train/y_train drop rows with any NaN in features or target.
    """
    feats = build_features(prices)
    cols = feature_columns(feats)
    labelled = feats.dropna(subset=cols + [TARGET])
    X = labelled[cols]
    y = labelled[TARGET]
    return X, y, feats, cols
