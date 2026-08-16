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

SENTIMENT_COL = "sentiment"
# A sentiment feature is only worth adding if it is actually present across the training
# window. Below this share of rows the column is mostly zeros at training time and non-zero
# at inference — the model would learn nothing from it and then be handed an input unlike
# anything it saw. Refusing is the honest option.
MIN_SENTIMENT_COVERAGE = 0.60


def sentiment_coverage(index: pd.Index, history: dict[str, float]) -> float:
    """Fraction of the given trading days for which a sentiment reading exists."""
    if len(index) == 0:
        return 0.0
    days = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in index}
    return len(days & set(history)) / len(days)


def attach_sentiment(df: pd.DataFrame, history: dict[str, float]) -> bool:
    """Add a `sentiment` column in place if coverage is sufficient. Returns whether it was added.

    The news API only serves ~4 weeks of history, so this is populated from readings the app
    has accumulated itself (`database.db.sentiment_history`). Until enough days exist, the
    column is deliberately **not** created: a feature that is zero for 95% of training rows
    teaches the model nothing and would be a lie dressed as a signal.
    """
    if not history or sentiment_coverage(df.index, history) < MIN_SENTIMENT_COVERAGE:
        return False
    days = pd.Index([pd.Timestamp(d).strftime("%Y-%m-%d") for d in df.index])
    values = pd.Series([history.get(d, np.nan) for d in days], index=df.index, dtype=float)
    # Carry the last known reading forward across gaps (weekends, days with no news), then
    # treat any remaining leading gap as neutral.
    df[SENTIMENT_COL] = values.ffill().fillna(0.0)
    return True


def build_features(prices: pd.DataFrame, sentiment: dict[str, float] | None = None) -> pd.DataFrame:
    """Return a DataFrame with engineered features, the raw indicators, and `target`.

    The last row has a NaN target (its next-day return is unknown) and is used for
    inference. Callers should drop NaN-target rows for training.

    `sentiment` is an optional {'YYYY-MM-DD': score} history. It becomes a model feature only
    when it covers enough of the window — see `attach_sentiment`.
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

    # Optional sentiment feature (only if we have accumulated enough history)
    if sentiment:
        attach_sentiment(df, sentiment)

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
        SENTIMENT_COL,      # present only when coverage passed the threshold
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
