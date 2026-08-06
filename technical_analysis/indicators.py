"""Technical indicators via the `ta` library (numpy-2 safe).

`compute_indicators` returns RAW indicator values (price units where applicable),
used both for charting overlays and as the basis for stationary ML features.
"""
from __future__ import annotations

import warnings

import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ta.momentum import RSIIndicator
    from ta.trend import EMAIndicator, MACD, SMAIndicator
    from ta.volatility import AverageTrueRange, BollingerBands


def compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of raw technical indicators aligned to `prices.index`."""
    close, high, low = prices["Close"], prices["High"], prices["Low"]
    out = pd.DataFrame(index=prices.index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["sma_20"] = SMAIndicator(close, window=20).sma_indicator()
        out["sma_50"] = SMAIndicator(close, window=50).sma_indicator()
        out["ema_12"] = EMAIndicator(close, window=12).ema_indicator()
        out["ema_26"] = EMAIndicator(close, window=26).ema_indicator()
        out["rsi_14"] = RSIIndicator(close, window=14).rsi()

        macd = MACD(close)
        out["macd"] = macd.macd()
        out["macd_signal"] = macd.macd_signal()
        out["macd_diff"] = macd.macd_diff()

        bb = BollingerBands(close, window=20, window_dev=2)
        out["bb_high"] = bb.bollinger_hband()
        out["bb_mid"] = bb.bollinger_mavg()
        out["bb_low"] = bb.bollinger_lband()
        out["bb_pctb"] = bb.bollinger_pband()
        out["bb_width"] = bb.bollinger_wband()

        out["atr_14"] = AverageTrueRange(high, low, close, window=14).average_true_range()

    return out
