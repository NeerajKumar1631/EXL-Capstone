"""Lightweight, LLM-free per-stock score for the screener.

Price-only (one cached prices fetch per ticker) so we can rank a whole index quickly.
Composite (0-100) = 0.45·momentum + 0.30·trend + 0.25·low-volatility.
"""
from __future__ import annotations

import numpy as np

from data_ingestion.prices import fetch_prices
from database import cache
from orchestration.schemas import ScoreCard
from technical_analysis.indicators import compute_indicators


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _pct_change(series, lookback: int) -> float:
    if len(series) <= lookback:
        return 0.0
    return float(series.iloc[-1] / series.iloc[-lookback - 1] - 1.0)


def quick_score(ticker: str, region: str | None = None, use_cache: bool = True) -> ScoreCard:
    """Score a single ticker. Raises on unusable data (caller counts failures)."""
    key = f"score_{ticker}"
    if use_cache:
        cached = cache.read_json(key)
        if cached:
            return ScoreCard(**cached)

    prices = fetch_prices(ticker)          # cached; raises InvalidTickerError on bad symbol
    close = prices["Close"]
    last = float(close.iloc[-1])

    ret_1m = _pct_change(close, 21)
    ret_3m = _pct_change(close, 63)
    logret = np.log(close / close.shift(1)).dropna()
    ann_vol = float(logret.std() * np.sqrt(252)) if len(logret) > 1 else 0.0

    ind = compute_indicators(prices)
    rsi = float(ind["rsi_14"].iloc[-1]) if not ind["rsi_14"].isna().all() else 50.0
    sma50 = ind["sma_50"].iloc[-1]
    trend_ratio = (last / sma50 - 1.0) if sma50 and not np.isnan(sma50) else 0.0

    momentum = _clip01(0.5 + ret_3m * 2.0) * 100          # +25% 3m ≈ top
    trend = _clip01(0.5 + trend_ratio * 5.0) * 100         # above/below SMA50
    low_vol = _clip01(1.0 - ann_vol / 0.6) * 100           # calmer = higher
    composite = 0.45 * momentum + 0.30 * trend + 0.25 * low_vol

    card = ScoreCard(
        ticker=ticker, composite=round(composite, 1), momentum=round(momentum, 1),
        trend=round(trend, 1), low_vol=round(low_vol, 1), last_close=round(last, 2),
        ret_1m=round(ret_1m, 4), ret_3m=round(ret_3m, 4), rsi=round(rsi, 1),
        annual_volatility=round(ann_vol, 4),
    )
    cache.write_json(key, card.model_dump(mode="json"))
    return card
