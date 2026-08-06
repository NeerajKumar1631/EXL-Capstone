"""Risk & History metrics — pure-quant, fast, and guarded against short/degenerate data.

Covers the user's "risk + how it got affected in the past": volatility, max drawdown,
beta, historical VaR, 52-week position, Sharpe-like ratio, biggest past moves, and the
rolling-vol / drawdown series for charts.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from orchestration.schemas import BigMove, RiskProfile

logger = get_logger("analytics.risk")

_TRADING_DAYS = 252


def _daily_log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).dropna()


def _max_drawdown(close: pd.Series) -> tuple[float, Optional[str], Optional[str], pd.Series]:
    running_max = close.cummax()
    dd = close / running_max - 1.0
    if dd.empty:
        return 0.0, None, None, dd
    trough = dd.idxmin()
    mdd = float(dd.min())
    # peak = the running-max date on/before the trough
    pre = close.loc[:trough]
    peak = pre.idxmax() if not pre.empty else None

    def _fmt(d) -> Optional[str]:
        return pd.Timestamp(d).strftime("%Y-%m-%d") if d is not None else None

    return mdd, _fmt(peak), _fmt(trough), dd


def compute_risk(
    prices: pd.DataFrame,
    benchmark_prices: Optional[pd.DataFrame] = None,
    benchmark_name: Optional[str] = None,
) -> RiskProfile:
    """Compute a RiskProfile from OHLCV history (+ optional benchmark for beta)."""
    notes: list[str] = []
    close = prices["Close"].astype(float)
    ret = _daily_log_returns(close)
    n = len(ret)

    ann_vol = float(ret.std() * np.sqrt(_TRADING_DAYS)) if n > 1 else 0.0
    if n < _TRADING_DAYS:
        notes.append(f"Only {n} days of returns — annualized figures are approximate.")

    mdd, peak, trough, dd_series = _max_drawdown(close)

    # Historical VaR (negative daily return at the tail)
    var_95 = float(np.percentile(ret, 5)) if n else 0.0
    var_99 = float(np.percentile(ret, 1)) if n else 0.0

    # Sharpe-like (rf=0), guarded against zero vol
    sharpe = float(ret.mean() * _TRADING_DAYS / ann_vol) if ann_vol > 1e-9 else None

    # Beta vs benchmark (aligned on common dates)
    beta: Optional[float] = None
    if benchmark_prices is not None and not benchmark_prices.empty:
        bret = _daily_log_returns(benchmark_prices["Close"].astype(float))
        joined = pd.concat([ret.rename("a"), bret.rename("b")], axis=1).dropna()
        if len(joined) >= 20 and joined["b"].var() > 1e-12:
            beta = float(joined["a"].cov(joined["b"]) / joined["b"].var())
        else:
            notes.append("Insufficient overlapping benchmark data — beta unavailable.")

    # 52-week window
    window = close.tail(_TRADING_DAYS)
    hi, lo = float(window.max()), float(window.min())
    last = float(close.iloc[-1])
    pos = float((last - lo) / (hi - lo)) if hi - lo > 1e-9 else 0.5

    # Biggest single-day moves
    ret_pct = (np.exp(ret) - 1.0) * 100.0
    up = ret_pct.sort_values(ascending=False).head(5)
    down = ret_pct.sort_values().head(5)
    biggest_up = [BigMove(date=pd.Timestamp(d).strftime("%Y-%m-%d"), pct=round(float(v), 2)) for d, v in up.items()]
    biggest_down = [BigMove(date=pd.Timestamp(d).strftime("%Y-%m-%d"), pct=round(float(v), 2)) for d, v in down.items()]

    # Rolling annualized vol series (for chart)
    roll = (ret.rolling(21).std() * np.sqrt(_TRADING_DAYS)).dropna()

    return RiskProfile(
        annual_volatility=round(ann_vol, 4),
        max_drawdown=round(mdd, 4),
        drawdown_peak=peak,
        drawdown_trough=trough,
        beta=None if beta is None else round(beta, 3),
        benchmark=benchmark_name,
        var_95=round(var_95, 4),
        var_99=round(var_99, 4),
        week52_high=round(hi, 2),
        week52_low=round(lo, 2),
        price_position_52w=round(pos, 3),
        sharpe_like=None if sharpe is None else round(sharpe, 3),
        biggest_up=biggest_up,
        biggest_down=biggest_down,
        rolling_vol_dates=[d.strftime("%Y-%m-%d") for d in roll.index],
        rolling_vol=[round(float(v), 4) for v in roll.values],
        drawdown_dates=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in dd_series.index],
        drawdown_series=[round(float(v), 4) for v in dd_series.values],
        notes=notes,
    )
