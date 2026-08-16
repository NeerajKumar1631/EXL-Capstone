"""Backtest the forecast as a trading rule, against buy-and-hold.

The Forecast page grades the model statistically (RMSE, directional accuracy, skill). None of
that answers the question a user actually has: **would following it have made money?**

This runs the only honest version available from the data we already hold — the holdout window,
where predictions were made without seeing the outcome:

    long when the predicted next-day return is positive, otherwise hold cash

and compares it to simply owning the stock over the same window.

Deliberate limitations, reported rather than buried:
- The window is short (~30 trading days), so the result is an anecdote, not evidence.
- Transaction costs are charged on every position change; the default is deliberately
  non-zero, because a costless backtest flatters any signal that trades often.
- This tests the **forecast**, not the LLM's Buy/Hold/Sell verdict — that needs historical
  news, which the news API does not provide.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from config.logging_config import get_logger
from orchestration.schemas import StrategyBacktest

logger = get_logger("forecast.strategy")

DEFAULT_COST_BPS = 5.0        # 5 basis points per trade, each way — a retail-ish assumption
_TRADING_DAYS = 252


def _annualised(total_return: float, n_days: int) -> Optional[float]:
    if n_days <= 0 or total_return <= -1.0:
        return None
    return float((1.0 + total_return) ** (_TRADING_DAYS / n_days) - 1.0)


def backtest(
    predicted_prices: list[float],
    actual_prices: list[float],
    cost_bps: float = DEFAULT_COST_BPS,
) -> Optional[StrategyBacktest]:
    """Compare 'follow the forecast' with buy-and-hold over the holdout window.

    Both series are prices: `actual_prices[i]` is what happened, `predicted_prices[i]` is what
    the model said would happen, for the same day. Returns None if there is too little data.
    """
    actual = np.asarray(actual_prices, dtype=float)
    predicted = np.asarray(predicted_prices, dtype=float)
    if len(actual) < 5 or len(actual) != len(predicted):
        return None

    # Day-over-day realised returns; the first day has no prior close in this window.
    actual_returns = actual[1:] / actual[:-1] - 1.0
    # The model's view for day i is predicted[i] against the previous *actual* close — the
    # only price it could have known at the time.
    predicted_returns = predicted[1:] / actual[:-1] - 1.0

    position = (predicted_returns > 0).astype(float)      # 1 = long, 0 = cash
    trades = int(np.sum(np.abs(np.diff(np.concatenate([[0.0], position])))))
    cost_per_trade = cost_bps / 10_000.0

    gross = position * actual_returns
    costs = np.abs(np.diff(np.concatenate([[0.0], position]))) * cost_per_trade
    net = gross - costs

    strategy_total = float(np.prod(1.0 + net) - 1.0)
    hold_total = float(np.prod(1.0 + actual_returns) - 1.0)

    days_in_market = int(np.sum(position))
    n = len(actual_returns)

    wins = int(np.sum((position > 0) & (actual_returns > 0)))
    taken = int(np.sum(position > 0))

    return StrategyBacktest(
        n_days=n,
        strategy_return=strategy_total,
        buy_hold_return=hold_total,
        excess_return=strategy_total - hold_total,
        strategy_annualised=_annualised(strategy_total, n),
        buy_hold_annualised=_annualised(hold_total, n),
        days_in_market=days_in_market,
        trades=trades,
        cost_bps=cost_bps,
        win_rate=(wins / taken) if taken else None,
        beat_buy_and_hold=strategy_total > hold_total,
        notes=_notes(n, trades, taken),
    )


def _notes(n_days: int, trades: int, taken: int) -> list[str]:
    notes = [
        f"Simulated over {n_days} trading days of held-out data, charging {DEFAULT_COST_BPS:.0f} "
        f"basis points per position change.",
        "This tests the price forecast only. The Buy/Hold/Sell verdict also uses news "
        "sentiment, which cannot be backtested without historical news.",
    ]
    if n_days < 60:
        notes.append(
            f"{n_days} days is far too short to judge a strategy — treat this as an "
            f"illustration, not evidence. A good result here is as likely to be luck as skill."
        )
    if taken == 0:
        notes.append("The model never signalled a long position in this window.")
    elif trades > n_days / 3:
        notes.append(f"The signal changed position {trades} times — costs matter a lot at that "
                     f"turnover, and real spreads would likely exceed the assumption above.")
    return notes
