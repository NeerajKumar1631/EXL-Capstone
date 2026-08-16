"""Grade past predictions against what the price actually did.

Every analysis is already persisted to the `runs` table with the price it predicted for the
next trading day. This module looks up what actually happened and scores it — the honest
counterpart to the forecast metrics, which are measured on a holdout rather than on live use.

Design notes:
- A run is only graded when we can line it up with a real price bar. If the bar the run was
  made from cannot be identified, it is counted as `unverifiable` rather than guessed at.
- A run whose next trading day has not happened yet is `pending`, not a miss.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from config.logging_config import get_logger
from data_ingestion.prices import fetch_prices
from database.db import all_runs
from orchestration.schemas import GradedRun, TrackRecord

logger = get_logger("analytics.track_record")

# The run stores the close it saw; require the bar we pick to match it this closely,
# otherwise we are not confident we found the right bar.
_CLOSE_TOLERANCE = 0.005      # 0.5%


def _find_origin_bar(close: pd.Series, run_date, last_close: float) -> Optional[int]:
    """Index of the bar the prediction was made from, or None if it can't be identified."""
    on_or_before = close.loc[:pd.Timestamp(run_date).normalize() + pd.Timedelta(days=1)]
    if on_or_before.empty:
        return None
    idx = len(on_or_before) - 1
    # Confirm by price: the run recorded the close it forecast from.
    for candidate in (idx, idx - 1):
        if candidate < 0:
            continue
        value = float(close.iloc[candidate])
        if value and abs(value - last_close) / value <= _CLOSE_TOLERANCE:
            return candidate
    return None


def evaluate(limit: int = 500) -> TrackRecord:
    """Score every saved prediction against the price that followed it."""
    runs = all_runs(limit)
    record = TrackRecord(total_runs=len(runs))
    if not runs:
        record.notes.append("No analyses have been saved yet.")
        return record

    prices: dict[str, pd.Series] = {}
    for run in runs:
        ticker = run["ticker"]
        if ticker not in prices:
            try:
                prices[ticker] = fetch_prices(ticker)["Close"]
            except Exception as exc:  # noqa: BLE001 - one bad ticker must not sink the page
                logger.warning("track record: no prices for %s: %s", ticker, exc)
                prices[ticker] = pd.Series(dtype=float)

        close = prices[ticker]
        if close.empty:
            record.unverifiable += 1
            continue

        origin = _find_origin_bar(close, run["created_at"], run["last_close"])
        if origin is None:
            record.unverifiable += 1
            continue
        if origin + 1 >= len(close):
            record.pending += 1          # the next trading day hasn't happened yet
            continue

        actual = float(close.iloc[origin + 1])
        base = run["last_close"]
        if not base or not actual:
            record.unverifiable += 1
            continue

        predicted_return = run["next_day_price"] / base - 1.0
        actual_return = actual / base - 1.0
        # A flat actual move is not a direction anyone can get right or wrong; treat the
        # sign match strictly so a 0% day counts against an up/down call.
        correct = (predicted_return >= 0) == (actual_return >= 0)

        record.graded.append(GradedRun(
            ticker=ticker,
            company=run["company"],
            when=run["created_at"].strftime("%Y-%m-%d %H:%M"),
            action=run["action"],
            confidence=run["confidence"],
            last_close=base,
            predicted_price=run["next_day_price"],
            actual_price=actual,
            predicted_return=predicted_return,
            actual_return=actual_return,
            direction_correct=correct,
            abs_pct_error=abs(run["next_day_price"] - actual) / actual,
        ))

    record.n_graded = len(record.graded)
    record.n_correct = sum(1 for g in record.graded if g.direction_correct)
    if record.n_graded:
        record.hit_rate = record.n_correct / record.n_graded
        record.mean_abs_pct_error = sum(g.abs_pct_error for g in record.graded) / record.n_graded

    if record.pending:
        record.notes.append(
            f"{record.pending} prediction(s) can't be scored yet — the next trading day "
            f"hasn't closed."
        )
    if record.unverifiable:
        record.notes.append(
            f"{record.unverifiable} prediction(s) couldn't be matched to a price bar and were "
            f"left out rather than guessed at."
        )
    if 0 < record.n_graded < 20:
        record.notes.append(
            f"Only {record.n_graded} prediction(s) scored so far — far too few to draw any "
            f"conclusion. Treat this as a running tally, not a measure of skill."
        )
    return record
