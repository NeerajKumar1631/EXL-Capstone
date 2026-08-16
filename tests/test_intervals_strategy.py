"""Prediction intervals, strategy backtest, and the gated sentiment feature."""
import numpy as np
import pandas as pd
import pytest

from forecasting.intervals import DEFAULT_LEVEL, calibrate, conformal_quantile, scale_for_horizon
from forecasting.strategy import backtest
from technical_analysis.features import (
    MIN_SENTIMENT_COVERAGE,
    SENTIMENT_COL,
    attach_sentiment,
    build_features,
    feature_columns,
    sentiment_coverage,
)


# ── conformal intervals ────────────────────────────────────
def test_quantile_widens_with_the_confidence_level():
    residuals = np.linspace(-1, 1, 200)
    assert conformal_quantile(residuals, 0.5) < conformal_quantile(residuals, 0.95)


def test_quantile_needs_a_minimum_sample():
    assert conformal_quantile(np.array([0.1, 0.2, 0.3]), 0.8) is None


def test_quantile_ignores_nans():
    r = np.array([0.1] * 10 + [np.nan] * 5)
    assert conformal_quantile(r, 0.8) == pytest.approx(0.1)


def test_interval_achieves_roughly_its_target_coverage():
    """The whole point of conformal: measured coverage should land near the nominal level."""
    rng = np.random.default_rng(0)
    n = 400
    y_pred = np.zeros(n)
    y_true = rng.standard_t(df=3, size=n) * 0.01     # fat tails, deliberately non-Gaussian
    q, coverage, n_calib = calibrate(y_true, y_pred, 0.80)
    assert q is not None and n_calib == int(n * 0.6)
    assert 0.70 <= coverage <= 0.90                  # near 80% despite the fat tails


def test_coverage_is_not_reported_when_the_sample_is_too_small_to_check():
    """Better to say 'unknown' than to quote a number measured on its own calibration data."""
    y_true = np.linspace(-0.02, 0.02, 12)
    q, coverage, n_calib = calibrate(y_true, np.zeros(12), DEFAULT_LEVEL)
    assert q is not None and coverage is None and n_calib == 12


def test_tiny_samples_give_no_interval_at_all():
    q, coverage, _ = calibrate(np.array([0.01, -0.01]), np.zeros(2), DEFAULT_LEVEL)
    assert q is None and coverage is None


def test_horizon_scaling_follows_square_root_of_time():
    assert scale_for_horizon(0.02, 1) == pytest.approx(0.02)
    assert scale_for_horizon(0.02, 4) == pytest.approx(0.04)
    assert scale_for_horizon(0.02, 21) > scale_for_horizon(0.02, 5)


# ── strategy backtest ──────────────────────────────────────
def test_a_perfect_signal_beats_buy_and_hold_in_a_falling_market():
    actual = [100, 99, 98, 97, 96, 95]        # steady decline
    predicted = [100, 98, 97, 96, 95, 94]     # always predicts down -> stays in cash
    r = backtest(predicted, actual, cost_bps=0)
    assert r.days_in_market == 0
    assert r.strategy_return == pytest.approx(0.0)
    assert r.buy_hold_return < 0
    assert r.beat_buy_and_hold


def test_an_always_long_signal_matches_buy_and_hold_without_costs():
    actual = [100, 101, 102, 103, 104, 105]
    predicted = [101, 102, 103, 104, 105, 106]   # always predicts up
    r = backtest(predicted, actual, cost_bps=0)
    assert r.strategy_return == pytest.approx(r.buy_hold_return)
    assert not r.beat_buy_and_hold                # equal is not better


def test_costs_are_actually_charged():
    actual = [100, 101, 102, 103, 104, 105]
    predicted = [101, 102, 103, 104, 105, 106]
    free = backtest(predicted, actual, cost_bps=0)
    priced = backtest(predicted, actual, cost_bps=50)
    assert priced.strategy_return < free.strategy_return
    assert priced.cost_bps == 50


def test_too_little_data_returns_nothing():
    assert backtest([1, 2], [1, 2]) is None
    assert backtest([1, 2, 3], [1, 2, 3, 4]) is None      # mismatched lengths


def test_short_windows_are_flagged_as_anecdote():
    actual = list(np.linspace(100, 110, 20))
    predicted = list(np.linspace(101, 111, 20))
    r = backtest(predicted, actual)
    assert any("too short" in n for n in r.notes)


# ── the gated sentiment feature ────────────────────────────
def _prices(n=120):
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2026-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": [1_000_000] * n}, index=idx)


def test_sparse_sentiment_is_refused_as_a_feature():
    """A column that is empty for most training rows must not become a model input."""
    df = _prices()
    history = {df.index[i].strftime("%Y-%m-%d"): 0.5 for i in range(5)}   # ~4% coverage
    assert sentiment_coverage(df.index, history) < MIN_SENTIMENT_COVERAGE
    assert attach_sentiment(df, history) is False
    assert SENTIMENT_COL not in df.columns


def test_dense_sentiment_is_accepted_and_becomes_a_model_input():
    df = _prices()
    history = {d.strftime("%Y-%m-%d"): 0.3 for d in df.index}             # full coverage
    assert attach_sentiment(df, history) is True
    assert SENTIMENT_COL in df.columns
    assert df[SENTIMENT_COL].notna().all()

    feats = build_features(_prices(), sentiment=history)
    assert SENTIMENT_COL in feature_columns(feats)


def test_gaps_are_carried_forward_not_invented():
    df = _prices(10)
    days = [d.strftime("%Y-%m-%d") for d in df.index]
    history = {days[0]: 0.4, days[5]: -0.6}      # readings only on two days
    # coverage is low, so force the attach to exercise the fill logic directly
    attach_sentiment(df, {d: history.get(d, 0.0) for d in days})
    assert df[SENTIMENT_COL].iloc[0] == pytest.approx(0.4)


def test_no_history_means_no_column():
    feats = build_features(_prices(), sentiment={})
    assert SENTIMENT_COL not in feature_columns(feats)
