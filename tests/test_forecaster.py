"""Forecasting orchestrator — the ML core, previously untested.

Uses the synthetic `synth_prices` fixture, so these run offline and fast.
"""
import numpy as np
import pandas as pd
import pytest

from forecasting.forecaster import (
    HORIZONS,
    ForecastError,
    _cache_key,
    _horizon_targets,
    run_forecast,
)


def test_insufficient_history_raises_rather_than_guessing(synth_prices):
    """A short series must fail loudly — a forecast from 20 rows would be meaningless."""
    with pytest.raises(ForecastError) as exc:
        run_forecast(synth_prices.head(20), "SHORT", use_cache=False)
    assert "insufficient history" in str(exc.value)


def test_horizon_targets_are_forward_looking_and_lose_the_tail(synth_prices):
    close = synth_prices["Close"]
    tgt = _horizon_targets(close, 5)
    # target at t is the cumulative return from t to t+5
    expected = np.log(close.iloc[5] / close.iloc[0])
    assert tgt.iloc[0] == pytest.approx(expected)
    assert tgt.iloc[-5:].isna().all()          # no future data exists for the last 5 rows


def test_cache_key_changes_with_the_last_bar_and_with_settings(synth_prices):
    from config.settings import settings

    full = _cache_key("X", synth_prices)
    assert _cache_key("X", synth_prices) == full            # deterministic
    assert _cache_key("X", synth_prices.iloc[:-1]) != full  # a new bar busts it
    assert _cache_key("Y", synth_prices) != full            # ticker is part of it

    original = settings.min_history_rows
    try:
        settings.min_history_rows = original + 1
        assert _cache_key("X", synth_prices) != full        # config change busts it
    finally:
        settings.min_history_rows = original


# ── The full forecast (one slow-ish run, reused by several assertions) ──
@pytest.fixture(scope="module")
def result(synth_prices_module):
    return run_forecast(synth_prices_module, "SYNTH", use_cache=False)


@pytest.fixture(scope="module")
def synth_prices_module():
    """Module-scoped copy of the synthetic prices (the shared fixture is function-scoped)."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2023-01-02", periods=320)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(idx))))
    return pd.DataFrame({
        "Open": close * 0.999, "High": close * 1.008, "Low": close * 0.992,
        "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, len(idx)),
    }, index=idx)


def test_produces_every_horizon_for_the_ensemble(result):
    horizons = {h.horizon for h in result.ensemble.horizons}
    assert horizons == {label for label, _ in HORIZONS}
    for h in result.ensemble.horizons:
        # price must be reconstructed from the return, not invented
        assert h.predicted_price == pytest.approx(result.last_close * np.exp(h.predicted_return))


def test_metrics_are_in_valid_ranges(result):
    for model in [*result.models, result.ensemble]:
        m = model.metrics
        assert m.rmse >= 0 and m.mae >= 0
        assert 0.0 <= m.directional_accuracy <= 1.0
    assert 0.0 <= result.baseline_metrics.directional_accuracy <= 1.0


def test_ensemble_weights_are_normalised_across_surviving_models(result):
    total = sum(m.weight for m in result.models)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_beats_baseline_agrees_with_the_skill_number(result):
    """The headline honesty flag must not disagree with the metric behind it."""
    assert result.beats_baseline == (result.ensemble.metrics.skill_vs_baseline > 0)


def test_backtest_arrays_line_up(result):
    n = len(result.backtest_dates)
    assert n > 0
    assert len(result.backtest_actual) == n and len(result.backtest_pred) == n


def test_a_losing_model_is_reported_not_hidden(result):
    """If the ensemble can't beat the naive baseline, that must be surfaced in notes."""
    if not result.beats_baseline:
        assert any("baseline" in note.lower() for note in result.notes)


def test_cache_round_trip_is_faithful(synth_prices_module, tmp_path):
    from config.settings import settings

    original = settings.cache_dir
    try:
        settings.cache_dir = tmp_path
        first = run_forecast(synth_prices_module, "CACHED", use_cache=True)
        second = run_forecast(synth_prices_module, "CACHED", use_cache=True)
        assert second.model_dump(mode="json") == first.model_dump(mode="json")
    finally:
        settings.cache_dir = original
