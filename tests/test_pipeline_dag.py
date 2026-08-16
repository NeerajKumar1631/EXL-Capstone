"""Orchestration DAG — graceful degradation and problem reporting.

`analyze()` is the single entry point and was previously only ever patched out. These tests
drive it with the network stages stubbed, so they run offline and assert the behaviour that
actually matters: one broken stage must degrade, not kill the analysis, and users must never
see raw exception text.
"""
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from agents.base import StageError
from orchestration.pipeline import _Problems, analyze
from orchestration.schemas import Article, MarketContext


@pytest.fixture
def prices():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2023-01-02", periods=300)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, len(idx))))
    return pd.DataFrame({
        "Open": close * 0.999, "High": close * 1.007, "Low": close * 0.993,
        "Close": close, "Volume": rng.integers(1_000_000, 4_000_000, len(idx)),
    }, index=idx)


@pytest.fixture
def articles():
    return [
        Article(title="Company beats earnings", url="http://x.com/1", source="Reuters",
                published_at=datetime(2026, 8, 10), snippet="Strong quarter."),
        Article(title="Analyst raises target", url="http://x.com/2", source="CNBC",
                published_at=datetime(2026, 8, 11), snippet="Target raised."),
    ]


def _run(prices, articles, **overrides):
    """Run analyze() with all network edges stubbed. `overrides` replace individual stubs."""
    stubs = {
        "orchestration.pipeline.resolve_company_name": lambda t: "Test Co",
        "orchestration.pipeline._safe_fetch_prices": lambda t: prices,
        "agents.pipeline_agents.fetch_prices": lambda t: prices,
        "agents.pipeline_agents.fetch_news": lambda t, c: articles,
        "agents.pipeline_agents.fetch_context": lambda t: MarketContext(),
    }
    stubs.update(overrides)
    patches = [patch(target, new=impl) for target, impl in stubs.items()]
    for p in patches:
        p.start()
    try:
        return analyze("TEST", use_llm=False)
    finally:
        for p in patches:
            p.stop()


def test_happy_path_produces_a_complete_result(prices, articles):
    r = _run(prices, articles)
    assert r.forecast is not None
    assert r.recommendation is not None and r.recommendation.action in ("Buy", "Hold", "Sell")
    assert r.risk is not None
    assert r.news is not None and r.news.summary
    assert r.errors == []


def test_missing_prices_is_fatal_and_explained_in_plain_english(prices, articles):
    def boom(_):
        raise RuntimeError("yfinance exploded")

    r = _run(prices, articles, **{"agents.pipeline_agents.fetch_prices": boom})
    assert r.forecast is None                     # prices are essential
    assert r.errors, "a fatal problem must be reported"
    joined = " ".join(r.errors)
    assert "couldn't find any price data" in joined
    assert "RuntimeError" not in joined           # no raw exception text for the user
    assert any("RuntimeError" in d for d in r.details)   # but kept for debugging


def test_news_failure_degrades_to_a_price_only_analysis(prices, articles):
    def boom(_t, _c):
        raise RuntimeError("news api down")

    r = _run(prices, articles, **{"agents.pipeline_agents.fetch_news": boom})
    assert r.forecast is not None and r.recommendation is not None   # analysis still completes
    assert any("news" in w.lower() for w in r.warnings)
    assert "RuntimeError" not in " ".join(r.warnings)
    assert any("RuntimeError" in d for d in r.details)


def test_context_failure_degrades_without_losing_the_recommendation(prices, articles):
    def boom(_):
        raise RuntimeError("info lookup failed")

    r = _run(prices, articles, **{"agents.pipeline_agents.fetch_context": boom})
    assert r.recommendation is not None
    assert r.context is not None                  # replaced with an empty MarketContext
    assert r.warnings


def test_no_news_still_yields_a_recommendation(prices):
    r = _run(prices, [])
    assert r.recommendation is not None
    assert r.news.n_collected == 0
    assert r.news.summary                          # falls back to a digest, never blank


def test_result_carries_the_prices_used(prices, articles):
    r = _run(prices, articles)
    assert r.prices is not None and len(r.prices) == len(prices)


# ── the problem collector ──────────────────────────────────
def test_problems_splits_user_message_from_technical_detail():
    p = _Problems()
    p.warn(StageError("We couldn't load the news.", "news: HTTPError: 503"))
    p.error(StageError("We couldn't build a forecast.", "forecast: ValueError: nope"))
    assert p.warnings == ["We couldn't load the news."]
    assert p.errors == ["We couldn't build a forecast."]
    assert p.details == ["news: HTTPError: 503", "forecast: ValueError: nope"]


def test_problems_accepts_a_plain_string_without_a_detail():
    p = _Problems()
    p.warn("Something mild happened.")
    assert p.warnings == ["Something mild happened."] and p.details == []
