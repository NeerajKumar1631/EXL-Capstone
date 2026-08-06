"""Offline unit tests for dedup, aggregation, news parsing, and the recommendation engine."""
from datetime import datetime
from unittest.mock import MagicMock, patch

from orchestration.schemas import (
    Article,
    ForecastResult,
    HorizonForecast,
    MarketContext,
    ModelForecast,
    ModelMetrics,
    SentimentSummary,
)


# ── dedup ─────────────────────────────────────────────
def test_dedup_removes_near_and_url_duplicates():
    from retrieval.dedup import deduplicate

    arts = [
        Article(title="Apple beats Q3 earnings", url="http://x.com/a"),
        Article(title="Apple beats Q3 earnings!!!", url="http://y.com/b"),   # near-dup title
        Article(title="Totally different headline", url="http://x.com/a?utm=1"),  # dup URL
        Article(title="A genuinely unique story", url="http://z.com/c"),
    ]
    out = deduplicate(arts)
    assert len(out) == 2
    assert {o.title for o in out} == {"Apple beats Q3 earnings", "A genuinely unique story"}


# ── sentiment aggregation ─────────────────────────────
def test_aggregate_is_credibility_weighted():
    from sentiment.aggregate import aggregate

    arts = [
        Article(title="t1", url="u1", sentiment_label="positive", sentiment_score=0.9, credibility=1.0),
        Article(title="t2", url="u2", sentiment_label="negative", sentiment_score=-0.5, credibility=0.5),
    ]
    s = aggregate(arts)
    assert abs(s.weighted_score - (0.9 * 1.0 - 0.5 * 0.5) / 1.5) < 1e-3  # aggregate rounds to 4dp
    assert s.label == "positive"
    assert s.n_positive == 1 and s.n_negative == 1


def test_aggregate_empty_is_neutral():
    from sentiment.aggregate import aggregate

    s = aggregate([])
    assert s.label == "neutral" and s.n_articles == 0


# ── Event Registry parsing (mocked HTTP) ──────────────
def test_event_registry_parse():
    from data_ingestion import news

    fake = {"articles": {"results": [{
        "title": "Apple wins big", "url": "http://a.com/1",
        "source": {"title": "Reuters"}, "dateTime": "2026-08-06T07:00:00Z",
        "body": "A long article body about Apple.",
    }]}}
    resp = MagicMock()
    resp.json.return_value = fake
    resp.raise_for_status.return_value = None
    with patch("data_ingestion.news.requests.post", return_value=resp):
        arts = news._from_event_registry("Apple Inc", 7, 10)
    assert len(arts) == 1
    assert arts[0].source == "Reuters"
    assert arts[0].credibility >= 0.9  # Reuters prior
    assert arts[0].published_at is not None


# ── recommendation engine ─────────────────────────────
def _forecast(beats: bool, ret: float) -> ForecastResult:
    mm = ModelMetrics(rmse=0.02, mae=0.01, mape=1.5, r2=0.0, directional_accuracy=0.5, skill_vs_baseline=0.0)
    ens = ModelForecast(name="Ensemble", weight=1.0, metrics=mm, horizons=[
        HorizonForecast(horizon="1d", horizon_days=1, predicted_return=ret, predicted_price=100 * (1 + ret)),
    ])
    return ForecastResult(ticker="X", last_close=100.0, as_of=datetime.now(), models=[ens],
                          ensemble=ens, baseline_metrics=mm, beats_baseline=beats, best_model="Ensemble")


def test_rule_based_returns_valid_recommendation():
    from recommendation.engine import recommend

    fc = _forecast(beats=False, ret=0.0)
    s = SentimentSummary(weighted_score=0.0, label="neutral", n_articles=0)
    r = recommend("Co", "X", fc, s, MarketContext(), [], use_llm=False)
    assert r.action in ("Buy", "Hold", "Sell")
    assert 0.0 <= r.confidence <= 1.0
    assert r.disclaimer


def test_rule_based_buys_on_strong_positive_signal():
    from recommendation.engine import recommend

    fc = _forecast(beats=True, ret=0.03)
    s = SentimentSummary(weighted_score=0.7, label="positive", n_articles=5, n_positive=5)
    r = recommend("Co", "X", fc, s, MarketContext(), [], use_llm=False)
    assert r.action == "Buy"


def test_grounding_drops_fabricated_urls():
    from recommendation.engine import _ground

    valid = {"http://real.com/a"}
    cleaned = _ground(["Grounded point (http://real.com/a)", "Made up (http://fake.com/x)"], valid)
    assert len(cleaned) == 1 and "real.com" in cleaned[0]
