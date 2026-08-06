from datetime import datetime
from unittest.mock import patch

from orchestration.schemas import (
    AnalysisResult,
    ForecastResult,
    HorizonForecast,
    ModelForecast,
    ModelMetrics,
    NewsResult,
    Recommendation,
    SentimentSummary,
)
from chat import tools


def _mini_result() -> AnalysisResult:
    mm = ModelMetrics(rmse=0.02, mae=0.01, mape=1.5, r2=0.0, directional_accuracy=0.55, skill_vs_baseline=0.02)
    ens = ModelForecast(name="Ensemble", weight=1.0, metrics=mm, horizons=[
        HorizonForecast(horizon="1d", horizon_days=1, predicted_return=0.012, predicted_price=101.2)])
    fc = ForecastResult(ticker="AAPL", last_close=100.0, as_of=datetime.now(), models=[ens],
                        ensemble=ens, baseline_metrics=mm, beats_baseline=True, best_model="Ensemble")
    news = NewsResult(summary="ok", sentiment=SentimentSummary(weighted_score=0.3, label="positive", n_articles=3))
    rec = Recommendation(action="Buy", confidence=0.6, thesis="ok")
    return AnalysisResult(ticker="AAPL", company_name="Apple Inc.", as_of=datetime.now(),
                          forecast=fc, news=news, recommendation=rec)


def test_analyze_text_is_grounded():
    with patch("orchestration.pipeline.analyze", return_value=_mini_result()):
        txt = tools._analyze_text("AAPL")
    assert "Apple Inc." in txt and "Buy" in txt and "$100.00" in txt
    assert "beats_naive_baseline=True" in txt


def test_analyze_text_handles_failure_gracefully():
    with patch("orchestration.pipeline.analyze", side_effect=RuntimeError("boom")):
        txt = tools._analyze_text("AAPL")
    assert "failed" in txt.lower() and "boom" in txt      # returns text, never raises


def test_compare_text_uses_compare():
    from orchestration.schemas import CompareItem, ComparisonResult

    fake = ComparisonResult(items=[
        CompareItem(ticker="AAPL", ok=True, action="Buy", next_day_return=0.01,
                    sentiment_label="positive", annual_volatility=0.28, beta=1.1),
        CompareItem(ticker="MSFT", ok=True, action="Hold", next_day_return=-0.002,
                    sentiment_label="neutral", annual_volatility=0.22, beta=0.9),
    ])
    with patch("compare.compare.compare", return_value=fake):
        txt = tools._compare_text("AAPL, MSFT")
    assert "AAPL" in txt and "MSFT" in txt and "verdict=Buy" in txt
