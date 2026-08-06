from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd

from orchestration.schemas import (
    AnalysisResult,
    ForecastResult,
    HorizonForecast,
    ModelForecast,
    ModelMetrics,
    NewsResult,
    SentimentSummary,
)


def _mini_result(ticker: str, ok: bool = True) -> AnalysisResult:
    if not ok:
        return AnalysisResult(ticker=ticker, as_of=datetime.now(), errors=[f"no data for {ticker}"])
    mm = ModelMetrics(rmse=0.02, mae=0.01, mape=1.5, r2=0.0, directional_accuracy=0.5, skill_vs_baseline=0.0)
    ens = ModelForecast(name="Ensemble", weight=1.0, metrics=mm, horizons=[
        HorizonForecast(horizon="1d", horizon_days=1, predicted_return=0.01, predicted_price=101.0)])
    fc = ForecastResult(ticker=ticker, last_close=100.0, as_of=datetime.now(), models=[ens],
                        ensemble=ens, baseline_metrics=mm, beats_baseline=True, best_model="Ensemble")
    idx = pd.date_range("2024-01-01", periods=130, freq="B")
    prices = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0,
                           "Close": np.linspace(100, 110, 130), "Volume": 1e6}, index=idx)
    news = NewsResult(summary="", sentiment=SentimentSummary(weighted_score=0.2, label="positive", n_articles=3))
    return AnalysisResult(ticker=ticker, company_name=f"{ticker} Inc", as_of=datetime.now(),
                          forecast=fc, news=news, prices=prices)


def _fake_analyze(ticker, use_llm=False):
    return _mini_result(ticker, ok=(ticker != "INVALID"))


def test_dedupe_order_and_rebase():
    from compare.compare import compare

    with patch("compare.compare.analyze", side_effect=_fake_analyze):
        cmp = compare(["AAPL", "aapl", "MSFT"])   # aapl dedupes into AAPL
    assert [i.ticker for i in cmp.items] == ["AAPL", "MSFT"]
    a = cmp.items[0]
    assert a.ok and a.rebased and abs(a.rebased[0] - 100.0) < 1e-6   # rebased to 100


def test_cap_at_three():
    from compare.compare import compare

    with patch("compare.compare.analyze", side_effect=_fake_analyze):
        cmp = compare(["A", "B", "C", "D"])
    assert len(cmp.items) == 3
    assert any("first 3" in n.lower() for n in cmp.notes)


def test_invalid_among_valid_is_flagged():
    from compare.compare import compare

    with patch("compare.compare.analyze", side_effect=_fake_analyze):
        cmp = compare(["AAPL", "INVALID"])
    by = {i.ticker: i for i in cmp.items}
    assert by["AAPL"].ok is True
    assert by["INVALID"].ok is False and by["INVALID"].error
