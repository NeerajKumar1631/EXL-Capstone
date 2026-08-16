from datetime import datetime

from orchestration.schemas import (
    AnalysisResult,
    Article,
    ForecastResult,
    HorizonForecast,
    ModelForecast,
    ModelMetrics,
    NewsResult,
    Recommendation,
    RiskProfile,
    SentimentSummary,
)
import pytest

from report.export import _pdf_safe, to_html, to_markdown, to_pdf


def _full_result() -> AnalysisResult:
    mm = ModelMetrics(rmse=0.02, mae=0.01, mape=1.5, r2=0.0, directional_accuracy=0.55, skill_vs_baseline=0.02)
    ens = ModelForecast(name="Ensemble", weight=1.0, metrics=mm, horizons=[
        HorizonForecast(horizon="1d", horizon_days=1, predicted_return=0.01, predicted_price=101.0)])
    fc = ForecastResult(ticker="AAPL", last_close=100.0, as_of=datetime.now(), models=[ens],
                        ensemble=ens, baseline_metrics=mm, beats_baseline=True, best_model="Ensemble")
    risk = RiskProfile(annual_volatility=0.28, max_drawdown=-0.33, drawdown_peak="2024-01-01",
                       drawdown_trough="2024-06-01", beta=1.1, var_95=-0.03)
    news = NewsResult(summary="Solid earnings.", sentiment=SentimentSummary(
        weighted_score=0.3, label="positive", n_articles=3),
        top_articles=[Article(title="Beats Q3", url="http://x.com/a", source="Reuters")])
    rec = Recommendation(action="Buy", confidence=0.62, thesis="Strong fundamentals.",
                         positive_factors=["Revenue up"], risks=["Valuation rich"])
    return AnalysisResult(ticker="AAPL", company_name="Apple Inc.", as_of=datetime.now(),
                          forecast=fc, news=news, risk=risk, recommendation=rec)


def test_markdown_has_all_sections():
    md = to_markdown(_full_result())
    for section in ["# StockSense AI", "Recommendation: **Buy**", "## Forecast",
                    "## Risk & History", "## News & Sentiment", "not financial advice"]:
        assert section in md
    assert "http://x.com/a" in md          # source link grounded


def test_markdown_handles_partial_result():
    r = AnalysisResult(ticker="XYZ", as_of=datetime.now(), errors=["no data"])
    md = to_markdown(r)                     # no forecast/news/risk/reco
    assert "# StockSense AI" in md and "not financial advice" in md   # no crash


def test_html_wraps_markdown():
    html = to_html(_full_result())
    assert html.startswith("<!doctype html>") and "StockSense" in html


# ── PDF export ────────────────────────────────────────
def test_pdf_safe_folds_typography_to_latin1():
    """fpdf2's built-in fonts are latin-1 only; an em dash used to crash the export."""
    out = _pdf_safe("Buy — 62% confidence · “quoted” … ≈ ₹100 ⚠")
    assert "—" not in out and "·" not in out and "…" not in out
    out.encode("latin-1")  # must not raise


def test_to_pdf_produces_a_real_pdf():
    pdf = to_pdf(_full_result())
    if pdf is None:
        pytest.skip("fpdf2 not installed — PDF export is optional")
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 500


def test_to_pdf_survives_a_partial_result():
    partial = AnalysisResult(ticker="ZZ", as_of=datetime.now(), errors=["nope"])
    pdf = to_pdf(partial)
    if pdf is None:
        pytest.skip("fpdf2 not installed — PDF export is optional")
    assert pdf.startswith(b"%PDF-")
