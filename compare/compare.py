"""Compare 2–3 stocks side-by-side (forecast + sentiment + risk), with a rebased overlay.

Reuses the full `analyze` pipeline per ticker (concurrently). Invalid tickers become a
flagged item rather than failing the whole comparison.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from config.logging_config import get_logger
from data_ingestion.markets import infer_region, normalize_ticker
from orchestration.pipeline import analyze
from orchestration.schemas import AnalysisResult, CompareItem, ComparisonResult

logger = get_logger("compare")

MAX_TICKERS = 3
_LOOKBACK = 120


def _to_item(ticker: str, result: AnalysisResult) -> CompareItem:
    if result.forecast is None:
        return CompareItem(ticker=ticker, ok=False,
                           error=" · ".join(result.errors) or "analysis failed")
    fc = result.forecast
    nd = fc.ensemble.next_day
    item = CompareItem(
        ticker=ticker,
        company=result.company_name,
        ok=True,
        action=result.recommendation.action if result.recommendation else None,
        confidence=result.recommendation.confidence if result.recommendation else 0.0,
        last_close=fc.last_close,
        next_day_return=nd.predicted_return if nd else 0.0,
        directional_accuracy=fc.ensemble.metrics.directional_accuracy,
        beats_baseline=fc.beats_baseline,
        sentiment_label=result.news.sentiment.label if result.news else "neutral",
        sentiment_score=result.news.sentiment.weighted_score if result.news else 0.0,
        annual_volatility=result.risk.annual_volatility if result.risk else 0.0,
        max_drawdown=result.risk.max_drawdown if result.risk else 0.0,
        beta=result.risk.beta if result.risk else None,
    )
    if result.prices is not None and not result.prices.empty:
        tail = result.prices["Close"].tail(_LOOKBACK)
        base = float(tail.iloc[0])
        if base:
            item.dates = [d.strftime("%Y-%m-%d") for d in tail.index]
            item.rebased = [round(float(v) / base * 100.0, 2) for v in tail.values]
    return item


def compare(tickers: list[str], use_llm: bool = False) -> ComparisonResult:
    """Analyze up to 3 tickers concurrently and assemble a side-by-side comparison."""
    notes: list[str] = []

    # normalize + dedupe (preserve order), cap at MAX_TICKERS
    seen: dict[str, None] = {}
    for t in tickers:
        norm = normalize_ticker(t, infer_region(t))
        if norm:
            seen.setdefault(norm, None)
    cleaned = list(seen)
    if len(cleaned) > MAX_TICKERS:
        notes.append(f"Comparing the first {MAX_TICKERS} of {len(cleaned)} tickers.")
        cleaned = cleaned[:MAX_TICKERS]
    if len(cleaned) < 2:
        notes.append("Add at least two tickers for a meaningful comparison.")

    items: list[CompareItem] = []
    if cleaned:
        with ThreadPoolExecutor(max_workers=len(cleaned)) as ex:
            futures = {ex.submit(analyze, t, use_llm): t for t in cleaned}
            results = {futures[f]: f.result() for f in futures}
        items = [_to_item(t, results[t]) for t in cleaned]  # preserve input order

    return ComparisonResult(items=items, notes=notes)
