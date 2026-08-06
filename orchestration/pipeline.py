"""Orchestration DAG: ticker -> AnalysisResult.

Runs the quant and news pipelines concurrently (I/O-bound), degrades gracefully via
each agent's `safe_run` (a failed stage becomes a warning, not a crash), and fuses
everything into a grounded recommendation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Optional

from agents.pipeline_agents import (
    ContextAgent,
    DataCollectionAgent,
    DedupAgent,
    ForecastAgent,
    NewsCollectionAgent,
    RecommendationAgent,
    RetrievalAgent,
    RiskAgent,
    SentimentAgent,
    SummarizationAgent,
)
from config.logging_config import get_logger
from config.settings import settings
from data_ingestion.markets import benchmark_for, infer_region
from data_ingestion.prices import fetch_prices, resolve_company_name
from orchestration.schemas import (
    AnalysisResult,
    Article,
    MarketContext,
    NewsResult,
    SentimentSummary,
)

logger = get_logger("pipeline")

_NEUTRAL = SentimentSummary(weighted_score=0.0, label="neutral", n_articles=0)


def _safe_fetch_prices(ticker: str):
    """Fetch prices without raising (used for the benchmark). Returns None on failure."""
    try:
        return fetch_prices(ticker)
    except Exception as exc:  # noqa: BLE001
        logger.warning("benchmark fetch failed for %s: %s", ticker, exc)
        return None


def _process_news(
    raw: list[Article], company: str, ticker: str, top_k: int, use_llm: bool, warns: list[str]
) -> tuple[NewsResult, list[Article]]:
    """dedup → rank → sentiment → aggregate → summarize, each degrading gracefully."""
    n_collected = len(raw)
    if not raw:
        return NewsResult(summary="No recent news was found.", sentiment=_NEUTRAL), []

    deduped, err = DedupAgent().safe_run(raw)
    if err:
        warns.append(err)
        deduped = raw
    n_after = len(deduped)

    query = f"{company} stock earnings guidance outlook results"
    top, err = RetrievalAgent().safe_run(query, deduped, top_k)
    if err or top is None:
        if err:
            warns.append(err)
        top = deduped[:top_k]

    sent_out, err = SentimentAgent().safe_run(top)
    if err or sent_out is None:
        warns.append(err or "sentiment failed")
        summary = _NEUTRAL
    else:
        top, summary = sent_out

    text, err = SummarizationAgent().safe_run(company, ticker, top, use_llm)
    if err or not text:
        if err:
            warns.append(err)
        text = "; ".join(a.title for a in top[:5])

    return (
        NewsResult(summary=text, sentiment=summary, top_articles=top,
                   n_collected=n_collected, n_after_dedup=n_after),
        top,
    )


def analyze(
    ticker: str,
    use_llm: bool = True,
    top_k: Optional[int] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> AnalysisResult:
    """Run the full analysis for a ticker and return a (possibly partial) AnalysisResult."""
    ticker = ticker.strip().upper()
    top_k = top_k or settings.news_top_k
    errors: list[str] = []
    warns: list[str] = []

    def note(msg: str) -> None:
        logger.info(msg)
        if progress:
            progress(msg)

    note(f"Resolving {ticker}…")
    company = resolve_company_name(ticker)

    region = infer_region(ticker)
    benchmark = benchmark_for(region)

    # Stage 1 — independent I/O concurrently: prices, news, context, benchmark.
    note("Fetching prices, news, and market context…")
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_prices = ex.submit(DataCollectionAgent().safe_run, ticker)
        f_news = ex.submit(NewsCollectionAgent().safe_run, ticker, company)
        f_ctx = ex.submit(ContextAgent().safe_run, ticker)
        f_bench = ex.submit(_safe_fetch_prices, benchmark)
        prices, perr = f_prices.result()
        raw_news, nerr = f_news.result()
        context, cerr = f_ctx.result()
        benchmark_prices = f_bench.result()

    if nerr:
        warns.append(nerr)
    if cerr:
        warns.append(cerr)
        context = MarketContext()
    context = context or MarketContext()

    if prices is None:
        # Prices are essential — return early with a clear error.
        errors.append(perr or f"No price data for {ticker}.")
        return AnalysisResult(ticker=ticker, company_name=company, as_of=datetime.now(),
                              errors=errors, warnings=warns)

    # Stage 2 — forecast, news-processing, and risk concurrently.
    note("Forecasting, analyzing sentiment, and measuring risk…")
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_fc = ex.submit(ForecastAgent().safe_run, prices, ticker)
        f_np = ex.submit(_process_news, raw_news or [], company, ticker, top_k, use_llm, warns)
        f_risk = ex.submit(RiskAgent().safe_run, prices, benchmark_prices, benchmark)
        forecast, ferr = f_fc.result()
        news_result, top_articles = f_np.result()
        risk, rkerr = f_risk.result()

    if ferr:
        errors.append(ferr)
    if rkerr:
        warns.append(rkerr)

    # Stage 3 — fuse into a recommendation (needs the forecast).
    recommendation = None
    if forecast is not None:
        note("Generating recommendation…")
        recommendation, rerr = RecommendationAgent().safe_run(
            company, ticker, forecast, news_result.sentiment, context, top_articles, use_llm,
        )
        if rerr:
            warns.append(rerr)
    else:
        warns.append("Forecast unavailable — recommendation skipped.")

    note("Done.")
    return AnalysisResult(
        ticker=ticker,
        company_name=company,
        as_of=datetime.now(),
        forecast=forecast,
        news=news_result,
        context=context,
        recommendation=recommendation,
        risk=risk,
        prices=prices,
        errors=errors,
        warnings=warns,
    )
