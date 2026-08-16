"""Orchestration DAG: ticker -> AnalysisResult.

Runs the quant and news pipelines concurrently (I/O-bound), degrades gracefully via
each agent's `safe_run` (a failed stage becomes a warning, not a crash), and fuses
everything into a grounded recommendation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Optional

from agents.base import StageError
from agents.pipeline_agents import (
    AnalystAgent,
    ContextAgent,
    DataCollectionAgent,
    DedupAgent,
    ForecastAgent,
    NewsCollectionAgent,
    RetrievalAgent,
    RiskAgent,
    SentimentAgent,
    SummarizationAgent,
)
from config.logging_config import get_logger
from config.settings import settings
from data_ingestion.markets import benchmark_for, infer_region
from data_ingestion.prices import fetch_prices, resolve_company_name
from llm.summarizer import headline_digest
from orchestration.schemas import (
    AnalysisResult,
    Article,
    MarketContext,
    NewsResult,
    SentimentSummary,
)

logger = get_logger("pipeline")

_NEUTRAL = SentimentSummary(weighted_score=0.0, label="neutral", n_articles=0)


class _Problems:
    """Collects what went wrong, split by audience.

    `errors`/`warnings` are plain sentences for the user; `details` keeps the matching
    exception text so it can sit behind a "technical details" expander rather than on the
    dashboard.
    """

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.details: list[str] = []

    def error(self, err) -> None:
        self.errors.append(str(err))
        self._detail(err)

    def warn(self, err) -> None:
        self.warnings.append(str(err))
        self._detail(err)

    def _detail(self, err) -> None:
        detail = getattr(err, "detail", None)
        if detail:
            self.details.append(detail)


def _safe_fetch_prices(ticker: str):
    """Fetch prices without raising (used for the benchmark). Returns None on failure."""
    try:
        return fetch_prices(ticker)
    except Exception as exc:  # noqa: BLE001
        logger.warning("benchmark fetch failed for %s: %s", ticker, exc)
        return None


def _prepare_news(
    raw: list[Article], company: str, top_k: int, problems: "_Problems"
) -> tuple[list[Article], SentimentSummary, int, int]:
    """dedup → rank → sentiment, each degrading gracefully.

    Deliberately stops short of summarization: the summary is an LLM call, and it is
    launched later alongside the recommendation so the two network waits overlap.
    Returns (top_articles, sentiment, n_collected, n_after_dedup).
    """
    n_collected = len(raw)
    if not raw:
        return [], _NEUTRAL, 0, 0

    deduped, err = DedupAgent().safe_run(raw)
    if err:
        problems.warn(err)
        deduped = raw
    n_after = len(deduped)

    query = f"{company} stock earnings guidance outlook results"
    top, err = RetrievalAgent().safe_run(query, deduped, top_k)
    if err or top is None:
        if err:
            problems.warn(err)
        top = deduped[:top_k]

    sent_out, err = SentimentAgent().safe_run(top)
    if err or sent_out is None:
        problems.warn(err or "We couldn't score the news sentiment, so it is treated as neutral.")
        sentiment = _NEUTRAL
    else:
        top, sentiment = sent_out

    return top, sentiment, n_collected, n_after


def analyze(
    ticker: str,
    use_llm: bool = True,
    top_k: Optional[int] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> AnalysisResult:
    """Run the full analysis for a ticker and return a (possibly partial) AnalysisResult."""
    ticker = ticker.strip().upper()
    top_k = top_k or settings.news_top_k
    problems = _Problems()

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
        problems.warn(nerr)
    if cerr:
        problems.warn(cerr)
        context = MarketContext()
    context = context or MarketContext()

    if prices is None:
        # Prices are essential — return early. A bad symbol is by far the likeliest cause,
        # so say that plainly rather than reusing the generic stage message.
        problems.error(StageError(
            f"We couldn't find any price data for {ticker}. Check the symbol — US symbols "
            f"look like AAPL, Indian ones like TCS.NS.",
            getattr(perr, "detail", "") or f"no price data for {ticker}",
        ))
        return AnalysisResult(ticker=ticker, company_name=company, as_of=datetime.now(),
                              errors=problems.errors, warnings=problems.warnings,
                              details=problems.details)

    # Stage 2 — forecast, news ranking/sentiment, and risk concurrently.
    note("Forecasting, analyzing sentiment, and measuring risk…")
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_fc = ex.submit(ForecastAgent().safe_run, prices, ticker)
        f_np = ex.submit(_prepare_news, raw_news or [], company, top_k, problems)
        f_risk = ex.submit(RiskAgent().safe_run, prices, benchmark_prices, benchmark)
        forecast, ferr = f_fc.result()
        top_articles, sentiment, n_collected, n_after_dedup = f_np.result()
        risk, rkerr = f_risk.result()

    if ferr:
        problems.error(ferr)
    if rkerr:
        problems.warn(rkerr)

    # Stage 3 — one LLM call returns the news summary AND the recommendation together.
    # Halves request usage (it matters on a free tier) and keeps the summary consistent
    # with the sentiment score, because the same prompt carries both.
    recommendation = None
    if forecast is not None:
        note("Writing the news summary and the recommendation…")
        out, aerr = AnalystAgent().safe_run(
            company, ticker, forecast, sentiment, context, top_articles, use_llm,
        )
        if aerr or out is None:
            problems.warn(aerr or "We couldn't generate the summary and recommendation.")
            text = headline_digest(company, top_articles)
        else:
            text, recommendation = out
    else:
        # No forecast means no recommendation to make, but the news is still worth summarizing.
        problems.warn("Without a forecast we can't make a Buy/Hold/Sell call, "
                      "so only the news is shown below.")
        note("Summarizing the news…")
        text, serr = SummarizationAgent().safe_run(company, ticker, top_articles, use_llm)
        if serr or not text:
            if serr:
                problems.warn(serr)
            text = headline_digest(company, top_articles)

    news_result = NewsResult(summary=text, sentiment=sentiment, top_articles=top_articles,
                             n_collected=n_collected, n_after_dedup=n_after_dedup)

    # Accumulate today's sentiment reading. The news API only serves ~4 weeks of history, so
    # this is the only way a sentiment feature can ever have a trainable past — see
    # `technical_analysis/features.attach_sentiment`.
    if sentiment.n_articles:
        from database.db import record_sentiment

        record_sentiment(ticker, datetime.now().strftime("%Y-%m-%d"),
                         sentiment.weighted_score, sentiment.label, sentiment.n_articles)

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
        errors=problems.errors,
        warnings=problems.warnings,
        details=problems.details,
    )
