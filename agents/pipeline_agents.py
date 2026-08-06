"""Thin Agent wrappers over the domain modules.

Each wraps a pre-existing library/module behind the uniform `Agent` interface
(logging + timing + `safe_run` graceful degradation) used by the orchestrator.
"""
from __future__ import annotations

import pandas as pd

from agents.base import Agent
from analytics.risk import compute_risk
from data_ingestion.context import fetch_context
from data_ingestion.news import fetch_news
from data_ingestion.prices import fetch_prices
from forecasting.forecaster import run_forecast
from llm.summarizer import summarize
from recommendation.engine import recommend
from retrieval.dedup import deduplicate
from retrieval.ranker import rank
from sentiment.aggregate import aggregate
from sentiment.finbert import score as finbert_score
from orchestration.schemas import (
    Article,
    ForecastResult,
    MarketContext,
    Recommendation,
    RiskProfile,
    SentimentSummary,
)


class DataCollectionAgent(Agent):
    name = "data_collection"

    def _run(self, ticker: str) -> pd.DataFrame:
        return fetch_prices(ticker)


class ForecastAgent(Agent):
    name = "forecast"

    def _run(self, prices: pd.DataFrame, ticker: str) -> ForecastResult:
        return run_forecast(prices, ticker)


class NewsCollectionAgent(Agent):
    name = "news_collection"

    def _run(self, ticker: str, company: str) -> list[Article]:
        return fetch_news(ticker, company)


class RiskAgent(Agent):
    name = "risk"

    def _run(self, prices, benchmark_prices=None, benchmark_name=None) -> RiskProfile:
        return compute_risk(prices, benchmark_prices, benchmark_name)


class DedupAgent(Agent):
    name = "dedup"

    def _run(self, articles: list[Article]) -> list[Article]:
        return deduplicate(articles)


class RetrievalAgent(Agent):
    name = "retrieval"

    def _run(self, query: str, articles: list[Article], top_k: int) -> list[Article]:
        return rank(query, articles, top_k)


class SentimentAgent(Agent):
    name = "sentiment"

    def _run(self, articles: list[Article]) -> tuple[list[Article], SentimentSummary]:
        scored = finbert_score(articles)
        return scored, aggregate(scored)


class SummarizationAgent(Agent):
    name = "summarization"

    def _run(self, company: str, ticker: str, articles: list[Article], use_llm: bool = True) -> str:
        return summarize(company, ticker, articles, use_llm=use_llm)


class ContextAgent(Agent):
    name = "context"

    def _run(self, ticker: str) -> MarketContext:
        return fetch_context(ticker)


class RecommendationAgent(Agent):
    name = "recommendation"

    def _run(
        self,
        company: str,
        ticker: str,
        forecast: ForecastResult,
        sentiment: SentimentSummary,
        context: MarketContext,
        articles: list[Article],
        use_llm: bool = True,
    ) -> Recommendation:
        return recommend(company, ticker, forecast, sentiment, context, articles, use_llm=use_llm)
