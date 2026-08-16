"""Recommendation engine: fuse forecast + sentiment + context into Buy/Hold/Sell.

Primary path: grounded Gemini reasoning (structured JSON), with a grounding post-check
that strips any fabricated URLs. Fallback path: a deterministic rule that combines the
forecast signal and weighted sentiment, so the app ALWAYS returns a recommendation.
"""
from __future__ import annotations

import hashlib
import math
import re

from pydantic import BaseModel, Field

from config.logging_config import get_logger
from database import cache
from llm.client import LLMUnavailable, get_llm
from llm.prompts import combined_prompt
from llm.summarizer import headline_digest
from orchestration.schemas import (
    Action,
    Article,
    DISCLAIMER,
    ForecastResult,
    MarketContext,
    Recommendation,
    SentimentSummary,
)

logger = get_logger("recommendation")

_URL_RE = re.compile(r"https?://\S+")
_CACHE_TTL_MINUTES = 24 * 60


def _cache_key(ticker: str, forecast: ForecastResult, sentiment: SentimentSummary,
               articles: list[Article], prefix: str = "llm_reco") -> str:
    """Key on the signals the LLM actually sees, so a cached call is only reused when
    the forecast, the sentiment and the article set are all effectively unchanged.

    Values are rounded rather than exact: gradient-boosted fits can wobble in the last
    few decimal places across threads, and that should not cause a cache miss.
    """
    nd = forecast.ensemble.next_day
    metrics = forecast.ensemble.metrics
    parts = [
        ticker,
        f"{nd.predicted_return:.4f}" if nd else "na",
        str(forecast.beats_baseline),
        f"{metrics.directional_accuracy:.3f}" if metrics else "na",
        f"{sentiment.weighted_score:.3f}",
        sentiment.label,
        "|".join(sorted(a.url for a in articles if a.url)),
    ]
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{ticker}_{digest}"


class RecommendationDraft(BaseModel):
    """LLM response schema (disclaimer is added by us, not the model)."""

    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: str
    positive_factors: list[str] = Field(default_factory=list)
    negative_factors: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)


class CombinedDraft(BaseModel):
    """One response carrying both the news summary and the recommendation.

    `news_summary` is declared first on purpose: the model writes the summary before it
    commits to a verdict, so the decision follows the evidence rather than preceding it.
    """

    news_summary: str
    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: str
    positive_factors: list[str] = Field(default_factory=list)
    negative_factors: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)


def _ground(items: list[str], valid_urls: set[str]) -> list[str]:
    """Drop any factor that cites a URL not present in the provided articles."""
    cleaned: list[str] = []
    for item in items:
        urls = _URL_RE.findall(item)
        if urls and not all(any(u.startswith(v) or v.startswith(u) for v in valid_urls) for u in urls):
            logger.warning("dropping ungrounded factor with fabricated URL: %s", item[:80])
            continue
        cleaned.append(item)
    return cleaned


def _finalize(draft: RecommendationDraft, articles: list[Article]) -> Recommendation:
    valid = {a.url for a in articles if a.url}
    action: Action = draft.action if draft.action in ("Buy", "Hold", "Sell") else "Hold"
    return Recommendation(
        action=action,
        confidence=float(max(0.0, min(1.0, draft.confidence))),
        thesis=draft.thesis.strip(),
        positive_factors=_ground(draft.positive_factors, valid),
        negative_factors=_ground(draft.negative_factors, valid),
        risks=_ground(draft.risks, valid),
        opportunities=_ground(draft.opportunities, valid),
        disclaimer=DISCLAIMER,
    )


def _rule_based(forecast: ForecastResult, sentiment: SentimentSummary) -> Recommendation:
    """Deterministic fallback when the LLM is unavailable."""
    nd = forecast.ensemble.next_day
    ret = nd.predicted_return if nd else 0.0
    fscore = math.tanh(ret * 20.0)                 # squash daily log-return to ~[-1,1]
    sscore = max(-1.0, min(1.0, sentiment.weighted_score))

    # Trust the price model less when it shows no skill over the baseline.
    fw, sw = (0.5, 0.5) if forecast.beats_baseline else (0.25, 0.75)
    signal = fw * fscore + sw * sscore

    if signal > 0.15:
        action: Action = "Buy"
    elif signal < -0.15:
        action = "Sell"
    else:
        action = "Hold"

    confidence = 0.35 + min(0.4, abs(signal) * 0.5)
    if not forecast.beats_baseline:
        confidence *= 0.8

    thesis = (
        f"Rule-based assessment (LLM unavailable). The ensemble projects a "
        f"{ret*100:+.2f}% next-day move to ${nd.predicted_price:.2f}; news sentiment is "
        f"{sentiment.label} ({sentiment.weighted_score:+.2f}). "
        + ("The price model beats the naive baseline, so it carries some weight. "
           if forecast.beats_baseline else
           "The price model does NOT beat a naive baseline, so this leans on sentiment and is low-confidence. ")
        + f"Net signal favors {action}."
    )
    pos, neg = [], []
    (pos if ret > 0 else neg).append(f"Ensemble forecast {ret*100:+.2f}% next day")
    (pos if sscore > 0 else neg).append(f"News sentiment {sentiment.label} ({sentiment.weighted_score:+.2f})")
    return Recommendation(
        action=action, confidence=round(confidence, 2), thesis=thesis,
        positive_factors=pos, negative_factors=neg,
        risks=["Daily price direction is near-random; do not over-rely on the point forecast."],
        opportunities=[], disclaimer=DISCLAIMER,
    )


def summarize_and_recommend(
    company: str,
    ticker: str,
    forecast: ForecastResult,
    sentiment: SentimentSummary,
    context: MarketContext,
    articles: list[Article],
    use_llm: bool = True,
) -> tuple[str, Recommendation]:
    """Produce the news summary AND the recommendation in a single Gemini call.

    Halves request usage versus two separate calls — which matters on a free tier, where the
    limit is 20 requests/day per model — and keeps the summary consistent with the sentiment
    score, because this prompt sees both. On failure it degrades to a headline digest plus the
    rule-based recommendation, so a caller always gets both values.
    """
    llm = get_llm()
    if use_llm and llm.available:
        key = _cache_key(ticker, forecast, sentiment, articles, prefix="llm_analyst")
        cached = cache.read_json(key, ttl_minutes=_CACHE_TTL_MINUTES)
        if isinstance(cached, dict) and cached.get("summary") and cached.get("recommendation"):
            try:
                logger.info("analyst output for %s served from cache", ticker)
                return str(cached["summary"]), Recommendation(**cached["recommendation"])
            except Exception:  # noqa: BLE001 - a stale/incompatible entry just regenerates
                logger.warning("cached analyst output for %s was unusable; regenerating", ticker)
        try:
            prompt = combined_prompt(company, ticker, forecast, sentiment, context, articles)
            draft = llm.generate_json(prompt, CombinedDraft)
            reco = _finalize(
                RecommendationDraft(**draft.model_dump(exclude={"news_summary"})), articles
            )
            summary = draft.news_summary.strip() or headline_digest(company, articles)
            cache.write_json(key, {"summary": summary,
                                   "recommendation": reco.model_dump(mode="json")})
            return summary, reco
        except LLMUnavailable as exc:
            logger.warning("analyst LLM unavailable, using fallbacks: %s", exc)

    return headline_digest(company, articles), _rule_based(forecast, sentiment)
