"""News summarization via Gemini, with a deterministic fallback."""
from __future__ import annotations

import hashlib

from config.logging_config import get_logger
from database import cache
from llm.client import LLMUnavailable, get_llm
from llm.prompts import summary_prompt
from orchestration.schemas import Article

logger = get_logger("llm.summarizer")

_CACHE_TTL_MINUTES = 24 * 60


def _cache_key(ticker: str, articles: list[Article]) -> str:
    """Key on the exact article set, so the summary is reused only while the news is unchanged."""
    fingerprint = "|".join(sorted((a.url or a.title) for a in articles))
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16]
    return f"llm_summary_{ticker}_{digest}"


def headline_digest(company: str, articles: list[Article]) -> str:
    """Deterministic no-LLM summary: leading headlines plus tone counts.

    Shared by `summarize()` and by the merged analyst call, so every path degrades to the
    same wording when Gemini is unavailable.
    """
    if not articles:
        return "No recent news was found for this company."
    pos = sum(1 for a in articles if a.sentiment_label == "positive")
    neg = sum(1 for a in articles if a.sentiment_label == "negative")
    heads = "; ".join(a.title for a in articles[:5])
    return (
        f"Recent headlines for {company}: {heads}. "
        f"Sentiment skews {'positive' if pos > neg else 'negative' if neg > pos else 'mixed'} "
        f"({pos} positive / {neg} negative of {len(articles)})."
    )


def summarize(company: str, ticker: str, articles: list[Article], use_llm: bool = True) -> str:
    """Summarize recent news. Falls back to a headline digest if the LLM is unavailable."""
    if not articles:
        return "No recent news was found for this company."
    llm = get_llm()
    if use_llm and llm.available:
        key = _cache_key(ticker, articles)
        cached = cache.read_json(key, ttl_minutes=_CACHE_TTL_MINUTES)
        if isinstance(cached, str) and cached.strip():
            logger.info("summary for %s served from cache", ticker)
            return cached
        try:
            text = llm.generate_text(summary_prompt(company, ticker, articles))
            if text.strip():
                cache.write_json(key, text)
                return text
        except LLMUnavailable as exc:
            logger.warning("summary LLM unavailable, using fallback: %s", exc)

    return headline_digest(company, articles)
