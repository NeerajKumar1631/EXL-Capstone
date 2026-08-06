"""News summarization via Gemini, with a deterministic fallback."""
from __future__ import annotations

from config.logging_config import get_logger
from llm.client import LLMUnavailable, get_llm
from llm.prompts import summary_prompt
from orchestration.schemas import Article

logger = get_logger("llm.summarizer")


def summarize(company: str, ticker: str, articles: list[Article], use_llm: bool = True) -> str:
    """Summarize recent news. Falls back to a headline digest if the LLM is unavailable."""
    if not articles:
        return "No recent news was found for this company."
    llm = get_llm()
    if use_llm and llm.available:
        try:
            return llm.generate_text(summary_prompt(company, ticker, articles))
        except LLMUnavailable as exc:
            logger.warning("summary LLM unavailable, using fallback: %s", exc)

    # Fallback: concise headline digest with tone counts.
    pos = sum(1 for a in articles if a.sentiment_label == "positive")
    neg = sum(1 for a in articles if a.sentiment_label == "negative")
    heads = "; ".join(a.title for a in articles[:5])
    return (
        f"Recent headlines for {company}: {heads}. "
        f"Sentiment skews {'positive' if pos > neg else 'negative' if neg > pos else 'mixed'} "
        f"({pos} positive / {neg} negative of {len(articles)})."
    )
