"""Aggregate per-article sentiment into a credibility-weighted summary."""
from __future__ import annotations

from orchestration.schemas import Article, SentimentSummary

_POS_THRESHOLD = 0.05
_NEG_THRESHOLD = -0.05


def aggregate(articles: list[Article]) -> SentimentSummary:
    """Credibility-weighted mean of signed sentiment scores → overall market mood."""
    scored = [a for a in articles if a.sentiment_label is not None]
    if not scored:
        return SentimentSummary(weighted_score=0.0, label="neutral", n_articles=0)

    total_w = sum(a.credibility for a in scored) or 1.0
    weighted = sum(a.sentiment_score * a.credibility for a in scored) / total_w

    if weighted > _POS_THRESHOLD:
        label = "positive"
    elif weighted < _NEG_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    return SentimentSummary(
        weighted_score=round(float(weighted), 4),
        label=label,
        n_articles=len(scored),
        n_positive=sum(1 for a in scored if a.sentiment_label == "positive"),
        n_negative=sum(1 for a in scored if a.sentiment_label == "negative"),
        n_neutral=sum(1 for a in scored if a.sentiment_label == "neutral"),
    )
