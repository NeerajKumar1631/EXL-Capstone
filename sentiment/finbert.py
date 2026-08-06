"""Financial sentiment via ProsusAI/finbert (CPU, cached singleton).

FinBERT has a 512-token limit, so inputs are truncated. Produces a signed compound
score per article: +confidence (positive), −confidence (negative), 0 (neutral).
"""
from __future__ import annotations

import os
import warnings
from functools import lru_cache

from config.logging_config import get_logger
from config.settings import settings
from orchestration.schemas import Article

logger = get_logger("sentiment.finbert")

_MODEL = "ProsusAI/finbert"


@lru_cache(maxsize=1)
def get_finbert():
    """Load FinBERT once (≈400 MB, ≈50 s first run, then cached on disk)."""
    if settings.hf_token:
        os.environ.setdefault("HF_TOKEN", settings.hf_token)
    from transformers import pipeline

    logger.info("loading %s", _MODEL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipeline("sentiment-analysis", model=_MODEL, truncation=True, max_length=512)


def score(articles: list[Article]) -> list[Article]:
    """Attach FinBERT sentiment (label, confidence, signed compound) to each article in place."""
    if not articles:
        return articles
    pipe = get_finbert()
    texts = [a.text[:2000] for a in articles]  # tokenizer truncates to 512 tokens
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = pipe(texts, truncation=True, max_length=512, batch_size=8)

    for art, res in zip(articles, results):
        label = str(res["label"]).lower()
        conf = float(res["score"])
        art.sentiment_label = label if label in ("positive", "negative", "neutral") else "neutral"
        art.sentiment_confidence = conf
        art.sentiment_score = conf if art.sentiment_label == "positive" else (
            -conf if art.sentiment_label == "negative" else 0.0
        )
    return articles
