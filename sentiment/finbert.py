"""Financial sentiment via ProsusAI/finbert (CPU, cached singleton).

FinBERT has a 512-token limit, so inputs are truncated. Produces a signed compound
score per article: +confidence (positive), −confidence (negative), 0 (neutral).
"""
from __future__ import annotations

import os
import threading
import warnings
from functools import lru_cache

from config.logging_config import get_logger
from config.settings import settings
from orchestration.schemas import Article

logger = get_logger("sentiment.finbert")

_MODEL = "ProsusAI/finbert"
_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _build_finbert():
    """Load FinBERT (≈400 MB, ≈50 s first run, then cached on disk)."""
    if settings.hf_token:
        os.environ.setdefault("HF_TOKEN", settings.hf_token)
    from transformers import pipeline

    logger.info("loading %s on CPU", _MODEL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # device="cpu" is required, not a preference — see embeddings/encoder.py. On Apple
        # Silicon transformers otherwise picks the MPS (GPU) backend, which is unsafe to
        # call from several threads at once and crashed the process. CPU is also what the
        # project documents.
        return pipeline("sentiment-analysis", model=_MODEL, truncation=True,
                        max_length=512, device="cpu")


def get_finbert():
    """Return the shared FinBERT pipeline, loading it at most once.

    Guarded for the same reason as the encoder: the warm-up thread and a live request can
    otherwise both miss the `lru_cache` and load a second copy at the same time.
    """
    with _LOAD_LOCK:
        return _build_finbert()


def is_loaded() -> bool:
    """True once the model is resident in memory."""
    return _build_finbert.cache_info().currsize > 0


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
