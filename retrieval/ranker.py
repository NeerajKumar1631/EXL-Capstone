"""Relevance ranking: BM25 (lexical) + semantic cosine, fused.

Filters noisy news down to the most relevant articles for a company before sentiment
and LLM stages.
"""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from config.logging_config import get_logger
from embeddings.encoder import embed
from orchestration.schemas import Article

logger = get_logger("retrieval.ranker")


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def rank(query: str, articles: list[Article], top_k: int, alpha: float = 0.6) -> list[Article]:
    """Return the top_k articles by fused relevance (alpha·semantic + (1-alpha)·lexical).

    Sets `relevance_score` on each returned article. Falls back gracefully if embeddings
    are unavailable (lexical-only).
    """
    if not articles:
        return []
    texts = [a.text for a in articles]

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm_scores = _minmax(bm25.get_scores(query.lower().split()))

    try:
        doc_emb = embed(texts)
        q_emb = embed([query])[0]
        sem_scores = _minmax(doc_emb @ q_emb)
    except Exception as exc:  # noqa: BLE001
        logger.warning("semantic ranking unavailable, using lexical only: %s", exc)
        sem_scores = bm_scores
        alpha = 0.0

    fused = alpha * sem_scores + (1.0 - alpha) * bm_scores
    for art, score in zip(articles, fused):
        art.relevance_score = float(score)

    return sorted(articles, key=lambda a: a.relevance_score, reverse=True)[:top_k]
