"""Sentence embeddings via all-MiniLM-L6-v2 (CPU, cached singleton)."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger("embeddings")

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_encoder():
    """Load the sentence-transformer once (≈90 MB, ≈50 s first run, then cached on disk)."""
    if settings.hf_token:
        os.environ.setdefault("HF_TOKEN", settings.hf_token)
    from sentence_transformers import SentenceTransformer

    logger.info("loading %s", _MODEL_NAME)
    return SentenceTransformer(_MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalized embeddings (so dot product = cosine similarity)."""
    if not texts:
        return np.zeros((0, 384), dtype=float)
    return np.asarray(
        get_encoder().encode(texts, normalize_embeddings=True, show_progress_bar=False),
        dtype=float,
    )
