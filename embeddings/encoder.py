"""Sentence embeddings via all-MiniLM-L6-v2 (CPU, cached singleton)."""
from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger("embeddings")

_MODEL_NAME = "all-MiniLM-L6-v2"
_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _build_encoder():
    """Load the sentence-transformer (≈90 MB, ≈50 s first run, then cached on disk)."""
    if settings.hf_token:
        os.environ.setdefault("HF_TOKEN", settings.hf_token)
    from sentence_transformers import SentenceTransformer

    logger.info("loading %s on CPU", _MODEL_NAME)
    # device="cpu" is required, not a preference. On Apple Silicon sentence-transformers
    # silently selects the MPS (GPU) backend, and MPS is not safe to call from several
    # Python threads at once — the pipeline runs embedding and forecasting concurrently,
    # which crashed the process with a pointer-authentication fault inside
    # `MetalShaderLibrary::exec_unary_kernel`. CPU also matches the documented design.
    return SentenceTransformer(_MODEL_NAME, device="cpu")


def get_encoder():
    """Return the shared encoder, loading it at most once.

    `lru_cache` alone is not enough: the background warm-up thread and an in-flight
    request can both miss the cache and construct a second copy of the model
    simultaneously. The lock makes the first load exclusive; later calls just take an
    uncontended lock.
    """
    with _LOAD_LOCK:
        return _build_encoder()


def is_loaded() -> bool:
    """True once the model is resident in memory."""
    return _build_encoder.cache_info().currsize > 0


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalized embeddings (so dot product = cosine similarity)."""
    if not texts:
        return np.zeros((0, 384), dtype=float)
    return np.asarray(
        get_encoder().encode(texts, normalize_embeddings=True, show_progress_bar=False),
        dtype=float,
    )
