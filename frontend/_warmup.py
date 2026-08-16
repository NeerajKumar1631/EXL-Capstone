"""Load the local ML models in the background as soon as the app starts.

FinBERT (~400 MB) and MiniLM (~90 MB) are `lru_cache` singletons, so they load once per
server process — but by default that happens inside the first analysis, so whoever clicks
first pays for it (~13 s, measured). Starting the load at boot moves it off the critical
path: the models load while the user is still choosing a ticker.

This only changes *when* the load happens, not how often.
"""
from __future__ import annotations

import threading

from config.logging_config import get_logger

logger = get_logger("warmup")

_lock = threading.Lock()
_started = False


def _load() -> None:
    from embeddings.encoder import get_encoder
    from sentiment.finbert import get_finbert

    for name, loader in (("MiniLM", get_encoder), ("FinBERT", get_finbert)):
        try:
            loader()
            logger.info("%s warmed up", name)
        except Exception as exc:  # noqa: BLE001 - warm-up must never break the app
            logger.warning("%s warm-up failed; it will load on demand instead: %s", name, exc)


def start() -> None:
    """Begin warming the models in a daemon thread. Cheap and safe to call on every rerun.

    The guard is module-level, not session state: the models are shared by the whole
    server process, so this must run once per process rather than once per visitor.
    """
    global _started
    with _lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_load, name="stocksense-warmup", daemon=True).start()


def ready() -> bool:
    """True once both models are resident, so the UI can say so honestly."""
    from embeddings.encoder import is_loaded as encoder_loaded
    from sentiment.finbert import is_loaded as finbert_loaded

    return encoder_loaded() and finbert_loaded()
