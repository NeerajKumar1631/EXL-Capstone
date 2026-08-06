"""Lightweight timing context manager for logging step durations."""
from __future__ import annotations

import time
from contextlib import contextmanager
from logging import Logger


@contextmanager
def timed(logger: Logger, label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s finished in %.2fs", label, elapsed)
