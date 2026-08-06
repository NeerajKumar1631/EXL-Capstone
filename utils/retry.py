"""Retry helper for flaky network calls (yfinance, news APIs)."""
from __future__ import annotations

from typing import Callable, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")


def network_retry(attempts: int = 3) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: retry a network call with exponential backoff on any Exception."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type(Exception),
    )
