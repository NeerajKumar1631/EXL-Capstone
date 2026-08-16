"""Uniform Agent interface.

Each capability in the pipeline is exposed as an `Agent` with a single `run()`
method, consistent logging, timing, and error handling. The heavy lifting is done
by the domain modules (which wrap pre-existing libraries); agents are thin
orchestration wrappers around them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from config.logging_config import get_logger
from utils.timing import timed

# What a user should be told when a stage fails. The technical text still goes to the log
# and to `StageError.detail`; nobody should have to read `ValueError` off a dashboard.
_FRIENDLY: dict[str, str] = {
    "data_collection": "We couldn't load price data for this stock.",
    "forecast": "We couldn't build a forecast from the available price history.",
    "news_collection": "We couldn't load recent news, so the analysis uses price data only.",
    "dedup": "We couldn't remove duplicate articles, so some news may repeat.",
    "retrieval": "We couldn't rank the news by relevance, so the most recent articles were used.",
    "sentiment": "We couldn't score the news sentiment, so it is treated as neutral.",
    "summarization": "We couldn't write a news summary, so headlines are shown instead.",
    "context": "We couldn't load fundamentals or market context for this stock.",
    "risk": "We couldn't calculate the risk profile for this stock.",
    "analyst": "We couldn't generate the summary and recommendation.",
}
_FALLBACK = "Part of the analysis could not be completed."


class StageError(str):
    """A plain-English failure message that also carries the technical detail.

    Subclasses `str` so existing call sites can keep treating it as one; use `.detail`
    when the underlying exception text is wanted.
    """

    detail: str

    def __new__(cls, message: str, detail: str) -> "StageError":
        obj = super().__new__(cls, message)
        obj.detail = detail
        return obj


class Agent(ABC):
    """Base class for all agents. Subclasses implement `_run`."""

    name: str = "agent"

    def __init__(self) -> None:
        self.logger = get_logger(self.name)

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Do the work. Implemented by subclasses."""

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the agent, logging start/finish and timing. Exceptions propagate."""
        self.logger.info("start")
        with timed(self.logger, self.name):
            return self._run(*args, **kwargs)

    def safe_run(self, *args: Any, **kwargs: Any) -> tuple[Optional[Any], Optional[StageError]]:
        """Run the agent but never raise: returns (result, error).

        Used by the orchestrator so one failing stage degrades gracefully instead of
        killing the whole analysis. The error reads as a plain sentence for the user and
        keeps the exception text on `.detail` for debugging.
        """
        try:
            return self.run(*args, **kwargs), None
        except Exception as exc:  # noqa: BLE001 - deliberate boundary
            detail = f"{self.name}: {type(exc).__name__}: {exc}"
            self.logger.exception(detail)
            return None, StageError(_FRIENDLY.get(self.name, _FALLBACK), detail)
