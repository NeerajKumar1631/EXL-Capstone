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

    def safe_run(self, *args: Any, **kwargs: Any) -> tuple[Optional[Any], Optional[str]]:
        """Run the agent but never raise: returns (result, error_message).

        Used by the orchestrator so one failing stage degrades gracefully instead
        of killing the whole analysis.
        """
        try:
            return self.run(*args, **kwargs), None
        except Exception as exc:  # noqa: BLE001 - deliberate boundary
            msg = f"{self.name} failed: {type(exc).__name__}: {exc}"
            self.logger.exception(msg)
            return None, msg
