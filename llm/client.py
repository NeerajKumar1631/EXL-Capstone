"""Gemini client wrapper.

Responsibilities:
- Try the primary model then fallbacks (handles retired models / per-model quota).
- Retry transient errors (429 quota, 503, timeouts) with backoff.
- Structured JSON output validated against a pydantic schema.
Callers should treat `LLMUnavailable` as "degrade to the rule-based path", never a crash.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger("llm")

T = TypeVar("T", bound=BaseModel)

_RETRYABLE = ("429", "resource_exhausted", "503", "unavailable", "500", "internal", "timeout", "deadline")
_MODEL_DEAD = ("404", "not_found", "no longer available", "not available", "permission", "401", "403")
_BACKOFF = (0.6, 1.5, 3.0)
_BACKOFF_BUDGET = sum(_BACKOFF)

# A 429 can mean two very different things:
#   - a short per-minute rate limit  -> worth retrying in a second or two
#   - a per-DAY free-tier quota      -> cannot possibly clear within our backoff budget
# Retrying the second kind wastes ~5s and 4 doomed round trips on *every* call for the
# rest of the day, so we park the model instead and fall straight through to the next one.
_PER_DAY_QUOTA = ("perday", "per day", "requests per day")
_RETRY_DELAY_RE = re.compile(r"retrydelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_DEFAULT_COOLDOWN_SECONDS = 15 * 60


def _quota_cooldown(message: str) -> Optional[float]:
    """Seconds to park a model after a 429, or None if a normal retry could still work."""
    low = message.lower()
    if not any(tok in low for tok in ("429", "resource_exhausted", "quota")):
        return None
    match = _RETRY_DELAY_RE.search(message)
    if match:
        delay = float(match.group(1))
        return delay if delay > _BACKOFF_BUDGET else None
    if any(tok in low for tok in _PER_DAY_QUOTA):
        return _DEFAULT_COOLDOWN_SECONDS
    return None


class LLMUnavailable(RuntimeError):
    """No Gemini model could satisfy the request (quota, retired, network, or no key)."""


class _ModelDead(Exception):
    """Internal: this specific model is unusable — move to the next in the chain."""


class LLMClient:
    def __init__(self) -> None:
        self._client = None
        self._cooldown: dict[str, float] = {}   # model -> monotonic deadline
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return settings.has_gemini

    def _parked(self, model: str) -> bool:
        """True while a model is cooling down after a quota error."""
        with self._lock:
            until = self._cooldown.get(model)
            if until is None:
                return False
            if time.monotonic() >= until:
                del self._cooldown[model]
                return False
            return True

    def _park(self, model: str, seconds: float) -> None:
        with self._lock:
            self._cooldown[model] = time.monotonic() + seconds
        logger.warning("%s is out of quota — skipping it for %.0fs", model, seconds)

    @property
    def _sdk(self):
        # Summarization and recommendation now call this concurrently, so build once.
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from google import genai

                    self._client = genai.Client(api_key=settings.gemini_api_key)
        return self._client

    def _call(self, model: str, contents: str, config):
        last: Optional[Exception] = None
        for attempt in range(len(_BACKOFF) + 1):
            try:
                return self._sdk.models.generate_content(model=model, contents=contents, config=config)
            except Exception as exc:  # noqa: BLE001
                text = str(exc)
                low = text.lower()
                last = exc
                if any(tok in low for tok in _MODEL_DEAD):
                    raise _ModelDead(text) from exc
                cooldown = _quota_cooldown(text)
                if cooldown is not None:
                    # A daily quota will not clear in seconds — park it and move on.
                    self._park(model, cooldown)
                    raise _ModelDead(text) from exc
                if attempt < len(_BACKOFF) and any(tok in low for tok in _RETRYABLE):
                    logger.warning("%s transient (%s); retry in %.1fs", model, type(exc).__name__, _BACKOFF[attempt])
                    time.sleep(_BACKOFF[attempt])
                    continue
                raise
        assert last is not None
        raise last

    def _run(self, prompt: str, config) -> object:
        if not self.available:
            raise LLMUnavailable("no Gemini API key configured")
        last: Optional[Exception] = None
        attempted = 0
        for model in settings.gemini_models:
            if self._parked(model):
                logger.info("skipping %s — still cooling down after a quota error", model)
                continue
            attempted += 1
            try:
                return self._call(model, prompt, config)
            except _ModelDead as exc:
                logger.warning("model %s unavailable, trying next: %.200s", model, exc)
                last = exc
            except Exception as exc:  # noqa: BLE001
                logger.warning("model %s failed: %.200s", model, exc)
                last = exc
        if attempted == 0:
            # Everything is quota-blocked: fail fast so callers degrade immediately
            # instead of waiting on calls we already know will be rejected.
            raise LLMUnavailable("all Gemini models are cooling down after quota errors")
        raise LLMUnavailable(f"all Gemini models failed: {last}")

    def generate_text(self, prompt: str, temperature: float = 0.4) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(temperature=temperature)
        resp = self._run(prompt, config)
        return (getattr(resp, "text", "") or "").strip()

    def generate_json(self, prompt: str, schema: Type[T], temperature: float = 0.3) -> T:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        resp = self._run(prompt, config)
        parsed = getattr(resp, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        try:
            return schema.model_validate(json.loads(getattr(resp, "text", "") or "{}"))
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"could not parse structured output: {exc}") from exc


_client: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
