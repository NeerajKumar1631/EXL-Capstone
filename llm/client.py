"""Gemini client wrapper.

Responsibilities:
- Try the primary model then fallbacks (handles retired models / per-model quota).
- Retry transient errors (429 quota, 503, timeouts) with backoff.
- Structured JSON output validated against a pydantic schema.
Callers should treat `LLMUnavailable` as "degrade to the rule-based path", never a crash.
"""
from __future__ import annotations

import json
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


class LLMUnavailable(RuntimeError):
    """No Gemini model could satisfy the request (quota, retired, network, or no key)."""


class _ModelDead(Exception):
    """Internal: this specific model is unusable — move to the next in the chain."""


class LLMClient:
    def __init__(self) -> None:
        self._client = None

    @property
    def available(self) -> bool:
        return settings.has_gemini

    @property
    def _sdk(self):
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
                msg = str(exc).lower()
                last = exc
                if any(tok in msg for tok in _MODEL_DEAD):
                    raise _ModelDead(str(exc)) from exc
                if attempt < len(_BACKOFF) and any(tok in msg for tok in _RETRYABLE):
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
        for model in settings.gemini_models:
            try:
                return self._call(model, prompt, config)
            except _ModelDead as exc:
                logger.warning("model %s unavailable, trying next: %s", model, exc)
                last = exc
            except Exception as exc:  # noqa: BLE001
                logger.warning("model %s failed: %s", model, exc)
                last = exc
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
