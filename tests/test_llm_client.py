"""LLM client — model fallback, retry policy, and quota handling.

Entirely offline: the Gemini SDK is replaced with scripted fakes. The behaviour under test is
what made analyses slow before v3 — a per-day quota being retried as if it were a blip.
"""
import time
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from llm.client import LLMClient, LLMUnavailable, _quota_cooldown

# Abridged from a real Gemini response.
PER_DAY_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current "
    "quota', 'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': "
    "[{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}, {'@type': "
    "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '50s'}]}}"
)
PER_MINUTE_429 = "429 RESOURCE_EXHAUSTED 'retryDelay': '2s'"


class Draft(BaseModel):
    value: str


@pytest.fixture(autouse=True)
def no_sleeping():
    """Retry backoff must not make the suite slow."""
    with patch("llm.client.time.sleep"):
        yield


@pytest.fixture
def client():
    """A client whose model chain is exactly ['model-a', 'model-b'].

    `gemini_models` is a computed property, so drive it through the fields it reads rather
    than patching the property itself.
    """
    from config.settings import settings

    saved = (settings.gemini_api_key, settings.gemini_model, settings.gemini_model_fallbacks)
    settings.gemini_api_key = "test-key"
    settings.gemini_model = "model-a"
    settings.gemini_model_fallbacks = ("model-b",)
    try:
        assert settings.gemini_models == ["model-a", "model-b"]
        yield LLMClient()
    finally:
        (settings.gemini_api_key, settings.gemini_model,
         settings.gemini_model_fallbacks) = saved


def _sdk(side_effects):
    """A fake genai client whose generate_content follows a script."""
    sdk = MagicMock()
    sdk.models.generate_content.side_effect = side_effects
    return sdk


# ── classifying a 429 ──────────────────────────────────────
def test_per_day_quota_is_parked_not_retried():
    cooldown = _quota_cooldown(PER_DAY_429)
    assert cooldown == 50.0            # taken from the API's own retryDelay


def test_short_rate_limit_stays_retryable():
    assert _quota_cooldown(PER_MINUTE_429) is None


def test_per_day_wording_without_a_delay_still_parks():
    assert _quota_cooldown("429 quota GenerateRequestsPerDayPerProject") == 15 * 60


def test_non_quota_errors_are_not_parked():
    assert _quota_cooldown("503 UNAVAILABLE overloaded") is None
    assert _quota_cooldown("deadline exceeded") is None


# ── fallback behaviour ─────────────────────────────────────
def test_falls_through_to_the_next_model_when_one_is_out_of_quota(client):
    ok = MagicMock(text="hello")
    client._client = _sdk([Exception(PER_DAY_429), ok])
    assert client.generate_text("hi") == "hello"
    assert client._parked("model-a")           # the dead model is remembered
    assert not client._parked("model-b")


def test_a_parked_model_is_skipped_on_the_next_call(client):
    ok = MagicMock(text="second")
    client._client = _sdk([Exception(PER_DAY_429), MagicMock(text="first"), ok])
    client.generate_text("one")                # discovers model-a is dead
    calls_before = client._client.models.generate_content.call_count
    client.generate_text("two")                # must not try model-a again
    used = [c.kwargs.get("model") for c in
            client._client.models.generate_content.call_args_list[calls_before:]]
    assert "model-a" not in used


def test_short_rate_limit_is_retried_on_the_same_model(client):
    ok = MagicMock(text="recovered")
    client._client = _sdk([Exception(PER_MINUTE_429), ok])
    assert client.generate_text("hi") == "recovered"
    assert not client._parked("model-a")       # a brief limit must not park the model


def test_all_models_exhausted_raises_so_callers_can_degrade(client):
    client._client = _sdk([Exception(PER_DAY_429), Exception(PER_DAY_429)])
    with pytest.raises(LLMUnavailable):
        client.generate_text("hi")


def test_everything_parked_fails_fast_without_calling_the_api(client):
    client._park("model-a", 300)
    client._park("model-b", 300)
    client._client = _sdk([MagicMock(text="should not be reached")])
    with pytest.raises(LLMUnavailable, match="cooling down"):
        client.generate_text("hi")
    assert client._client.models.generate_content.call_count == 0


def test_cooldown_expires(client):
    client._park("model-a", 300)
    assert client._parked("model-a")
    client._cooldown["model-a"] = time.monotonic() - 1     # pretend it elapsed
    assert not client._parked("model-a")


def test_no_api_key_is_unavailable_rather_than_a_crash():
    from config.settings import settings

    saved = settings.gemini_api_key
    settings.gemini_api_key = ""
    try:
        with pytest.raises(LLMUnavailable, match="no Gemini API key"):
            LLMClient().generate_text("hi")
    finally:
        settings.gemini_api_key = saved


# ── structured output ──────────────────────────────────────
def test_structured_output_uses_the_parsed_object(client):
    client._client = _sdk([MagicMock(parsed=Draft(value="ok"), text="{}")])
    assert client.generate_json("hi", Draft).value == "ok"


def test_structured_output_falls_back_to_parsing_the_text(client):
    resp = MagicMock(parsed=None, text='{"value": "from-text"}')
    client._client = _sdk([resp])
    assert client.generate_json("hi", Draft).value == "from-text"


def test_unparseable_structured_output_raises_llm_unavailable(client):
    client._client = _sdk([MagicMock(parsed=None, text="not json at all")])
    with pytest.raises(LLMUnavailable, match="structured output"):
        client.generate_json("hi", Draft)
