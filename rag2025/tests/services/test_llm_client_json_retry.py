"""TDD: json_mode retry storm bound + read-timeout safety cap (S10 perf fix).

Reproduces the ns24 gateway latency tail observed in the benchmark:
- router_classify p95=11.6s, max=59.98s driven by tenacity retry storm
  (stop_after_attempt(4) × exp backoff 1-8s) on json_mode empty-body responses.
- Read timeout: httpx client must have a bounded read timeout (≤30s) so a
  stalled gateway response fails at ~30s rather than hanging.

NO live gateway — every transport call is mocked.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Make services importable.
RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

# Ensure at least one provider is configured so _build_providers returns >0.
os.environ.setdefault("NS24_API_KEY", "sk-test-fake")

from src.services.llm_client import (  # noqa: E402
    CHAT_MAX_ATTEMPTS,
    JSON_MODE_MAX_ATTEMPTS,
    LLM_READ_TIMEOUT_S,
    ProviderConfig,
    UnifiedLLMClient,
)


# --- Helpers ---


def _make_provider(name: str = "primary") -> ProviderConfig:
    return ProviderConfig(
        name=name,
        api_key="sk-test-fake",
        base_url="https://example.test/v1",
        model="deepseek-v4-pro",
        priority=0,
    )


def _empty_body_response() -> MagicMock:
    """A response whose .choices[0].message.content is empty — the ns24
    json_mode failure mode that produces 'Expecting value: line 1 column 1'."""
    msg = MagicMock()
    msg.content = ""
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _good_json_response(payload: dict) -> MagicMock:
    """A streaming response that yields a single chunk with the JSON payload.

    Matches the SSE happy-path: when stream=True, the code iterates
    `async for ev in stream` and reads ev.choices[0].delta.content.
    """
    delta = MagicMock()
    delta.content = '{"ok": true, "v": 1}'
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = "stop"
    ev = MagicMock()
    ev.choices = [choice]
    return ev


# --- Module-level constants ---


def test_constants_are_bounded():
    """The retry constants must encode the perf fix: chat keeps 4, json_mode drops to 2."""
    assert CHAT_MAX_ATTEMPTS == 4, "chat path must keep 4 attempts (gen can hit transient 5xx)"
    assert JSON_MODE_MAX_ATTEMPTS == 2, (
        "json_mode must cap at 2 to prevent the 30-60s retry storm on empty body"
    )
    assert LLM_READ_TIMEOUT_S <= 30.0, "read timeout must be ≤30s to bound stalled responses"


# --- Read-timeout safety cap ---


def test_http_client_has_bounded_read_timeout():
    """httpx.AsyncClient must be constructed with a bounded read timeout (≤30s)."""
    with patch.dict(os.environ, {"NS24_API_KEY": "sk-test-fake"}):
        c = UnifiedLLMClient()

    timeout = c._http_client.timeout
    # httpx.Timeout: read is the most important cap (stalled body read).
    # The Timeout object exposes the per-phase value directly.
    assert timeout.read is not None, "read timeout must not be None/unbounded"
    assert timeout.read <= 30.0, f"read timeout must be ≤30s, got {timeout.read}"
    # Connect should also be bounded (≤10s) to fail fast on bad gateways.
    assert timeout.connect is not None
    assert timeout.connect <= 10.0
    # No phase may be None (which means 'unbounded' in httpx semantics).
    assert timeout.write is not None
    assert timeout.pool is not None


# --- chat_json retry storm bound (the latency tail fix) ---


@pytest.mark.asyncio
async def test_chat_json_per_provider_budget_then_fallback_provider():
    """Per provider: json_mode retries stop at JSON_MODE_MAX_ATTEMPTS, then one
    non-json attempt budget of CHAT_MAX_ATTEMPTS, then the next provider.
    Every physical call is one ledger record, so recorded attempts == calls."""
    from openai import APIError
    from services import llm_client as lc
    from services.llm_client import ProviderConfig, TokenLedger

    primary = ProviderConfig("ns24", "sk-test-fake", "https://example.test/v1", "gpt-6-luna", 0, None)
    fallback = ProviderConfig(
        "ns24", "sk-test-fake", "https://example.test/v1", "claude-haiku-5-5", 1, "gpt-6-luna"
    )
    c = UnifiedLLMClient(providers=[primary, fallback])
    c.token_ledger = TokenLedger()

    calls_by_model: dict = {}

    async def always_fails(**kwargs):
        calls_by_model[kwargs["model"]] = calls_by_model.get(kwargs["model"], 0) + 1
        raise APIError("an internal error occurred", request=MagicMock(), body=None)

    original_get_client = c._get_client

    def patched_get_client(provider):
        client = original_get_client(provider)
        client.chat.completions.create = AsyncMock(side_effect=always_fails)
        return client

    c._get_client = patched_get_client

    with pytest.raises(RuntimeError):
        await c.chat_json("hi", system_message="be json", max_tokens=8)

    per_provider = lc.JSON_MODE_MAX_ATTEMPTS + lc.CHAT_MAX_ATTEMPTS
    assert calls_by_model["gpt-6-luna"] == per_provider
    assert calls_by_model["claude-haiku-5-5"] == per_provider, "fallback provider was not reached"
    recorded = c.token_ledger.snapshot()
    assert len(recorded) == sum(calls_by_model.values())
    assert {u.model for u in recorded} == {"gpt-6-luna", "claude-haiku-5-5"}
    await c.close()


@pytest.mark.asyncio
async def test_chat_path_retains_wider_retry_budget():
    """chat (non-json) path must NOT be reduced to 2 attempts. Streaming
    generation legitimately hits transient 5xx — keep CHAT_MAX_ATTEMPTS=4."""
    from services import llm_client as lc
    from services.llm_client import ProviderConfig

    primary = ProviderConfig("ns24", "sk-test-fake", "https://example.test/v1", "gpt-6-luna", 0, None)
    c = UnifiedLLMClient(providers=[primary])

    call_count = 0

    from openai import APIError

    async def always_fails(**_kwargs):
        nonlocal call_count
        call_count += 1
        raise APIError("an internal error occurred", request=MagicMock(), body=None)

    original_get_client = c._get_client

    def patched_get_client(p):
        client = original_get_client(p)
        client.chat.completions.create = AsyncMock(side_effect=always_fails)
        return client

    c._get_client = patched_get_client

    with pytest.raises(RuntimeError):
        await c.chat("hi", max_tokens=8)

    # Per provider, chat must use CHAT_MAX_ATTEMPTS (not JSON_MODE_MAX_ATTEMPTS).
    # This single-provider client isolates the per-provider budget.
    assert call_count == lc.CHAT_MAX_ATTEMPTS, (
        f"chat path must make {lc.CHAT_MAX_ATTEMPTS} attempts, got {call_count}"
    )
    assert call_count > lc.JSON_MODE_MAX_ATTEMPTS

    await c.close()


# --- json_mode success path (no regression) ---


@pytest.mark.asyncio
async def test_chat_json_first_attempt_success_no_regression():
    """A successful json_mode call on the FIRST attempt must return the
    parsed dict with no retry, no error."""
    with patch.dict(os.environ, {"NS24_API_KEY": "sk-test-fake"}):
        c = UnifiedLLMClient()

    provider = c._providers[0] if c._providers else _make_provider()
    if not c._providers:
        c._providers = [provider]

    call_count = 0

    async def good(**_kwargs):
        nonlocal call_count
        call_count += 1
        # Return an async iterator over a single chunk (SSE shape).
        async def _aiter():
            yield _good_json_response({"ok": True, "v": 1})

        class _Stream:
            def __aiter__(self):
                return _aiter()

        return _Stream()

    original_get_client = c._get_client

    def patched_get_client(p):
        client = original_get_client(p)
        client.chat.completions.create = AsyncMock(side_effect=good)
        return client

    c._get_client = patched_get_client

    result = await c.chat_json("hi", system_message="be json", max_tokens=8)

    assert result == {"ok": True, "v": 1}
    assert call_count == 1, (
        f"first-attempt success must not retry, got {call_count} calls"
    )

    await c.close()
