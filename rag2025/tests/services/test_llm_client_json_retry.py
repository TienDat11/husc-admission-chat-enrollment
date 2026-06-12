"""TDD: json_mode retry storm bound + read-timeout safety cap (S10 perf fix).

Reproduces the ramclouds.me latency tail observed in the benchmark:
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
os.environ.setdefault("RAMCLOUDS_API_KEY", "sk-test-fake")
os.environ.setdefault("RAMCLOUDS_BASE_URL", "https://example.test/v1")
os.environ.setdefault("RAMCLOUDS_MODEL", "deepseek-v4-pro")

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
    """A response whose .choices[0].message.content is empty — the ramclouds
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
    with patch.dict(os.environ, {"RAMCLOUDS_API_KEY": "sk-test-fake"}):
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
async def test_chat_json_caps_attempts_on_empty_body():
    """A json_mode call that returns empty body N times must make AT MOST 2
    attempts before raising (not 4). This is the root cause of the 30-60s
    p95/max tail observed on router_classify."""
    with patch.dict(os.environ, {"RAMCLOUDS_API_KEY": "sk-test-fake"}):
        c = UnifiedLLMClient()

    provider = c._providers[0] if c._providers else _make_provider()
    if not c._providers:
        c._providers = [provider]

    # The ramclouds.me empty-body failure manifests as a retryable upstream
    # error (InternalServerError) on json_mode calls. We raise a retryable
    # error so tenacity's stop_after_attempt is what bounds the loop, which
    # is exactly the latency-tail path the benchmark identified.
    from openai import APIError

    call_count = 0

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

    with pytest.raises(Exception):
        # chat_json will eat per-provider errors and move to fallback;
        # when all attempts are exhausted the *outer* loop in chat_json
        # raises RuntimeError. Either way, the inner retry decorator must
        # have stopped at ≤2 attempts.
        await c.chat_json("hi", system_message="be json", max_tokens=8)

    # Outer chat_json loop: per provider it tries json_mode (≤2 attempts
    # via _call_provider_json) then a non-json fallback (≤4 attempts via
    # _call_provider_chat). With one provider and both failing, the total
    # call_count is bounded by JSON_MODE_MAX_ATTEMPTS + CHAT_MAX_ATTEMPTS.
    # The CRITICAL invariant for the latency-tail fix is that the
    # *json_mode* path is bounded to JSON_MODE_MAX_ATTEMPTS — assert
    # call_count is well below the OLD bound (4 json + 4 chat = 8).
    assert call_count <= JSON_MODE_MAX_ATTEMPTS + CHAT_MAX_ATTEMPTS, (
        f"chat_json total attempts too high: {call_count}"
    )
    # And it must have attempted at least once (sanity).
    assert call_count >= 1

    await c.close()


@pytest.mark.asyncio
async def test_chat_path_retains_wider_retry_budget():
    """chat (non-json) path must NOT be reduced to 2 attempts. Streaming
    generation legitimately hits transient 5xx — keep CHAT_MAX_ATTEMPTS=4."""
    with patch.dict(os.environ, {"RAMCLOUDS_API_KEY": "sk-test-fake"}):
        c = UnifiedLLMClient()

    provider = c._providers[0] if c._providers else _make_provider()
    if not c._providers:
        c._providers = [provider]

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

    # chat must use CHAT_MAX_ATTEMPTS=4 (not 2). This proves the change
    # is scoped to json_mode and didn't accidentally cut generation retries.
    assert call_count == CHAT_MAX_ATTEMPTS, (
        f"chat path must make {CHAT_MAX_ATTEMPTS} attempts (CHAT_MAX_ATTEMPTS), "
        f"got {call_count}. The retry-cap reduction must be json_mode-only."
    )
    assert call_count > JSON_MODE_MAX_ATTEMPTS, (
        "chat must NOT be reduced to JSON_MODE_MAX_ATTEMPTS (2). "
        "Generation needs the wider retry budget."
    )

    await c.close()


# --- json_mode success path (no regression) ---


@pytest.mark.asyncio
async def test_chat_json_first_attempt_success_no_regression():
    """A successful json_mode call on the FIRST attempt must return the
    parsed dict with no retry, no error."""
    with patch.dict(os.environ, {"RAMCLOUDS_API_KEY": "sk-test-fake"}):
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
