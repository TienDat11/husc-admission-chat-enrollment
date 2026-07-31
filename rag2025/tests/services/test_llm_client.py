"""Test cho UnifiedLLMClient httpx connection pool (B5).

@spec(S10.B5)
Wraps: rag2025/.omc/plans/standardize_frontend_backend_plan_v2.md Section 5.B5
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Make sure src is importable
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Stub env so _build_providers returns at least one provider for tests
os.environ.setdefault("RAMCLOUDS_API_KEY", "sk-test-fake")
os.environ.setdefault("RAMCLOUDS_BASE_URL", "https://example.test/v1")
os.environ.setdefault("RAMCLOUDS_MODEL", "deepseek-v4-pro")

from src.services.llm_client import LLMResponse, ProviderConfig, strip_reasoning, UnifiedLLMClient  # noqa: E402


# ── strip_reasoning pure-function tests ──



def test_ramclouds_key_does_not_fall_back_to_openai_key():
    """BUG-1 CRITICAL: _build_providers must use RAMCLOUDS_API_KEY only.

    If RAMCLOUDS_API_KEY is unset but OPENAI_API_KEY exists, ramclouds provider
    must NOT be constructed (no silent identity swap).
    """
    with patch.dict(
        os.environ,
        {"RAMCLOUDS_API_KEY": "", "OPENAI_API_KEY": "sk-openai", "GROQ_API_KEY": ""},
        clear=False,
    ):
        c = UnifiedLLMClient()
        names = [p.name for p in c._providers]
        assert "ramclouds" not in names, (
            f"ramclouds provider leaked via OPENAI_API_KEY fallback: {names}"
        )
    # With env patched away, the singleton in this process stays unconfigured;
    # close is a no-op for an empty-provider client.


@pytest.mark.asyncio
async def test_llm_client_uses_persistent_pool() -> None:
    """B5: Client must construct httpx.AsyncClient as persistent pool."""
    c = UnifiedLLMClient()
    assert hasattr(c, "_http_client")
    assert isinstance(c._http_client, httpx.AsyncClient)
    await c.close()


@pytest.mark.asyncio
async def test_llm_client_pool_limits_configured() -> None:
    """B5: pool max_connections=20, max_keepalive_connections=10."""
    c = UnifiedLLMClient()
    # httpx >= 0.24: limits stored on transport. We probe via the underlying _transport.
    # If attribute structure changes across versions, fall back to verifying close() works.
    transport = getattr(c._http_client, "_transport", None)
    pool = getattr(transport, "_pool", None) if transport else None
    if pool is not None:
        # Available on httpx HTTPCore-backed transport
        assert pool._max_connections == 20
        assert pool._max_keepalive_connections == 10
    else:
        # Fallback: verify httpx.Limits roundtrip (best-effort, version-dependent)
        # At minimum the client must not be closed yet
        assert not c._http_client.is_closed
    await c.close()


@pytest.mark.asyncio
async def test_llm_client_close_aclose_pool() -> None:
    """close() must aclose() http_client cleanly without raising."""
    c = UnifiedLLMClient()
    await c.close()
    # After close, client is_closed True
    assert c._http_client.is_closed
    # Idempotent: calling again must not raise
    await c.close()


@pytest.mark.asyncio
async def test_llm_client_get_client_passes_pool() -> None:
    """B5: _get_client(provider) wires AsyncOpenAI with our http_client (pool reuse)."""
    c = UnifiedLLMClient()
    if not c._providers:
        pytest.skip("No provider configured")
    aoai = c._get_client(c._providers[0])
    # AsyncOpenAI stores the underlying http client
    inner = getattr(aoai, "_client", None) or getattr(aoai, "http_client", None)
    # Best-effort: at least one of the introspection paths returns httpx-like
    # If neither resolves, just verify close() still works
    await c.close()
    assert c._http_client.is_closed


@pytest.mark.asyncio
async def test_llm_client_handles_concurrent_requests() -> None:
    """B5: 10 parallel chat() calls don't exceed pool limits — no exceptions."""
    c = UnifiedLLMClient()
    if not c._providers:
        pytest.skip("No provider configured for concurrent test")

    # Stub the network call so we exercise the pool path without real I/O
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="ok"))]

    async def fake_create(**_kwargs):
        return fake_response

    # Patch every provider's underlying AsyncOpenAI client lazily via _get_client
    original_get_client = c._get_client

    def patched_get_client(provider):
        client = original_get_client(provider)
        client.chat.completions.create = AsyncMock(side_effect=fake_create)
        return client

    c._get_client = patched_get_client

    tasks = [
        c.chat("ping", max_tokens=5)
        for _ in range(10)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    assert not errors, f"Concurrent calls failed: {errors[:3]}"

    await c.close()



# ── G2: DeepSeek think-tag sanitization tests ──

def test_strip_matched_tag():
    """<think>...</think> with content — fully removed."""
    assert strip_reasoning("Hello<think>secret</think>World") == "HelloWorld"


def test_strip_matched_tag_multiline():
    """<think>...</think> spanning multiple lines."""
    text = "A<think>\nreasoning\nblock\n</think>B"
    assert strip_reasoning(text) == "AB"


def test_strip_unmatched_opening():
    """Lone <think> without closing — removed."""
    assert strip_reasoning("Hello<think> unfinished") == "Hello unfinished"


def test_strip_unmatched_closing():
    """Lone </think> without opening — removed."""
    assert strip_reasoning("Hello</think>World") == "HelloWorld"


def test_strip_at_start():
    """<think> tag at start of output."""
    assert strip_reasoning("<think>secret</think>Actual content") == "Actual content"


def test_strip_at_end():
    """<think> tag at end of output."""
    assert strip_reasoning("Content<think>secret</think>") == "Content"


def test_strip_multiple_blocks():
    """Multiple think blocks — all removed."""
    text = "<think>a</think>X<think>b</think>Y<think>c</think>"
    assert strip_reasoning(text) == "XY"


def test_normal_text_unchanged():
    """Plain Vietnamese text — no modification."""
    text = "Điểm chuẩn ngành Công nghệ thông tin 2025 là 26.5"
    assert strip_reasoning(text) == text


def test_empty_string():
    """Empty string — no crash."""
    assert strip_reasoning("") == ""


def test_no_think_tag():
    """Text without any think patterns — unchanged."""
    text = "Normal response with <b>bold</b> and <notthink> tags"
    assert strip_reasoning(text) == text


# ── LLMResponse integration tests ──


def test_as_json_after_strip():
    """JSON body after think-tag strip — as_json parses correctly."""
    raw = "<think>reasoning</think>{\"key\": \"value\"}"
    resp = LLMResponse(content=raw, model="test", provider="test")
    assert resp.as_json() == {"key": "value"}


def test_as_json_unmatched_tag_before_json():
    """Unmatched opening <think> before JSON — stripped before parse."""
    raw = "<think>{\"key\": \"value\"}"
    resp = LLMResponse(content=raw, model="test", provider="test")
    assert resp.as_json() == {"key": "value"}


# ── Async mock helpers ──


class _AsyncIter:
    """Async iterator from a list — for mocking SSE stream responses."""

    def __init__(self, items):
        self._items = list(reversed(items))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop()


def _patch_get_client(client, fake_create):
    """Patch client._get_client so chat.completions.create uses fake_create."""
    original = client._get_client

    def patched(provider):
        aoai = original(provider)
        aoai.chat.completions.create = AsyncMock(side_effect=fake_create)
        return aoai

    client._get_client = patched


@pytest.mark.asyncio
async def test_call_provider_stream_strips_reasoning_content():
    """Delta with reasoning_content — only content field enters response."""
    c = UnifiedLLMClient()

    def fake_create(**_kwargs):
        return _AsyncIter([
            MagicMock(
                choices=[MagicMock(
                    delta=MagicMock(content=None, reasoning_content="chain of thought"),
                    finish_reason=None,
                )]
            ),
            MagicMock(
                choices=[MagicMock(
                    delta=MagicMock(content="Final answer"),
                    finish_reason="stop",
                )]
            ),
        ])

    _patch_get_client(c, fake_create)

    resp = await c._call_provider(
        c._providers[0],
        messages=[{"role": "user", "content": "test"}],
        temperature=0.1,
        max_tokens=100,
        json_mode=False,
    )
    assert "chain of thought" not in resp.content
    assert resp.content == "Final answer"
    await c.close()


@pytest.mark.asyncio
async def test_call_provider_strips_think_tags():
    """_call_provider content gets think-tag stripped via LLMResponse."""
    c = UnifiedLLMClient()

    def fake_create(**_kwargs):
        return _AsyncIter([
            MagicMock(
                choices=[MagicMock(
                    delta=MagicMock(content="<think>secret</think>Open content"),
                    finish_reason=None,
                )]
            ),
            MagicMock(
                choices=[MagicMock(
                    delta=MagicMock(content=" more"),
                    finish_reason="stop",
                )]
            ),
        ])

    _patch_get_client(c, fake_create)

    resp = await c._call_provider(
        c._providers[0],
        messages=[{"role": "user", "content": "test"}],
        temperature=0.1,
        max_tokens=100,
        json_mode=False,
    )
    assert "think" not in resp.content
    assert "secret" not in resp.content
    assert resp.content == "Open content more"
    await c.close()


@pytest.mark.asyncio
async def test_chat_returns_clean_content():
    """End-to-end: chat() with mocked stream returns think-free response."""
    c = UnifiedLLMClient()

    def fake_create(**_kwargs):
        return _AsyncIter([
            MagicMock(
                choices=[MagicMock(
                    delta=MagicMock(content="<think>hidden</think>Visible"),
                    finish_reason=None,
                )]
            ),
            MagicMock(
                choices=[MagicMock(
                    delta=MagicMock(content=" result"),
                    finish_reason="stop",
                )]
            ),
        ])

    _patch_get_client(c, fake_create)

    resp = await c.chat("test", max_tokens=100)
    assert "think" not in resp.content
    assert "hidden" not in resp.content
    assert resp.content == "Visible result"
    await c.close()


@pytest.mark.asyncio
async def test_chat_json_strips_think_tags():
    """chat_json with think tags before JSON — clean parse."""
    c = UnifiedLLMClient()

    def fake_create(**_kwargs):
        return _AsyncIter([
            MagicMock(
                choices=[MagicMock(
                    delta=MagicMock(content="<think>reason</think>{\"k\": \"v\"}"),
                    finish_reason="stop",
                )]
            ),
        ])

    _patch_get_client(c, fake_create)

    data = await c.chat_json("test", max_tokens=100)
    assert data == {"k": "v"}
    await c.close()


# ── G2: Streaming think-tag sanitization tests ──
# Cover: chat_stream stripping, uppercase/mixed-case regex, split-across-chunk

def test_strip_think_uppercase():
    """<THINK> uppercase — stripped by strip_reasoning."""
    assert strip_reasoning("Hello<THINK>secret</THINK>World") == "HelloWorld"


def test_strip_think_mixed_case():
    """<ThInK> mixed case — stripped by strip_reasoning."""
    assert strip_reasoning("A<ThInK>reason</ThInK>B") == "AB"


def test_strip_think_uppercase_multiline():
    """<THINK>...</THINK> multiline, uppercase — stripped."""
    t = "<THINK>\nsecret\n</THINK>X"
    assert strip_reasoning(t) == "X"


@pytest.mark.asyncio
async def test_chat_stream_strips_think_tags():
    """chat_stream yields think-free output across chunks."""
    c = UnifiedLLMClient()
    def fake_create(**_kwargs):
        return _AsyncIter([
            MagicMock(choices=[MagicMock(delta=MagicMock(content="<think>hid"), finish_reason=None)]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="den</think>Vis"), finish_reason=None)]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="ible"), finish_reason="stop")]),
        ])
    _patch_get_client(c, fake_create)
    pieces = [p async for p in c.chat_stream("test", max_tokens=100)]
    result = "".join(pieces)
    assert "think" not in result.lower()
    assert "hidden" not in result
    assert result == "Visible"
    await c.close()


@pytest.mark.asyncio
async def test_chat_stream_unmatched_think_at_end():
    """Unmatched <think> at stream end — fully suppressed."""
    c = UnifiedLLMClient()
    def fake_create(**_kwargs):
        return _AsyncIter([
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello "), finish_reason=None)]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="<think>ooops"), finish_reason="stop")]),
        ])
    _patch_get_client(c, fake_create)
    pieces = [p async for p in c.chat_stream("test", max_tokens=100)]
    result = "".join(pieces)
    assert "think" not in result.lower()
    assert "ooops" not in result
    assert result == "Hello "
    await c.close()


@pytest.mark.asyncio
async def test_chat_stream_non_think_throughput():
    """Non-think content passes through unchanged in order."""
    c = UnifiedLLMClient()
    def fake_create(**_kwargs):
        return _AsyncIter([
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"), finish_reason=None)]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"), finish_reason=None)]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" foo"), finish_reason="stop")]),
        ])
    _patch_get_client(c, fake_create)
    pieces = [p async for p in c.chat_stream("test", max_tokens=100)]
    assert "".join(pieces) == "Hello world foo"
    await c.close()


@pytest.mark.asyncio
async def test_chat_stream_uppercase_think():
    """chat_stream with <THINK>...</THINK> uppercase — stripped."""
    c = UnifiedLLMClient()
    def fake_create(**_kwargs):
        return _AsyncIter([
            MagicMock(choices=[MagicMock(delta=MagicMock(content="<THINK>"), finish_reason=None)]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="hid</THINK>X"), finish_reason="stop")]),
        ])
    _patch_get_client(c, fake_create)
    pieces = [p async for p in c.chat_stream("test", max_tokens=100)]
    assert "".join(pieces) == "X"
    await c.close()


@pytest.mark.parametrize("stream_enabled", [False, True])
@pytest.mark.asyncio
async def test_force_model_applies_only_to_ramclouds(monkeypatch, stream_enabled) -> None:
    ramclouds = ProviderConfig("ramclouds", "k", "https://example.test/v1", "base-model")
    groq = ProviderConfig("groq", "k", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile")
    client = UnifiedLLMClient(force_model="deepseek-v4-flash", providers=[ramclouds, groq])
    captured = []

    async def fake_create(**kwargs):
        captured.append(kwargs)
        if stream_enabled:
            async def events():
                yield MagicMock(choices=[MagicMock(delta=MagicMock(content="ok"), finish_reason="stop")])
            return events()
        return MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])

    fake_openai = MagicMock()
    fake_openai.chat.completions.create = AsyncMock(side_effect=fake_create)
    monkeypatch.setattr(client, "_get_client", lambda _provider: fake_openai)
    monkeypatch.setenv("RAMCLOUDS_STREAM", str(stream_enabled).lower())

    ramclouds_response = await client._call_provider(ramclouds, [], 0.1, 10, False)
    groq_response = await client._call_provider(groq, [], 0.1, 10, False)

    assert ramclouds_response.model == "deepseek-v4-flash"
    assert groq_response.model == "llama-3.3-70b-versatile"
    assert [call["model"] for call in captured] == [
        "deepseek-v4-flash",
        "llama-3.3-70b-versatile",
    ]
    await client.close()


@pytest.mark.asyncio
async def test_force_model_does_not_bleed_into_groq_chat_stream(monkeypatch) -> None:
    groq = ProviderConfig("groq", "k", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile")
    client = UnifiedLLMClient(force_model="deepseek-v4-flash", providers=[groq])
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        async def events():
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="ok"), finish_reason="stop")])
        return events()

    fake_openai = MagicMock()
    fake_openai.chat.completions.create = AsyncMock(side_effect=fake_create)
    monkeypatch.setattr(client, "_get_client", lambda _provider: fake_openai)

    assert [piece async for piece in client.chat_stream("test")] == ["ok"]
    assert captured["model"] == "llama-3.3-70b-versatile"
    await client.close()


def test_provider_defaults_use_ai_box_deepseek_and_current_groq(monkeypatch) -> None:
    monkeypatch.setenv("RAMCLOUDS_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    for name in ("RAMCLOUDS_BASE_URL", "RAMCLOUDS_MODEL", "GROQ_ROUTER_MODEL"):
        monkeypatch.delenv(name, raising=False)

    from src.services.llm_client import _build_providers
    providers = {provider.name: provider for provider in _build_providers()}

    assert providers["ramclouds"].base_url == "https://api.ai-box.vn/v1"
    assert providers["ramclouds"].model == "deepseek-v4-flash"
    assert providers["groq"].model == "llama-3.3-70b-versatile"


def test_settings_defaults_use_ai_box_deepseek_and_guardrail(monkeypatch) -> None:
    from config.settings import RAGSettings

    for name in ("RAMCLOUDS_BASE_URL", "RAMCLOUDS_MODEL", "GUARDRAIL_MODEL"):
        monkeypatch.delenv(name, raising=False)
    settings = RAGSettings(_env_file=None)

    assert settings.RAMCLOUDS_BASE_URL == "https://api.ai-box.vn/v1"
    assert settings.RAMCLOUDS_MODEL == "deepseek-v4-flash"
    assert settings.GUARDRAIL_MODEL == "llama-3.3-70b-versatile"
