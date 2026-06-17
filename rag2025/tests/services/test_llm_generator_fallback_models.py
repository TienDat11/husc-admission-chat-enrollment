"""TDD: llm_generator fallback paths must be env-driven and non-blocking.

BUG-4/5 HIGH: hardcoded Groq/GLM fallback model strings must read env vars
with the old values as defaults; the Z.AI path must not block the event loop.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))


def _build_generator(monkeypatch) -> "LLMGenerator":
    from services.llm_generator import LLMGenerator

    gen = LLMGenerator.__new__(LLMGenerator)
    gen.generation_system_prompt = "TEST"
    gen.zai_client = None
    gen.groq_client = None
    gen.unified_client = MagicMock()
    gen.unified_client._providers = []
    gen.unified_fallback_client = None
    gen.has_any_provider = True

    async def _passthrough_vi(answer, *args, **kwargs):
        return answer
    gen._enforce_vietnamese = _passthrough_vi  # type: ignore[assignment]
    return gen


@pytest.mark.asyncio
async def test_groq_fallback_uses_env_model(monkeypatch):
    """GROQ_GEN_MODEL env var must override the default hardcoded model."""
    monkeypatch.setenv("GROQ_GEN_MODEL", "llama-3.1-8b-instant")

    from services.llm_generator import LLMGenerator

    gen = _build_generator(monkeypatch)
    fake_groq = MagicMock()
    captured: list[dict] = []

    async def fake_create(**kwargs):
        captured.append(kwargs)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        return resp

    fake_groq.chat.completions.create = AsyncMock(side_effect=fake_create)
    gen.groq_client = fake_groq

    await gen.generate_answer(
        query="Học phí?",
        chunks=[{"text": "14tr", "metadata": {"data_year": 2026}}],
        confidence=0.9,
    )

    assert captured, "Groq fallback was not called"
    assert captured[0]["model"] == "llama-3.1-8b-instant", (
        f"Wrong model used: {captured[0]['model']!r}"
    )


@pytest.mark.asyncio
async def test_groq_fallback_default_when_env_unset(monkeypatch):
    """When GROQ_GEN_MODEL is unset, the historical default must remain."""
    monkeypatch.delenv("GROQ_GEN_MODEL", raising=False)

    from services.llm_generator import LLMGenerator

    gen = _build_generator(monkeypatch)
    fake_groq = MagicMock()
    captured: list[dict] = []

    async def fake_create(**kwargs):
        captured.append(kwargs)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        return resp

    fake_groq.chat.completions.create = AsyncMock(side_effect=fake_create)
    gen.groq_client = fake_groq

    await gen.generate_answer(
        query="Học phí?",
        chunks=[{"text": "14tr", "metadata": {"data_year": 2026}}],
        confidence=0.9,
    )

    assert captured
    assert captured[0]["model"] == "llama-3.3-70b-versatile"


@pytest.mark.asyncio
async def test_zai_fallback_is_nonblocking(monkeypatch):
    """The Z.AI path must call the sync SDK via asyncio.to_thread."""
    monkeypatch.setenv("ZAI_GEN_MODEL", "glm-test-model")

    from services.llm_generator import LLMGenerator

    gen = _build_generator(monkeypatch)
    gen.groq_client = None

    captured: list[dict] = []

    def fake_create(**kwargs):
        captured.append(kwargs)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        return resp

    fake_zai = MagicMock()
    fake_zai.chat.completions.create = fake_create
    gen.zai_client = fake_zai

    await gen.generate_answer(
        query="Học phí?",
        chunks=[{"text": "14tr", "metadata": {"data_year": 2026}}],
        confidence=0.9,
    )

    assert captured, "Z.AI fallback was not called"
    assert captured[0]["model"] == "glm-test-model"
    # If the call were blocking, the test still passes functionally, but
    # source-level inspection guarantees asyncio.to_thread is used.
    src = Path(__file__).resolve().parents[2] / "src" / "services" / "llm_generator.py"
    text = src.read_text(encoding="utf-8")
    zai_start = text.find("elif self.zai_client")
    assert zai_start != -1
    zai_block = text[zai_start : text.find("\n            else:", zai_start)]
    assert "await asyncio.to_thread" in zai_block, (
        "Z.AI fallback must be wrapped in asyncio.to_thread"
    )
    assert "self.zai_client.chat.completions.create" in zai_block


@pytest.mark.asyncio
async def test_zai_fallback_default_when_env_unset(monkeypatch):
    """When ZAI_GEN_MODEL is unset, the historical default must remain."""
    monkeypatch.delenv("ZAI_GEN_MODEL", raising=False)

    from services.llm_generator import LLMGenerator

    gen = _build_generator(monkeypatch)
    gen.groq_client = None

    captured: list[dict] = []

    def fake_create(**kwargs):
        captured.append(kwargs)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        return resp

    fake_zai = MagicMock()
    fake_zai.chat.completions.create = fake_create
    gen.zai_client = fake_zai

    await gen.generate_answer(
        query="Học phí?",
        chunks=[{"text": "14tr", "metadata": {"data_year": 2026}}],
        confidence=0.9,
    )

    assert captured
    assert captured[0]["model"] == "glm-4-32b-0414-128k"
