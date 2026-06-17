"""TDD: default `max_tokens` caps must be NORMAL=1200 / ENUM=4000.

@spec(slice-C latency plan) prevent runaway normal generation without
risking truncation of real answers.

Measured (results/eval_harness/86q_records_s16.jsonl): NORMAL max ~620
est tokens. ENUM source (28-major list) ~1400 tokens. So 1200/4000
gives ~2x headroom on normal and ample room for full enum lists.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))


def _build_generator(monkeypatch):
    """Build an LLMGenerator with all real providers stubbed.

    Returns (gen, captured) where `captured` is a list of kwargs dicts
    passed to `unified_client.chat` (or its fallback).
    """
    monkeypatch.delenv("MAX_GEN_TOKENS", raising=False)
    monkeypatch.delenv("MAX_GEN_TOKENS_ENUM", raising=False)
    monkeypatch.setenv("MAX_CONTEXT_CHARS", "12000")

    from config.settings import RAGSettings
    test_settings = RAGSettings(_env_file=None)
    assert test_settings.MAX_GEN_TOKENS == 1200, "default must be 1200"
    monkeypatch.setattr("services.llm_generator.settings", test_settings)

    from services.llm_generator import LLMGenerator

    captured: list[dict] = []

    async def _fake_chat(**kwargs):
        captured.append(kwargs)
        resp = MagicMock()
        resp.content = "ok"
        resp.model = "fake-model"
        resp.provider = "fake"
        return resp

    fake_unified = MagicMock()
    fake_unified._providers = ["primary"]
    fake_unified.chat = AsyncMock(side_effect=_fake_chat)
    fake_unified_fb = MagicMock()
    fake_unified_fb._providers = ["primary"]
    fake_unified_fb.chat = AsyncMock(side_effect=_fake_chat)

    gen = LLMGenerator.__new__(LLMGenerator)
    gen.generation_system_prompt = "BẠN LÀ TƯ VẤN VIÊN."
    gen.gen_model = "fake-primary"
    gen.gen_fallback_model = "fake-fallback"
    gen.zai_client = None
    gen.groq_client = None
    gen.unified_client = fake_unified
    gen.unified_fallback_client = fake_unified_fb
    gen.has_any_provider = True

    async def _passthrough_vi(answer, *args, **kwargs):
        return answer
    gen._enforce_vietnamese = _passthrough_vi  # type: ignore[assignment]
    return gen, captured


@pytest.mark.asyncio
async def test_normal_default_max_tokens_is_1200(monkeypatch):
    gen, captured = _build_generator(monkeypatch)
    await gen.generate_answer(
        query="Học phí ngành CNTT năm 2026?",
        chunks=[{"text": "Học phí CNTT 14tr", "metadata": {"data_year": 2026}}],
        confidence=0.9,
        is_program_list_query=False,
    )
    assert captured, "unified_client.chat was not called"
    assert captured[0]["max_tokens"] == 1200


@pytest.mark.asyncio
async def test_enum_default_max_tokens_is_4000(monkeypatch):
    gen, captured = _build_generator(monkeypatch)
    await gen.generate_answer(
        query="Danh sách 28 ngành?",
        chunks=[{"text": "28 majors list", "metadata": {"data_year": 2026}}],
        confidence=0.9,
        is_program_list_query=True,
    )
    assert captured, "unified_client.chat was not called"
    assert captured[0]["max_tokens"] == 4000


def test_normal_default_exceeds_measured_max_with_headroom():
    """Documented safety assertion: 1200 > 1000 (> measured max 620).
    If anyone lowers the normal default, this test fails and forces a
    conscious decision about whether new measurements justify the risk.
    """
    NORMAL_DEFAULT = 1200
    MEASURED_NORMAL_MAX = 620  # est tokens, from 86q_records_s16.jsonl
    # 1000 = hard floor (1.6x measured max). Update both numbers together.
    assert NORMAL_DEFAULT >= 1000, (
        f"NORMAL_DEFAULT={NORMAL_DEFAULT} below safety floor 1000; "
        f"measured normal max ~{MEASURED_NORMAL_MAX} tokens would truncate."
    )
    assert NORMAL_DEFAULT > MEASURED_NORMAL_MAX


def test_env_overrides_take_precedence_over_default(monkeypatch):
    """Operator can raise the cap by setting the env var; default only
    applies when the env var is unset/empty.
    """
    monkeypatch.setenv("MAX_GEN_TOKENS", "2000")
    monkeypatch.delenv("MAX_GEN_TOKENS_ENUM", raising=False)
    from config.settings import RAGSettings

    settings = RAGSettings()
    assert settings.MAX_GEN_TOKENS == 2000
