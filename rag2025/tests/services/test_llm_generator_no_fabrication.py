"""TDD pre-flight: anti-fallback retry must NOT fabricate when current-year context missing.

@spec(S13.6) regression test for Type-5/Type-6 hallucination class.

This test was written BEFORE the fix to prove the bug exists. After the fix
(P5-4 deletes llm_generator.py:309-345 anti-fallback retry block), the test
must PASS.
"""
from __future__ import annotations
import re
import os
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import sys

import pytest


# Make services importable.
RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))


_FABRICATED_NUMERIC = re.compile(r"\b\d{2,3}[\.,]\d{3}(\.\d{3})?\b")
_FABRICATED_TR_TRIEU = re.compile(r"\b\d{1,3}\s*tr(iệu)?\b", re.IGNORECASE)


def _build_chunks_only_2025() -> list[dict]:
    """Context only contains 2025 data — current-year (2026) is missing."""
    return [
        {
            "text": "Học phí năm 2025 ngành Công nghệ thông tin: 14.000.000 VNĐ/năm.",
            "summary": "Học phí 2025",
            "metadata": {
                "data_year": 2025,
                "source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=59",
                "source": "TB59 2025",
                "info_type": "hoc_phi",
                "notification_id": 59,
                "chunk_method": "rule_v2",
            },
        }
    ]


@pytest.mark.asyncio
async def test_no_fabrication_when_only_2025_context_for_2026_hoc_phi_query(monkeypatch):
    """If user asks 2026 học phí but context only has 2025 chunks, the answer
    MUST NOT contain fabricated 2026 numerics, AND provider MUST NOT be tagged
    'anti-fallback-retry'. Either:
      a) graceful fallback ('chưa có thông tin chính thức năm 2026') OR
      b) explicit disclaimer ('năm 2025 ... còn năm 2026 chưa công bố') without numerics.
    """
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")

    from services.llm_generator import LLMGenerator

    gen = LLMGenerator.__new__(LLMGenerator)
    gen.generation_system_prompt = "BẠN LÀ TƯ VẤN VIÊN."
    gen.zai_client = None
    gen.groq_client = None
    gen.has_any_provider = True

    fake_unified = MagicMock()
    fake_unified._providers = ["primary"]
    # Fabricated answer that mentions 2026 + a number — represents what
    # anti-fallback-retry would have produced.
    bad_response = MagicMock()
    bad_response.content = "Học phí năm 2026 ngành CNTT là 18.000.000 VNĐ/năm."
    bad_response.model = "deepseek-v4-pro"
    bad_response.provider = "test"
    fake_unified.chat = AsyncMock(return_value=bad_response)
    gen.unified_client = fake_unified

    # Disable Vietnamese enforcement retry by stubbing it.
    async def _passthrough_vi(answer, *args, **kwargs):
        return answer
    gen._enforce_vietnamese = _passthrough_vi  # type: ignore[assignment]

    chunks = _build_chunks_only_2025()
    result = await gen.generate_answer(
        query="Học phí năm 2026 ngành CNTT bao nhiêu?",
        chunks=chunks,
        confidence=0.6,
    )
    answer = result["answer"]
    answer_lower = answer.lower()
    provider = result.get("provider", "")

    # Conjunctive guard 1: if answer mentions 2026, it MUST be a safe fallback.
    if "2026" in answer_lower:
        assert (
            "chưa có" in answer_lower
            or "chưa được công bố" in answer_lower
            or "chưa có thông tin" in answer_lower
            or "chưa công bố" in answer_lower
        ), (
            f"Answer mentions 2026 without safe fallback wording: {answer!r}"
        )

    # Conjunctive guard 2: provider must NOT have anti-fallback-retry tag.
    assert "anti-fallback-retry" not in provider, (
        f"anti-fallback-retry should be removed; got provider={provider!r}"
    )


@pytest.mark.asyncio
async def test_no_fabricated_numerics_when_risky_intent_lacks_current_year(monkeypatch):
    """Stronger assertion: when risky_intent + no current-year chunk, answer
    must NOT contain fabricated numeric patterns at all (no '18.000.000', no
    '18 triệu', etc.).
    """
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")

    from services.llm_generator import LLMGenerator

    gen = LLMGenerator.__new__(LLMGenerator)
    gen.generation_system_prompt = "BẠN LÀ TƯ VẤN VIÊN."
    gen.zai_client = None
    gen.groq_client = None
    gen.has_any_provider = True

    fake_unified = MagicMock()
    fake_unified._providers = ["primary"]
    bad_response = MagicMock()
    bad_response.content = "Năm 2026 học phí dự kiến 18.000.000 VNĐ/năm theo trường."
    bad_response.model = "deepseek-v4-pro"
    bad_response.provider = "test"
    fake_unified.chat = AsyncMock(return_value=bad_response)
    gen.unified_client = fake_unified

    async def _passthrough_vi(answer, *args, **kwargs):
        return answer
    gen._enforce_vietnamese = _passthrough_vi  # type: ignore[assignment]

    result = await gen.generate_answer(
        query="Học phí 2026 ngành Toán?",
        chunks=_build_chunks_only_2025(),
        confidence=0.5,
    )
    answer = result["answer"]
    provider = result.get("provider", "")

    # No fabricated decimal-grouped numerics.
    assert not _FABRICATED_NUMERIC.search(answer), (
        f"Risky-intent answer contains fabricated numeric: {answer!r}"
    )
    # No fabricated triệu pattern.
    assert not _FABRICATED_TR_TRIEU.search(answer), (
        f"Risky-intent answer contains 'triệu' fabrication: {answer!r}"
    )
    # Provider should reflect graceful fallback OR pristine LLM (no retry).
    assert "anti-fallback-retry" not in provider


@pytest.mark.asyncio
async def test_anti_fallback_retry_string_removed_from_source(monkeypatch):
    """Source-code regression: rag2025/src/services/llm_generator.py must NOT
    contain the literal 'anti-fallback-retry' identifier anymore (P5-4 deletes it).
    """
    src_path = Path(__file__).resolve().parents[2] / "src" / "services" / "llm_generator.py"
    text = src_path.read_text(encoding="utf-8")
    assert "anti-fallback-retry" not in text, (
        "anti-fallback-retry was still found in llm_generator.py — P5-4 fix incomplete"
    )


@pytest.mark.asyncio
async def test_risky_intent_with_current_year_chunk_does_NOT_trigger_fallback(monkeypatch):
    """HIGH-1 regression: when a 2026 chunk IS present, fallback must NOT fire.

    This proves the type-normalized comparison correctly detects current-year
    chunks even when stored as string.
    """
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")

    from services.llm_generator import LLMGenerator

    gen = LLMGenerator.__new__(LLMGenerator)
    gen.generation_system_prompt = "TEST"
    gen.zai_client = None
    gen.groq_client = None
    gen.has_any_provider = True

    fake_unified = MagicMock()
    fake_unified._providers = ["primary"]
    good_response = MagicMock()
    good_response.content = "Học phí 2026 ngành CNTT là 600.000 VNĐ/tín chỉ theo TB74."
    good_response.model = "deepseek-v4-pro"
    good_response.provider = "test"
    fake_unified.chat = AsyncMock(return_value=good_response)
    gen.unified_client = fake_unified

    async def _passthrough_vi(answer, *args, **kwargs):
        return answer
    gen._enforce_vietnamese = _passthrough_vi  # type: ignore[assignment]

    chunks_2026 = [
        {
            "text": "Học phí năm 2026 ngành Công nghệ thông tin: 600.000 VNĐ/tín chỉ.",
            "summary": "Học phí 2026",
            "metadata": {
                "data_year": 2026,
                "source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=74",
                "source": "TB74",
                "info_type": "hoc_phi",
                "notification_id": 74,
                "chunk_method": "rule_v2",
            },
        }
    ]
    result = await gen.generate_answer(
        query="Học phí năm 2026 ngành CNTT bao nhiêu?",
        chunks=chunks_2026,
        confidence=0.9,
    )
    answer = result["answer"]
    provider = result.get("provider", "")
    assert "chưa có" not in answer.lower(), f"Fallback fired when 2026 chunk present: {answer}"
    assert "risky-intent-fallback" not in provider, f"Fallback tag added when chunk present: {provider}"


@pytest.mark.asyncio
async def test_data_year_as_string_does_not_disable_fallback_check(monkeypatch):
    """HIGH-1 regression: data_year stored as string '2026' should be detected
    as the current-year chunk, so fallback must NOT fire."""
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")

    from services.llm_generator import LLMGenerator

    gen = LLMGenerator.__new__(LLMGenerator)
    gen.generation_system_prompt = "TEST"
    gen.zai_client = None
    gen.groq_client = None
    gen.has_any_provider = True

    fake_unified = MagicMock()
    fake_unified._providers = ["primary"]
    good_response = MagicMock()
    good_response.content = "Học phí 2026: 600.000đ/tín chỉ theo TB74."
    good_response.model = "deepseek-v4-pro"
    good_response.provider = "test"
    fake_unified.chat = AsyncMock(return_value=good_response)
    gen.unified_client = fake_unified

    async def _passthrough_vi(answer, *args, **kwargs):
        return answer
    gen._enforce_vietnamese = _passthrough_vi  # type: ignore[assignment]

    # data_year stored as STRING (common when JSON deserializer keeps it as str).
    chunks = [
        {
            "text": "Học phí năm 2026: 600.000đ/tín chỉ.",
            "metadata": {
                "data_year": "2026",  # ← STRING, not int
                "info_type": "hoc_phi",
                "notification_id": 74,
                "source_url": "https://x",
            },
        }
    ]
    result = await gen.generate_answer(
        query="Học phí 2026 ngành CNTT?",
        chunks=chunks,
        confidence=0.8,
    )
    provider = result.get("provider", "")
    assert "risky-intent-fallback" not in provider, (
        f"String-typed data_year should be matched. Provider={provider}"
    )
