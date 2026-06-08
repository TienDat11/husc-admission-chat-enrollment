# @spec(UW1) test_empty_retrieval_guard — substantive answer with 0 grounding
"""TDD: UW1 — defensive BE guard for the "0 grounding, substantive answer"
hallucination risk.

Contract (pinned by `.omc/plans/ultrawork_metric_harden_plan.md` §T2):

  * If `generate_answer` is about to return a NON-abstain, NON-clarification
    answer while `chunks` is empty (zero retrieved grounding), REPLACE the
    answer with the standard abstain string (``_ABSTAIN_STRING``).
  * Exempt (guard MUST NOT fire):
      - the answer is already the standard abstain string;
      - the answer is a vague-clarification template
        ("chưa đủ rõ" / "vui lòng cho biết cụ thể") — this is the
        intentional chunk-less path for `hyde_auto_answer` / vague queries.
  * When `chunks` is non-empty, the answer passes through unchanged
    (the existing contact-keyword guard still applies on top).

LLM is mocked. The guard operates on the final answer string
post-generation, post-season-aware fallback, post-contact-keyword guard.

Mocking pattern mirrors `tests/services/test_abstain_hardening.py::_make_gen`:
bypass `LLMGenerator.__init__`, wire a single `AsyncMock` `unified_client`,
and stub `_enforce_vietnamese` as a passthrough.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))


# Standard abstain phrasing used elsewhere in `llm_generator.py`.
# Pin it as the canonical string the guard MUST emit.
ABSTAIN_STRING = "Tôi không tìm thấy thông tin này trong tài liệu hiện có."


# ---------------------------------------------------------------------------
# Builders (mirror test_abstain_hardening.py)
# ---------------------------------------------------------------------------

def _make_gen(answer_text: str = "Câu trả lời mặc định từ mô hình."):
    """Construct an LLMGenerator with a mocked unified client + passthrough VI."""
    from services.llm_generator import LLMGenerator

    gen = LLMGenerator.__new__(LLMGenerator)
    gen.generation_system_prompt = "TEST SYSTEM PROMPT"
    gen.zai_client = None
    gen.groq_client = None
    gen.has_any_provider = True
    gen.gen_model = "test-model"
    gen.gen_fallback_model = "test-fallback"
    gen.unified_fallback_client = None

    fake_unified = MagicMock()
    fake_unified._providers = ["primary"]
    resp = MagicMock()
    resp.content = answer_text
    resp.model = "test-model"
    resp.provider = "test"
    fake_unified.chat = AsyncMock(return_value=resp)
    gen.unified_client = fake_unified

    async def _passthrough_vi(answer, *args, **kwargs):
        return answer

    gen._enforce_vietnamese = _passthrough_vi  # type: ignore[assignment]

    # Stub admission-context resolver so the season-aware fallback doesn't
    # pull in a live retriever.
    def _stub_context(chunks):
        from services.season import SeasonPhase
        return 2026, SeasonPhase.IN_SEASON, True

    gen._resolve_admission_context = _stub_context  # type: ignore[assignment]
    return gen


def _chunk(text: str, year: int = 2026) -> dict:
    """Synthetic retrieved chunk with the given text and a current-year tag."""
    return {
        "text": text,
        "summary": "",
        "metadata": {
            "data_year": year,
            "source_url": "https://tuyensinh.husc.edu.vn/",
            "source": "TB_test",
            "info_type": "general",
        },
    }


# ---------------------------------------------------------------------------
# Case (a) — chunks=[] + substantive draft → MUST abstain
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_substantive_answer_with_empty_chunks_abstains():
    """The genuine hallucination-prevention case: the model produced a
    substantive answer (e.g. "Điểm chuẩn ngành CNTT là 24 điểm.") but
    there are ZERO retrieved chunks to ground it. Guard MUST replace
    the answer with the standard abstain string (no fabrication)."""
    gen = _make_gen(
        answer_text="Điểm chuẩn ngành CNTT là 24 điểm.",
    )
    result = await gen.generate_answer(
        query="Điểm chuẩn ngành CNTT năm 2026?",
        chunks=[],
        confidence=0.6,
    )
    answer = result["answer"]
    assert answer == ABSTAIN_STRING, (
        f"Guard must replace substantive 0-grounding answer with abstain "
        f"string; got {answer!r}"
    )


# ---------------------------------------------------------------------------
# Case (b) — chunks=[] + clarification/auto_answer draft → unchanged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clarification_answer_with_empty_chunks_is_exempt():
    """The `hyde_auto_answer` (vague-clarification) route is INTENTIONALLY
    chunk-less. The guard MUST NOT replace the clarification text. Two
    marker variants are pinned (matches the templates used by
    `query_router.py` and the season-aware fallback)."""
    clarification = (
        "Câu hỏi của bạn hiện chưa đủ rõ, vui lòng cho biết cụ thể "
        "ngành/trường bạn quan tâm để mình hỗ trợ chính xác hơn."
    )
    gen = _make_gen(answer_text=clarification)
    result = await gen.generate_answer(
        query="Trường có xét tổ hợp.....không?",
        chunks=[],
        confidence=0.3,
    )
    answer = result["answer"]
    assert answer == clarification, (
        f"Clarification/auto_answer path is exempt; got {answer!r}"
    )
    assert ABSTAIN_STRING not in answer, (
        "Clarification text must not be replaced by abstain string"
    )


@pytest.mark.asyncio
async def test_clarification_marker_chua_du_ro_is_exempt():
    """Pinned marker variant: just the 'chưa đủ rõ' clarification
    template (e.g. when `query_router.hyde_auto_answer` returns a
    short clarification)."""
    clarification = "Câu hỏi chưa đủ rõ, bạn vui lòng mô tả chi tiết hơn."
    gen = _make_gen(answer_text=clarification)
    result = await gen.generate_answer(
        query="Ngành... xét những tổ hợp nào ạ?",
        chunks=[],
        confidence=0.2,
    )
    answer = result["answer"]
    assert answer == clarification, (
        f"chưa đủ rõ clarification must be exempt; got {answer!r}"
    )


@pytest.mark.asyncio
async def test_clarification_marker_vui_long_cho_biet_is_exempt():
    """Pinned marker variant: the alternative clarification phrasing
    'vui lòng cho biết cụ thể' used by some templates."""
    clarification = (
        "Vui lòng cho biết cụ thể ngành học và năm học bạn quan tâm."
    )
    gen = _make_gen(answer_text=clarification)
    result = await gen.generate_answer(
        query="xét tuyển?",
        chunks=[],
        confidence=0.2,
    )
    answer = result["answer"]
    assert answer == clarification, (
        f"vui lòng cho biết cụ thể clarification must be exempt; "
        f"got {answer!r}"
    )


# ---------------------------------------------------------------------------
# Case (c) — non-empty chunks + substantive answer → unchanged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_substantive_answer_with_chunks_passes_through():
    """Normal grounded path: chunks present, model produced a substantive
    answer — guard MUST NOT fire. The answer body is preserved verbatim."""
    body = "Điểm chuẩn ngành CNTT năm 2025 là 24.5."
    gen = _make_gen(answer_text=body)
    chunks = [_chunk("Điểm chuẩn ngành CNTT các năm gần đây.")]
    result = await gen.generate_answer(
        query="Điểm chuẩn ngành CNTT?",
        chunks=chunks,
        confidence=0.8,
    )
    answer = result["answer"]
    assert answer == body, (
        f"Guard must NOT fire on a grounded substantive answer; "
        f"got {answer!r}"
    )
    assert ABSTAIN_STRING not in answer, (
        "Grounded answer must not be replaced by abstain string"
    )


# ---------------------------------------------------------------------------
# Case (d) — meta: confirm the pinned clarify-marker shape (frozen contract)
# ---------------------------------------------------------------------------

def test_clarify_markers_constant_shape_is_pinned():
    """The exact clarification-marker set is part of the contract (UW1
    spec): 'chưa đủ rõ' and 'vui lòng cho biết cụ thể'. Pin them so
    accidental edits break the test rather than silently regress."""
    from services.llm_generator import LLMGenerator

    marker_attr = "_CLARIFY_MARKERS"
    assert hasattr(LLMGenerator, marker_attr), (
        f"LLMGenerator.{marker_attr} must be defined so the guard can "
        f"detect the chunk-less clarification path"
    )
    markers = getattr(LLMGenerator, marker_attr)
    if isinstance(markers, re.Pattern):
        pattern_src = markers.pattern
    else:
        # If implemented as a tuple of literal strings, normalize.
        assert hasattr(markers, "__iter__"), (
            f"_CLARIFY_MARKERS must be a re.Pattern or iterable of str; "
            f"got {type(markers)!r}"
        )
        pattern_src = "|".join(markers)
    for marker in ("chưa đủ rõ", "vui lòng cho biết cụ thể"):
        assert marker in pattern_src, (
            f"_CLARIFY_MARKERS missing marker {marker!r}: {pattern_src!r}"
        )
