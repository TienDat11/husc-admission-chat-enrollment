# @spec(S15.6 / ADR-F / CF-5) test_abstain_hardening — contact-keyword guard
"""TDD: S15.6 / ADR-F / CF-5 — narrow abstain guard.

Contract (pinned by plan §S15.6 + CF-5):

  * Define `_CONTACT_KEYWORDS = re.compile(
        r"\\b(zalo|group|nhóm|fanpage|facebook|hotline|email|sđt|số điện thoại)\\b",
        re.IGNORECASE
    )`.
  * If the QUERY matches a contact keyword AND none of the retrieved chunk
    TEXTS contain the matched term (case-insensitive substring) → REPLACE
    the answer with the standard abstain string.
  * Narrow: never affects normal questions (no contact keyword in query).

LLM is mocked. The guard must fire regardless of what the model returns
(it operates on the final answer string post-generation). For positive
path the model output MUST be preserved verbatim.

Mocking pattern mirrors `tests/services/test_generator_season.py::_make_gen`:
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


# Standard abstain phrasing used elsewhere in `llm_generator.py` (e.g. line 428
# in the LLM-error path and the `_get_fallback_prompt` rule #5). Pin it as the
# canonical string the guard MUST emit.
ABSTAIN_STRING = "Tôi không tìm thấy thông tin này trong tài liệu hiện có."


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _make_gen(answer_text: str = "Câu trả lời mặc định từ mô hình."):
    """Construct an LLMGenerator with a mocked unified client + passthrough VI.

    Mirrors the pattern in `tests/services/test_generator_season.py::_make_gen`.
    """
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

    # Stub admission-context resolver so the season-aware fallback (which
    # runs AFTER the abstain guard) doesn't pull in a live retriever.
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
# Case 1 — contact keyword in query, term absent from chunks → ABSTAIN
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contact_keyword_in_query_term_absent_from_chunks_abstains():
    """The msg055 regression: query asks about zalo/group, no retrieved
    chunk mentions zalo or group → answer MUST be replaced by the standard
    abstain string (no fabrication)."""
    gen = _make_gen(
        # The model would happily fabricate here ("Zalo OA + mùa cao điểm
        # tháng 5-9") — the guard MUST short-circuit that.
        answer_text="Trường có Zalo OA và fanpage, mùa cao điểm tháng 5-9.",
    )
    chunks = [
        _chunk("Học phí năm 2026 ngành CNTT là 600.000 VNĐ/tín chỉ."),
        _chunk("Thời gian đăng ký xét tuyển từ tháng 3 đến tháng 6."),
    ]
    result = await gen.generate_answer(
        query="Trường có group/zalo tư vấn không?",
        chunks=chunks,
        confidence=0.6,
    )
    answer = result["answer"]
    assert answer == ABSTAIN_STRING, (
        f"Guard must replace fabricated zalo answer with abstain string; "
        f"got {answer!r}"
    )


# ---------------------------------------------------------------------------
# Case 2 — contact keyword in query, term PRESENT in a chunk → NORMAL
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contact_keyword_in_query_term_present_in_chunk_preserves_answer():
    """When a retrieved chunk actually mentions zalo, the guard MUST NOT
    fire — the model is allowed to answer normally based on the context.
    (S16.5 update: the answer body text is preserved verbatim; the URL
    post-guard may strip ungrounded URL tokens but MUST NOT replace the
    answer with the standard abstain string.)"""
    body = "Trường có Zalo OA chính thức để tư vấn tuyển sinh."
    gen = _make_gen(answer_text=body)
    chunks = [_chunk("Liên hệ tư vấn qua Zalo OA của trường theo link đính kèm.")]
    result = await gen.generate_answer(
        query="Trường có zalo tư vấn không?",
        chunks=chunks,
        confidence=0.9,
    )
    answer = result["answer"]
    assert answer == body, (
        f"Guard must NOT fire when a chunk mentions zalo; got {answer!r}"
    )
    assert ABSTAIN_STRING not in answer, (
        "Answer must not be replaced by abstain string when term is present"
    )


# ---------------------------------------------------------------------------
# Case 3 — normal query (no contact keyword) → guard never fires
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_normal_query_without_contact_keyword_guard_does_not_fire():
    """A regular admissions question (no contact keyword) must pass
    through unchanged. The guard is keyword-scoped on purpose."""
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
        f"Guard must NOT fire on a normal query; got {answer!r}"
    )
    assert ABSTAIN_STRING not in answer, (
        "Normal query answer must not be replaced by abstain string"
    )


# ---------------------------------------------------------------------------
# Case 4 — meta: confirm the pinned regex shape (frozen contract)
# ---------------------------------------------------------------------------

def test_contact_keywords_regex_shape_is_pinned():
    """The exact keyword set + regex flags are part of the contract
    (CF-5: 'trigger keywords = zalo|group|nhóm|fanpage|facebook|hotline|
    email|sđt|số điện thoại'). Pin them so accidental edits break the
    test rather than silently regress."""
    from services.llm_generator import LLMGenerator

    rx = LLMGenerator._CONTACT_KEYWORDS
    assert isinstance(rx, re.Pattern), (
        f"_CONTACT_KEYWORDS must be a compiled re.Pattern; got {type(rx)!r}"
    )
    assert rx.flags & re.IGNORECASE, (
        "_CONTACT_KEYWORDS must use re.IGNORECASE"
    )
    # Pin the keyword set exactly (CF-5 spec).
    pattern_src = rx.pattern
    for kw in [
        r"zalo",
        r"group",
        r"nhóm",
        r"fanpage",
        r"facebook",
        r"hotline",
        r"email",
        r"sđt",
        r"số điện thoại",  # CF-5: literal spaces, NOT \s+
    ]:
        assert kw in pattern_src, (
            f"_CONTACT_KEYWORDS missing keyword token {kw!r}: {pattern_src!r}"
        )
    assert pattern_src.startswith(r"\b") and pattern_src.endswith(r"\b"), (
        f"_CONTACT_KEYWORDS must use \\b word boundaries: {pattern_src!r}"
    )
