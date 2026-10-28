# @spec(T11) evidence-based abstention — policy unit tests
"""Ticket 11 acceptance matrix, tested directly at the policy seam.

The spec pins abstention tests to the HTTP layer *because* abstention must
fire on the production path; the production wiring lives in
``LLMGenerator.generate_answer`` (pre-generation gate), whose HTTP-side
behavior is covered by the pinned season/fabrication/hardening suites.
This file covers the decision function itself: each acceptance criterion
gets its own rule, plus the cross-cutting invariants (never-empty refusal,
ABSTAIN_STRING parity with the generator).
"""
from __future__ import annotations

import sys
from pathlib import Path

RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from services import abstention_policy as ap
from services.abstention_policy import (
    ACTION_ABSTAIN,
    ACTION_ANSWER,
    ACTION_CONFLICT,
    REASON_LINEAGE_MISSING,
    REASON_NO_EVIDENCE,
    REASON_OUT_OF_SCOPE_MAJOR,
    REASON_UNRESOLVED_CONFLICT,
    REASON_YEAR_MISMATCH,
    evaluate_evidence,
)
from services.season import SeasonPhase


def _chunk(text="mô tả", *, year=None, record_id=None, source=None, source_url=None):
    md = {}
    if year is not None:
        md["data_year"] = year
    if record_id is not None:
        md["record_id"] = record_id
    if source is not None:
        md["source"] = source
    if source_url is not None:
        md["source_url"] = source_url
    return {"text": text, "summary": text, "metadata": md}


# ---------------------------------------------------------------------------
# Criterion 1 — zero evidence abstains (for factual queries)
# ---------------------------------------------------------------------------

def test_zero_evidence_current_year_factual_abstains():
    d = evaluate_evidence(
        "Điểm chuẩn năm 2026 ngành 7480201?",
        [],
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
    )
    assert d.action == ACTION_ABSTAIN
    assert d.reason_code == REASON_NO_EVIDENCE
    assert d.message  # never empty


def test_zero_evidence_risky_intent_without_year_abstains():
    d = evaluate_evidence(
        "Học phí ngành công nghệ thông tin bao nhiêu?",
        [],
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
    )
    assert d.action == ACTION_ABSTAIN
    assert d.reason_code == REASON_NO_EVIDENCE


def test_zero_evidence_gap_year_uses_unpublished_wording():
    d = evaluate_evidence(
        "Điểm chuẩn năm 2027?",
        [],
        current_year=2026,
        season_phase=SeasonPhase.PRE_SEASON_GAP,
    )
    assert d.action == ACTION_ABSTAIN
    assert "chưa được công bố" in d.message
    assert "2027" in d.message


def test_zero_evidence_historical_year_says_not_found():
    d = evaluate_evidence(
        "Điểm chuẩn năm 2023?",
        [],
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
    )
    assert d.action == ACTION_ABSTAIN
    assert "không tìm thấy" in d.message
    # Honest: 2023 WAS published; it is simply absent from the corpus.
    assert "chưa được công bố" not in d.message


def test_vague_nonfactual_query_without_evidence_is_left_to_router():
    # The clarify/auto-answer path is owned by the router + UW1 guard;
    # the policy must not override it.
    d = evaluate_evidence("chào bạn", [], current_year=2026,
                          season_phase=SeasonPhase.IN_SEASON)
    assert d.action == ACTION_ANSWER


# ---------------------------------------------------------------------------
# Criterion 2 — only other-year evidence abstains for a current-year question
# ---------------------------------------------------------------------------

def test_other_year_only_evidence_abstains_for_current_year():
    chunks = [_chunk("Học phí 14.000.000đ", year=2025, source="TB59 2025")]
    d = evaluate_evidence(
        "Học phí 2026 ngành 7480201?",
        chunks,
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
    )
    assert d.action == ACTION_ABSTAIN
    assert d.reason_code == REASON_YEAR_MISMATCH
    # Criterion 6: states the verified portion (the year) + related source.
    assert "2025" in d.message
    assert "TB59 2025" in d.message
    assert d.verified_parts == ("năm 2025",)
    assert d.source_labels == ("TB59 2025",)


def test_matching_year_evidence_answers():
    chunks = [_chunk("Học phí 14.000.000đ", year=2026)]
    d = evaluate_evidence(
        "Học phí 2026 ngành 7480201?",
        chunks,
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
    )
    assert d.action == ACTION_ANSWER


def test_year_rule_fails_open_on_unlabeled_chunks():
    # Absence of a year label is not proof of mismatch (multiturn fixtures
    # carry empty metadata).
    chunks = [{"text": "Học phí là 14.000.000đ", "metadata": {}}]
    d = evaluate_evidence(
        "Học phí 2026?",
        chunks,
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
    )
    assert d.action == ACTION_ANSWER


def test_cross_year_comparison_is_not_year_bound():
    chunks = [_chunk("Học phí 2025: 13.000.000đ", year=2025)]
    d = evaluate_evidence(
        "So sánh học phí năm 2025 và 2026",
        chunks,
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
    )
    # cross_year queries deliberately span years -> no mismatch refusal.
    assert d.action == ACTION_ANSWER


def test_zero_sentinel_year_fails_open_like_unlabeled():
    # Production husc_chunks stores data_year == 0 for unknown-year rows
    # (test_ingest_husc_chunks_schema pins this). 0 must read as "no year",
    # not as a real year that "proves" a mismatch against 2026.
    chunks = [_chunk("Học phí là 14.000.000đ", year=0)]
    d = evaluate_evidence(
        "Học phí 2026?",
        chunks,
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
    )
    assert d.action == ACTION_ANSWER


def test_gap_season_defers_to_sanctioned_disclaimer():
    # PRE_SEASON_GAP: the season block's GAP_DISCLAIMER is the pinned
    # behavior; the policy must not pre-empt it with a refusal.
    chunks = [_chunk("Học phí 2025: 13.000.000đ", year=2025)]
    d = evaluate_evidence(
        "Học phí 2026?",
        chunks,
        current_year=2026,
        season_phase=SeasonPhase.PRE_SEASON_GAP,
    )
    assert d.action == ACTION_ANSWER


# ---------------------------------------------------------------------------
# Criterion 3 — unresolved same-year conflict: abstain/conflict, never pick
# ---------------------------------------------------------------------------

class _FakeConflict:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _conflict(year=2026, ma="7480201", field="hoc_phi", statements=None,
              status="open"):
    return _FakeConflict(
        admission_year=year,
        ma_nganh=ma,
        field=field,
        statements=statements or {
            "TB59 2025": "14.000.000",
            "TB60 2025": "16.000.000",
        },
        status=status,
    )


def test_same_year_open_conflict_returns_conflict_state():
    d = evaluate_evidence(
        "Học phí 2026 ngành 7480201?",
        [_chunk("Học phí 14.000.000đ", year=2026)],
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
        unresolved_conflicts=[_conflict()],
    )
    assert d.action == ACTION_CONFLICT
    assert d.reason_code == REASON_UNRESOLVED_CONFLICT
    # Never silently picks: both statements are surfaced.
    assert "14.000.000" in d.message and "16.000.000" in d.message


def test_resolved_conflict_does_not_fire():
    d = evaluate_evidence(
        "Học phí 2026 ngành 7480201?",
        [_chunk("Học phí 14.000.000đ", year=2026)],
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
        unresolved_conflicts=[_conflict(status="resolved")],
    )
    assert d.action == ACTION_ANSWER


def test_conflict_for_other_year_does_not_fire():
    d = evaluate_evidence(
        "Học phí 2026 ngành 7480201?",
        [_chunk("Học phí 14.000.000đ", year=2026)],
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
        unresolved_conflicts=[_conflict(year=2025)],
    )
    assert d.action == ACTION_ANSWER


def test_injected_conflict_source_is_used_at_the_seam():
    ap.configure_conflict_source(lambda: [_conflict()])
    try:
        d = evaluate_evidence(
            "Học phí 2026 ngành 7480201?",
            [_chunk("Học phí 14.000.000đ", year=2026)],
            current_year=2026,
            season_phase=SeasonPhase.IN_SEASON,
        )
        assert d.action == ACTION_CONFLICT
    finally:
        ap.configure_conflict_source(None)


# ---------------------------------------------------------------------------
# Criterion 4 — missing citation lineage abstains
# ---------------------------------------------------------------------------

def test_canonical_chunk_without_source_lineage_abstains():
    chunks = [_chunk("Điểm chuẩn 20.", year=2026, record_id="rec-1")]
    d = evaluate_evidence(
        "Điểm chuẩn 2026?",
        chunks,
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
    )
    assert d.action == ACTION_ABSTAIN
    assert d.reason_code == REASON_LINEAGE_MISSING


def test_canonical_chunk_with_source_lineage_answers():
    chunks = [_chunk(
        "Điểm chuẩn 20.",
        year=2026,
        record_id="rec-1",
        source_url="https://tuyensinh.husc.edu.vn/thongbao.php?id=59",
    )]
    d = evaluate_evidence(
        "Điểm chuẩn 2026?",
        chunks,
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
    )
    assert d.action == ACTION_ANSWER


def test_legacy_chunk_without_canonical_markers_is_exempt():
    # Pre-canonical corpus chunks carry no record_id: lineage rule must not
    # abstain the whole legacy corpus (would break every golden test).
    chunks = [_chunk("Điểm chuẩn 20.", year=2026)]
    d = evaluate_evidence(
        "Điểm chuẩn 2026?",
        chunks,
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
    )
    assert d.action == ACTION_ANSWER


# ---------------------------------------------------------------------------
# Criterion 5 — out-of-scope major abstains
# ---------------------------------------------------------------------------

def test_unknown_major_code_for_current_year_abstains():
    chunks = [_chunk("Học phí CNTT", year=2026)]
    d = evaluate_evidence(
        "Học phí 2026 ngành 7729001?",
        chunks,
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
        major_whitelist=["7480201"],
    )
    assert d.action == ACTION_ABSTAIN
    assert d.reason_code == REASON_OUT_OF_SCOPE_MAJOR
    assert "7729001" in d.message


def test_known_major_code_answers():
    chunks = [_chunk("Học phí CNTT", year=2026)]
    d = evaluate_evidence(
        "Học phí 2026 ngành 7480201?",
        chunks,
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
        major_whitelist=["7480201"],
    )
    assert d.action == ACTION_ANSWER


def test_out_of_scope_rule_skipped_for_historical_year():
    # "ngành X năm 2024" is a legitimate historical question.
    chunks = [_chunk("Học phí", year=2024)]
    d = evaluate_evidence(
        "Học phí 2024 ngành 7729001?",
        chunks,
        current_year=2026,
        season_phase=SeasonPhase.IN_SEASON,
        major_whitelist=["7480201"],
    )
    assert d.reason_code != REASON_OUT_OF_SCOPE_MAJOR


# ---------------------------------------------------------------------------
# Criterion 6 — refusals are never empty; parity with the generator string
# ---------------------------------------------------------------------------

def test_abstain_string_parity_with_generator():
    from services.llm_generator import LLMGenerator

    assert ap.ABSTAIN_STRING == LLMGenerator._ABSTAIN_STRING


def test_every_refusal_carries_a_message():
    cases = [
        ("Điểm chuẩn 2026?", []),
        ("Học phí 2026?", [_chunk("x", year=2025)]),
        ("Học phí 2026?", [_chunk("x", year=2026, record_id="r")]),
        ("Học phí 2026 ngành 7729001?", [_chunk("x", year=2026)]),
    ]
    for q, ch in cases:
        d = evaluate_evidence(
            q, ch, current_year=2026,
            season_phase=SeasonPhase.IN_SEASON,
            major_whitelist=["7480201"],
        )
        assert d.action != ACTION_ANSWER, q
        assert d.message.strip(), f"empty refusal for {q!r}"

# ---------------------------------------------------------------------------
# The generate_answer seam: the refusal fires BEFORE any provider call
# ---------------------------------------------------------------------------

import pytest
from unittest.mock import AsyncMock, MagicMock


def _bare_generator(answer_text="Học phí 2026 là 18.000.000đ."):
    """LLMGenerator with a mocked unified client; chat() records calls."""
    from services.llm_generator import LLMGenerator

    gen = LLMGenerator.__new__(LLMGenerator)
    gen.generation_system_prompt = "BAN LA TU VAN VIEN."
    gen.gen_model = "fake-model"
    gen.has_any_provider = True
    gen.unified_fallback_client = None
    gen.gen_fallback_model = "none"

    fake = MagicMock()
    fake._providers = ["primary"]
    resp = MagicMock()
    resp.content = answer_text
    resp.model = "fake"
    resp.provider = "test"
    fake.chat = AsyncMock(return_value=resp)
    gen.unified_client = fake

    async def _passthrough_vi(answer, *args, **kwargs):
        return answer
    gen._enforce_vietnamese = _passthrough_vi
    return gen, fake


@pytest.mark.asyncio
async def test_gate_refuses_without_calling_the_model():
    gen, fake = _bare_generator()
    result = await gen.generate_answer(
        query="Điểm chuẩn 2026 của trường là bao nhiêu?",
        chunks=[],
        confidence=0.5,
    )
    assert fake.chat.await_count == 0, "model must NOT be called pre-generation"
    assert result["provider"].startswith("evidence-policy:")
    assert result["generation_failed"] is False
    assert result["answer"].strip()


@pytest.mark.asyncio
async def test_gate_year_mismatch_refuses_before_model(monkeypatch):
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    gen, fake = _bare_generator()
    # Stub the season resolver seam the existing suites already use.
    from services.season import SeasonPhase as _SP
    gen._resolve_admission_context = lambda chunks: (2026, _SP.IN_SEASON, False)
    result = await gen.generate_answer(
        query="Học phí 2026 của trường là bao nhiêu?",
        chunks=[_chunk("Học phí 14.000.000đ", year=2025, source="TB59 2025")],
        confidence=0.5,
    )
    assert fake.chat.await_count == 0
    assert result["provider"] == "evidence-policy:year-mismatch"
    # Criterion 6 at the seam: refusal still carries the related sources.
    assert [s["data_year"] for s in result["sources"]] == ["2025"]
    assert "14.000.000" not in result["answer"], (
        "refusal must be metadata-only, never echo chunk numerics"
    )


@pytest.mark.asyncio
async def test_sufficient_evidence_still_calls_the_model():
    gen, fake = _bare_generator()
    from services.season import SeasonPhase as _SP
    gen._resolve_admission_context = lambda chunks: (2026, _SP.IN_SEASON, True)
    result = await gen.generate_answer(
        query="Học phí 2026 của trường là bao nhiêu?",
        chunks=[_chunk("Học phí 14.000.000đ", year=2026, source="TB59 2026")],
        confidence=0.5,
    )
    assert fake.chat.await_count == 1
    assert not result["provider"].startswith("evidence-policy:")


@pytest.mark.asyncio
async def test_stream_gate_yields_refusal_without_streaming(monkeypatch):
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    gen, fake = _bare_generator()
    from services.season import SeasonPhase as _SP
    gen._resolve_admission_context = lambda chunks: (2026, _SP.IN_SEASON, False)
    pieces = [
        p async for p in gen.generate_answer_stream(
            query="Học phí 2026?",
            chunks=[_chunk("Học phí 14.000.000đ", year=2025)],
            confidence=0.5,
        )
    ]
    assert len(pieces) == 1
    assert "14.000.000" not in pieces[0]
    assert not fake.chat_stream.called
