"""S19 major-scope guardrail tests.

Mock GROQ like the existing guardrail tests so the LLM is never called.
Verify the deterministic _major_out_of_scope layer blocks admission
questions about majors HUSC does NOT offer (e.g. Kinh tế, Luật, Y khoa)
while keeping real HUSC majors (CNTT, Công nghệ bán dẫn, Khoa học dữ
liệu) and generic admission questions in scope.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RAG_ROOT = Path(__file__).resolve().parents[2]
SCOPE_FILE = RAG_ROOT / "data" / "major_codes" / "husc_majors_current.json"


class _FakeLLM:
    """Stands in for the ns24 guardrail role client; records chat_json calls."""

    def __init__(self, is_in_scope: bool, counter=None):
        self._is_in_scope = is_in_scope
        self._counter = counter
        self.token_ledger = None
        self._providers = []

    async def chat_json(self, **kwargs):
        if self._counter is not None:
            self._counter["calls"] += 1
        return {"is_in_scope": self._is_in_scope, "reason": "mocked"}


def _build_service_with_mock_llm(monkeypatch, is_in_scope: bool, call_counter=None):
    """Build a GuardrailService whose ns24 guardrail client is a fake.

    The service calls ``self._llm.chat_json``; we inject a fake there and
    mark a model present so the LLM path is reachable.
    """
    from config.settings import RAGSettings
    from services import guardrail as guardrail_mod

    guardrail_mod.GuardrailService.clear_major_scope_cache_for_testing()
    svc = guardrail_mod.GuardrailService(RAGSettings(_env_file=None))
    svc._llm = _FakeLLM(is_in_scope, call_counter)
    svc._model = "fake-guardrail-model"
    return svc, guardrail_mod


def _llm_call_counter():
    return {"calls": 0}


# ---------------------------------------------------------------------------
# 1. "Ngành Kinh Tế điểm chuẩn bao nhiêu?" → blocked, NO LLM call
# ---------------------------------------------------------------------------
def test_kinh_te_blocked_deterministically_no_llm(monkeypatch):
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_llm(monkeypatch, is_in_scope=True, call_counter=counter)
    decision = asyncio.run(svc.precheck("Ngành Kinh Tế điểm chuẩn bao nhiêu?"))
    assert decision.is_in_scope is False
    assert decision.internal_code == "NOT_IN_HUSC_SCOPE"
    assert decision.reason == "major_not_offered"
    assert counter["calls"] == 0, "major-scope must block before any LLM call"
    # Helpful answer mentions HUSC + link
    assert "HUSC" in decision.short_answer
    assert "tuyensinh.husc.edu.vn" in decision.short_answer


# ---------------------------------------------------------------------------
# 2. "Ngành Luật HUSC học phí?" → blocked even though HUSC is present
# ---------------------------------------------------------------------------
def test_luat_blocked_even_with_husc_alias(monkeypatch):
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_llm(monkeypatch, is_in_scope=True, call_counter=counter)
    decision = asyncio.run(svc.precheck("Ngành Luật HUSC học phí?"))
    assert decision.is_in_scope is False
    assert decision.reason == "major_not_offered"
    assert counter["calls"] == 0


# ---------------------------------------------------------------------------
# 3. "Học phí ngành Công nghệ thông tin?" → NOT blocked, keyword fast-path
# ---------------------------------------------------------------------------
def test_cntt_in_scope_via_keyword(monkeypatch):
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_llm(monkeypatch, is_in_scope=False, call_counter=counter)
    decision = asyncio.run(svc.precheck("Học phí ngành Công nghệ thông tin?"))
    assert decision.is_in_scope is True
    assert decision.reason == "in_scope_keyword"
    # LLM not consulted (keyword path short-circuited in_scope)
    assert counter["calls"] == 0


# ---------------------------------------------------------------------------
# 4. "Vật lý học - Chương trình Công nghệ bán dẫn có gì hay?" → NOT blocked
# ---------------------------------------------------------------------------
def test_ban_dan_2026_major_in_scope(monkeypatch):
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_llm(monkeypatch, is_in_scope=True, call_counter=counter)
    decision = asyncio.run(
        svc.precheck("Vật lý học - Chương trình Công nghệ bán dẫn có gì hay?")
    )
    assert decision.is_in_scope is True
    # Either keyword fast-path or LLM-driven in_scope is acceptable — both
    # mean the major-scope layer did NOT block. Reason 'major_not_offered'
    # would mean we failed the test.
    assert decision.reason != "major_not_offered"


# ---------------------------------------------------------------------------
# 5. "Điểm chuẩn ngành Khoa học dữ liệu 2026?" → NOT blocked
# ---------------------------------------------------------------------------
def test_khoa_hoc_du_lieu_in_scope(monkeypatch):
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_llm(monkeypatch, is_in_scope=False, call_counter=counter)
    decision = asyncio.run(svc.precheck("Điểm chuẩn ngành Khoa học dữ liệu 2026?"))
    assert decision.is_in_scope is True
    assert decision.reason == "in_scope_keyword"


# ---------------------------------------------------------------------------
# 6. Generic admission questions → NOT blocked
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q", [
    "học phí bao nhiêu?",
    "thời gian xét tuyển?",
    "điểm chuẩn trường?",
    "HUSC có những ngành nào?",
])
def test_generic_admission_questions_not_blocked(monkeypatch, q):
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_llm(monkeypatch, is_in_scope=False, call_counter=counter)
    decision = asyncio.run(svc.precheck(q))
    assert decision.is_in_scope is True, f"generic question should be in_scope: {q!r}"
    # LLM not consulted (no major phrase + admission keyword)
    assert counter["calls"] == 0, f"LLM should not be called for: {q!r}"


# ---------------------------------------------------------------------------
# 7. Dynamic mtime reload: adding a major to the JSON unblocks new queries
# ---------------------------------------------------------------------------
def test_dynamic_mtime_reload_adds_new_major(monkeypatch, tmp_path):
    """Write a temp husc_majors_current.json with an extra 'Ngành Vũ trụ học'
    major, then assert a query about it is NOT blocked after reload.
    Also assert a non-listed major ("Ngành Kinh tế") is still blocked.
    """
    from services import guardrail as guardrail_mod

    # Backup original file
    original = SCOPE_FILE.read_text(encoding="utf-8")
    try:
        data = json.loads(original)
        data["majors"].append(
            {
                "ma_nganh": "9999999",
                "ten": "Vũ trụ học",
                "aliases": ["vu tru hoc", "vũ trụ"],
            }
        )
        # Force the loader to pick up the new file by bumping mtime.
        SCOPE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        # Touch with a clearly newer timestamp
        import os
        new_mtime = SCOPE_FILE.stat().st_mtime + 1
        os.utime(SCOPE_FILE, (new_mtime, new_mtime))

        counter = _llm_call_counter()
        svc, mod = _build_service_with_mock_llm(
            monkeypatch, is_in_scope=True, call_counter=counter
        )

        # New major (in the updated file) — NOT blocked
        decision_ok = asyncio.run(svc.precheck("Ngành Vũ trụ học điểm chuẩn?"))
        assert decision_ok.is_in_scope is True, (
            f"vũ trụ học should be allowed after dynamic reload: {decision_ok.reason}"
        )

        # Major not in file — still blocked
        decision_bad = asyncio.run(svc.precheck("Ngành Kinh tế điểm chuẩn?"))
        assert decision_bad.is_in_scope is False
        assert decision_bad.reason == "major_not_offered"
    finally:
        # Restore the original file
        SCOPE_FILE.write_text(original, encoding="utf-8")
        guardrail_mod.GuardrailService.clear_major_scope_cache_for_testing()


# ---------------------------------------------------------------------------
# 8. Fallback: missing/corrupt file → seed still blocks Kinh tế, allows CNTT
# ---------------------------------------------------------------------------
def test_fallback_when_scope_file_missing(monkeypatch, tmp_path):
    """If husc_majors_current.json is missing/corrupt the loader falls back
    to the tuition json (and finally the hard-coded seed) and still:
      - blocks Kinh tế (denylist catches it)
      - allows CNTT (allowlist has it)
      - never raises
    """
    from services import guardrail as guardrail_mod

    original = SCOPE_FILE.read_text(encoding="utf-8")
    try:
        # Corrupt the file
        SCOPE_FILE.write_text("{ this is not valid json", encoding="utf-8")
        import os
        os.utime(SCOPE_FILE, (SCOPE_FILE.stat().st_mtime + 5,) * 2)

        counter = _llm_call_counter()
        svc, mod = _build_service_with_mock_llm(
            monkeypatch, is_in_scope=True, call_counter=counter
        )

        # Denylist-driven block (loader irrelevant here)
        d_kt = asyncio.run(svc.precheck("Ngành Kinh tế điểm chuẩn?"))
        assert d_kt.is_in_scope is False
        assert d_kt.reason == "major_not_offered"

        # CNTT — real major, fallback allowlist still has it
        d_cntt = asyncio.run(svc.precheck("Học phí ngành Công nghệ thông tin?"))
        assert d_cntt.is_in_scope is True
        assert d_cntt.reason == "in_scope_keyword"

        # Y khoa — denylist
        d_y = asyncio.run(svc.precheck("Ngành Y đa khoa điểm chuẩn?"))
        assert d_y.is_in_scope is False
        assert d_y.reason == "major_not_offered"
    finally:
        SCOPE_FILE.write_text(original, encoding="utf-8")
        guardrail_mod.GuardrailService.clear_major_scope_cache_for_testing()


# ---------------------------------------------------------------------------
# 9. PII still wins first (CCCD + "ngành Kinh tế" → SENSITIVE_PII_DETECTED)
# ---------------------------------------------------------------------------
def test_pii_wins_over_major_scope(monkeypatch):
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_llm(
        monkeypatch, is_in_scope=False, call_counter=counter
    )
    decision = asyncio.run(
        svc.precheck("Số CCCD 012345678901 của tôi, hỏi điểm chuẩn ngành Kinh tế")
    )
    assert decision.is_in_scope is False
    assert decision.internal_code == "SENSITIVE_PII_DETECTED"
    assert decision.pii_detected is True
    # Major-scope never reached → LLM not called
    assert counter["calls"] == 0


# ---------------------------------------------------------------------------
# 10. "Ngành Y đa khoa" → blocked (Y/Medicine)
# ---------------------------------------------------------------------------
def test_y_da_khoa_blocked(monkeypatch):
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_llm(
        monkeypatch, is_in_scope=True, call_counter=counter
    )
    decision = asyncio.run(svc.precheck("Ngành Y đa khoa điểm chuẩn?"))
    assert decision.is_in_scope is False
    assert decision.reason == "major_not_offered"
    assert counter["calls"] == 0


# ---------------------------------------------------------------------------
# 11. diacritic-folded negative — no diacritics in user query
# ---------------------------------------------------------------------------
def test_kinh_te_no_diacritics_blocked(monkeypatch):
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_llm(
        monkeypatch, is_in_scope=True, call_counter=counter
    )
    decision = asyncio.run(svc.precheck("nganh kinh te hoc phi bao nhieu?"))
    assert decision.is_in_scope is False
    assert decision.reason == "major_not_offered"
    assert counter["calls"] == 0


# ---------------------------------------------------------------------------
# 12. Pure major-name extraction sanity (unit-level, no precheck)
# ---------------------------------------------------------------------------
def test_extract_major_phrase_patterns(monkeypatch):
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_llm(
        monkeypatch, is_in_scope=True, call_counter=counter
    )
    # "ngành X ..."
    assert svc._extract_major_phrase("Ngành Kinh tế điểm chuẩn bao nhiêu?") == "Kinh tế"
    # "ngành X năm YYYY" — Pattern 1
    assert svc._extract_major_phrase("Điểm chuẩn ngành Luật năm 2026?") == "Luật"
    # "điểm chuẩn ngành X" — Pattern 3 (tail before ngành)
    assert svc._extract_major_phrase("Điểm chuẩn ngành Khoa học dữ liệu 2026?") == "Khoa học dữ liệu"
    # No specific major → None (generic admission question)
    assert svc._extract_major_phrase("học phí bao nhiêu?") is None
    assert svc._extract_major_phrase("thời gian xét tuyển?") is None
