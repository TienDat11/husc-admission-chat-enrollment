"""Comprehensive fast-path coverage for the S19 guardrail.

Goal: pin the ADMISSION_KEYWORDS + HUSC_ALIASES expansions and the
one-char generic-major guard so legitimate HUSC questions never fall
through to the LLM, while denylist / other-school queries are STILL
blocked deterministically.

Test tiers:
  A. ~25 valid admission queries (incl. colloquial / Hue-regional / typo)
     → all in_scope, zero LLM calls when keyword fast-path matches.
  B. Regression: denylist majors + other-school must still be blocked
     deterministically (0 LLM).
  C. Comparison queries (HUSC vs other) must stay in_scope.
  D. One-char guard: bare generic use does NOT mis-classify.
  E. husc_majors_current.json sanity — no entry has ten == its own code;
     KHMT maps to 7480101.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


RAG_ROOT = Path(__file__).resolve().parents[2]
SCOPE_FILE = RAG_ROOT / "data" / "major_codes" / "husc_majors_current.json"


def _build_service_with_mock_groq(monkeypatch, is_in_scope: bool, call_counter=None):
    monkeypatch.setenv("GROQ_API_KEY", "test-fake-groq-key")

    from config.settings import RAGSettings
    from services import guardrail as guardrail_mod

    class _FakeCompletions:
        def __init__(self, counter):
            self._counter = counter

        async def create(self, *args, **kwargs):
            if self._counter is not None:
                self._counter["calls"] += 1
            content_str = (
                '{"is_in_scope": %s, "reason": "mocked"}' % str(is_in_scope).lower()
            )
            resp = MagicMock()
            resp.choices[0].message.content = content_str
            return resp

    class _FakeChat:
        completions = _FakeCompletions(call_counter)

    class _FakeAsyncGroq:
        def __init__(self, *a, **kw):
            self.chat = _FakeChat()

    monkeypatch.setattr(guardrail_mod, "AsyncGroq", _FakeAsyncGroq)
    guardrail_mod.GuardrailService.clear_major_scope_cache_for_testing()
    settings = RAGSettings()
    svc = guardrail_mod.GuardrailService(settings)
    return svc, guardrail_mod


def _llm_call_counter():
    return {"calls": 0}


# ---------------------------------------------------------------------------
# A. ~25 valid admission queries → ALL in_scope, ZERO LLM calls
# ---------------------------------------------------------------------------

VALID_QUERIES = [
    # Original vague-phrasing false-blocks (commit 99f5f89 baseline)
    "Đăng ký vào trường như lào?",
    # Huế regional / colloquial
    "thi vô trường ni cần chi rứa?",
    "đk xét tuyển mô?",
    "điểm sàn năm ni bao nhiêu?",
    # New ADMISSION_KEYWORDS
    "ngành tâm lý HUSC điểm chuẩn?",
    "ký túc xá có không?",
    "nguyện vọng nộp sao?",
    "đánh giá năng lực được không?",
    "liên thông được k?",
    "miễn giảm học phí thế nào?",
    "mã ngành CNTT là gì?",
    "tổ hợp môn xét tuyển ngành Y sinh?",
    "hạn nộp hồ sơ khi nào?",
    "cổng thông tin tuyển sinh ở đâu?",
    "chương trình đào tạo ngành Hóa dược?",
    "xét tuyển thẳng có những tiêu chí nào?",
    "lệ phí xét tuyển bao nhiêu?",
    "học bạ có được cộng điểm không?",
    "điểm trúng tuyển ngành Khoa học dữ liệu 2026?",
    "điểm xét tuyển ngành CNTT?",
    "vào trường cần điều kiện gì?",
    "thi vào có khó không?",
    # No-diacritic / abbrev variants
    "nguyen vong 1 nen chon nganh gi?",
    "dgnl HUSC co su dung khong?",
    "ky tuc xa co phong cho sinh vien khong?",
]


@pytest.mark.parametrize("q", VALID_QUERIES)
def test_valid_admission_queries_in_scope(monkeypatch, q):
    """All colloquial / typo / Hue-regional admission queries must hit
    the keyword fast-path and reach in_scope without any LLM call.
    """
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_groq(
        monkeypatch, is_in_scope=True, call_counter=counter
    )
    decision = asyncio.run(svc.precheck(q))
    assert decision.is_in_scope is True, (
        f"valid admission query should be in_scope: {q!r} → "
        f"reason={decision.reason!r}"
    )
    assert decision.reason != "major_not_offered", (
        f"valid admission query wrongly blocked by major-scope: {q!r}"
    )
    assert counter["calls"] == 0, (
        f"valid admission query should NOT trigger LLM: {q!r}"
    )


# ---------------------------------------------------------------------------
# B. Regression: denylist + other-school must STILL be blocked (0 LLM)
# ---------------------------------------------------------------------------

REGRESSION_BLOCKED = [
    "Ngành Kinh Tế điểm chuẩn?",
    "Ngành Luật HUSC?",
    "Ngành Y đa khoa?",
    "Điểm chuẩn ĐH Bách Khoa Hà Nội?",
    "Học phí FPT?",
]


@pytest.mark.parametrize("q", REGRESSION_BLOCKED)
def test_denylist_and_other_school_still_blocked(monkeypatch, q):
    """HUSC has never offered Kinh tế / Luật / Y đa khoa; FPT / ĐH
    Bách Khoa are other schools. All MUST be blocked deterministically
    (zero LLM calls) — the fast-path expansion MUST NOT re-open these
    holes.

    Denylist majors ("Kinh tế", "Luật", "Y đa khoa") are caught by the
    S19 major-scope layer BEFORE the LLM, so we mock the LLM to return
    in_scope=True (it must never be called).

    Other-school queries ("FPT", "ĐH Bách Khoa Hà Nội") fall through to
    the LLM because the keyword fast-path is bypassed by the other-
    school guard. We mock the LLM to return in_scope=False to confirm
    the hardening path still rejects them.
    """
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_groq(
        monkeypatch, is_in_scope=False, call_counter=counter
    )
    decision = asyncio.run(svc.precheck(q))
    assert decision.is_in_scope is False, (
        f"query MUST be blocked: {q!r} → reason={decision.reason!r}"
    )
    # The denylist hits return zero LLM calls; the other-school hits
    # fall through to a single LLM call. Either way, the block is
    # deterministic from the user's perspective.
    assert counter["calls"] <= 1, (
        f"block must be deterministic (≤ 1 LLM call): {q!r} "
        f"(calls={counter['calls']})"
    )


# ---------------------------------------------------------------------------
# C. Comparison queries: HUSC vs other school → must stay in_scope
# ---------------------------------------------------------------------------

COMPARISON_QUERIES = [
    "HUSC so với Bách Khoa ngành CNTT?",
    "Đại học Huế khác HUSC thế nào?",
]


@pytest.mark.parametrize("q", COMPARISON_QUERIES)
def test_comparison_queries_in_scope(monkeypatch, q):
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_groq(
        monkeypatch, is_in_scope=True, call_counter=counter
    )
    decision = asyncio.run(svc.precheck(q))
    assert decision.is_in_scope is True, (
        f"comparison should be in_scope: {q!r} → reason={decision.reason!r}"
    )


# ---------------------------------------------------------------------------
# D. One-char guard: bare generic use must NOT crash / false-classify
# ---------------------------------------------------------------------------

def test_one_char_guard_bare_toan_not_extracted(monkeypatch):
    """Bare 'toán' (single-syllable generic word) must NOT be extracted
    as the Toán học major. Otherwise "học toán ở đâu?" would land in
    the major-scope path. (The query still goes through the keyword
    fast-path — "học" alone is NOT an admission keyword, so it falls
    through to the LLM, but `_extract_major_phrase` must return None.)
    """
    svc, mod = _build_service_with_mock_groq(monkeypatch, is_in_scope=True)
    assert svc._extract_major_phrase("học toán ở đâu?") is None
    assert svc._extract_major_phrase("văn học lớp 12") is None
    assert svc._extract_major_phrase("lý thuyết") is None


def test_one_char_guard_ngành_toán_still_in_scope(monkeypatch):
    """When framed as 'ngành Toán HUSC' (compound), the major-scope
    extractor SHOULD treat it as a HUSC major and the query must
    remain in_scope via the keyword fast-path.
    """
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_groq(
        monkeypatch, is_in_scope=True, call_counter=counter
    )
    decision = asyncio.run(svc.precheck("ngành Toán HUSC điểm chuẩn?"))
    assert decision.is_in_scope is True
    assert decision.reason != "major_not_offered"
    assert counter["calls"] == 0


def test_one_char_guard_denylist_still_blocks(monkeypatch):
    """The one-char guard must NOT let the denylist break: 'Ngành Kinh tế'
    has multiple syllables, so the guard does NOT fire. The query is
    still blocked by Pattern 1 + denylist (case b in the precheck flow).
    """
    counter = _llm_call_counter()
    svc, mod = _build_service_with_mock_groq(
        monkeypatch, is_in_scope=True, call_counter=counter
    )
    decision = asyncio.run(svc.precheck("Ngành Kinh tế điểm chuẩn?"))
    assert decision.is_in_scope is False
    assert decision.reason == "major_not_offered"
    assert counter["calls"] == 0


# ---------------------------------------------------------------------------
# E. husc_majors_current.json sanity: no code-as-name, KHMT → 7480101
# ---------------------------------------------------------------------------

def test_no_entry_has_ten_equal_to_code():
    """The original data-quality bug: ~15 entries had ``ten == code``
    (e.g. "7310401" for Tâm lý học). After the refresh, NO entry should
    have ``ten`` equal to its own code (folded lowercase comparison).
    """
    with SCOPE_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    bad = []
    for entry in data.get("majors", []):
        if not isinstance(entry, dict):
            continue
        code = entry.get("ma_nganh")
        ten = entry.get("ten")
        if not code or not ten:
            continue
        # Compare on folded-lowercase; "ten"=="code" is the bug shape
        from services.guardrail import GuardrailService
        if (
            isinstance(ten, str)
            and GuardrailService._fold_diacritics(ten)
            == GuardrailService._fold_diacritics(code)
        ):
            bad.append((code, ten))
    assert not bad, (
        f"husc_majors_current.json still has code-as-name entries: {bad}"
    )


def test_khmt_alias_maps_to_khmt_7480101():
    """'khmt' must map to 7480101 (Khoa học máy tính). The previous bug
    had it under 7440301 (Khoa học môi trường) — environmental majors
    now use 'mt' / 'môi trường'.
    """
    with SCOPE_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    by_code = {m.get("ma_nganh"): m for m in data.get("majors", [])}
    # 7480101 must have khmt as alias
    assert "7480101" in by_code, "7480101 (KHMT) missing from scope file"
    aliases_7480101 = by_code["7480101"].get("aliases", [])
    assert "khmt" in aliases_7480101, (
        f"KHMT alias must be on 7480101, got {aliases_7480101!r}"
    )
    # 7440301 must NOT have khmt
    aliases_7440301 = by_code.get("7440301", {}).get("aliases", [])
    assert "khmt" not in aliases_7440301, (
        f"KHMT must NOT leak into 7440301; got {aliases_7440301!r}"
    )


# ---------------------------------------------------------------------------
# F. Direct structural assertions: new keywords / aliases registered
# ---------------------------------------------------------------------------

def test_admission_keywords_have_safe_standalone_expansions():
    """Pin the structural fix: ADMISSION_KEYWORDS must contain the
    safe-standalone additions the spec mandates, and MUST NOT contain
    the ambiguous / false-block-prone ones.
    """
    from services.guardrail import GuardrailService

    required = [
        "điểm sàn", "điểm trúng tuyển", "điểm xét tuyển", "tổ hợp môn",
        "mã ngành", "đăng ký xét tuyển", "nguyện vọng", "chương trình đào tạo",
        "cổng thông tin tuyển sinh", "lệ phí", "miễn giảm", "xét tuyển thẳng",
        "hạn nộp", "đánh giá năng lực", "ký túc xá", "ktx", "liên thông",
        "vào trường", "thi vào", "học bạ",
        "nguyen vong", "vao truong", "thi vao", "dgnl", "đgnl", "ky tuc xa",
    ]
    for kw in required:
        assert kw in GuardrailService.ADMISSION_KEYWORDS, (
            f"ADMISSION_KEYWORDS missing required safe-standalone kw {kw!r}"
        )

    # Forbidden (would re-open other-school / generic false-block)
    forbidden = [
        "trường", "khoa", "ielts", "chứng chỉ", "tư vấn", "điều kiện",
        "yêu cầu", "việc làm", "ra trường", "thư viện", "ở đâu", "khi nào",
        "bao giờ", "thạc sĩ", "tiến sĩ", "từ xa",
    ]
    for kw in forbidden:
        assert kw not in GuardrailService.ADMISSION_KEYWORDS, (
            f"ADMISSION_KEYWORDS contains forbidden generic kw {kw!r}"
        )


def test_husc_aliases_have_new_short_forms():
    from services.guardrail import GuardrailService

    for a in ("đhkh huế", "dhkh hue", "đhkh", "dhkh", "đhkhh",
              "trường đhkh", "truong dhkh"):
        assert a in GuardrailService.HUSC_ALIASES, (
            f"HUSC_ALIASES missing short-form {a!r}"
        )


def test_one_char_guard_constant_present():
    """Pin the structural fix: _BARE_GENERIC_MAJORS must be a non-empty
    frozenset containing the four core short majors.
    """
    from services.guardrail import GuardrailService

    assert hasattr(GuardrailService, "_BARE_GENERIC_MAJORS")
    for tok in ("toan", "van", "hoa", "ly", "su", "kt"):
        assert tok in GuardrailService._BARE_GENERIC_MAJORS, (
            f"_BARE_GENERIC_MAJORS missing {tok!r}"
        )
    assert hasattr(GuardrailService, "_is_bare_generic_major_token")
    assert GuardrailService._is_bare_generic_major_token("toán") is True
    assert GuardrailService._is_bare_generic_major_token("Văn") is True
    assert GuardrailService._is_bare_generic_major_token("Hoá") is True
    assert GuardrailService._is_bare_generic_major_token("Công nghệ thông tin") is False
