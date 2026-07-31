"""TDD: V5-R013 / FIX 1 — guardrail must NOT block in-scope admission-process
queries that the major extractor was wrongly capturing as a "major name".

Root cause: 6 STILL_BLOCKED in-scope queries (msg016/020/023/029/041/065) had
their major-phrase capture (e.g. "có những phương thức", "xét những",
"Em muốn tìm hiểu chính sách") pass the major gates but fail the
allowlist → block. The phrase IS generic admission-process language, not
a major name.

Fix: extend ``_ADMISSION_LANGUAGE_PHRASES`` + ``ADMISSION_KEYWORDS`` to
cover admission-domain language (phương thức tuyển sinh, tổ hợp, chứng
chỉ tiếng anh, chính sách học bổng, yêu cầu tiếng anh đầu vào, quy
đổi điểm, học bạ, tốt nghiệp). AND harden ``_major_out_of_scope`` so
that when the query IS admission-intent AND the captured phrase is NOT
in the allowlist, we collapse it to in-scope (NOT a real major name —
it's just generic admission-domain language that survived the gates).

Regression: 4 real out-of-scope majors (Y khoa, Răng Hàm Mặt, Luật kinh
tế, Dược) MUST still be blocked. Plus 10 OK queries from previous fix
must remain in-scope.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# V5-R013 STILL_BLOCKED → in-scope (post-fix)
STILL_BLOCKED_NOW_IN_SCOPE = [
    # msg016 — phương thức tuyển sinh (admission-domain, not a major)
    "Trường có những phương thức tuyển sinh nào?",
    # msg020 — tổ hợp xét tuyển
    "Ngành... xét những tổ hợp nào ạ?",
    # msg023 — học bạ tổ hợp khác
    "Em thi THPT tổ hợp khác. Mà em xét học bạ tổ hợp khác được không ạ?",
    # msg029 — yêu cầu tiếng anh đầu vào
    "Trường mình có yêu cầu tiếng anh đầu vào không ạ?",
    # msg041 — quy đổi điểm chứng chỉ tiếng anh
    "Cách quy đổi điểm chứng chỉ tiếng anh như thế nào?",
    # msg065 — chính sách học bổng
    "Em muốn tìm hiểu chính sách học bổng của trường?",
]


# OOS regression — DO NOT collapse these to in-scope
REAL_OUT_OF_SCOPE = [
    "Trường có đào tạo ngành Y khoa không?",
    "ngành Răng Hàm Mặt",
    "ngành Luật kinh tế",
    "Trường đào tạo ngành Dược không?",
    "HUSC có ngành Dược không?",
]


# Previously-fixed in-scope queries — must STAY in-scope
PREVIOUSLY_FIXED_IN_SCOPE = [
    "Em muốn tìm hiểu về các ngành đào tạo của trường mình",
    "Em muốn tìm hiểu thông tin về các ngành đào tạo ở trường, thì xem ở đâu ạ?",
    "cho em hỏi các ngành thì mỗi tổ hợp sẽ chỉ có 1 lựa chọn là 1 phương thức xét thôi ạ",
    "Cùng 1 ngành học thì mình được phép xét 2 phương thức k ạ. Dùng điểm thi tốt nghiệp và học bạ",
    "Chỉ tiêu ngành đưa ra liệu có thể thay đổi hay có thể tăng chỉ tiêu được không ad?",
    "Trường mình có quy đổi điểm giữa các tổ hợp không ạ ?",
    "Trường có yêu cầu, bắt buộc phải chọn ngành của trường là NV1 không?",
    "Em muốn tìm hiểu về các ngành đào tạo của trường?",
    "điểm chuẩn ngành CNTT 2026",
    "học phí ngành Khoa học dữ liệu",
]


def _build_service(monkeypatch):
    """Build a GuardrailService with the Groq client mocked out so no
    network call happens during precheck()."""
    monkeypatch.setenv("GROQ_API_KEY", "test-fake-groq-key")

    from config.settings import RAGSettings
    from services import guardrail as guardrail_mod

    class _FakeChat:
        class completions:
            @staticmethod
            async def create(*args, **kwargs):
                raise RuntimeError(
                    "LLM must NOT be called for fast-path / denylist hits"
                )

    class _FakeAsyncGroq:
        def __init__(self, *a, **kw):
            self.chat = _FakeChat()

    monkeypatch.setattr(guardrail_mod, "AsyncGroq", _FakeAsyncGroq)
    guardrail_mod.GuardrailService.clear_major_scope_cache_for_testing()
    settings = RAGSettings()
    return guardrail_mod.GuardrailService(settings)


@pytest.mark.parametrize("query", STILL_BLOCKED_NOW_IN_SCOPE)
def test_v5r013_still_blocked_now_in_scope(monkeypatch, query: str) -> None:
    """V5-R013 FIX 1: 6 STILL_BLOCKED in-scope admission queries must pass."""
    svc = _build_service(monkeypatch)
    decision = asyncio.run(svc.precheck(query))
    assert decision.is_in_scope is True, (
        f"V5-R013 regression: STILL_BLOCKED query still blocked.\n"
        f"  query={query!r}\n"
        f"  internal_code={decision.internal_code!r}\n"
        f"  reason={decision.reason!r}"
    )


@pytest.mark.parametrize("query", REAL_OUT_OF_SCOPE)
def test_v5r013_real_out_of_scope_still_blocked(monkeypatch, query: str) -> None:
    """V5-R013 regression: 4 real out-of-scope majors MUST still be blocked."""
    svc = _build_service(monkeypatch)
    decision = asyncio.run(svc.precheck(query))
    assert decision.is_in_scope is False, (
        f"V5-R013 regression: real out-of-scope major NOT blocked.\n"
        f"  query={query!r}\n"
        f"  decision={decision!r}"
    )
    assert decision.internal_code in {"major_not_offered", "NOT_IN_HUSC_SCOPE"}, (
        f"V5-R013 regression: wrong code for OOS {query!r}: {decision.internal_code!r}"
    )


@pytest.mark.parametrize("query", PREVIOUSLY_FIXED_IN_SCOPE)
def test_v5r013_previous_in_scope_still_in_scope(monkeypatch, query: str) -> None:
    """V5-R013 regression: previously-fixed in-scope queries must remain in-scope."""
    svc = _build_service(monkeypatch)
    decision = asyncio.run(svc.precheck(query))
    assert decision.is_in_scope is True, (
        f"V5-R013 regression: previously-fixed in-scope now blocked.\n"
        f"  query={query!r}\n"
        f"  reason={decision.reason!r}"
    )
    assert decision.internal_code != "major_not_offered"