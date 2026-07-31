"""TDD: REGRESSION-FIX — "Trường có ngành Dược không?" must be OOS (HUSC
does not teach Dược), even after the V5-R013 hardened bypass. The fix
adds _denylist_token_is_admission_language + _phrase_hits_denylist guards
that distinguish:
  Lớp A — "Trường có ngành <TÊN NGÀNH CỤ THỂ> không?" when the named
  major is NOT in the HUSC allowlist (Dược, Y khoa, Răng Hàm Mặt,
  Luật kinh tế, Điều dưỡng, …). Must BLOCK.
  Lớp B — admission-domain language questions (phương thức tuyển sinh,
  tổ hợp, chứng chỉ tiếng anh, chính sách học bổng, …). Must PASS.

Root cause of the regression: the hardened admission-intent bypass
(`_is_admission_intent_query(query)` AND phrase_early is None) was
firing on "Trường có ngành Dược không?" because the major extractor
collapsed the "Dược không" capture to None via _is_all_stopwords, even
though "Dược" is a real denylist major noun. The fix adds a guard that
falls through to the denylist branch whenever the denylist substring on
the whole query is a bare-major-noun reference (not part of admission-
domain language).
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# OOS — must be BLOCKED. HUSC does not teach these majors.
OOS_QUERIES = [
    "Trường có ngành Dược không?",
    "Trường có đào tạo ngành Y khoa không?",
    "ngành Răng Hàm Mặt",
    "ngành Luật kinh tế",
    "Trường có ngành Điều dưỡng không?",
]


# IN-SCOPE — 6 admission-domain language queries (msg016/020/023/029/041/065).
IN_SCOPE_DOMAIN_QUERIES = [
    # msg016 — phương thức tuyển sinh
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


# IN-SCOPE — 10 previously-fixed queries must STAY in-scope.
IN_SCOPE_OK_QUERIES = [
    # msg001
    "Em muốn tìm hiểu về các ngành đào tạo của trường mình",
    # msg015
    "Em muốn tìm hiểu thông tin về các ngành đào tạo ở trường, thì xem ở đâu ạ?",
    # msg019
    "cho em hỏi các ngành thì mỗi tổ hợp sẽ chỉ có 1 lựa chọn là 1 phương thức xét thôi ạ",
    # msg024
    "Cùng 1 ngành học thì mình được phép xét 2 phương thức k ạ. Dùng điểm thi tốt nghiệp và học bạ",
    # msg049
    "Chỉ tiêu ngành đưa ra liệu có thể thay đổi hay có thể tăng chỉ tiêu được không ad?",
    # msg051
    "Trường mình có quy đổi điểm giữa các tổ hợp không ạ ?",
    # msg052
    "Trường có yêu cầu, bắt buộc phải chọn ngành của trường là NV1 không?",
    # msg064
    "Em muốn tìm hiểu về các ngành đào tạo của trường?",
    # msg071
    "điểm chuẩn ngành CNTT 2026",
    # msg075
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


@pytest.mark.parametrize("query", OOS_QUERIES)
def test_duoc_regression_oos_blocked(monkeypatch, query: str) -> None:
    """REGRESSION-FIX: OOS queries naming a major HUSC does not teach
    MUST be blocked (is_in_scope=False). Includes Dược, Y khoa, Răng
    Hàm Mặt, Luật kinh tế, Điều dưỡng.
    """
    svc = _build_service(monkeypatch)
    decision = asyncio.run(svc.precheck(query))
    assert decision.is_in_scope is False, (
        f"REGRESSION: OOS major NOT blocked.\n"
        f"  query={query!r}\n"
        f"  decision={decision!r}"
    )


@pytest.mark.parametrize("query", IN_SCOPE_DOMAIN_QUERIES)
def test_duoc_regression_domain_in_scope(monkeypatch, query: str) -> None:
    """REGRESSION-FIX: 6 admission-domain language queries (msg016/020/
    023/029/041/065) MUST remain in-scope. They are process questions
    not major references.
    """
    svc = _build_service(monkeypatch)
    decision = asyncio.run(svc.precheck(query))
    assert decision.is_in_scope is True, (
        f"REGRESSION: domain query blocked.\n"
        f"  query={query!r}\n"
        f"  decision={decision!r}"
    )


@pytest.mark.parametrize("query", IN_SCOPE_OK_QUERIES)
def test_duoc_regression_previous_in_scope(monkeypatch, query: str) -> None:
    """REGRESSION-FIX: 10 previously-fixed in-scope queries must STAY
    in-scope after the Dược fix."""
    svc = _build_service(monkeypatch)
    decision = asyncio.run(svc.precheck(query))
    assert decision.is_in_scope is True, (
        f"REGRESSION: previously-fixed in-scope now blocked.\n"
        f"  query={query!r}\n"
        f"  decision={decision!r}"
    )
