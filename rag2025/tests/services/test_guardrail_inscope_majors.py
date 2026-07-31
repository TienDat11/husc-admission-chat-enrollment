"""TDD: guardrail must NOT block in-scope queries that mention "ngành" generically.

Bug-3 root cause: ``GuardrailService._extract_major_phrase`` Pattern 1
matches ``ngành`` and captures the ENTIRE tail (e.g. "đào tạo của trường mình"),
then ``_is_likely_major_phrase`` returns True (≥1 token ≥3 chars), so
``_major_out_of_scope`` blocks 8 in-scope sentences with ``major_not_offered``.

Fix target: queries that mention "ngành" but no specific major name must
fall through to the keyword fast-path / LLM as in-scope.

8 in-scope sentences (from results/86q_thi_sinh_results_post_xcheck_fix.json):
    msg001, msg015, msg049, msg051, msg052, msg064, msg071, msg075

Regression guard: real out-of-scope majors (Y khoa, Răng Hàm Mặt, Luật
kinh tế) MUST still be blocked — do not open the guardrail.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


IN_SCOPE_QUERIES = [
    # msg001 — generic list of majors
    "Em muốn tìm hiểu về các ngành đào tạo của trường mình",
    # msg015 — generic, xem ở đâu
    "Em muốn tìm hiểu thông tin về các ngành đào tạo ở trường, thì xem ở đâu ạ?",
    # msg049 — generic about phương thức xét per tổ hợp
    "cho em hỏi các ngành thì mỗi tổ hợp sẽ chỉ có 1 lựa chọn là 1 phương thức xét thôi ạ",
    # msg051 — xét 2 phương thức per ngành
    "Cùng 1 ngành học thì mình được phép xét 2 phương thức k ạ. Dùng điểm thi tốt nghiệp và học bạ",
    # msg052 — học song ngành 2 trường
    "Em muốn học song ngành ở 2 trường khác nhau có được không?",
    # msg064 — chỉ tiêu ngành thay đổi
    "Chỉ tiêu ngành đưa ra liệu có thể thay đổi hay có thể tăng chỉ tiêu được không ad?",
    # msg071 — quy đổi điểm giữa các tổ hợp
    "Trường mình có quy đổi điểm giữa các tổ hợp không ạ ?",
    # msg075 — NV1 ngành
    "Trường có yêu cầu, bắt buộc phải chọn ngành của trường là NV1 không?",
]


# Real out-of-scope majors — guardrail MUST still block these.
OUT_OF_SCOPE_QUERIES = [
    "Trường có đào tạo ngành Y khoa không?",
    "ngành Răng Hàm Mặt",
    "ngành Luật kinh tế",
]


def _build_service(monkeypatch):
    """Build a GuardrailService with the Groq client mocked out so no
    network call happens during precheck(). Mirrors the helper in
    test_guardrail_fastpath_comprehensive.py."""
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


@pytest.mark.parametrize("query", IN_SCOPE_QUERIES)
def test_in_scope_ngành_generic_not_blocked(monkeypatch, query: str) -> None:
    """In-scope queries mentioning 'ngành' generically must NOT be blocked."""
    svc = _build_service(monkeypatch)
    decision = asyncio.run(svc.precheck(query))
    assert decision.is_in_scope is True, (
        f"BUG-3 regression: in-scope query blocked.\n"
        f"  query={query!r}\n"
        f"  internal_code={decision.internal_code!r}\n"
        f"  reason={decision.reason!r}"
    )
    assert decision.internal_code != "major_not_offered", (
        f"BUG-3 regression: in-scope hit major_not_offered for {query!r}"
    )


@pytest.mark.parametrize("query", OUT_OF_SCOPE_QUERIES)
def test_out_of_scope_ngành_real_majors_still_blocked(monkeypatch, query: str) -> None:
    """Regression guard: real out-of-scope majors MUST still be blocked."""
    svc = _build_service(monkeypatch)
    decision = asyncio.run(svc.precheck(query))
    assert decision.is_in_scope is False, (
        f"Guardrail regression: out-of-scope major NOT blocked.\n"
        f"  query={query!r}\n"
        f"  decision={decision!r}"
    )
    # Either major-specific or generic NOT_IN_HUSC_SCOPE — both mean blocked.
    assert decision.internal_code in {"major_not_offered", "NOT_IN_HUSC_SCOPE"}, (
        f"Guardrail regression: wrong code for out-of-scope {query!r}: "
        f"{decision.internal_code!r}"
    )