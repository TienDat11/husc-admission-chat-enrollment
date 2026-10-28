"""T10 — refusal trailer (hotline + serving release date) on student abstentions.

Acceptance criteria:
 1. Out-of-scope question through the real /v2 seam → answer carries hotline
    and the release date currently being served.
 2. The footer appears exactly once even though the refusal crosses several
    layers (router → main → response payload).
 3. An in-scope SUCCESS answer must NOT carry the footer.
 4. When no release is configured, refusal_footer() still emits the hotline,
    omits the date line, and does not crash.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from src.services import abstention_policy as ap
# llm_generator.py uses a local `from services import abstention_policy` —
# same file, separate module object under importlib. Configure both so the
# release-date source lands whichever instance the path under test reads.
from services import abstention_policy as ap_services


def _set_release(source):
    """Configure the release-date source on BOTH module instances."""
    ap.configure_release_date_source(source)
    ap_services.configure_release_date_source(source)


HOTLINE = "📞 Hotline:"
LINK = "🌐 https://tuyensinh.husc.edu.vn"
DATE_MARKER = "Thông tin cập nhật lần cuối"


# ---------------------------------------------------------------------------
# Harness — mirrors tests/routers/test_v2_abstention_http.py so every change
# is exercised on the REAL handlers with a stubbed router/LLM.
# ---------------------------------------------------------------------------


def _make_doc(text: str, *, year: str, source: str | None = None):
    meta = {"data_year": year}
    if source is not None:
        meta["source"] = source
    return SimpleNamespace(text=text, metadata=meta, score=0.7,
                           chunk_id="c", source=source)


def _make_rag_result(query: str, documents, *, is_in_scope: bool):
    router = SimpleNamespace(
        skip_retrieval=False, auto_answer=None, step_back_query="",
        intent="general", complexity=1, reasoning="",
        hyde_variants=[], is_in_scope=is_in_scope,
    )
    return SimpleNamespace(
        query=query, route="padded_rag", documents=documents,
        router_result=router, ppr_scores={}, latency_ms=12.5, confidence=0.5,
    )


def _make_guardrail():
    d = SimpleNamespace(
        is_in_scope=True, internal_code="SUCCESS", reason="",
        short_answer="", data_gap_hints=[], pii_detected=False,
    )
    svc = MagicMock()
    svc.precheck = AsyncMock(return_value=d)
    svc.classify_no_result = AsyncMock(return_value=d)
    svc.public_status = MagicMock(side_effect=lambda code: code)
    svc.expose_internal = MagicMock(return_value=False)
    svc._pii_warning_answer = MagicMock(return_value="pii")
    svc._contains_sensitive_pii = MagicMock(return_value=False)
    return svc


def _make_generator(answer_text: str):
    from services.llm_generator import LLMGenerator
    from services.season import SeasonPhase

    gen = LLMGenerator.__new__(LLMGenerator)
    gen.generation_system_prompt = "T"
    gen.has_any_provider = True
    gen.gen_model = "t"
    gen.gen_fallback_model = "t"
    gen.unified_fallback_client = None

    fake = MagicMock()
    fake._providers = ["primary"]
    resp = MagicMock()
    resp.content = answer_text
    resp.model = "t"
    resp.provider = "t"
    fake.chat = AsyncMock(return_value=resp)
    gen.unified_client = fake

    async def _nop(a, *a_, **k_):
        return a

    gen._enforce_vietnamese = _nop
    gen._resolve_admission_context = lambda c: (2026, SeasonPhase.IN_SEASON, True)
    return gen, fake.chat


def _build_app(module, *, guardrail, pipeline, generator):
    module.guardrail_service = guardrail
    module.unified_pipeline = pipeline
    module.lancedb_retriever_service = None
    module.embedding_service = None
    module.hybrid_search_service = None
    module.reranker_service = None
    module.query_cache = None
    module.llm_generator_service = generator
    app = FastAPI()
    app.state.unanswered_queue = MagicMock()  # T07 hook dep; not T10 scope
    app.post("/v2/query", response_model=module.UnifiedQueryResponse)(
        module.unified_query)
    app.post("/v2/query/stream")(module.unified_query_stream)
    return app


def _client(documents, answer_text="x", *, is_in_scope: bool = True):
    spec = importlib.util.spec_from_file_location("main", RAG_SRC / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(RAG_SRC))
    spec.loader.exec_module(module)

    generator, fake_chat = _make_generator(answer_text)
    pipeline = MagicMock()
    pipeline.query = AsyncMock(side_effect=lambda **kw: _make_rag_result(
        kw.get("user_query", ""), documents, is_in_scope=is_in_scope,
    ))
    pipeline._graphrag = SimpleNamespace(graph_stats={})
    pipeline._router = MagicMock(
        route=AsyncMock(return_value=SimpleNamespace(
            skip_retrieval=False, auto_answer=None, step_back_query="",
            intent="general", complexity=1, reasoning="",
            hyde_variants=[], is_in_scope=is_in_scope,
        ))
    )
    return TestClient(_build_app(module, guardrail=_make_guardrail(),
                                 pipeline=pipeline, generator=generator)), fake_chat


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------


def test_1_oos_question_carries_hotline_and_serving_release_date():
    """AC1 — out-of-scope on the real /v2 path → hotline + serving release date."""
    _set_release(lambda: "2026-09-01")
    try:
        client, chat = _client([], answer_text="x", is_in_scope=False)
        r = client.post("/v2/query",
                        json={"query": "Hôm nay bóng đá HUSC thắng không?", "top_k": 5})
        assert r.status_code == 200
        body = r.json()
        assert body["status_code"] == "NOT_IN_HUSC_SCOPE"
        assert chat.await_count == 0, "model must not be called for out-of-scope"
        answer = body["answer"]
        assert HOTLINE in answer
        assert "Phòng Đào tạo" in answer
        assert LINK in answer
        assert "01/09/2026" in answer  # dd/mm/yyyy of the served release
    finally:
        _set_release(None)


def test_2_footer_appears_exactly_once_across_layers():
    """AC2 — footer exactly once even crossing several layers."""
    _set_release(lambda: "2026-09-01")
    try:
        client, chat = _client([], answer_text="x", is_in_scope=False)
        r = client.post("/v2/query",
                        json={"query": "Giá cổ phiếu Techcomstock hôm nay?", "top_k": 5})
        body = r.json()
        assert body["answer"].count(HOTLINE) == 1, "no duplicated footer"
        assert body["answer"].count(LINK) == 1
        assert body["answer"].count(DATE_MARKER) == 1
    finally:
        _set_release(None)


def test_3_in_scope_success_has_no_footer():
    """AC3 — an in-scope SUCCESS answer must NOT carry the footer."""
    _set_release(lambda: "2026-09-01")
    try:
        client, chat = _client(
            [_make_doc("Học phí ngành CNTT 2026 là 18.000.000đ.",
                       year="2026", source="TB60 2026")],
            answer_text="Học phí ngành CNTT 2026 là 18.000.000đ.",
            is_in_scope=True,
        )
        r = client.post("/v2/query",
                        json={"query": "Học phí ngành CNTT năm 2026 là bao nhiêu?"})
        assert r.status_code == 200
        body = r.json()
        assert body["status_code"] == "SUCCESS"
        assert chat.await_count == 1, "model must run for grounded question"
        assert HOTLINE not in body["answer"]
        assert LINK not in body["answer"]
    finally:
        _set_release(None)


def test_4_no_serving_release_still_shows_hotline():
    """AC4 — release_date() None → hotline present, no date line, no crash."""
    _set_release(None)
    client, chat = _client([], answer_text="x", is_in_scope=False)
    try:
        r = client.post("/v2/query",
                        json={"query": "Tiền đạo Liverpool thi đấu ở câu lạc bộ nào?"})
        assert r.status_code == 200
        body = r.json()
        answer = body["answer"]
        assert HOTLINE in answer
        assert "Phòng Đào tạo" in answer
        assert LINK in answer
        assert DATE_MARKER not in answer, "no last-updated line when nothing served"
        assert chat.await_count == 0
    finally:
        _set_release(None)


def test_5_policy_refusals_also_get_the_footer():
    """Sanity — policy abstentions (no evidence / year mismatch) end with footer."""
    _set_release(lambda: "2026-09-01")
    try:
        client, chat = _client([], answer_text="x", is_in_scope=True)
        # No chunks + factual current-year query → REASON_NO_EVIDENCE abstain.
        r = client.post("/v2/query",
                        json={"query": "Điểm chuẩn ngành CNTT năm 2026?", "top_k": 5})
        assert r.status_code == 200
        assert chat.await_count == 0, "policy must refuse before model call"
        assert HOTLINE in r.json()["answer"]
        assert "Phòng Đào tạo" in r.json()["answer"]
        assert "01/09/2026" in r.json()["answer"]
    finally:
        _set_release(None)
