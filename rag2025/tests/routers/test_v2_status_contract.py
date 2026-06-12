"""Tests for /v2/query status contract (Group 2 — G2-T1).

@spec(G2)

These tests verify the UnifiedQueryResponse (a.k.a. /v2/query) carries the
guardrail / no-result status fields that the FE ChatLayout banner + gap-hint
UX depends on (status_code, status_reason, data_gap_hints,
internal_status_code, pii_detected).

Approach: build a thin FastAPI app that mounts the *real* /v2/query handler
from src/main.py, but injects a stubbed guardrail_service + a minimal
unified_pipeline + retriever before the handler runs. This matches the
pattern of test_meta.py / test_admin.py (load the module via importlib,
wire dependencies on app.state) but inverts it: we need to wire global
service variables inside main.py rather than app.state. The fastest seam is
to monkeypatch the module-level globals (guardrail_service,
unified_pipeline, lancedb_retriever_service, embedding_service) BEFORE the
TestClient hits the endpoint. TestClient triggers the startup event, so we
pre-set globals and then force the app to skip the heavy startup by
deferring to our pre-set values.
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


SRC = Path(__file__).resolve().parents[2] / "src"
MAIN_PATH = SRC / "main.py"

# Sys-path bootstrap mirrors test_v2_uses_generate_answer.py.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_main(monkeypatch):
    """Load src/main.py as a stand-alone module, but pre-stub the heavy
    services the startup event would otherwise require (LanceDB, GraphRAG,
    embeddings, generator). Then re-mount just the /v2/query handler on a
    fresh FastAPI app to avoid pulling the entire /query pipeline.
    """
    monkeypatch.setenv("ADMIN_API_TOKEN", "")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("RAMCLOUDS_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("ZAI_API_KEY", "")

    spec = importlib.util.spec_from_file_location("v2_status_contract_test", MAIN_PATH)
    assert spec and spec.loader, "could not build import spec for src/main.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_fake_guardrail(*, is_in_scope: bool, internal_code: str,
                          reason: str = "fake_reason",
                          hints: list[str] | None = None,
                          pii: bool = False,
                          short_answer: str = "Câu trả lời OOS") -> MagicMock:
    """Build a fake guardrail_service whose precheck + classify_no_result
    return predictable GuardrailDecision-shaped values.
    """
    decision = SimpleNamespace(
        is_in_scope=is_in_scope,
        internal_code=internal_code,
        reason=reason,
        short_answer=short_answer,
        data_gap_hints=hints if hints is not None else ["HUSC chưa có dữ liệu ngành X", "Bổ sung chunk mới"],
        pii_detected=pii,
    )
    svc = MagicMock()
    svc.precheck = AsyncMock(return_value=decision)
    svc.classify_no_result = AsyncMock(return_value=decision)
    svc.public_status = MagicMock(side_effect=lambda code: "SUCCESS" if code == "SUCCESS" else code)
    svc.expose_internal = MagicMock(return_value=True)
    return svc


def _make_fake_generator(answer: str = "Đây là câu trả lời tổng hợp.",
                          provider: str = "fake") -> MagicMock:
    gen = MagicMock()
    gen.generate_answer = AsyncMock(return_value={
        "answer": answer,
        # generate_answer emits enriched SourceChip dicts (id/title/url/
        # snippet/data_year) — UnifiedQueryResponse.sources is List[SourceChip].
        "sources": [{
            "id": "s1",
            "title": "Nguồn 1",
            "url": None,
            "snippet": "trích đoạn",
            "data_year": "2026",
        }],
        "confidence": 0.8,
        "provider": provider,
        "chunks_used": 1,
    })
    return gen


def _make_fake_pipeline_and_retriever(rag_result: SimpleNamespace) -> tuple:
    pipeline = MagicMock()
    pipeline.query = AsyncMock(return_value=rag_result)
    pipeline._graphrag = SimpleNamespace(graph_stats={"nodes": 1, "edges": 0})
    # The /v2 hot path now kicks off `unified_pipeline._router.route(query)`
    # CONCURRENTLY with baseline retrieval (latency refactor). The router
    # must therefore be awaitable here — wire it as an AsyncMock returning
    # the same router_result the pipeline reports, so the pass-through is
    # faithful and `create_task(...)` gets a real coroutine.
    pipeline._router = MagicMock()
    pipeline._router.route = AsyncMock(return_value=rag_result.router_result)

    retriever = MagicMock()
    result = SimpleNamespace(
        is_success=True,
        documents=[],
    )
    retriever.retrieve = MagicMock(return_value=result)
    return pipeline, retriever


def _build_app_with_handler(main_module, *, guardrail, pipeline, retriever, embedding, generator):
    """Mount just the /v2/query handler on a fresh app, after wiring the
    module-level service globals. We do NOT register startup — the handler
    reads the globals synchronously.
    """
    main_module.guardrail_service = guardrail
    main_module.unified_pipeline = pipeline
    main_module.lancedb_retriever_service = retriever
    main_module.embedding_service = embedding
    main_module.llm_generator_service = generator

    app = FastAPI()
    # Re-mount only the /v2/query endpoint to keep tests fast.
    app.post("/v2/query", response_model=main_module.UnifiedQueryResponse)(main_module.unified_query)
    return app


@pytest.fixture
def env(monkeypatch):
    """Pre-stubbed services bundle for tests that need a wired app."""
    rag_result = SimpleNamespace(
        query="học phí ngành CNTT?",
        route="padded_rag",
        documents=[],
        router_result=SimpleNamespace(
            skip_retrieval=False,
            auto_answer=None,
            step_back_query="step",
            intent="general",
            complexity=1,
            reasoning="ok",
            hyde_variants=[],
        ),
        ppr_scores={},
        latency_ms=12.5,
        confidence=0.5,
    )
    return SimpleNamespace(
        main_module=_load_main(monkeypatch),
        guardrail=None,  # set per-test
        pipeline=None,
        retriever=None,
        embedding=MagicMock(),
        generator=None,
        rag_result=rag_result,
    )


# ────────────────────────────────────────────────────────────────────────
# G2-T1: status contract shape
# ────────────────────────────────────────────────────────────────────────

def test_v2_out_of_scope_query_returns_non_success_status(env):
    """@spec(G2) — Out-of-scope query must short-circuit on guardrail and
    return a non-SUCCESS status_code with populated data_gap_hints.

    RED today: the /v2 handler does not call guardrail_service.precheck, so
    the field is the default SUCCESS and data_gap_hints is empty.
    """
    env.guardrail = _make_fake_guardrail(
        is_in_scope=False,
        internal_code="NOT_IN_HUSC_SCOPE",
        reason="out_of_scope_fake",
        hints=["Câu hỏi nằm ngoài phạm vi tuyển sinh HUSC"],
    )
    env.pipeline, env.retriever = _make_fake_pipeline_and_retriever(env.rag_result)
    env.generator = _make_fake_generator()

    main_module = env.main_module
    app = _build_app_with_handler(
        main_module,
        guardrail=env.guardrail,
        pipeline=env.pipeline,
        retriever=env.retriever,
        embedding=env.embedding,
        generator=env.generator,
    )
    client = TestClient(app)
    res = client.post("/v2/query", json={"query": "thời tiết Hà Nội hôm nay?", "top_k": 5})

    assert res.status_code == 200, res.text
    body = res.json()
    # (a) OOS short-circuit
    assert body["status_code"] != "SUCCESS", body
    assert body["status_code"] == "NOT_IN_HUSC_SCOPE"
    assert isinstance(body["data_gap_hints"], list)
    assert len(body["data_gap_hints"]) >= 1
    assert body["status_reason"] == "out_of_scope_fake"
    # The pipeline was NOT called for the OOS path.
    env.pipeline.query.assert_not_awaited()


def test_v2_response_contains_all_five_status_fields(env):
    """@spec(G2) — Even on a normal/happy-path response, the contract
    surface must include all 5 fields that the FE ChatLayout depends on.
    """
    env.guardrail = _make_fake_guardrail(
        is_in_scope=True,
        internal_code="SUCCESS",
        reason="in_scope",
        hints=[],
    )
    env.pipeline, env.retriever = _make_fake_pipeline_and_retriever(env.rag_result)
    env.generator = _make_fake_generator()

    main_module = env.main_module
    app = _build_app_with_handler(
        main_module,
        guardrail=env.guardrail,
        pipeline=env.pipeline,
        retriever=env.retriever,
        embedding=env.embedding,
        generator=env.generator,
    )
    client = TestClient(app)
    res = client.post("/v2/query", json={"query": "học phí ngành CNTT?", "top_k": 5})

    assert res.status_code == 200, res.text
    body = res.json()
    for key in (
        "status_code",
        "status_reason",
        "data_gap_hints",
        "internal_status_code",
        "pii_detected",
    ):
        assert key in body, f"missing {key!r} in /v2 response: {body}"


def test_v2_normal_path_defaults_status_to_success(env):
    """@spec(G2) — When the query is in-scope and yields chunks, status
    fields must default to SUCCESS / no reason / no hints / no internal
    code / pii_detected=False.
    """
    env.guardrail = _make_fake_guardrail(
        is_in_scope=True,
        internal_code="SUCCESS",
    )
    env.pipeline, env.retriever = _make_fake_pipeline_and_retriever(env.rag_result)
    env.generator = _make_fake_generator()

    main_module = env.main_module
    app = _build_app_with_handler(
        main_module,
        guardrail=env.guardrail,
        pipeline=env.pipeline,
        retriever=env.retriever,
        embedding=env.embedding,
        generator=env.generator,
    )
    client = TestClient(app)
    res = client.post("/v2/query", json={"query": "học phí ngành CNTT?", "top_k": 5})

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status_code"] == "SUCCESS"
    assert body["status_reason"] in (None, "")
    assert body["data_gap_hints"] == []
    assert body["internal_status_code"] is None
    assert body["pii_detected"] is False
    # Existing /v2 contract keys intact
    for key in ("query", "route", "answer", "sources", "confidence", "router_info", "latency_ms", "trace_id"):
        assert key in body, f"REGRESSION: existing {key!r} key removed by G2 change"


def test_v2_no_result_path_classifies_status(env):
    """@spec(G2) — When retrieval yields zero documents, classify_no_result
    must populate the status fields (mimics /query 726-736).

    RED today: /v2 does not call classify_no_result, so the default
    SUCCESS persists even when documents is empty.
    """
    env.guardrail = _make_fake_guardrail(
        is_in_scope=True,
        internal_code="INSUFFICIENT_DATA",
        reason="no_data_fake",
        hints=["Bổ sung thêm dữ liệu ngành X", "Cập nhật dữ liệu tuyển sinh 2026"],
    )
    env.pipeline, env.retriever = _make_fake_pipeline_and_retriever(env.rag_result)
    env.generator = _make_fake_generator()

    main_module = env.main_module
    app = _build_app_with_handler(
        main_module,
        guardrail=env.guardrail,
        pipeline=env.pipeline,
        retriever=env.retriever,
        embedding=env.embedding,
        generator=env.generator,
    )
    client = TestClient(app)
    res = client.post("/v2/query", json={"query": "ngành XYZ123 không tồn tại?", "top_k": 5})

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status_code"] == "INSUFFICIENT_DATA"
    assert body["status_reason"] == "no_data_fake"
    assert len(body["data_gap_hints"]) >= 1
    # classify_no_result was awaited at least once
    env.guardrail.classify_no_result.assert_awaited()


def test_v2_pii_detection_propagates_to_response(env):
    """@spec(G2) — pii_detected=True on the precheck must propagate to the
    /v2 response (FE may mask/redact accordingly).
    """
    env.guardrail = _make_fake_guardrail(
        is_in_scope=False,
        internal_code="SENSITIVE_PII_DETECTED",
        reason="pii_fake",
        hints=["Ẩn bớt thông tin định danh trước khi gửi câu hỏi"],
        pii=True,
    )
    env.pipeline, env.retriever = _make_fake_pipeline_and_retriever(env.rag_result)
    env.generator = _make_fake_generator()

    main_module = env.main_module
    app = _build_app_with_handler(
        main_module,
        guardrail=env.guardrail,
        pipeline=env.pipeline,
        retriever=env.retriever,
        embedding=env.embedding,
        generator=env.generator,
    )
    client = TestClient(app)
    res = client.post("/v2/query", json={"query": "CCCD tôi là 012345678901, học phí?", "top_k": 5})

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["pii_detected"] is True
    assert body["status_code"] == "SENSITIVE_PII_DETECTED"
