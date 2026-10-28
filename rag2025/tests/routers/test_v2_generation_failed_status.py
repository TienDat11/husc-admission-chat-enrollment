"""FIX-05 (ticket 44) — degraded generation must never be published as SUCCESS.

@spec(FIX-05)

Invariant under test: when the answer step RAN but produced no model output
(generator reports `generation_failed=True`, or its `provider` tag starts with
`Fallback(`), the /v2 surface must report the degraded status
(`GENERATION_FAILED` / `generation_failed` / internal `GENERATION_FAILED`) with
`answer_confidence = null`, on BOTH the JSON and the SSE paths.

The retrieval confidence (`confidence`) keeps its own meaning and stays a
number — the two must not be blurred.

Harness: identical to test_v2_status_contract.py — load src/main.py as a
stand-alone module, wire fake guardrail / pipeline / retriever / generator
globals, and mount the REAL /v2 handlers on a fresh app.
"""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[2] / "src"
MAIN_PATH = SRC / "main.py"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FALLBACK_ANSWER = (
    "He thong đã tìm thay tài lieu liên quan nhung dich vu tao câu tra loi "
    "đang tam thoi không kha dung. Vui lòng thu lai sau."
)


def _load_main(monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("NS24_API_KEY", "")
    spec = importlib.util.spec_from_file_location("fix05_generation_failed_test", MAIN_PATH)
    assert spec and spec.loader, "could not build import spec for src/main.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _doc(text: str = "Học phí ngành CNTT là 15 triệu đồng/năm.") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        score=0.83,
        chunk_id="c1",
        source="hocphi.pdf",
        metadata={"summary": text, "data_year": "2026"},
    )


def _fake_guardrail(*, expose_internal: bool = True, no_result_code: str = "SUCCESS"):
    decision = SimpleNamespace(
        is_in_scope=True,
        internal_code=no_result_code,
        reason="insufficient_data_fake" if no_result_code != "SUCCESS" else "ok",
        short_answer="",
        data_gap_hints=["Thiếu dữ liệu"] if no_result_code != "SUCCESS" else [],
        pii_detected=False,
    )
    svc = MagicMock()
    svc.precheck = AsyncMock(return_value=decision)
    svc.classify_no_result = AsyncMock(return_value=decision)
    svc.public_status = MagicMock(side_effect=lambda code: "SUCCESS" if code == "SUCCESS" else code)
    svc.expose_internal = MagicMock(return_value=expose_internal)
    svc._contains_sensitive_pii = MagicMock(return_value=False)
    svc._pii_warning_answer = MagicMock(return_value="Ẩn thông tin định danh.")
    return svc


def _fake_generator(answer: str, provider: str, *, include_flag: bool = True) -> MagicMock:
    payload = {
        "answer": answer,
        "sources": [{
            "id": "s1", "title": "Nguồn 1", "url": None,
            "snippet": "trích đoạn", "data_year": "2026",
        }],
        "confidence": 0.83,
        "provider": provider,
        "chunks_used": 1,
    }
    if include_flag:
        payload["generation_failed"] = provider.startswith("Fallback(")
    gen = MagicMock()
    gen.generate_answer = AsyncMock(return_value=payload)
    return gen


def _fake_pipeline(documents: list, *, skip_retrieval: bool = False, auto_answer=None):
    router_result = SimpleNamespace(
        skip_retrieval=skip_retrieval,
        auto_answer=auto_answer,
        step_back_query="step",
        intent="general",
        complexity=1,
        reasoning="ok",
        hyde_variants=[],
        is_in_scope=True,
    )
    rag_result = SimpleNamespace(
        query="học phí ngành CNTT?",
        route="padded_rag",
        documents=documents,
        router_result=router_result,
        ppr_scores={},
        latency_ms=12.5,
        confidence=0.83,
    )
    pipeline = MagicMock()
    pipeline.query = AsyncMock(return_value=rag_result)
    pipeline._graphrag = SimpleNamespace(graph_stats={"nodes": 1, "edges": 0})
    pipeline._router = MagicMock()
    pipeline._router.route = AsyncMock(return_value=router_result)
    retriever = MagicMock()
    retriever.retrieve = MagicMock(return_value=SimpleNamespace(is_success=True, documents=documents))
    return pipeline, retriever


def _app(main_module, *, guardrail, pipeline, retriever, generator):
    main_module.guardrail_service = guardrail
    main_module.unified_pipeline = pipeline
    main_module.lancedb_retriever_service = retriever
    main_module.embedding_service = MagicMock()
    main_module.llm_generator_service = generator
    app = FastAPI()
    app.post("/v2/query", response_model=main_module.UnifiedQueryResponse)(main_module.unified_query)
    app.post("/v2/query/stream")(main_module.unified_query_stream)
    return app


def _parse_sse(raw: str) -> list[dict[str, str]]:
    frames: list[dict[str, str]] = []
    for chunk in raw.split("\n\n"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        ev = {"event": "", "data": ""}
        for line in chunk.splitlines():
            if line.startswith("event:"):
                ev["event"] = line[len("event:"):].strip()
            elif line.startswith("data:"):
                payload = line[len("data:"):].strip()
                ev["data"] = f"{ev['data']}{payload}" if ev["data"] else payload
        if ev["event"] or ev["data"]:
            frames.append(ev)
    return frames


@pytest.fixture
def main_module(monkeypatch):
    return _load_main(monkeypatch)


# ────────────────────────────────────────────────────────────────────────
# The invariant
# ────────────────────────────────────────────────────────────────────────

def test_fallback_answer_with_documents_is_not_success(main_module):
    """Documents exist, the LLM is dead → GENERATION_FAILED, not SUCCESS.

    This is the exact production defect: retrieval succeeded (so the
    empty-retrieval classifier never fires) and the fallback string used to
    be published as SUCCESS with a retrieval confidence attached.
    """
    pipeline, retriever = _fake_pipeline([_doc()])
    app = _app(
        main_module,
        guardrail=_fake_guardrail(),
        pipeline=pipeline,
        retriever=retriever,
        generator=_fake_generator(FALLBACK_ANSWER, "Fallback(Error)"),
    )
    body = TestClient(app).post("/v2/query", json={"query": "học phí ngành CNTT?"}).json()

    assert body["status_code"] == "GENERATION_FAILED", body
    assert body["status_reason"] == "generation_failed", body
    assert body["internal_status_code"] == "GENERATION_FAILED", body
    assert body["answer"] == FALLBACK_ANSWER
    # answer confidence must not exist for a non-model answer …
    assert body["answer_confidence"] is None, body
    # … while retrieval confidence keeps its own (retrieval) meaning.
    assert body["confidence"] == pytest.approx(0.83), body
    assert body["data_gap_hints"] == []


def test_provider_tag_alone_triggers_degraded_status(main_module):
    """The `generation_failed` flag is not the only signal: a suffixed
    provider tag (`Fallback(Error) + season-fallback`) must also be caught,
    so a caller that predates the flag cannot leak SUCCESS."""
    pipeline, retriever = _fake_pipeline([_doc()])
    app = _app(
        main_module,
        guardrail=_fake_guardrail(),
        pipeline=pipeline,
        retriever=retriever,
        generator=_fake_generator(
            FALLBACK_ANSWER, "Fallback(Error) + season-fallback", include_flag=False
        ),
    )
    body = TestClient(app).post("/v2/query", json={"query": "học phí ngành CNTT?"}).json()

    assert body["status_code"] == "GENERATION_FAILED", body
    assert body["answer_confidence"] is None, body


def test_no_provider_configured_is_also_degraded(main_module):
    """`Fallback(NoProvider)` — the no-provider terminal path — is degraded too."""
    pipeline, retriever = _fake_pipeline([_doc()])
    app = _app(
        main_module,
        guardrail=_fake_guardrail(),
        pipeline=pipeline,
        retriever=retriever,
        generator=_fake_generator(FALLBACK_ANSWER, "Fallback(NoProvider)"),
    )
    body = TestClient(app).post("/v2/query", json={"query": "học phí ngành CNTT?"}).json()

    assert body["status_code"] == "GENERATION_FAILED", body
    assert body["answer_confidence"] is None, body


def test_stream_and_non_stream_agree_on_degraded_status(main_module):
    """SSE meta must carry the same status surface as the JSON response."""
    pipeline, retriever = _fake_pipeline([_doc()])
    app = _app(
        main_module,
        guardrail=_fake_guardrail(),
        pipeline=pipeline,
        retriever=retriever,
        generator=_fake_generator(FALLBACK_ANSWER, "Fallback(Error)"),
    )
    client = TestClient(app)

    non_stream = client.post("/v2/query", json={"query": "học phí ngành CNTT?"}).json()

    res = client.post("/v2/query/stream", json={"query": "học phí ngành CNTT?"})
    assert res.status_code == 200, res.text
    frames = _parse_sse(res.text)
    meta = json.loads(next(f for f in frames if f["event"] == "meta")["data"])
    done = json.loads(next(f for f in frames if f["event"] == "done")["data"])

    for field in (
        "status_code",
        "status_reason",
        "internal_status_code",
        "answer_confidence",
        # Ticket 17: retrieval confidence rides the meta frame too. It is a
        # different quantity from answer_confidence (it survives a dead
        # generator), and the FE renders it onto every source chip — so a
        # missing key here is not a cosmetic omission, it is a chip carrying
        # a fabricated number.
        "confidence",
    ):
        assert meta[field] == non_stream[field], (
            f"stream meta {field}={meta[field]!r} disagrees with "
            f"non-stream {non_stream[field]!r}"
        )
    assert meta["status_code"] == "GENERATION_FAILED", meta
    assert done["answer"] == FALLBACK_ANSWER


# ────────────────────────────────────────────────────────────────────────
# Non-regression: the healthy paths keep their shape
# ────────────────────────────────────────────────────────────────────────

def test_healthy_generation_still_reports_success_with_answer_confidence(main_module):
    pipeline, retriever = _fake_pipeline([_doc()])
    app = _app(
        main_module,
        guardrail=_fake_guardrail(),
        pipeline=pipeline,
        retriever=retriever,
        generator=_fake_generator(
            "Học phí ngành CNTT là 15 triệu đồng/năm.", "mimo-v2.5-pro (ramclouds)"
        ),
    )
    body = TestClient(app).post("/v2/query", json={"query": "học phí ngành CNTT?"}).json()

    assert body["status_code"] == "SUCCESS", body
    assert body["status_reason"] is None, body
    # A model wrote this answer, so an answer-level confidence exists and is
    # the groundedness score — never a retrieval number.
    assert body["answer_confidence"] == pytest.approx(body["groundedness_score"]), body


def test_canned_auto_answer_has_no_answer_confidence(main_module):
    """skip_retrieval + auto_answer: nothing failed, but no model wrote it
    either, so answer_confidence stays null."""
    pipeline, retriever = _fake_pipeline(
        [], skip_retrieval=True, auto_answer="Vui lòng liên hệ Phòng Tuyển sinh HUSC."
    )
    app = _app(
        main_module,
        guardrail=_fake_guardrail(),
        pipeline=pipeline,
        retriever=retriever,
        generator=_fake_generator("unused", "unused"),
    )
    body = TestClient(app).post("/v2/query", json={"query": "liên hệ ở đâu?"}).json()

    assert body["status_code"] == "SUCCESS", body
    assert body["answer_confidence"] is None, body


def test_data_gap_classification_wins_over_generation_failure(main_module):
    """Zero documents + a data-gap classification stays the data-gap status:
    the actionable diagnosis is the missing data, and it is not SUCCESS
    either, so the FIX-05 invariant still holds."""
    pipeline, retriever = _fake_pipeline([])
    app = _app(
        main_module,
        guardrail=_fake_guardrail(no_result_code="INSUFFICIENT_DATA"),
        pipeline=pipeline,
        retriever=retriever,
        generator=_fake_generator(FALLBACK_ANSWER, "Fallback(Error)"),
    )
    body = TestClient(app).post("/v2/query", json={"query": "học phí ngành CNTT?"}).json()

    assert body["status_code"] == "INSUFFICIENT_DATA", body
    assert body["status_code"] != "SUCCESS"
    assert body["answer_confidence"] is None, body


def test_legacy_query_endpoint_also_refuses_success_on_fallback(main_module):
    """The same invariant on the legacy /query surface."""
    import numpy as np

    embedding = MagicMock()
    embedding.encode_batch = MagicMock(
        return_value=np.zeros((1, 4), dtype=np.float32)
    )
    retriever = MagicMock()
    retriever.retrieve = MagicMock(
        return_value=SimpleNamespace(
            documents=[], is_success=True, error_type=None, error_message=None
        )
    )
    enhancer = MagicMock()
    enhancer.enhance_query = AsyncMock(return_value={
        "variants": ["học phí ngành CNTT?"],
        "original_query": "học phí ngành CNTT?",
        "detected_intent": "tuition",
        "query_type": "specific",
        "top_k": 5,
    })
    main_module.guardrail_service = _fake_guardrail()
    main_module.llm_generator_service = _fake_generator(FALLBACK_ANSWER, "Fallback(Error)")
    main_module.embedding_service = embedding
    main_module.query_enhancer_service = enhancer
    main_module.lancedb_retriever_service = retriever
    main_module.hybrid_search_service = None
    main_module.reranker_service = None
    main_module.query_cache = None

    app = FastAPI()
    app.post("/query", response_model=main_module.QueryResponse)(main_module.query)
    res = TestClient(app).post(
        "/query", json={"query": "học phí ngành CNTT?", "force_rag_only": False}
    )
    assert res.status_code == 200, res.text
    body = res.json()

    assert body["status_code"] == "GENERATION_FAILED", body
    assert body["status_reason"] == "generation_failed", body


# ---------------------------------------------------------------------------
# T11 policy refusal — the third kind of answer
#
# A refusal is not a failure (it ships as SUCCESS, intentionally), but no model
# wrote it, so it must not carry `answer_confidence`. `model_answer` was
# computed as `not generation_failed`, which is True for a refusal, so the
# payload published a groundedness number as the refusal's answer confidence.
# Measured live before the fix: a year-mismatch refusal shipped
# `answer_confidence = 0.8696876764297485` for policy-authored text, and the FE
# renders that field (useChatStream.ts reads `answer_confidence`).
#
# Mutation witness: revert `model_answer` to `not generation_failed` in
# `_synthesize_v2_answer` and both refusal tests below go RED.
# ---------------------------------------------------------------------------

REFUSAL_ANSWER = (
    "Chưa có căn cứ chính thức cho học phí năm 2024; tài liệu chỉ có dữ liệu "
    "năm 2026. Hệ thống không tự suy đoán số liệu giữa các năm."
)

def test_policy_refusal_has_no_answer_confidence(main_module):
    """A refusal is intentional, not a failure — and not a model answer."""
    pipeline, retriever = _fake_pipeline([_doc()])
    app = _app(
        main_module,
        guardrail=_fake_guardrail(),
        pipeline=pipeline,
        retriever=retriever,
        generator=_fake_generator(REFUSAL_ANSWER, "evidence-policy:year-mismatch"),
    )
    body = TestClient(app).post("/v2/query", json={"query": "học phí 2024?"}).json()

    # Intentional refusal: NOT degraded, so it keeps the SUCCESS surface …
    assert body["status_code"] == "SUCCESS", body
    assert body["provider"] == "evidence-policy:year-mismatch", body
    assert body["answer"] == REFUSAL_ANSWER, body
    # … but no model wrote it, so it must not claim a model confidence.
    assert body["answer_confidence"] is None, body
    # Retrieval confidence keeps its own meaning, as on every other path.
    assert body["confidence"] == pytest.approx(0.83), body

def test_policy_conflict_refusal_has_no_answer_confidence(main_module):
    """The conflict branch is model-free too, and must behave identically."""
    pipeline, retriever = _fake_pipeline([_doc()])
    app = _app(
        main_module,
        guardrail=_fake_guardrail(),
        pipeline=pipeline,
        retriever=retriever,
        generator=_fake_generator(
            "Hai nguồn tuyển sinh chính thức đang mâu thuẫn về: học phí …",
            "evidence-policy:unresolved-conflict",
        ),
    )
    body = TestClient(app).post("/v2/query", json={"query": "học phí CNTT?"}).json()

    assert body["provider"] == "evidence-policy:unresolved-conflict", body
    assert body["answer_confidence"] is None, body

def test_model_answer_still_carries_answer_confidence(main_module):
    """Complement: the fix must not null out a real model answer's confidence.

    Without this the fix could "pass" by making `model_answer` always False.
    """
    pipeline, retriever = _fake_pipeline([_doc()])
    app = _app(
        main_module,
        guardrail=_fake_guardrail(),
        pipeline=pipeline,
        retriever=retriever,
        generator=_fake_generator(
            "Học phí ngành CNTT năm 2026 là 18.000.000 đồng theo TB60 2026.",
            "gemini-3.8-flash (ramclouds)",
        ),
    )
    body = TestClient(app).post("/v2/query", json={"query": "học phí CNTT?"}).json()

    assert body["status_code"] == "SUCCESS", body
    assert body["answer_confidence"] == pytest.approx(body["groundedness_score"]), body
    assert body["answer_confidence"] is not None, body
