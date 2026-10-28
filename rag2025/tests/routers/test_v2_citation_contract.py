"""Ticket 10 — the citation contract through the HTTP layer.

@spec(10) "Test seam: HTTP POST /v2/query" — the /v2 response must carry
``citations`` populated from the generator (JSON path) and the SSE ``meta``
frame must carry the same list (stream path). Missing pages must not break
either response (acceptance #3 at the surface).

Harness: identical to test_v2_generation_failed_status.py — load src/main.py
as a stand-alone module, wire fake guardrail / pipeline / retriever /
generator globals, and mount the REAL /v2 handlers. The fake generator runs
the REAL build_citations over the chunks it was handed, so the assertions
measure the whole chain: generator dict -> _synthesize_v2_answer pass-through
-> response_payload -> UnifiedQueryResponse -> JSON / SSE meta.
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

from services.citation import build_citations  # noqa: E402

_ANSWER = "Điểm chuẩn ngành CNTT 22,5; Kinh doanh 24,75."


def _load_main(monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("NS24_API_KEY", "")
    spec = importlib.util.spec_from_file_location("t10_citation_http_test", MAIN_PATH)
    assert spec and spec.loader, "could not build import spec for src/main.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _doc(text: str, *, chunk_id: str, doc_id: str, page: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        score=0.83,
        chunk_id=chunk_id,
        source=doc_id,
        metadata={
            "summary": text,
            "data_year": "2026",
            "record_id": f"rec-{doc_id}",
            "record_version_key": f"rec-{doc_id}@v1",
            "source_document_id": doc_id,
            "source_page": page,
            "source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=99",
        },
    )


# Two majors in two documents; doc-biz deliberately has NO page (criterion 3).
DOCS = [
    _doc("CNTT | A00 | 22,5", chunk_id="c-it", doc_id="doc-it", page=4),
    _doc("Kinh doanh | D01 | 24,75", chunk_id="c-biz", doc_id="doc-biz", page=None),
]


def _fake_guardrail():
    decision = SimpleNamespace(
        is_in_scope=True,
        internal_code="SUCCESS",
        reason="ok",
        short_answer="",
        data_gap_hints=[],
        pii_detected=False,
    )
    svc = MagicMock()
    svc.precheck = AsyncMock(return_value=decision)
    svc.classify_no_result = AsyncMock(return_value=decision)
    svc.public_status = MagicMock(return_value="SUCCESS")
    svc.expose_internal = MagicMock(return_value=True)
    svc._contains_sensitive_pii = MagicMock(return_value=False)
    return svc


def _fake_generator() -> MagicMock:
    """Runs the REAL citation builder over the chunks /v2 handed it."""

    async def _gen(*, query: str, chunks: list, confidence: float, **kwargs):
        cites = build_citations(_ANSWER, chunks)
        return {
            "answer": _ANSWER,
            "sources": [{"id": c["chunk_id"], "title": c["text"][:80], "url": None,
                         "snippet": c["text"], "data_year": "2026"} for c in chunks],
            "citations": cites,
            "confidence": confidence,
            "provider": "test-model",
            "chunks_used": len(chunks),
            "generation_failed": False,
        }

    gen = MagicMock()
    gen.generate_answer = AsyncMock(side_effect=_gen)
    return gen


def _fake_pipeline():
    router_result = SimpleNamespace(
        skip_retrieval=False, auto_answer=None, step_back_query="step",
        intent="general", complexity=1, reasoning="ok", hyde_variants=[],
        is_in_scope=True,
    )
    rag_result = SimpleNamespace(
        query="điểm chuẩn", route="padded_rag", documents=DOCS,
        router_result=router_result, ppr_scores={}, latency_ms=12.5, confidence=0.83,
    )
    pipeline = MagicMock()
    pipeline.query = AsyncMock(return_value=rag_result)
    pipeline._graphrag = SimpleNamespace(graph_stats={"nodes": 1, "edges": 0})
    pipeline._router = MagicMock()
    pipeline._router.route = AsyncMock(return_value=router_result)
    retriever = MagicMock()
    retriever.retrieve = MagicMock(return_value=SimpleNamespace(is_success=True, documents=DOCS))
    return pipeline, retriever


def _app(main_module):
    main_module.guardrail_service = _fake_guardrail()
    pipeline, retriever = _fake_pipeline()
    main_module.unified_pipeline = pipeline
    main_module.lancedb_retriever_service = retriever
    main_module.embedding_service = MagicMock()
    main_module.llm_generator_service = _fake_generator()
    app = FastAPI()
    app.post("/v2/query", response_model=main_module.UnifiedQueryResponse)(main_module.unified_query)
    app.post("/v2/query/stream")(main_module.unified_query_stream)
    return app


@pytest.fixture
def client(monkeypatch):
    main_module = _load_main(monkeypatch)
    return TestClient(_app(main_module))


def _parse_sse(raw: str) -> list[dict]:
    frames = []
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


def test_v2_json_response_carries_per_claim_citations(client):
    r = client.post("/v2/query", json={"query": "điểm chuẩn hai ngành?"})
    assert r.status_code == 200, r.text
    body = r.json()
    cites = body["citations"]
    assert isinstance(cites, list) and len(cites) == 2, (
        f"one citation per comparison cell, got {cites}"
    )
    by_value = {c["value"]: c for c in cites}
    assert by_value["22.5"]["source_document_id"] == "doc-it"
    assert by_value["22.5"]["source_page"] == 4
    assert by_value["24.75"]["source_document_id"] == "doc-biz"
    # Criterion 4 at the surface: the table did not collapse to one source.
    assert len({c["source_document_id"] for c in cites}) == 2
    # Criterion 3 at the surface: the page-less claim is still a valid 200.
    assert by_value["24.75"]["source_page"] is None
    assert by_value["24.75"]["grounded"] is True


def test_v2_sse_meta_frame_carries_same_citations(client):
    with client.stream("POST", "/v2/query/stream", json={"query": "điểm chuẩn hai ngành?"}) as r:
        assert r.status_code == 200, r.read()
        raw = "".join(r.iter_text())
    frames = _parse_sse(raw)
    metas = [f for f in frames if f["event"] == "meta"]
    assert metas, "stream must emit a meta frame"
    meta = json.loads(metas[-1]["data"])
    assert "citations" in meta, "meta frame must carry citations"
    assert [c["value"] for c in meta["citations"]] == ["22.5", "24.75"]
    assert {c["source_document_id"] for c in meta["citations"]} == {"doc-it", "doc-biz"}


def test_citations_field_defaults_to_empty_list(client):
    """A response whose synthesis produced no citations (e.g. canned answer)
    is still schema-valid — the field is optional-by-default, not required."""
    r = client.post("/v2/query", json={"query": "điểm chuẩn hai ngành?"})
    body = r.json()
    assert "citations" in body
    # The model itself validates an omitted key:
    import importlib.util

    spec = importlib.util.spec_from_file_location("t10_model_check", MAIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    payload = {
        "query": "q", "route": "r", "answer": "a", "sources": [], "confidence": 0.5,
        "router_info": {}, "latency_ms": 1.0, "trace_id": "t",
    }
    model = module.UnifiedQueryResponse(**payload)
    assert model.citations == []
