"""T15 AC3 — HTTP seam: a tombstoned legacy row is absent from
`sources` on ALL THREE retrieval routes.

Drives the real HTTP surface (`POST /v2/query` with
`force_route=padded_rag` and `force_route=graph_rag`, plus
`POST /v3/query` hybrid) through the real retrieval path —
the REAL `LanceDBRetriever` (dense belt + fetch_by_id(s)
belt) and the REAL `HybridSearchService` (BM25 corpus build
belt + per-request sparse filter) — stubbing ONLY the LLM /
embedding / reranker / guardrail dependencies, per the seam
principle. The service-path mapping in
`tests/services/test_tombstone_three_routes.py` is a unit
companion; this file proves the routes the FE actually calls.

The `/v2/recommend` and `/v2/whatif` endpoints are NOT tested
here: they are pure-arithmetic (`src/routers/recommender.py`)
and never touch LanceDB, so they cannot serve a tombstoned row.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
MAIN_PATH = SRC / "main.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from services.lancedb_retrieval import (  # noqa: E402
    LanceDBRetriever,
    LanceDBRetrieverConfig,
    RetrievalResult,
)

EMBEDDING_DIM = 4

TOMBSTONE_ID = "tombstoned-legacy"
NORMAL_ID = "normal-legacy"


# ---------------------------------------------------------------------------
# Real retriever over a fake LanceDB table (the ONLY fake data source)
# ---------------------------------------------------------------------------


def _row(cid: str, is_tombstone: Optional[bool]) -> Dict[str, Any]:
    meta = {"data_year": 2026}
    return {
        "text": f"text {cid}",
        "source": "synthetic",
        "chunk_id": cid,
        "id": cid,
        "metadata_json": json.dumps(meta),
        "is_tombstone": is_tombstone,
        "_distance": 0.5,
    }


def _make_real_retriever(rows: List[Dict[str, Any]]) -> LanceDBRetriever:
    """Real `LanceDBRetriever` whose adapter returns `rows` from a
    fake table carrying the `is_tombstone` column — the dense arm
    passes the server-side tombstone gate and the post-fetch belt
    runs for real."""
    retriever = LanceDBRetriever.__new__(LanceDBRetriever)
    retriever._config = LanceDBRetrieverConfig(
        uri="memory://",
        table_name="t",
        embedding_dim=EMBEDDING_DIM,
        default_top_k=5,
    )
    adapter = MagicMock()
    adapter.ensure_table = MagicMock()
    adapter.has_tombstone_column = MagicMock(return_value=True)
    adapter.search = MagicMock(
        side_effect=lambda query_vector, top_k, where_clause=None: list(rows)
    )
    adapter.fetch_rows_by_ids = MagicMock(
        side_effect=lambda ids: [
            r for r in rows if r.get("id") in set(ids or [])
        ]
    )
    adapter.fetch_rows_by_doc_ids = MagicMock(return_value=[])
    retriever._adapter = adapter
    return retriever


# ---------------------------------------------------------------------------
# Main module + real services, LLM/embedding stubbed
# ---------------------------------------------------------------------------


def _build_main_module(monkeypatch):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    repo_root = MAIN_PATH.resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    monkeypatch.setenv("ADMIN_API_TOKEN", "")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    monkeypatch.delenv("APP_ENV", raising=False)
    spec = importlib.util.spec_from_file_location(
        "tombstone_three_routes_http_test", MAIN_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _router_result(*, route_value: str = "padded_rag", in_scope: bool = True):
    return SimpleNamespace(
        skip_retrieval=False,
        auto_answer=None,
        step_back_query=None,
        intent="factoid",
        complexity=1,
        reasoning="test",
        hyde_variants=[],
        route=SimpleNamespace(value=route_value),
        is_in_scope=in_scope,
        sub_queries=None,
    )


def _install_fakes(main_module, rows: List[Dict[str, Any]]):
    """Wire the REAL retriever + REAL hybrid service; stub only
    LLM / embedding / reranker / guardrail."""
    retriever = _make_real_retriever(rows)
    main_module.lancedb_retriever_service = retriever

    embedding = MagicMock()
    embedding.encode_query = MagicMock(
        return_value=SimpleNamespace(tolist=lambda: [0.0] * EMBEDDING_DIM)
    )
    main_module.embedding_service = embedding

    # Real hybrid service over the real retriever (BM25 corpus build
    # reads adapter._table — a fake table shaped for build_bm25_index).
    import pandas as pd

    from services.hybrid_search import HybridSearchService

    hybrid = HybridSearchService(retriever, _fake_settings())
    table = MagicMock()
    table.schema.names = [
        "text", "source", "chunk_id", "metadata_json",
        "id", "release_id", "is_tombstone",
    ]
    df = pd.DataFrame(
        {
            "text": [r["text"] for r in rows],
            "source": [r["source"] for r in rows],
            "chunk_id": [r["chunk_id"] for r in rows],
            "metadata_json": [r["metadata_json"] for r in rows],
            "id": [r["id"] for r in rows],
            "release_id": ["R1" for _ in rows],
            "is_tombstone": [r["is_tombstone"] for r in rows],
        }
    )
    select_result = MagicMock()
    select_result.to_pandas.return_value = df
    table.select.return_value = select_result
    table.to_pandas.return_value = df
    retriever._adapter._table = table
    assert hybrid.build_bm25_index() is True
    main_module.hybrid_search_service = hybrid

    guardrail = MagicMock()
    guardrail.precheck = AsyncMock(
        return_value=SimpleNamespace(
            is_in_scope=True,
            internal_code="SUCCESS",
            reason="ok",
            public_code="SUCCESS",
            short_answer="",
            pii_detected=False,
            data_gap_hints=[],
        )
    )
    guardrail.expose_internal = MagicMock(return_value=False)
    guardrail._pii_warning_answer = MagicMock(return_value="Vui lòng ẩn PII.")
    guardrail._contains_sensitive_pii = MagicMock(return_value=False)
    guardrail.classify_no_result = AsyncMock(
        return_value=SimpleNamespace(internal_code="SUCCESS", reason="ok", data_gap_hints=[])
    )
    guardrail.public_status = MagicMock(side_effect=lambda code: code)
    main_module.guardrail_service = guardrail
    main_module.query_cache = None
    from services.graphrag_retriever import RAGResult, UnifiedRAGPipeline

    pipeline = MagicMock(spec=UnifiedRAGPipeline)


    async def _pipeline_query(user_query, baseline_docs, top_k, **kwargs):
        # Real `RAGResult` dataclass: `force_route` in main.py calls
        # `dataclasses.replace()` on it, which rejects SimpleNamespace.
        docs = list(baseline_docs[:top_k])
        return RAGResult(
            query=user_query,
            route="padded_rag",
            documents=docs,
            router_result=_router_result(),
            ppr_scores={},
            latency_ms=1.0,
            confidence=(docs[0].score if docs else 0.0),
        )

    pipeline.query = AsyncMock(side_effect=_pipeline_query)
    pipeline._graphrag = MagicMock()
    pipeline._graphrag.graph_stats = {"nodes": 0, "edges": 0}

    async def _graphrag_retrieve(*, query, router_result, baseline_docs, top_k, **kw):
        # The graph route's chunk-serving read: the real fetch_by_ids belt.
        fetched = retriever.fetch_by_ids([d.chunk_id for d in baseline_docs])
        docs = list(fetched.values())
        return docs, {d.chunk_id: 0.5 for d in docs}

    pipeline._graphrag.retrieve = AsyncMock(side_effect=_graphrag_retrieve)
    pipeline._router = MagicMock()
    pipeline._router.route = AsyncMock(return_value=_router_result())
    main_module.unified_pipeline = pipeline

    reranker = MagicMock()
    reranker.enabled = False
    main_module.reranker_service = reranker

    captured: Dict[str, Any] = {"generator_chunks": None}

    async def _generate_answer(*, query, chunks, confidence,
                               is_program_list_query, history=None):
        captured["generator_chunks"] = list(chunks)
        return {
            "answer": "test answer",
            "sources": [
                {
                    "id": c.get("chunk_id") or "?",
                    "title": c.get("chunk_id") or "?",
                    "url": None,
                    "snippet": (c.get("text") or "")[:120],
                    "data_year": "2026",
                }
                for c in chunks
            ],
            "confidence": confidence,
            "provider": "fake",
            "chunks_used": len(chunks),
        }

    generator = MagicMock()
    generator.generate_answer = AsyncMock(side_effect=_generate_answer)
    main_module.llm_generator_service = generator

    return {"captured": captured, "retriever": retriever, "hybrid": hybrid}


def _fake_settings():
    s = MagicMock()
    s.HYBRID_FUSION_DENSE_WEIGHT = 0.6
    s.HYBRID_FUSION_SPARSE_WEIGHT = 0.4
    return s


async def _post_v2(main_module, query: str, *, force_route: str, top_k: int = 5):
    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/v2/query",
            json={"query": query, "top_k": top_k, "force_route": force_route},
        )


async def _post_v3(main_module, query: str, *, top_k: int = 5,
                   candidate_pool: int = 20):
    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/v3/query",
            json={
                "query": query,
                "top_k": top_k,
                "candidate_pool": candidate_pool,
            },
        )


@pytest.fixture
def main_module(monkeypatch):
    return _build_main_module(monkeypatch)


ROWS = [_row(NORMAL_ID, False), _row(TOMBSTONE_ID, True)]


class TestThreeRoutesTombstoneAbsent:
    """AC3: the tombstoned legacy row must be absent from `sources`
    on padded_rag, graph_rag, AND hybrid (v3)."""

    @pytest.mark.asyncio
    async def test_padded_rag_sources_exclude_tombstone(self, main_module):
        fakes = _install_fakes(main_module, ROWS)
        resp = await _post_v2(
            main_module, "học phí ngành CNTT", force_route="padded_rag"
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        served = {c["id"] for c in body["sources"]}
        assert NORMAL_ID in served
        assert TOMBSTONE_ID not in served, (
            f"padded_rag served a tombstoned row: {served}"
        )
        gen_ids = {c.get("chunk_id") for c in fakes["captured"]["generator_chunks"]}
        assert TOMBSTONE_ID not in gen_ids

    @pytest.mark.asyncio
    async def test_graph_rag_sources_exclude_tombstone(self, main_module):
        fakes = _install_fakes(main_module, ROWS)
        resp = await _post_v2(
            main_module, "so sánh điểm chuẩn các ngành", force_route="graph_rag"
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        served = {c["id"] for c in body["sources"]}
        assert TOMBSTONE_ID not in served, (
            f"graph_rag served a tombstoned row: {served}"
        )
        gen_ids = {c.get("chunk_id") for c in fakes["captured"]["generator_chunks"]}
        assert TOMBSTONE_ID not in gen_ids

    @pytest.mark.asyncio
    async def test_hybrid_v3_sources_exclude_tombstone(self, main_module):
        fakes = _install_fakes(main_module, ROWS)
        resp = await _post_v3(main_module, "học phí ngành CNTT")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # /v3 `sources` is the set of source labels of the served chunks.
        assert body["sources"] == ["synthetic"], (
            f"hybrid v3 served a tombstoned row: {body['sources']}"
        )
        assert body["chunks_used"] == 1, (
            f"expected exactly the normal chunk to survive, got "
            f"{body['chunks_used']}"
        )
