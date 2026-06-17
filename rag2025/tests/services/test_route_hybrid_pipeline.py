"""Tests for S16.1 — 3-way routing pipeline (AMF-1 / Wave A).

Contract:
  - `QueryRoute.HYBRID` added.
  - `UnifiedRAGPipeline.query()` branches 3 ways based on router_result.route:
      * GRAPH_RAG + docs → `retrieve(..., expander_m=8)`
      * HYBRID     + docs → `retrieve(..., expander_m=5)`
      * PADDED_RAG       → `docs[:top_k]` (NO retrieve() call)
  - `GraphRAGRetriever.retrieve()` accepts an `expander_m: Optional[int]=None`
    param that OVERRIDES the comparison-heuristic M when provided.
  - For graph route + comparison seeds, the effective M is
    `max(expander_m, _EXPANDER_M_COMPARISON)` so a comparison still gets 8.
  - Backward-compat: `retrieve()` with no `expander_m` preserves today's
    comparison-heuristic behavior (M=8 on comparison, M=5 otherwise).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.graphrag_retriever import (  # noqa: E402
    _EXPANDER_M_COMPARISON,
    _EXPANDER_M_DEFAULT,
    GraphRAGRetriever,
)
from src.services.lancedb_retrieval import RetrievedDocument  # noqa: E402
from src.services.query_router import (  # noqa: E402
    QueryRoute,
    RouterResult,
)
from src.services.router_cache import get_router_cache  # noqa: E402
import src.services.router_cache as router_cache_mod  # noqa: E402
import src.services.graphrag_retriever as graphrag_mod  # noqa: E402
import src.services.query_router as query_router_mod  # noqa: E402
import src.services.graphrag_retriever as _gret_mod  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────


class _FakeNER:
    def __init__(self, seeds: Optional[List[str]] = None) -> None:
        self._seeds = list(seeds or [])

    def extract_from_query(self, query: str) -> List[str]:
        return list(self._seeds)

    async def extract(self, chunk):
        class _R:
            is_success = False
            entities: list = []

        return _R()


class _FakeGraph:
    def __init__(self, ppr_map: Optional[Dict[str, float]] = None) -> None:
        self._ppr_map = dict(ppr_map or {})

    def ppr_scores_by_chunk(self, seeds, alpha: float = 0.85):
        return dict(self._ppr_map)

    def stats(self):
        return {"nodes": 0, "edges": 0}


def _doc(cid: str, score: float = 1.0) -> RetrievedDocument:
    return RetrievedDocument(
        text=f"text-{cid}", source=cid, chunk_id=cid, metadata={}, score=score, point_id=cid,
    )


def _router_result(route: QueryRoute, query: str = "ngành CNTT") -> RouterResult:
    return RouterResult(
        original_query=query,
        step_back_query=query,
        hypothetical_doc="hyde",
        hyde_variants=[],
        route=route,
        complexity=2,
        intent="nganh_hoc",
        reasoning="test",
    )


@pytest.fixture(autouse=True)
def _reset_router_cache(monkeypatch):
    """Reset router cache between tests."""
    monkeypatch.setattr(router_cache_mod, "_router_cache", None)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# QueryRoute.HYBRID
# ─────────────────────────────────────────────────────────────────────────────


def test_query_route_has_hybrid_member():
    """The QueryRoute enum must have a HYBRID member equal to "hybrid"."""
    assert hasattr(QueryRoute, "HYBRID"), "QueryRoute.HYBRID must be defined"
    assert QueryRoute.HYBRID.value == "hybrid"
    # The original members are still present.
    assert QueryRoute.PADDED_RAG.value == "padded_rag"
    assert QueryRoute.GRAPH_RAG.value == "graph_rag"


# ─────────────────────────────────────────────────────────────────────────────
# GraphRAGRetriever.retrieve() — expander_m param
# ─────────────────────────────────────────────────────────────────────────────


def test_retrieve_expander_m_8_used_for_8_ppr_only_chunks():
    """expander_m=8 → at most 8 fetcher calls (overrides default 5)."""
    ner = _FakeNER(seeds=["NGANH:cntt"])  # 1 seed → not comparison
    ppr_map = {f"c_{i:02d}": (10 - i) * 0.1 for i in range(8)}
    graph = _FakeGraph(ppr_map=ppr_map)
    doc_map = {cid: _doc(cid) for cid in ppr_map}

    fetcher_calls: List[str] = []

    def fetcher(cid: str) -> Optional[RetrievedDocument]:
        fetcher_calls.append(cid)
        return doc_map.get(cid)

    retriever = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_fetcher=fetcher,
    )
    asyncio.run(
        retriever.retrieve("q", _router_result(QueryRoute.GRAPH_RAG), [], top_k=5, expander_m=8)
    )
    # Without expander_m=8, M would be 5 (default). With it, M=8.
    assert len(fetcher_calls) == 8, (
        f"expander_m=8 must cap fetches at 8, got {len(fetcher_calls)}"
    )


def test_retrieve_expander_m_5_default_for_5_ppr_only_chunks():
    """expander_m=5 (the hybrid-route value) → at most 5 fetcher calls."""
    ner = _FakeNER(seeds=["NGANH:cntt"])
    ppr_map = {f"c_{i:02d}": (10 - i) * 0.1 for i in range(8)}
    graph = _FakeGraph(ppr_map=ppr_map)
    doc_map = {cid: _doc(cid) for cid in ppr_map}

    fetcher_calls: List[str] = []

    def fetcher(cid: str) -> Optional[RetrievedDocument]:
        fetcher_calls.append(cid)
        return doc_map.get(cid)

    retriever = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_fetcher=fetcher,
    )
    asyncio.run(
        retriever.retrieve("q", _router_result(QueryRoute.HYBRID), [], top_k=5, expander_m=5)
    )
    assert len(fetcher_calls) == 5


def test_retrieve_no_expander_m_preserves_legacy_comparison_heuristic():
    """No expander_m + 2 NGANH seeds (comparison detected) → M=8 (legacy)."""
    ner = _FakeNER(seeds=["NGANH:cntt", "NGANH:ktpm"])
    ppr_map = {f"c_{i:02d}": (20 - i) * 0.05 for i in range(10)}
    graph = _FakeGraph(ppr_map=ppr_map)
    doc_map = {cid: _doc(cid) for cid in ppr_map}

    fetcher_calls: List[str] = []

    def fetcher(cid: str) -> Optional[RetrievedDocument]:
        fetcher_calls.append(cid)
        return doc_map.get(cid)

    retriever = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_fetcher=fetcher,
    )
    asyncio.run(
        retriever.retrieve("q", _router_result(QueryRoute.GRAPH_RAG), [], top_k=5)
    )
    assert len(fetcher_calls) == _EXPANDER_M_COMPARISON == 8


def test_retrieve_no_expander_m_single_seed_uses_default_5():
    """No expander_m + 1 seed (no comparison) → M=5 (default)."""
    ner = _FakeNER(seeds=["NGANH:cntt"])
    ppr_map = {f"c_{i:02d}": (20 - i) * 0.05 for i in range(10)}
    graph = _FakeGraph(ppr_map=ppr_map)
    doc_map = {cid: _doc(cid) for cid in ppr_map}

    fetcher_calls: List[str] = []

    def fetcher(cid: str) -> Optional[RetrievedDocument]:
        fetcher_calls.append(cid)
        return doc_map.get(cid)

    retriever = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_fetcher=fetcher,
    )
    asyncio.run(
        retriever.retrieve("q", _router_result(QueryRoute.GRAPH_RAG), [], top_k=5)
    )
    assert len(fetcher_calls) == _EXPANDER_M_DEFAULT == 5


def test_retrieve_graph_route_comparison_uses_max_of_expander_m_and_8():
    """Graph route + comparison seeds + expander_m=5 → effective M = max(5, 8) = 8."""
    ner = _FakeNER(seeds=["NGANH:cntt", "NGANH:ktpm"])  # comparison seeds
    ppr_map = {f"c_{i:02d}": (20 - i) * 0.05 for i in range(10)}
    graph = _FakeGraph(ppr_map=ppr_map)
    doc_map = {cid: _doc(cid) for cid in ppr_map}

    fetcher_calls: List[str] = []

    def fetcher(cid: str) -> Optional[RetrievedDocument]:
        fetcher_calls.append(cid)
        return doc_map.get(cid)

    retriever = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_fetcher=fetcher,
    )
    asyncio.run(
        retriever.retrieve("q", _router_result(QueryRoute.GRAPH_RAG), [], top_k=5, expander_m=5)
    )
    # Comparison heuristic M=8 wins over caller-passed M=5.
    assert len(fetcher_calls) == 8, (
        f"graph+comparison must use max(expander_m, 8)=8, got {len(fetcher_calls)}"
    )


def test_retrieve_hybrid_route_expander_m_5_ignores_comparison_heuristic():
    """Hybrid route: caller-passed expander_m wins over comparison heuristic."""
    ner = _FakeNER(seeds=["NGANH:cntt", "NGANH:ktpm"])  # 2 NGANH seeds
    ppr_map = {f"c_{i:02d}": (20 - i) * 0.05 for i in range(10)}
    graph = _FakeGraph(ppr_map=ppr_map)
    doc_map = {cid: _doc(cid) for cid in ppr_map}

    fetcher_calls: List[str] = []

    def fetcher(cid: str) -> Optional[RetrievedDocument]:
        fetcher_calls.append(cid)
        return doc_map.get(cid)

    retriever = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_fetcher=fetcher,
    )
    # Hybrid caller passes expander_m=5; the comparison heuristic (8) must NOT
    # override because we're not on the graph route.
    asyncio.run(
        retriever.retrieve("q", _router_result(QueryRoute.HYBRID), [], top_k=5, expander_m=5)
    )
    assert len(fetcher_calls) == 5, (
        f"hybrid+expander_m=5 must cap at 5 even with comparison seeds; "
        f"got {len(fetcher_calls)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# UnifiedRAGPipeline.query() — 3-branch dispatch
# ─────────────────────────────────────────────────────────────────────────────


def _build_pipeline(route: QueryRoute) -> tuple:
    """Build a UnifiedRAGPipeline with all dependencies mocked.

    Returns (pipeline, retrieve_mock) so the test can inspect call args.
    """
    # Build a fake graphrag with a controllable retrieve coroutine.
    fake_graphrag = MagicMock()
    fake_graphrag.retrieve = AsyncMock(
        return_value=([_doc("c_graph", 1.0)], {"c_graph": 0.5})
    )

    fake_router = MagicMock()
    fake_router.route = AsyncMock(return_value=_router_result(route, query="ngành CNTT"))

    fake_llm = MagicMock()

    # Import locally to avoid module-level side effects
    from src.services.graphrag_retriever import UnifiedRAGPipeline

    pipeline = UnifiedRAGPipeline(
        router=fake_router, graphrag=fake_graphrag, llm=fake_llm,
    )
    return pipeline, fake_graphrag


def test_pipeline_graph_route_calls_retrieve_with_expander_m_8():
    """GRAPH_RAG + docs → retrieve called with expander_m=8."""
    pipeline, fake_graphrag = _build_pipeline(QueryRoute.GRAPH_RAG)
    baseline = [_doc("a", 1.0), _doc("b", 0.8)]
    result = asyncio.run(pipeline.query("ngành CNTT", baseline_docs=baseline, top_k=5))

    fake_graphrag.retrieve.assert_awaited_once()
    kwargs = fake_graphrag.retrieve.await_args.kwargs
    # expander_m may be passed positionally or as keyword — check both
    args = fake_graphrag.retrieve.await_args.args
    if "expander_m" in kwargs:
        assert kwargs["expander_m"] == 8
    else:
        # The 5th positional arg (after query, router_result, baseline_docs, top_k)
        assert len(args) >= 5, f"expected positional expander_m, got args={args}"
        assert args[4] == 8, f"expander_m positional = {args[4]}"


def test_pipeline_hybrid_route_calls_retrieve_with_expander_m_5():
    """HYBRID + docs → retrieve called with expander_m=5."""
    pipeline, fake_graphrag = _build_pipeline(QueryRoute.HYBRID)
    baseline = [_doc("a", 1.0), _doc("b", 0.8)]
    result = asyncio.run(pipeline.query("ngành CNTT", baseline_docs=baseline, top_k=5))

    fake_graphrag.retrieve.assert_awaited_once()
    kwargs = fake_graphrag.retrieve.await_args.kwargs
    args = fake_graphrag.retrieve.await_args.args
    if "expander_m" in kwargs:
        assert kwargs["expander_m"] == 5
    else:
        assert len(args) >= 5
        assert args[4] == 5


def test_pipeline_padded_route_does_NOT_call_retrieve():
    """PADDED_RAG → retrieve() is NEVER awaited (pure vector, docs[:top_k])."""
    pipeline, fake_graphrag = _build_pipeline(QueryRoute.PADDED_RAG)
    baseline = [_doc("a", 1.0), _doc("b", 0.8), _doc("c", 0.6), _doc("d", 0.4), _doc("e", 0.2)]
    result = asyncio.run(pipeline.query("học phí ngành CNTT", baseline_docs=baseline, top_k=3))

    fake_graphrag.retrieve.assert_not_awaited()
    # Documents are clamped to top_k from the baseline only.
    assert [d.chunk_id for d in result.documents] == ["a", "b", "c"]
    assert result.route == "padded_rag"


def test_pipeline_padded_route_uses_baseline_only_no_fusion():
    """PADDED_RAG → ppr_scores stays empty; no GraphRAG fusion ran."""
    pipeline, fake_graphrag = _build_pipeline(QueryRoute.PADDED_RAG)
    baseline = [_doc("a", 1.0), _doc("b", 0.5)]
    result = asyncio.run(pipeline.query("học phí ngành CNTT", baseline_docs=baseline, top_k=2))

    assert result.ppr_scores == {}
    assert result.documents[0].chunk_id == "a"


def test_pipeline_backward_compat_no_expander_m():
    """If pipeline.query is called WITHOUT top_k change, the expander_m dispatch
    is still wired (3 branches). Smoke check: padded → no retrieve call."""
    pipeline, fake_graphrag = _build_pipeline(QueryRoute.PADDED_RAG)
    asyncio.run(pipeline.query("query", baseline_docs=[_doc("x", 1.0)]))
    fake_graphrag.retrieve.assert_not_awaited()


def test_pipeline_graph_route_propagates_retrieve_results():
    """GRAPH_RAG → final docs come from the retrieve() return value."""
    pipeline, fake_graphrag = _build_pipeline(QueryRoute.GRAPH_RAG)
    fake_graphrag.retrieve = AsyncMock(
        return_value=([_doc("from_graph", 0.7)], {"from_graph": 0.5})
    )
    baseline = [_doc("a", 1.0)]
    result = asyncio.run(pipeline.query("ngành CNTT", baseline_docs=baseline, top_k=3))

    assert [d.chunk_id for d in result.documents] == ["from_graph"]
    assert result.ppr_scores == {"from_graph": 0.5}


def test_pipeline_hybrid_route_propagates_retrieve_results():
    """HYBRID → final docs come from the retrieve() return value (M=5 path)."""
    pipeline, fake_graphrag = _build_pipeline(QueryRoute.HYBRID)
    fake_graphrag.retrieve = AsyncMock(
        return_value=([_doc("from_hybrid", 0.6)], {"from_hybrid": 0.4})
    )
    baseline = [_doc("a", 1.0)]
    result = asyncio.run(pipeline.query("ngành CNTT", baseline_docs=baseline, top_k=3))

    assert [d.chunk_id for d in result.documents] == ["from_hybrid"]
