"""Tests for S15.2 — Multi-seed comparison bump (ADR-B + CF-4).

Contract pinned:
  - Trigger = `≥2 distinct entities of type NGANH or TO_HOP` among the seeds
    (node_id prefix `NGANH:` or `TO_HOP:`).
  - On trigger: expander M 5→**8** and the final top_k +**3**.
  - Single-entity (or <2 NGANH/TO_HOP): M=5, top_k unchanged.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.graphrag_retriever import GraphRAGRetriever  # noqa: E402
from src.services.lancedb_retrieval import RetrievedDocument  # noqa: E402
from src.services.query_router import QueryRoute, RouterResult  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Shared fakes (kept local to avoid cross-file coupling)
# ─────────────────────────────────────────────────────────────────────────────


class _FakeNER:
    def __init__(self, seeds: List[str]) -> None:
        self._seeds = list(seeds)

    def extract_from_query(self, query: str) -> List[str]:
        return list(self._seeds)

    async def extract(self, chunk):
        class _R:
            is_success = False
            entities: list = []
        return _R()


class _FakeGraph:
    def __init__(self, ppr_map: Dict[str, float]) -> None:
        self._ppr_map = dict(ppr_map)

    def ppr_scores_by_chunk(self, seeds, alpha: float = 0.85):
        return dict(self._ppr_map)

    def stats(self):
        return {"nodes": 0, "edges": 0}


class _RecordingFetcher:
    def __init__(self, doc_map: Dict[str, RetrievedDocument]) -> None:
        self._doc_map = dict(doc_map)
        self.calls: List[str] = []

    def __call__(self, chunk_id: str) -> Optional[RetrievedDocument]:
        self.calls.append(chunk_id)
        return self._doc_map.get(chunk_id)


def _router(q: str = "so sánh ngành A và B") -> RouterResult:
    return RouterResult(
        original_query=q,
        step_back_query=q,
        hypothetical_doc="hyde",
        hyde_variants=[],
        route=QueryRoute.GRAPH_RAG,
        complexity=4,
        intent="so_sanh",
        reasoning="test",
    )


def _doc(cid: str, score: float = 1.0) -> RetrievedDocument:
    return RetrievedDocument(
        text=f"text-{cid}", source=cid, chunk_id=cid, metadata={}, score=score, point_id=cid,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_two_nganh_seeds_trigger_M_bump_to_8_and_topk_plus_3():
    """≥2 NGANH seeds → M=8 expander budget + top_k bumped by +3."""
    ner = _FakeNER(seeds=["NGANH:cntt", "NGANH:ktpm"])
    # 12 PPR-only candidates, NONE in baseline; descending scores so order is
    # deterministic.
    ppr_map = {f"c_inj_{i:02d}": (20 - i) * 0.05 for i in range(12)}
    graph = _FakeGraph(ppr_map=ppr_map)
    doc_map = {cid: _doc(cid) for cid in ppr_map}
    fetcher = _RecordingFetcher(doc_map=doc_map)

    retriever = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_fetcher=fetcher
    )
    # Baseline = empty so all 8 injected docs survive into output.
    reranked, _ = asyncio.run(
        retriever.retrieve("q", _router(), baseline_docs=[], top_k=5)
    )

    # M bumped to 8 → exactly 8 fetcher calls.
    assert len(fetcher.calls) == 8, (
        f"comparison path must bump M 5→8, got {len(fetcher.calls)} fetches"
    )
    # top_k bumped +3 → up to 8 final docs (5 + 3).
    assert len(reranked) == 8, (
        f"comparison path must bump final top_k 5→8, got {len(reranked)}"
    )


def test_two_tohop_seeds_also_trigger_comparison():
    """≥2 TO_HOP seeds also count as comparison (CF-4 wording)."""
    ner = _FakeNER(seeds=["TO_HOP:a00", "TO_HOP:b00"])
    ppr_map = {f"c_inj_{i:02d}": (20 - i) * 0.05 for i in range(12)}
    graph = _FakeGraph(ppr_map=ppr_map)
    doc_map = {cid: _doc(cid) for cid in ppr_map}
    fetcher = _RecordingFetcher(doc_map=doc_map)

    retriever = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_fetcher=fetcher
    )
    reranked, _ = asyncio.run(
        retriever.retrieve("q", _router(), baseline_docs=[], top_k=5)
    )

    assert len(fetcher.calls) == 8
    assert len(reranked) == 8


def test_single_seed_uses_default_M5_and_topk_unchanged():
    """1 seed → no comparison bump → M=5 and top_k unchanged."""
    ner = _FakeNER(seeds=["NGANH:cntt"])
    ppr_map = {f"c_inj_{i:02d}": (20 - i) * 0.05 for i in range(12)}
    graph = _FakeGraph(ppr_map=ppr_map)
    doc_map = {cid: _doc(cid) for cid in ppr_map}
    fetcher = _RecordingFetcher(doc_map=doc_map)

    retriever = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_fetcher=fetcher
    )
    reranked, _ = asyncio.run(
        retriever.retrieve("q", _router(), baseline_docs=[], top_k=5)
    )

    assert len(fetcher.calls) == 5, (
        f"single-seed path must keep M=5, got {len(fetcher.calls)} fetches"
    )
    assert len(reranked) == 5, (
        f"single-seed path must keep top_k=5, got {len(reranked)}"
    )


def test_mixed_one_nganh_one_diem_does_not_trigger():
    """A NGANH + a non-NGANH/TO_HOP entity should NOT trigger (count NGANH/TO_HOP only)."""
    ner = _FakeNER(seeds=["NGANH:cntt", "DIEM_CHUAN:24.5"])
    ppr_map = {f"c_inj_{i:02d}": (20 - i) * 0.05 for i in range(12)}
    graph = _FakeGraph(ppr_map=ppr_map)
    doc_map = {cid: _doc(cid) for cid in ppr_map}
    fetcher = _RecordingFetcher(doc_map=doc_map)

    retriever = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_fetcher=fetcher
    )
    reranked, _ = asyncio.run(
        retriever.retrieve("q", _router(), baseline_docs=[], top_k=5)
    )

    assert len(fetcher.calls) == 5, (
        f"single NGANH+other should NOT bump; got {len(fetcher.calls)}"
    )
    assert len(reranked) == 5
