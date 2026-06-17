"""Tests for S15.1 — GraphRAG expander-fusion (ADR-A + MUST-FIX#1/#2 + CF-1).

Contract pinned by these tests:
  1. `chunk_fetcher=None` → identical to pre-S15 rerank-only behavior.
  2. Empty seeds → fetcher NEVER called.
  3. When seeds extracted + PPR returns chunk_ids NOT in baseline, the expander
     fetches them via `chunk_fetcher`, appends with `score=0.0` and
     `metadata["graph_injected"]=True`, BEFORE fusion + sort.
  4. M cap respected (default 5).
  5. Injected chunks do NOT auto-rank #1 from a sentinel score — MUST-FIX#1
     neutralization gives them `β·ppr` only, so a strong baseline (high rrf)
     still outranks the injected one.
  6. The injected chunk DOES reach the reranked output (proves expander-fusion).
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
# Fakes
# ─────────────────────────────────────────────────────────────────────────────


class _FakeNER:
    """NER stub: `extract_from_query` returns a fixed seed list; `extract`
    coroutine is never invoked because regex returns non-empty.
    """

    def __init__(self, seeds: List[str]) -> None:
        self._seeds = list(seeds)
        self.extract_calls = 0

    def extract_from_query(self, query: str) -> List[str]:
        return list(self._seeds)

    async def extract(self, chunk):
        # Should NOT be hit when seeds is non-empty.
        self.extract_calls += 1

        class _R:
            is_success = False
            entities: list = []

        return _R()


class _FakeGraph:
    """Graph stub returning a controllable ppr_scores_by_chunk mapping."""

    def __init__(self, ppr_map: Dict[str, float]) -> None:
        self._ppr_map = dict(ppr_map)
        self.calls: List[List[str]] = []

    def ppr_scores_by_chunk(self, seeds, alpha: float = 0.85):
        self.calls.append(list(seeds))
        return dict(self._ppr_map)

    def stats(self):
        return {"nodes": 0, "edges": 0}


class _RecordingFetcher:
    """Records call args, returns a doc per chunk_id from a fixture map.
    Unknown ids → None (mirrors fetch_by_id miss).
    """

    def __init__(self, doc_map: Dict[str, RetrievedDocument]) -> None:
        self._doc_map = dict(doc_map)
        self.calls: List[str] = []

    def __call__(self, chunk_id: str) -> Optional[RetrievedDocument]:
        self.calls.append(chunk_id)
        return self._doc_map.get(chunk_id)


def _router(query: str = "ngành CNTT") -> RouterResult:
    return RouterResult(
        original_query=query,
        step_back_query=query,
        hypothetical_doc="hyde",
        hyde_variants=[],
        route=QueryRoute.GRAPH_RAG,
        complexity=3,
        intent="nganh_hoc",
        reasoning="test",
    )


def _doc(cid: str, score: float, text: str = "", metadata: Optional[dict] = None) -> RetrievedDocument:
    return RetrievedDocument(
        text=text or f"text-{cid}",
        source=f"src-{cid}",
        chunk_id=cid,
        metadata=metadata or {},
        score=score,
        point_id=cid,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_chunk_fetcher_none_is_backward_compatible():
    """`chunk_fetcher=None` → no expansion; behavior identical to pre-S15."""
    ner = _FakeNER(seeds=["NGANH:cntt"])
    graph = _FakeGraph(ppr_map={"c1": 0.9, "c_extra": 0.7})
    retriever = GraphRAGRetriever(
        ner_service=ner, graph=graph, alpha=0.6, beta=0.4, chunk_fetcher=None
    )

    docs = [_doc("c1", 2.0), _doc("c2", 1.0)]
    reranked, ppr_scores = asyncio.run(
        retriever.retrieve("q", _router(), baseline_docs=docs, top_k=5)
    )

    # No new chunk_ids — only baseline chunk_ids survive.
    out_ids = [d.chunk_id for d in reranked]
    assert set(out_ids).issubset({"c1", "c2"}), out_ids
    # ppr_scores still includes c_extra (PPR returns ALL graph chunks), but it
    # is NOT promoted into reranked because no fetcher was supplied.
    assert "c_extra" in ppr_scores
    assert all(d.chunk_id != "c_extra" for d in reranked)
    # No graph_injected marker on any output doc.
    assert all(not (d.metadata or {}).get("graph_injected") for d in reranked)


def test_no_fetch_when_seeds_empty():
    """If `_seed_entities_from_query` returns []→fetcher NEVER called."""
    ner = _FakeNER(seeds=[])  # regex empty
    graph = _FakeGraph(ppr_map={})  # not even consulted
    fetcher = _RecordingFetcher(doc_map={})

    retriever = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_fetcher=fetcher
    )
    docs = [_doc("c1", 1.0), _doc("c2", 0.5)]
    reranked, ppr_scores = asyncio.run(
        retriever.retrieve("q", _router(), baseline_docs=docs, top_k=5)
    )

    assert fetcher.calls == [], f"fetcher must not be called: {fetcher.calls}"
    assert ppr_scores == {}
    assert [d.chunk_id for d in reranked] == ["c1", "c2"]


def test_injected_chunk_reaches_output_with_graph_injected_marker():
    """PPR-only chunk (NOT in baseline) is fetched, marked, and surfaces."""
    ner = _FakeNER(seeds=["NGANH:cntt"])
    # PPR returns c1 (in baseline) AND c_inj (NOT in baseline, high PPR).
    graph = _FakeGraph(ppr_map={"c1": 0.4, "c_inj": 0.9})
    inj_doc = _doc("c_inj", 1.0, text="injected text", metadata={"data_year": 2026})
    fetcher = _RecordingFetcher(doc_map={"c_inj": inj_doc})

    retriever = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_fetcher=fetcher
    )
    docs = [_doc("c1", 1.0)]  # baseline with one weak doc
    reranked, ppr_scores = asyncio.run(
        retriever.retrieve("q", _router(), baseline_docs=docs, top_k=5)
    )

    out_ids = [d.chunk_id for d in reranked]
    assert "c_inj" in out_ids, f"injected chunk must surface: {out_ids}"
    assert fetcher.calls == ["c_inj"], fetcher.calls

    # The injected doc carries the graph_injected marker.
    inj_out = next(d for d in reranked if d.chunk_id == "c_inj")
    assert (inj_out.metadata or {}).get("graph_injected") is True
    # Original metadata preserved (merged).
    assert (inj_out.metadata or {}).get("data_year") == 2026


def test_injected_chunk_does_not_auto_rank_first_from_sentinel():
    """MUST-FIX#1: fetcher returns score=1.0 sentinel; expander MUST neutralize
    it to 0.0 so fused = β·ppr only. A strong baseline doc (high rrf) with no
    PPR weight must STILL outrank the injected chunk.
    """
    ner = _FakeNER(seeds=["NGANH:cntt"])
    # Strong baseline doc has NO PPR weight; injected has modest PPR.
    graph = _FakeGraph(ppr_map={"c_inj": 0.3})
    inj_doc = _doc("c_inj", 1.0)  # sentinel score from fetch_by_id
    fetcher = _RecordingFetcher(doc_map={"c_inj": inj_doc})

    retriever = GraphRAGRetriever(
        ner_service=ner, graph=graph, alpha=0.6, beta=0.4, chunk_fetcher=fetcher
    )
    # c_strong: high baseline rrf (becomes 1.0 after normalize); zero PPR.
    docs = [_doc("c_strong", 10.0), _doc("c_weak", 1.0)]
    reranked, _ = asyncio.run(
        retriever.retrieve("q", _router(), baseline_docs=docs, top_k=5)
    )

    out_ids = [d.chunk_id for d in reranked]
    assert "c_inj" in out_ids
    # Strong baseline MUST outrank injected:
    strong_idx = out_ids.index("c_strong")
    inj_idx = out_ids.index("c_inj")
    assert strong_idx < inj_idx, (
        f"c_strong (rrf-dominant) must outrank c_inj (sentinel-neutralized); "
        f"got order {out_ids}"
    )

    # Sanity: c_inj's fused score = β · ppr_only ≈ 0.4 * 0.3 / 0.3 = 0.4
    # (after PPR normalization within the helper). c_strong's = α · 1.0 = 0.6.
    inj_doc_out = next(d for d in reranked if d.chunk_id == "c_inj")
    strong_doc_out = next(d for d in reranked if d.chunk_id == "c_strong")
    assert strong_doc_out.score > inj_doc_out.score


def test_m_cap_respected_default_5():
    """Expander default M=5 → at most 5 fetches even when PPR has more."""
    ner = _FakeNER(seeds=["NGANH:cntt"])  # 1 seed → NOT comparison → M=5
    # 8 PPR-only chunk_ids, all with descending scores, NONE in baseline.
    ppr_map = {f"c_inj_{i}": (10 - i) * 0.1 for i in range(8)}
    graph = _FakeGraph(ppr_map=ppr_map)
    doc_map = {cid: _doc(cid, 1.0) for cid in ppr_map}
    fetcher = _RecordingFetcher(doc_map=doc_map)

    retriever = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_fetcher=fetcher
    )
    docs: List[RetrievedDocument] = []  # empty baseline
    asyncio.run(retriever.retrieve("q", _router(), baseline_docs=docs, top_k=5))

    assert len(fetcher.calls) == 5, (
        f"M cap (5) must bound fetcher invocations, got {len(fetcher.calls)}: "
        f"{fetcher.calls}"
    )
    # The top-5 PPR ids should be fetched.
    expected_top5 = sorted(ppr_map, key=ppr_map.__getitem__, reverse=True)[:5]
    assert set(fetcher.calls) == set(expected_top5)


def test_fetcher_miss_does_not_block_other_injections():
    """If fetcher returns None for one id, expander continues with the rest."""
    ner = _FakeNER(seeds=["NGANH:cntt"])
    graph = _FakeGraph(ppr_map={"c_missing": 0.9, "c_ok": 0.8})
    # c_missing maps to None → fetch miss.
    fetcher = _RecordingFetcher(doc_map={"c_ok": _doc("c_ok", 1.0)})

    retriever = GraphRAGRetriever(
        ner_service=ner, graph=graph, chunk_fetcher=fetcher
    )
    reranked, _ = asyncio.run(
        retriever.retrieve("q", _router(), baseline_docs=[], top_k=5)
    )

    out_ids = [d.chunk_id for d in reranked]
    assert "c_ok" in out_ids
    assert "c_missing" not in out_ids
