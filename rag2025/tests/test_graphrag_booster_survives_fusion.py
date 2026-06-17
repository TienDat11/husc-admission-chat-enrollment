"""Regression test: aggregation-booster chunks MUST survive the GraphRAG
fusion top-k slice.

Bug: GraphRAGRetriever.retrieve() truncated fused results to `effective_top_k`
by FUSED score, which could drop `metadata["booster_injected"]=True` docs
(aggregation summary chunks like `liet_ke_28_nganh_2026_v2`,
`bang_xep_hang_chi_tieu_2026_v2`, `tuyensinh_overview_2026_v2`) before they
reached `apply_rerank_cutoff` (where the CF-2 exemption lives).

After the fix, fusion selection ALWAYS retains docs carrying either
`booster_injected` or `graph_injected` metadata, regardless of fused-score
rank. Queries with NO injected docs are unchanged (top_k truncation is
byte-identical to before the fix).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.entities import Chunk, Entity, EntityType, ExtractionResult, RelationType, Triple
from src.domain.graph import KnowledgeGraphBuilder
from src.services.graphrag_retriever import GraphRAGRetriever
from src.services.lancedb_retrieval import RetrievedDocument
from src.services.query_router import QueryRoute, RouterResult


class _FailingNER:
    async def extract(self, chunk: Chunk) -> ExtractionResult:
        return ExtractionResult(chunk_id=chunk.chunk_id, error="ner failed")

    def extract_from_query(self, query: str):
        return []


def _build_graph():
    builder = KnowledgeGraphBuilder()
    entities = [
        Entity(text="CNTT", entity_type=EntityType.NGANH, normalized="cntt", chunk_id="c1"),
    ]
    triples = [
        Triple(
            head="NGANH:cntt",
            relation=RelationType.CO_TO_HOP,
            tail="TO_HOP:a00",
            chunk_id="c1",
        )
    ]
    builder.add_entities(entities)
    builder.add_triples(triples)
    return builder.build()


def _router_result() -> RouterResult:
    return RouterResult(
        original_query="query",
        step_back_query="step back",
        hypothetical_doc="hyde",
        hyde_variants=["v1", "v2"],
        route=QueryRoute.GRAPH_RAG,
        complexity=4,
        intent="tra_cuu",
        reasoning="test",
    )


def _doc(cid: str, score: float, metadata=None) -> RetrievedDocument:
    return RetrievedDocument(
        text=f"t-{cid}",
        source=cid,
        chunk_id=cid,
        metadata=metadata or {},
        score=score,
        point_id=cid,
    )


def test_booster_injected_doc_survives_fusion_topk_truncation():
    """A `booster_injected` doc with a low fused score must still reach the
    retriever output. Construct a scenario with >= top_k OTHER docs outranking
    the booster-injected doc so the pre-fix [:effective_top_k] slice would
    have dropped it.
    """
    kg = _build_graph()
    retriever = GraphRAGRetriever(ner_service=_FailingNER(), graph=kg, alpha=0.6, beta=0.4)

    # 5 outrankers + 1 booster-injected doc = 6 baseline candidates.
    # top_k=5 → pre-fix the booster-injected doc would be dropped by the
    # [:5] fused-score slice.
    outrankers = [_doc(f"rank_{i}", score=10.0 - float(i)) for i in range(5)]
    booster_meta = {
        "booster_injected": True,
        "booster_kind": "aggregation",
    }
    booster = _doc(
        "liet_ke_28_nganh_2026_v2",
        score=0.01,  # tiny fused score → would be sliced out pre-fix
        metadata=booster_meta,
    )
    baseline = outrankers + [booster]

    reranked, _ppr = asyncio.run(
        retriever.retrieve(
            query="Chi tiết các ngành đào tạo HUSC",
            router_result=_router_result(),
            baseline_docs=baseline,
            top_k=5,
        )
    )

    ids = [d.chunk_id for d in reranked]
    assert "liet_ke_28_nganh_2026_v2" in ids, (
        f"booster_injected doc must survive the fusion top-k slice; "
        f"got ids={ids}"
    )
    # And it must keep its tag through the pipeline.
    booster_out = next(d for d in reranked if d.chunk_id == "liet_ke_28_nganh_2026_v2")
    assert (booster_out.metadata or {}).get("booster_injected") is True


def test_graph_injected_doc_survives_through_cutoff():
    """A `graph_injected` doc (CF-1 expander output) is protected at the
    `apply_rerank_cutoff` stage, NOT force-kept at the fusion top-k slice.

    Design (S15.10 contract): the expander admits up to M PPR-only chunks and
    fusion bounds the output to `effective_top_k` (top_k or top_k+3 on
    comparison). graph_injected docs rank naturally by β·ppr within that
    bound; their guaranteed survival lives in `apply_rerank_cutoff` (which
    EXEMPTS booster_injected + graph_injected from the relative-score drop).
    Force-keeping graph_injected at the fusion slice would break the bounded
    expander budget (test_comparison_e2e S15.10). So we assert the real
    guarantee: a graph_injected doc that made it into the reranked pool is
    NOT dropped by the precision cutoff even with a near-zero score.
    """
    from src.services.graphrag_retriever import apply_rerank_cutoff

    # A graph_injected doc with score 0.0 alongside high-score docs: the
    # relative cutoff (0.3 * top) would drop it WITHOUT the CF-1 exemption.
    docs = [
        _doc("hi_1", 0.9),
        _doc("hi_2", 0.8),
        _doc("c_graph_inj", 0.0, metadata={"graph_injected": True}),
    ]
    kept = apply_rerank_cutoff(docs, ratio=0.3, min_keep=2)
    ids = [d.chunk_id for d in kept]
    assert "c_graph_inj" in ids, (
        f"graph_injected doc must survive apply_rerank_cutoff (CF-1 exemption); "
        f"got ids={ids}"
    )


def test_no_injected_docs_topk_truncation_unchanged():
    """For queries with NO injected docs, the top_k truncation must behave
    exactly as before: result length == top_k, and order is by fused score
    DESC.
    """
    kg = _build_graph()
    retriever = GraphRAGRetriever(ner_service=_FailingNER(), graph=kg, alpha=0.6, beta=0.4)

    docs = [
        _doc("c1", 6.0),  # tops fused-score list
        _doc("c2", 5.0),
        _doc("c3", 4.0),
        _doc("c4", 3.0),
        _doc("c5", 2.0),
        _doc("c6", 1.0),  # must be cut by [:5]
    ]

    reranked, _ppr = asyncio.run(
        retriever.retrieve(
            query="bất kỳ",
            router_result=_router_result(),
            baseline_docs=docs,
            top_k=5,
        )
    )

    assert len(reranked) == 5, (
        f"no injected docs → result must be exactly top_k=5; got {len(reranked)}"
    )
    # Highest fused score first; pre-fix order is preserved.
    scores = [d.score for d in reranked]
    assert scores == sorted(scores, reverse=True), (
        f"non-injected ranking must remain DESC by fused score; got {scores}"
    )
    assert "c6" not in [d.chunk_id for d in reranked], (
        f"lowest-score doc must be cut by [:top_k]; got {[d.chunk_id for d in reranked]}"
    )
    assert "c1" in [d.chunk_id for d in reranked]
