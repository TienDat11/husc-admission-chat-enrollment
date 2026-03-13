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


def _build_graph():
    builder = KnowledgeGraphBuilder()
    entities = [
        Entity(text="CNTT", entity_type=EntityType.NGANH, normalized="cntt", chunk_id="c1"),
        Entity(text="A00", entity_type=EntityType.TO_HOP, normalized="a00", chunk_id="c2"),
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
        intent="so_sanh",
        reasoning="test",
    )


def test_ppr_cache_reuses_results_and_clear_caches():
    kg = _build_graph()

    seeds_ab = ["TO_HOP:a00", "NGANH:cntt"]
    seeds_ba = ["NGANH:cntt", "TO_HOP:a00"]

    first = kg.personalized_pagerank(seeds_ab, alpha=0.85)
    second = kg.personalized_pagerank(seeds_ba, alpha=0.85)

    assert first == second
    assert len(kg._ppr_cache) == 1

    kg.clear_caches()
    assert len(kg._ppr_cache) == 0


def test_ppr_returns_empty_when_convergence_fails(monkeypatch):
    kg = _build_graph()

    import src.domain.graph as graph_module

    def _raise(*args, **kwargs):
        raise graph_module.nx.PowerIterationFailedConvergence(100)

    monkeypatch.setattr(graph_module.nx, "pagerank", _raise)

    scores = kg.personalized_pagerank(["NGANH:cntt"], alpha=0.85)
    assert scores == {}


def test_graphrag_fusion_falls_back_to_baseline_when_no_ppr_scores():
    kg = _build_graph()
    retriever = GraphRAGRetriever(ner_service=_FailingNER(), graph=kg, alpha=0.6, beta=0.4)

    docs = [
        RetrievedDocument(text="doc1", source="s1", chunk_id="c1", score=2.0),
        RetrievedDocument(text="doc2", source="s2", chunk_id="c2", score=1.0),
    ]

    reranked, ppr_scores = asyncio.run(
        retriever.retrieve(
            query="so sánh ngành",
            router_result=_router_result(),
            baseline_docs=docs,
            top_k=2,
        )
    )

    assert ppr_scores == {}
    assert [d.chunk_id for d in reranked] == ["c1", "c2"]
    assert reranked[0].score == pytest.approx(1.0)
    assert reranked[1].score == pytest.approx(0.5)
