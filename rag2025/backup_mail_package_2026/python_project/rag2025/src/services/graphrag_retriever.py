"""
Unified RAG Pipeline – PaddedRAG + GraphRAG with Smart Routing

Entry point for all queries. Routes automatically:
  simple query  → PaddedRAG (BM25 + dense + cross-encoder)
  multihop/comparative → GraphRAG (PPR fusion)

Usage:
    pipeline = UnifiedRAGPipeline.from_disk()
    result = await pipeline.query("Ngành CNTT có điểm chuẩn bao nhiêu?")

Scalability:
    - Graph is loaded once at startup (in-memory NetworkX MultiDiGraph)
    - Incremental graph updates: call pipeline.update_graph(new_chunks)
    - Router uses UnifiedLLMClient → swap model by changing .env only
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from loguru import logger

from src.domain.entities import Chunk
from src.domain.graph import KnowledgeGraph, KnowledgeGraphBuilder
from src.services.llm_client import UnifiedLLMClient, get_llm_client
from src.services.ner_service import NERService
from src.services.query_router import QueryRoute, RouterResult, SmartQueryRouter
from src.services.lancedb_retrieval import RetrievedDocument
from config.settings import RAGSettings


GRAPH_PATH = Path(__file__).parent.parent.parent / "data" / "graph" / "knowledge_graph.graphml"


@dataclass
class RAGResult:
    """Unified result from either PaddedRAG or GraphRAG pipeline.

    Attributes:
        query: Original user query.
        route: Which pipeline was used.
        documents: Retrieved documents (reranked).
        router_result: Full router output (step-back, HyDE, classification).
        ppr_scores: Graph PPR scores (empty if padded_rag route).
        latency_ms: End-to-end latency in milliseconds.
        confidence: Top document score (0–1).
    """
    query: str
    route: str
    documents: List[RetrievedDocument]
    router_result: RouterResult
    ppr_scores: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "route": self.route,
            "documents": [d.to_dict() for d in self.documents],
            "router": {
                "step_back": self.router_result.step_back_query,
                "intent": self.router_result.intent,
                "complexity": self.router_result.complexity,
                "reasoning": self.router_result.reasoning,
                "hyde_variants": self.router_result.hyde_variants,
            },
            "latency_ms": round(self.latency_ms, 1),
            "confidence": round(self.confidence, 4),
        }


class GraphRAGRetriever:
    """Graph-augmented retriever: fuses PaddedRAG scores with PPR graph scores.

    Fusion formula:
        score(chunk) = α·rrf_score + β·ppr_score     α+β = 1.0

    Args:
        ner_service: For extracting query entities.
        graph: Pre-loaded KnowledgeGraph.
        alpha: Weight for vector/RRF score (default 0.6).
        beta: Weight for PPR graph score (default 0.4).
    """

    def __init__(
        self,
        ner_service: NERService,
        graph: KnowledgeGraph,
        alpha: float = 0.6,
        beta: float = 0.4,
        ppr_alpha: float = 0.85,
    ) -> None:
        if abs(alpha + beta - 1.0) > 1e-6:
            raise ValueError(f"alpha + beta must equal 1.0, got {alpha + beta}")
        self._ner = ner_service
        self._graph = graph
        self._alpha = alpha
        self._beta = beta
        self._ppr_alpha = ppr_alpha

    async def _seed_entities_from_query(self, query: str, router: RouterResult) -> List[str]:
        """Extract entity node_ids for PPR seeding.

        Uses original + step-back + HyDE variants for richer seed coverage.
        """
        combined_parts = [query, router.step_back_query] + router.hyde_variants[:3]
        combined = "\n".join([p for p in combined_parts if p])
        chunk = Chunk(chunk_id="__query__", text=combined)
        result = await self._ner.extract(chunk)
        if not result.is_success:
            return []

        seen: Set[str] = set()
        seeds: List[str] = []
        for entity in result.entities:
            if entity.node_id in seen:
                continue
            seen.add(entity.node_id)
            seeds.append(entity.node_id)
        return seeds

    def _normalize_scores(self, docs: List[RetrievedDocument]) -> Dict[str, float]:
        if not docs:
            return {}
        scores = {d.chunk_id: d.score for d in docs if d.chunk_id}
        max_s = max(scores.values(), default=1.0)
        return {k: v / max_s for k, v in scores.items()} if max_s > 0 else scores

    async def retrieve(
        self,
        query: str,
        router_result: RouterResult,
        baseline_docs: List[RetrievedDocument],
        top_k: int = 5,
    ) -> Tuple[List[RetrievedDocument], Dict[str, float]]:
        """Fuse baseline PaddedRAG docs with PPR graph scores.

        Args:
            query: Original query.
            router_result: Router output (has step_back, entities, etc.).
            baseline_docs: Documents from PaddedRAG pipeline.
            top_k: Final number of documents.

        Returns:
            (reranked_docs, ppr_scores_by_chunk)
        """
        seed_entities = await self._seed_entities_from_query(query, router_result)
        ppr_scores = (
            self._graph.ppr_scores_by_chunk(seed_entities, alpha=self._ppr_alpha)
            if seed_entities
            else {}
        )
        rrf_scores = self._normalize_scores(baseline_docs)

        fused: Dict[str, float] = {}
        for doc in baseline_docs:
            if not doc.chunk_id:
                continue
            base_score = rrf_scores.get(doc.chunk_id, 0.0)
            graph_score = ppr_scores.get(doc.chunk_id, 0.0)

            if ppr_scores:
                fused_score = self._alpha * base_score + self._beta * graph_score
            else:
                fused_score = base_score

            fused[doc.chunk_id] = fused_score

        doc_by_id = {d.chunk_id: d for d in baseline_docs if d.chunk_id}
        sorted_ids = sorted(fused, key=fused.__getitem__, reverse=True)[:top_k]

        reranked = [
            RetrievedDocument(
                text=doc_by_id[cid].text,
                source=doc_by_id[cid].source,
                chunk_id=cid,
                metadata=doc_by_id[cid].metadata,
                score=fused[cid],
                point_id=doc_by_id[cid].point_id,
            )
            for cid in sorted_ids
            if cid in doc_by_id
        ]

        logger.info(
            f"GraphRAG fusion: {len(reranked)} results "
            f"(seeds={len(seed_entities)}, ppr_hits={len(ppr_scores)})"
        )
        return reranked, ppr_scores

    def update_graph(self, new_graph: KnowledgeGraph) -> None:
        """Hot-swap graph (for incremental updates without restart)."""
        old_stats = self._graph.stats()
        self._graph = new_graph
        new_stats = new_graph.stats()
        logger.info(
            f"Graph updated: {old_stats['nodes']}→{new_stats['nodes']} nodes, "
            f"{old_stats['edges']}→{new_stats['edges']} edges"
        )

    @property
    def graph_stats(self) -> Dict:
        return self._graph.stats()


class UnifiedRAGPipeline:
    """Top-level pipeline: routes and executes PaddedRAG or GraphRAG.

    This is the single entry point for all query handling.

    Args:
        router: SmartQueryRouter instance.
        graphrag: GraphRAGRetriever instance.
        llm: UnifiedLLMClient for answer generation.
    """

    def __init__(
        self,
        router: SmartQueryRouter,
        graphrag: GraphRAGRetriever,
        llm: Optional[UnifiedLLMClient] = None,
    ) -> None:
        self._router = router
        self._graphrag = graphrag
        self._llm = llm or get_llm_client()

    @classmethod
    def from_disk(
        cls,
        graph_path: Optional[Path] = None,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> "UnifiedRAGPipeline":
        """Factory: load graph from disk and initialize full pipeline.

        Falls back to empty graph if .graphml not found.
        Run scripts/build_graph.py first to populate the graph.
        """
        path = graph_path or GRAPH_PATH
        llm = get_llm_client()
        ner = NERService(llm=llm)
        settings = RAGSettings()
        router = SmartQueryRouter(
            llm=llm,
            simple_complexity_threshold=settings.GRAPHRAG_SIMPLE_THRESHOLD,
        )

        if path.exists():
            graph = KnowledgeGraph.load(path)
        else:
            logger.warning(
                f"Graph not found at {path} – GraphRAG will use PPR=0. "
                "Run: python scripts/build_graph.py"
            )
            graph = KnowledgeGraph.empty()

        graphrag = GraphRAGRetriever(
            ner_service=ner,
            graph=graph,
            alpha=alpha,
            beta=beta,
            ppr_alpha=settings.GRAPHRAG_PPR_ALPHA,
        )
        return cls(router=router, graphrag=graphrag, llm=llm)

    async def query(
        self,
        user_query: str,
        baseline_docs: Optional[List[RetrievedDocument]] = None,
        top_k: int = 5,
    ) -> RAGResult:
        """Route and execute query.

        Args:
            user_query: Raw user query.
            baseline_docs: Pre-computed PaddedRAG documents (if None, pipeline
                           returns routing info only – integrate with main.py).
            top_k: Number of final documents.

        Returns:
            RAGResult with documents, routing metadata, and latency.
        """
        t0 = time.perf_counter()

        # Step 1: Route (HyDE + Step-Back + Classification)
        router_result = await self._router.route(user_query)

        # Step 2: Retrieve based on route
        docs = baseline_docs or []
        ppr_scores: Dict[str, float] = {}

        if router_result.route == QueryRoute.GRAPH_RAG and docs:
            docs, ppr_scores = await self._graphrag.retrieve(
                query=user_query,
                router_result=router_result,
                baseline_docs=docs,
                top_k=top_k,
            )

        latency = (time.perf_counter() - t0) * 1000
        confidence = docs[0].score if docs else 0.0

        result = RAGResult(
            query=user_query,
            route=router_result.route.value,
            documents=docs[:top_k],
            router_result=router_result,
            ppr_scores=ppr_scores,
            latency_ms=latency,
            confidence=confidence,
        )

        logger.info(
            f"Pipeline: route={result.route}, docs={len(result.documents)}, "
            f"latency={result.latency_ms:.0f}ms, confidence={result.confidence:.3f}"
        )
        return result

    async def incremental_update(self, new_chunks: List[Chunk]) -> None:
        """Add new data to graph without full rebuild.

        Args:
            new_chunks: New Chunk objects to process and add to graph.
        """
        logger.info(f"Incremental update: {len(new_chunks)} new chunks")
        ner = NERService(llm=self._llm)
        results = await ner.extract_batch(new_chunks)

        builder = KnowledgeGraphBuilder.from_graph(self._graphrag._graph)
        for r in results:
            if r.is_success:
                builder.add_entities(r.entities)
                builder.add_triples(r.triples)

        new_graph = builder.build()
        self._graphrag.update_graph(new_graph)

        # Persist updated graph
        GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_graph.save(GRAPH_PATH)
        logger.info("Incremental update complete – graph persisted")


# Module-level singleton
_pipeline: Optional[UnifiedRAGPipeline] = None


def get_pipeline() -> UnifiedRAGPipeline:
    """Get or create the singleton UnifiedRAGPipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = UnifiedRAGPipeline.from_disk()
    return _pipeline
