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

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Set

from loguru import logger

from src.domain.entities import Chunk
from src.domain.graph import KnowledgeGraph, KnowledgeGraphBuilder
from src.services.llm_client import UnifiedLLMClient, get_llm_client
from src.services.ner_service import NERService
from src.services.query_router import QueryRoute, RouterResult, SmartQueryRouter
from src.services.lancedb_retrieval import RetrievedDocument
from config.settings import RAGSettings


GRAPH_PATH = Path(__file__).parent.parent.parent / "data" / "graph" / "knowledge_graph.graphml"

# ─── S15.1 / S15.2 / S15.5 constants ───────────────────────────────────────────
# Expander budget (M): max number of PPR-only chunks to admit into the candidate
# pool via `chunk_fetcher`. Default 5; bumped to 8 on comparison-detection.
_EXPANDER_M_DEFAULT = 5
_EXPANDER_M_COMPARISON = 8

# Top-k bump applied on comparison-detection (≥2 NGANH/TO_HOP seeds).
_TOPK_BUMP_COMPARISON = 3

# Prefixes that count toward the "≥2 distinct entities of type NGANH or TO_HOP"
# comparison-trigger from CF-4.
_COMPARISON_SEED_PREFIXES: Tuple[str, ...] = ("NGANH:", "TO_HOP:")


def _is_comparison_seed_set(seeds: List[str]) -> bool:
    """CF-4: comparison-trigger = ≥2 distinct seeds whose node_id starts with
    `NGANH:` or `TO_HOP:`. Other entity types (DIEM_CHUAN, HOC_PHI, …) do not
    contribute.
    """
    distinct = {
        s
        for s in seeds
        if s and s.startswith(_COMPARISON_SEED_PREFIXES)
    }
    return len(distinct) >= 2


def apply_rerank_cutoff(
    docs: List[RetrievedDocument],
    ratio: float,
    min_keep: int,
) -> List[RetrievedDocument]:
    """ADR-E / CF-1 / CF-2 precision cutoff (pure helper).

    Drops docs whose `.score < ratio * top_score`. Always keeps at least
    `min_keep` docs (filled from the input head — caller is expected to pass
    docs sorted DESC by score). NEVER drops a doc carrying either
    `metadata["booster_injected"]` (CF-2 aggregation-booster exemption) or
    `metadata["graph_injected"]` (CF-1 GraphRAG-expander exemption).

    Args:
        docs: Reranked docs from the retriever, sorted DESC by score.
        ratio: Cutoff ratio. ``ratio <= 0`` → cutoff disabled.
        min_keep: Minimum number of docs to return. The cutoff never returns
            fewer than this (capped by ``len(docs)``).

    Returns:
        New list — input is never mutated.
    """
    if not docs:
        return []
    if ratio <= 0:
        return list(docs)

    top_score = max((d.score for d in docs), default=0.0)
    if top_score <= 0:
        # CF-5 div-by-zero guard: nothing has meaningful score → keep all.
        return list(docs)

    threshold = ratio * top_score
    kept: List[RetrievedDocument] = []
    for d in docs:
        md = d.metadata or {}
        is_exempt = bool(md.get("booster_injected") or md.get("graph_injected"))
        if is_exempt or d.score >= threshold:
            kept.append(d)

    if len(kept) < min_keep:
        # Backfill from input head (assumed sorted DESC) until min_keep reached.
        kept_ids = {id(d) for d in kept}
        for d in docs:
            if id(d) in kept_ids:
                continue
            kept.append(d)
            kept_ids.add(id(d))
            if len(kept) >= min_keep:
                break
    return kept


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
        chunk_fetcher: Optional[Callable[[str], Optional[RetrievedDocument]]] = None,
        chunk_batch_fetcher: Optional[Callable[[List[str]], Dict[str, RetrievedDocument]]] = None,
    ) -> None:
        if abs(alpha + beta - 1.0) > 1e-6:
            raise ValueError(f"alpha + beta must equal 1.0, got {alpha + beta}")
        self._ner = ner_service
        self._graph = graph
        self._alpha = alpha
        self._beta = beta
        self._ppr_alpha = ppr_alpha
        # ADR-A / MUST-FIX#2: optional expander hook. Default None preserves
        # pre-S15 rerank-only behavior (backward-compat).
        self._chunk_fetcher = chunk_fetcher
        # Latency-fix: optional batched expander hook. When set, the expander
        # collapses M per-id scans of LanceDB into ONE scan via isin. When
        # None, the per-id `chunk_fetcher` loop is used (backward-compat).
        self._chunk_batch_fetcher = chunk_batch_fetcher

    async def _seed_entities_from_query(self, query: str, router: RouterResult) -> List[str]:
        """Extract entity node_ids for PPR seeding.

        Strategy: regex-first (fast, deterministic), then LLM fallback only if needed.
        """
        # Step 1: regex-first (fast, deterministic)
        regex_seeds = self._ner.extract_from_query(query)
        seen: Set[str] = set(regex_seeds)
        seeds: List[str] = list(regex_seeds)

        # Step 2: also try on combined (step_back + hyde variants)
        combined_parts = [router.step_back_query] + router.hyde_variants[:3]
        combined = "\n".join([p for p in combined_parts if p])
        if combined:
            for sid in self._ner.extract_from_query(combined):
                if sid not in seen:
                    seen.add(sid)
                    seeds.append(sid)

        # Step 3: only LLM fallback if regex found nothing
        if not seeds:
            full_text = f"{query}\n{combined}" if combined else query
            chunk = Chunk(chunk_id="__query__", text=full_text)
            result = await self._ner.extract(chunk)
            if result.is_success:
                for entity in result.entities:
                    if entity.node_id not in seen:
                        seen.add(entity.node_id)
                        seeds.append(entity.node_id)

        logger.info(f"Seed entities: {len(seeds)} found (regex={len(regex_seeds)})")
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
        expander_m: Optional[int] = None,
    ) -> Tuple[List[RetrievedDocument], Dict[str, float]]:
        """Fuse baseline PaddedRAG docs with PPR graph scores.

        S15.1 (ADR-A + MUST-FIX#1/#2 + CF-1): when `self._chunk_fetcher` is
        wired, admit top-M PPR-only chunks (NOT in baseline) into the
        candidate pool BEFORE `_normalize_scores`. Each injected doc has its
        base score neutralized to 0.0 (MUST-FIX#1 — so fused = β·ppr only,
        not the fetch_by_id sentinel 1.0) and is tagged
        `metadata["graph_injected"]=True` (CF-1 — exempts it from S15.5
        precision cutoff).

        S15.2 (ADR-B + CF-4): when seeds contain ≥2 distinct entities of
        type NGANH or TO_HOP, bump expander budget M 5→8 and final top_k +3.

        S16.1 (AMF-1): `expander_m` (optional) overrides the comparison
        heuristic. Caller-passed M wins EXCEPT on the graph route where a
        comparison still demands M≥8 (so effective M = max(expander_m,
        _EXPANDER_M_COMPARISON) when the graph route runs into comparison
        seeds). Backward-compat: omit `expander_m` → today's behavior
        (heuristic M=5 default, M=8 on comparison seeds).

        Args:
            query: Original query.
            router_result: Router output (has step_back, entities, etc.).
            baseline_docs: Documents from PaddedRAG pipeline.
            top_k: Final number of documents (may be bumped on comparison).
            expander_m: Optional explicit expander budget. When provided,
                OVERRIDES the comparison heuristic for non-graph routes.
                For graph route, the effective M is
                max(expander_m, _EXPANDER_M_COMPARISON) if comparison seeds
                are present, else `expander_m`.

        Returns:
            (reranked_docs, ppr_scores_by_chunk)
        """
        seed_entities = await self._seed_entities_from_query(query, router_result)
        ppr_scores = (
            self._graph.ppr_scores_by_chunk(seed_entities, alpha=self._ppr_alpha)
            if seed_entities
            else {}
        )

        # CF-4 comparison-detection — bumps both expander budget and final top_k.
        is_comparison = _is_comparison_seed_set(seed_entities)
        # S16.1 / AMF-1: per-route expander_m override.
        #   - expander_m None   → legacy heuristic (M=8 on comparison else M=5)
        #   - expander_m int    → caller M wins, except on the GRAPH route
        #                         where a comparison still demands M≥8
        #                         (so effective M = max(expander_m, 8)).
        if expander_m is None:
            expander_m_eff = _EXPANDER_M_COMPARISON if is_comparison else _EXPANDER_M_DEFAULT
        else:
            if router_result.route == QueryRoute.GRAPH_RAG and is_comparison:
                # Graph route on a comparison must keep M≥8 so a comparison
                # never loses recall; caller M is the floor.
                expander_m_eff = max(int(expander_m), _EXPANDER_M_COMPARISON)
            else:
                # Hybrid / padded + non-graph: caller-passed M is authoritative.
                expander_m_eff = int(expander_m)
        effective_top_k = top_k + _TOPK_BUMP_COMPARISON if is_comparison else top_k

        # ─── S15.1 GraphRAG EXPANDER ────────────────────────────────────────
        # Admit top-M PPR-only chunks (NOT in baseline) via chunk_fetcher.
        # MUST happen BEFORE _normalize_scores so doc_by_id covers them.
        working_docs: List[RetrievedDocument] = list(baseline_docs)
        # Expander runs if EITHER fetcher hook is wired. The batch hook
        # (latency-fix) takes precedence when both are set.
        expander_enabled = (
            self._chunk_fetcher is not None or self._chunk_batch_fetcher is not None
        )
        if expander_enabled and ppr_scores:
            baseline_ids: Set[str] = {
                d.chunk_id for d in baseline_docs if d.chunk_id
            }
            # Highest PPR-only chunk_ids first.
            ranked_ppr_ids = sorted(
                ppr_scores, key=ppr_scores.__getitem__, reverse=True
            )
            # Pre-compute the candidate list (ranked, not-in-baseline, capped).
            # This MUST run BEFORE deciding which fetcher path to take, so the
            # per-id and batch paths admit IDENTICAL chunks in the same order.
            candidate_ids: List[str] = []
            for cid in ranked_ppr_ids:
                if len(candidate_ids) >= expander_m_eff:
                    break
                if cid in baseline_ids:
                    continue
                candidate_ids.append(cid)

            injected_count = 0
            if self._chunk_batch_fetcher is not None and candidate_ids:
                # Latency-fix: ONE call returns {cid: doc} for the whole
                # capped admit list. M scans → 1 scan.
                try:
                    fetched_map = self._chunk_batch_fetcher(candidate_ids)
                except Exception as exc:  # never block the retrieve path
                    logger.warning(
                        f"chunk_batch_fetcher({len(candidate_ids)} ids) raised: {exc}"
                    )
                    fetched_map = {}
                for cid in candidate_ids:
                    fetched = fetched_map.get(cid) if fetched_map else None
                    if fetched is None:
                        continue
                    # MUST-FIX#1: neutralize sentinel score → 0.0 so fused = β·ppr.
                    # CF-1: tag graph_injected so S15.5 cutoff never drops it.
                    merged_meta = dict(fetched.metadata or {})
                    merged_meta["graph_injected"] = True
                    working_docs.append(
                        RetrievedDocument(
                            text=fetched.text,
                            source=fetched.source,
                            chunk_id=fetched.chunk_id or cid,
                            metadata=merged_meta,
                            score=0.0,
                            point_id=fetched.point_id,
                        )
                    )
                    injected_count += 1
            elif self._chunk_fetcher is not None:
                # Backward-compat: per-id loop (unchanged).
                for cid in candidate_ids:
                    try:
                        fetched = self._chunk_fetcher(cid)
                    except Exception as exc:  # never block the retrieve path
                        logger.warning(f"chunk_fetcher({cid}) raised: {exc}")
                        continue
                    if fetched is None:
                        continue
                    # MUST-FIX#1: neutralize sentinel score → 0.0 so fused = β·ppr.
                    # CF-1: tag graph_injected so S15.5 cutoff never drops it.
                    merged_meta = dict(fetched.metadata or {})
                    merged_meta["graph_injected"] = True
                    working_docs.append(
                        RetrievedDocument(
                            text=fetched.text,
                            source=fetched.source,
                            chunk_id=fetched.chunk_id or cid,
                            metadata=merged_meta,
                            score=0.0,
                            point_id=fetched.point_id,
                        )
                    )
                    injected_count += 1
            if injected_count:
                logger.info(
                    f"GraphRAG expander: admitted {injected_count} PPR-only "
                    f"chunks (M={expander_m_eff}, comparison={is_comparison})"
                )

        rrf_scores = self._normalize_scores(working_docs)

        fused: Dict[str, float] = {}
        for doc in working_docs:
            if not doc.chunk_id:
                continue
            base_score = rrf_scores.get(doc.chunk_id, 0.0)
            graph_score = ppr_scores.get(doc.chunk_id, 0.0)

            if ppr_scores:
                fused_score = self._alpha * base_score + self._beta * graph_score
            else:
                fused_score = base_score

            fused[doc.chunk_id] = fused_score

        doc_by_id = {d.chunk_id: d for d in working_docs if d.chunk_id}
        # Exempt ONLY booster_injected (CF-2 aggregation summaries) from the
        # fused-score top-k slice — those are deliberately-injected full-list
        # chunks that must never be evicted before generation (fixes the
        # "generic vs detailed major list" bug). graph_injected (CF-1 expander)
        # docs are NOT force-kept here: they rank naturally by β·ppr and are
        # already protected at the later apply_rerank_cutoff stage (line 101).
        # Force-keeping graph_injected here would break the comparison
        # top_k+3 bounded-output contract (test_comparison_e2e S15.10).
        exempt_ids = {
            cid
            for cid, doc in doc_by_id.items()
            if (doc.metadata or {}).get("booster_injected")
        }
        sorted_full = sorted(fused, key=fused.__getitem__, reverse=True)
        top_ids = sorted_full[:effective_top_k]
        # Union in any exempt doc not already in the top-k, keeping its
        # fused-score rank order. Non-exempt ranking/ordering is unchanged
        # for queries with no injected docs.
        ordered_extras = [cid for cid in sorted_full if cid in exempt_ids and cid not in top_ids]
        sorted_ids = list(top_ids) + ordered_extras

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
            f"(seeds={len(seed_entities)}, ppr_hits={len(ppr_scores)}, "
            f"comparison={is_comparison}, effective_top_k={effective_top_k})"
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
        settings = RAGSettings()

        # Separate LLM clients for router vs NER (extraction).
        # Router model = effective_router_model (RAMCLOUDS_ROUTER_MODEL or
        # falls back to RAMCLOUDS_HYDE_MODEL — true no-op default per ADR-D).
        # Lets us pin a stronger model for routing/HyDE/classify without
        # slowing generation (S15.4).
        router_llm = get_llm_client(force_model=settings.effective_router_model)
        ner_llm = get_llm_client(force_model=settings.RAMCLOUDS_NER_MODEL)

        ner = NERService(llm=ner_llm)
        router = SmartQueryRouter(
            llm=router_llm,
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

        # MUST-FIX#2: bind chunk_fetcher to the shared LanceDB singleton so
        # the GraphRAG expander (S15.1) can admit PPR-only chunks via the
        # SAME table handle used by main.py / eval_run_instrumented.py — no
        # second DB handle, no schema drift.
        chunk_fetcher: Optional[Callable[[str], Optional[RetrievedDocument]]]
        chunk_batch_fetcher: Optional[Callable[[List[str]], Dict[str, RetrievedDocument]]]
        try:
            from src.services.lancedb_retrieval import get_retriever as _get_retriever
            _lance = _get_retriever()
            chunk_fetcher = _lance.fetch_by_id
            # Latency-fix: also bind the batched sibling so the expander can
            # collapse M per-id scans into ONE scan via `fetch_by_ids`.
            chunk_batch_fetcher = _lance.fetch_by_ids
        except Exception as exc:
            logger.warning(
                f"Could not bind chunk_fetcher: {exc} – GraphRAG expander disabled"
            )
            chunk_fetcher = None
            chunk_batch_fetcher = None

        graphrag = GraphRAGRetriever(
            ner_service=ner,
            graph=graph,
            alpha=alpha,
            beta=beta,
            ppr_alpha=settings.GRAPHRAG_PPR_ALPHA,
            chunk_fetcher=chunk_fetcher,
            chunk_batch_fetcher=chunk_batch_fetcher,
        )
        return cls(router=router, graphrag=graphrag, llm=router_llm)

    async def query(
        self,
        user_query: str,
        baseline_docs: Optional[List[RetrievedDocument]] = None,
        top_k: int = 5,
        router_result: Optional[RouterResult] = None,
    ) -> RAGResult:
        """Route and execute query.

        Args:
            user_query: Raw user query.
            baseline_docs: Pre-computed PaddedRAG documents (if None, pipeline
                           returns routing info only – integrate with main.py).
            top_k: Number of final documents.
            router_result: Pre-computed routing result (skip re-routing if provided).

        Returns:
            RAGResult with documents, routing metadata, and latency.
        """
        t0 = time.perf_counter()

        # Step 1: Route (HyDE + Step-Back + Classification)
        if router_result is None:
            router_result = await self._router.route(user_query)

        # Step 1.5: HyDE auto-answer / vague-reject short-circuit.
        # Nếu router quyết định skip retrieval (auto_answer set) → trả luôn kết quả
        # rỗng documents để main.py / eval script biết dùng auto_answer thay cho LLM.
        if router_result.skip_retrieval and router_result.auto_answer:
            latency = (time.perf_counter() - t0) * 1000
            logger.info(
                f"Pipeline: skip_retrieval=True intent={router_result.intent} "
                f"latency={latency:.0f}ms (auto-answer)"
            )
            return RAGResult(
                query=user_query,
                route=router_result.route.value,
                documents=[],
                router_result=router_result,
                ppr_scores={},
                latency_ms=latency,
                confidence=1.0,
            )

        # Step 2: Retrieve based on route (S16.1 — 3-way dispatch)
        #   - PADDED_RAG  → pure vector, baseline_docs[:top_k], NO retrieve call.
        #   - HYBRID      → vector + PPR fusion + expander M=5 (default budget).
        #   - GRAPH_RAG   → vector + PPR fusion + expander M=8 (+ comparison bump).
        # The per-route expander_m is passed so the GraphRAGRetriever can
        # override the comparison-heuristic M (AMF-1). For graph+comparison
        # seeds, GraphRAGRetriever floors M at 8 internally.
        docs = baseline_docs or []
        ppr_scores: Dict[str, float] = {}

        if router_result.route == QueryRoute.GRAPH_RAG and docs:
            docs, ppr_scores = await self._graphrag.retrieve(
                query=user_query,
                router_result=router_result,
                baseline_docs=docs,
                top_k=top_k,
                expander_m=8,
            )
            # GraphRAG.retrieve() already applied its own effective_top_k
            # (may be top_k + 3 on comparison-detection — ADR-B/CF-4). Do NOT
            # re-slice with [:top_k] here, which would undo the comparison
            # bump.
            final_docs = list(docs)
        elif router_result.route == QueryRoute.HYBRID and docs:
            docs, ppr_scores = await self._graphrag.retrieve(
                query=user_query,
                router_result=router_result,
                baseline_docs=docs,
                top_k=top_k,
                expander_m=5,
            )
            # Hybrid uses the default top_k (no comparison bump).
            final_docs = list(docs)
        else:
            # PADDED_RAG / no-doc paths: pure vector; baseline list can be
            # longer than top_k (the eval loop seeds 15 docs), so clamp here.
            final_docs = list(docs[:top_k])

        # ── S15.5 precision cutoff (ADR-E + MUST-FIX#4 + CF-1/CF-2/CF-5) ──
        # Applied at the pipeline boundary on final_docs (which carry a real
        # `.score`). The booster_injected + graph_injected exemptions are
        # enforced inside `apply_rerank_cutoff` so this stays a single concern.
        try:
            ratio = float(os.getenv("RERANK_KEEP_RATIO", "0.3"))
        except ValueError:
            ratio = 0.3
        try:
            min_keep = int(os.getenv("RERANK_MIN_KEEP", "2"))
        except ValueError:
            min_keep = 2
        final_docs = apply_rerank_cutoff(final_docs, ratio=ratio, min_keep=min_keep)

        latency = (time.perf_counter() - t0) * 1000
        confidence = final_docs[0].score if final_docs else 0.0

        result = RAGResult(
            query=user_query,
            route=router_result.route.value,
            documents=final_docs,
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
