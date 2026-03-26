"""
HybridSearchService — dense + BM25 sparse retrieval with RRF fusion.

Works natively with RetrievedDocument (no adapter needed).
Integrates directly with LanceDBRetriever.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from services.lancedb_retrieval import LanceDBRetriever, RetrievedDocument, RetrievalResult
from config.settings import RAGSettings

logger = logging.getLogger(__name__)


class HybridSearchService:
    """
    Hybrid dense + BM25 retrieval with RRF fusion.

    Lifecycle:
      1. __init__(lancedb_retriever, settings)
      2. build_bm25_index() -> bool   (call once at startup)
      3. await retrieve(query, query_vector, top_k) -> RetrievalResult
    """

    def __init__(self, lancedb_retriever: LanceDBRetriever, settings: RAGSettings) -> None:
        self._retriever = lancedb_retriever
        self._settings = settings
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_docs: List[RetrievedDocument] = []

    # ------------------------------------------------------------------
    # Index Build
    # ------------------------------------------------------------------

    def build_bm25_index(self) -> bool:
        """
        Scan LanceDB, build in-memory BM25 index.

        Uses column-selective to_pandas() to avoid loading 4096-dim vectors.
        Returns True on success, False on failure (caller should set service = None).
        """
        try:
            adapter = self._retriever._adapter
            table = adapter._table

            cols = ["text", "sparse_terms", "source", "chunk_id"]
            available_cols = table.schema.names
            load_cols = [c for c in cols if c in available_cols]

            df = table.to_pandas(columns=load_cols)

            if df.empty:
                logger.warning("HybridSearchService: LanceDB table is empty, BM25 not built")
                return False

            tokenized_corpus: List[List[str]] = []
            self._corpus_docs = []

            for _, row in df.iterrows():
                sparse_terms = row.get("sparse_terms")
                if sparse_terms and isinstance(sparse_terms, list) and len(sparse_terms) > 0:
                    tokens = [str(t) for t in sparse_terms]
                else:
                    raw = str(row.get("text", "")).lower().split()
                    tokens = raw if raw else [""]

                tokenized_corpus.append(tokens)
                self._corpus_docs.append(
                    RetrievedDocument(
                        text=str(row.get("text", "")),
                        source=str(row.get("source", "")),
                        chunk_id=str(row.get("chunk_id", "")),
                        metadata={},
                        score=0.0,
                        point_id=None,
                    )
                )

            self._bm25 = BM25Okapi(tokenized_corpus)
            logger.info(
                f"HybridSearchService: BM25 index built — {len(self._corpus_docs)} chunks"
            )
            return True

        except Exception as e:
            logger.error(f"HybridSearchService: BM25 build failed: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Hybrid retrieval: dense search + BM25 sparse -> RRF fusion.

        Falls back to dense-only if BM25 index unavailable.
        NOTE: retrieve() is async but calls sync LanceDB — no run_in_threadpool needed
        because LanceDB is embedded (in-process); there is no network or kernel I/O
        that would block the event loop. If LanceDB is ever switched to remote mode, revisit.
        """
        dense_result: RetrievalResult = self._retriever.retrieve(
            query_vector=query_vector,
            top_k=top_k * 2,
            metadata_filter=metadata_filter,
        )
        dense_docs: List[RetrievedDocument] = dense_result.documents

        if self._bm25 is None:
            logger.warning("HybridSearchService: BM25 not ready, using dense-only")
            return dense_result

        sparse_docs = self._bm25_search(query, top_k=top_k * 2)
        fused = self._rrf_fusion(dense_docs, sparse_docs, top_k=top_k)
        confidence = sum(d.score for d in fused) / len(fused) if fused else 0.0
        return RetrievalResult(documents=fused, confidence=confidence)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, top_k: int) -> List[RetrievedDocument]:
        """Return top-k docs from BM25 index as RetrievedDocument list."""
        tokens = query.lower().split()
        if not tokens:
            logger.warning("_bm25_search called with empty token list — returning []")
            return []

        scores = self._bm25.get_scores(tokens)
        scored = sorted(zip(scores, self._corpus_docs), key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scored[:top_k]:
            if score <= 0:
                break
            results.append(
                RetrievedDocument(
                    text=doc.text,
                    source=doc.source,
                    chunk_id=doc.chunk_id,
                    metadata=doc.metadata,
                    score=float(score),
                    point_id=doc.point_id,
                )
            )
        return results

    def _rrf_fusion(
        self,
        dense_docs: List[RetrievedDocument],
        sparse_docs: List[RetrievedDocument],
        top_k: int,
        k: int = 60,
    ) -> List[RetrievedDocument]:
        """
        Reciprocal Rank Fusion (RRF) over dense + sparse result lists.

        RRF score = sum(weight_i / (k + rank_i))
        k=60 is the standard parameter from Cormack et al. SIGIR 2009.
        """
        dense_weight = self._settings.HYBRID_FUSION_DENSE_WEIGHT
        sparse_weight = self._settings.HYBRID_FUSION_SPARSE_WEIGHT

        scores: Dict[str, float] = {}
        docs_map: Dict[str, RetrievedDocument] = {}

        for rank, doc in enumerate(dense_docs, start=1):
            cid = doc.chunk_id
            scores[cid] = scores.get(cid, 0.0) + dense_weight / (k + rank)
            if cid not in docs_map:
                docs_map[cid] = doc

        for rank, doc in enumerate(sparse_docs, start=1):
            cid = doc.chunk_id
            scores[cid] = scores.get(cid, 0.0) + sparse_weight / (k + rank)
            if cid not in docs_map:
                docs_map[cid] = doc

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

        fused = []
        for cid in sorted_ids[:top_k]:
            doc = docs_map[cid]
            fused.append(
                RetrievedDocument(
                    text=doc.text,
                    source=doc.source,
                    chunk_id=doc.chunk_id,
                    metadata=doc.metadata,
                    score=scores[cid],
                    point_id=doc.point_id,
                )
            )

        return fused
