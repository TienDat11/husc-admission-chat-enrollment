"""
Hybrid Retriever Module (Epic 3.1)

Implements hybrid retrieval pipeline:
1. Query Rewrite (multi-query generation)
2. Dense Search (vector similarity)
3. Sparse Search (BM25)
4. Fusion (Reciprocal Rank Fusion)
5. Reranking (Cross-encoder)
6. Confidence Scoring
"""
from typing import Any, Dict, List, Tuple

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config.settings import RAGSettings
from services.embedding import EmbeddingService
from services.vector_store import SearchResult, VectorStore


class HybridRetriever:
    """
    Hybrid retrieval with dense + sparse + reranking.
    
    Pipeline:
    1. Query → Dense (vector) + Sparse (BM25) search
    2. Fusion: RRF with weight_dense=0.6, weight_sparse=0.4
    3. Rerank top 50 with cross-encoder
    4. Compute ensemble confidence score
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        settings: RAGSettings,
    ):
        """
        Initialize hybrid retriever.

        Args:
            embedding_service: EmbeddingService instance
            vector_store: VectorStore instance
            settings: RAGSettings instance
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.settings = settings

        # BM25 initialization (lazy)
        self.bm25: BM25Okapi | None = None
        self.bm25_corpus_ids: List[str] = []

        # Load reranker
        logger.info(f"Loading reranker: {settings.RERANKER_MODEL}")
        self.reranker = CrossEncoder(settings.RERANKER_MODEL)

        logger.info("HybridRetriever initialized")

    def build_bm25_index(self, corpus_texts: List[str], corpus_ids: List[str]) -> None:
        """
        Build BM25 index from corpus.

        Args:
            corpus_texts: List of text strings
            corpus_ids: List of corresponding IDs
        """
        logger.info(f"Building BM25 index for {len(corpus_texts)} documents...")

        # Tokenize corpus
        tokenized_corpus = [self._tokenize(text) for text in corpus_texts]

        # Build BM25
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_corpus_ids = corpus_ids

        logger.info("BM25 index built successfully")

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Basic tokenization: lowercase + split
        tokens = text.lower().split()

        # Vietnamese stopwords (basic set)
        stopwords = {
            "và", "các", "của", "có", "được", "cho", "trong", "là",
            "một", "này", "để", "với", "theo", "từ", "đã", "sẽ",
            "không", "khi", "bằng", "the", "a", "an", "is", "are",
        }

        # Filter stopwords and short tokens
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]

        return tokens

    def _dense_search(
        self, query: str, top_k: int
    ) -> Tuple[List[SearchResult], np.ndarray]:
        """
        Dense vector search.

        Args:
            query: Query string
            top_k: Number of results

        Returns:
            Tuple of (search results, scores array)
        """
        # Encode query
        query_vector = self.embedding_service.encode_query(query)

        # Search vector store
        results = self.vector_store.search(query_vector, top_k=top_k)

        # Extract scores
        scores = np.array([r.score for r in results])

        # Safe top score extraction (defensive coding)
        top_score = scores[0] if len(scores) > 0 else 0.0
        
        logger.debug(
            f"Dense search: {len(results)} results, top score: {top_score:.3f}"
        )

        return results, scores

    def _sparse_search(self, query: str, top_k: int) -> Tuple[List[str], np.ndarray]:
        """
        Sparse BM25 search.

        Args:
            query: Query string
            top_k: Number of results

        Returns:
            Tuple of (result IDs, scores array)
        """
        if self.bm25 is None:
            logger.warning("BM25 index not built, returning empty results")
            return [], np.array([])

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Get IDs and scores
        result_ids = [self.bm25_corpus_ids[i] for i in top_indices]
        result_scores = scores[top_indices]

        # Safe top score extraction (defensive coding)
        top_score = result_scores[0] if len(result_scores) > 0 else 0.0
        
        logger.debug(
            f"Sparse search: {len(result_ids)} results, top score: {top_score:.3f}"
        )

        return result_ids, result_scores

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_ids: List[str],
        weight_dense: float = 0.6,
        weight_sparse: float = 0.4,
    ) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion (RRF).

        Args:
            dense_results: Dense search results
            sparse_ids: Sparse search result IDs
            weight_dense: Weight for dense scores
            weight_sparse: Weight for sparse scores

        Returns:
            List of (id, fused_score) tuples sorted by score (descending)
        """
        k = 60  # RRF constant

        # Build score dictionaries
        scores: Dict[str, float] = {}

        # Dense scores
        for rank, result in enumerate(dense_results):
            chunk_id = result.metadata.get("id", "")
            if chunk_id:
                scores[chunk_id] = weight_dense / (k + rank + 1)

        # Sparse scores
        for rank, chunk_id in enumerate(sparse_ids):
            if chunk_id in scores:
                scores[chunk_id] += weight_sparse / (k + rank + 1)
            else:
                scores[chunk_id] = weight_sparse / (k + rank + 1)

        # Sort by fused score
        fused_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        logger.debug(f"RRF fusion: {len(fused_results)} unique results")

        return fused_results

    def _rerank(
        self, query: str, candidates: List[SearchResult], top_k: int
    ) -> Tuple[List[SearchResult], np.ndarray]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: Query string
            candidates: List of candidate results
            top_k: Number of results to return

        Returns:
            Tuple of (reranked results, reranker scores)
        """
        if not candidates:
            return [], np.array([])

        # Prepare query-document pairs
        pairs = [(query, result.text) for result in candidates]

        # Score with cross-encoder
        rerank_scores = self.reranker.predict(pairs)

        # Sort by reranker score
        sorted_indices = np.argsort(rerank_scores)[::-1][:top_k]

        # Build reranked results
        reranked_results = [candidates[i] for i in sorted_indices]
        reranked_scores = rerank_scores[sorted_indices]

        # Update scores in results
        for result, score in zip(reranked_results, reranked_scores):
            result.score = float(score)

        # Safe top score extraction (defensive coding)
        top_score = reranked_scores[0] if len(reranked_scores) > 0 else 0.0
        
        logger.debug(
            f"Reranking: {len(reranked_results)} results, top score: {top_score:.3f}"
        )

        return reranked_results, reranked_scores

    def _compute_ensemble_confidence(
        self,
        dense_scores: np.ndarray,
        sparse_scores: np.ndarray,
        rerank_scores: np.ndarray,
    ) -> float:
        """
        Compute ensemble confidence score.

        Formula: 0.4 × dense_max + 0.3 × sparse_max + 0.3 × rerank_max

        Args:
            dense_scores: Dense search scores
            sparse_scores: Sparse search scores
            rerank_scores: Reranker scores

        Returns:
            Confidence score (0-1)
        """
        # Get max scores
        dense_max = float(np.max(dense_scores)) if len(dense_scores) > 0 else 0.0
        sparse_max = float(np.max(sparse_scores)) if len(sparse_scores) > 0 else 0.0
        rerank_max = float(np.max(rerank_scores)) if len(rerank_scores) > 0 else 0.0

        # Normalize rerank scores to 0-1 (sigmoid-like)
        rerank_max_norm = 1 / (1 + np.exp(-rerank_max))

        # Ensemble confidence
        confidence = 0.4 * dense_max + 0.3 * sparse_max + 0.3 * rerank_max_norm

        logger.debug(
            f"Ensemble confidence: {confidence:.3f} "
            f"(dense={dense_max:.3f}, sparse={sparse_max:.3f}, "
            f"rerank={rerank_max_norm:.3f})"
        )

        return confidence

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> Tuple[List[SearchResult], float]:
        """
        Hybrid retrieval pipeline.

        Args:
            query: Query string
            top_k: Number of final results to return

        Returns:
            Tuple of (search results, ensemble confidence)
        """
        logger.info(f"Retrieving for query: {query[:100]}...")

        # Step 1: Dense search
        dense_results, dense_scores = self._dense_search(
            query, top_k=self.settings.TOP_K_DENSE
        )

        # Step 2: Sparse search
        sparse_ids, sparse_scores = self._sparse_search(
            query, top_k=self.settings.TOP_K_SPARSE
        )

        # Step 3: RRF fusion
        fused_results = self._reciprocal_rank_fusion(
            dense_results, sparse_ids, weight_dense=0.6, weight_sparse=0.4
        )

        # Step 4: Get candidates for reranking
        candidate_ids = [chunk_id for chunk_id, _ in fused_results[: self.settings.MAX_RERANK]]

        # Retrieve full candidates from vector store
        candidates: List[SearchResult] = []
        for chunk_id in candidate_ids:
            # Find in dense results
            for result in dense_results:
                if result.metadata.get("id") == chunk_id:
                    candidates.append(result)
                    break

        logger.debug(f"Prepared {len(candidates)} candidates for reranking")

        # Step 5: Rerank
        reranked_results, rerank_scores = self._rerank(query, candidates, top_k=top_k)

        # Step 6: Compute ensemble confidence
        confidence = self._compute_ensemble_confidence(
            dense_scores, sparse_scores, rerank_scores
        )

        logger.info(
            f"Retrieved {len(reranked_results)} results with confidence {confidence:.3f}"
        )

        return reranked_results, confidence


def rewrite_query(query: str, num_rewrites: int = 3) -> List[str]:
    """
    Generate multiple query rewrites.

    Simple template-based approach (can be enhanced with LLM later).

    Args:
        query: Original query
        num_rewrites: Number of rewrites to generate

    Returns:
        List of query rewrites (including original)
    """
    rewrites = [query]

    # Template-based rewrites
    templates = [
        f"Explain {query}",
        f"What is {query}",
        f"Information about {query}",
    ]

    rewrites.extend(templates[:num_rewrites - 1])

    return rewrites
