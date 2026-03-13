"""
BGE Multi-Vector Retriever Service (Retrieval Layer)

Advanced retrieval with:
- BGE-M3 embedding model (1024-dim, multilingual)
- Qdrant vector store
- Score boosting for near-matches
- Vietnamese reranker (optional)
"""
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from loguru import logger
import os


@dataclass
class SearchResult:
    """Search result wrapper for Qdrant results"""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[np.ndarray] = None


class BGEMultiVectorRetriever:
    """
    BGE Multi-Vector Retrieval with score boosting
    """

    def __init__(self):
        # Load BGE model (BAAI/bge-m3 - best for Vietnamese)
        logger.info("Loading BGE-M3 embedding model...")
        self.model = SentenceTransformer('BAAI/bge-m3')
        self.embedding_dim = 1024

        # Qdrant client
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url:
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_key
            )
            self.collection_name = os.getenv("QDRANT_COLLECTION", "hue_admissions_2025")
            logger.info(f"Connected to Qdrant: {qdrant_url}")
        else:
            logger.warning("No QDRANT_URL configured, using fallback to NumPy store")
            self.qdrant_client = None
            self.collection_name = None

        # Weights for multi-vector fusion
        self.dense_weight = 0.7
        self.sparse_weight = 0.3

        # Load reranker (optional - skip if memory issues)
        self.reranker = None
        logger.info("Reranker disabled for memory optimization")

        logger.info("BGE Multi-Vector Retriever initialized")

    async def retrieve(
        self,
        query_enhanced: str,
        original_query: str,
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Main retrieval with score boosting

        Returns:
            (chunks, confidence_score)
        """
        logger.info(f"Retrieving for: {original_query}")

        # 1. Encode query
        query_vector = self.model.encode(query_enhanced, normalize_embeddings=True)

        # 2. Search Qdrant (get top_k * 2 for reranking)
        if self.qdrant_client is None:
            logger.warning("Qdrant not available, returning empty results")
            return [], 0.0

        # Use search_batch API (compatible with all Qdrant versions)
        # This returns a list of lists of ScoredPoint
        try:
            search_batch_results = self.qdrant_client.search_batch(
                collection_name=self.collection_name,
                queries=[query_vector.tolist()],
                limit=top_k * 2,
                with_payload=True,
                with_vector=False
            )
            # Get first (and only) query results
            search_results = search_batch_results[0] if search_batch_results else []
        except Exception as e:
            logger.error(f"Qdrant search_batch failed: {e}")
            # Fallback: use query (different API)
            try:
                response = self.qdrant_client.query(
                    collection_name=self.collection_name,
                    query=query_vector.tolist(),
                    limit=top_k * 2,
                    with_payload=True,
                    with_vector=False
                )
                # Extract points from response if it's a QueryResponse
                if hasattr(response, 'points'):
                    search_results = response.points
                else:
                    search_results = list(response) if response else []
            except Exception as e2:
                logger.error(f"Qdrant query also failed: {e2}")
                return [], 0.0

        # Convert to our SearchResult format
        wrapped_results = []
        for r in search_results:
            wrapped_results.append(SearchResult(
                id=str(r.id),
                score=float(r.score),
                payload=dict(r.payload) if hasattr(r, 'payload') else {},
                vector=None
            ))

        # 3. Apply score boosting (CRITICAL!)
        boosted_results = self._apply_score_boosting(
            wrapped_results,
            original_query,
            query_vector
        )

        # 4. Rerank top results (skip if reranker not available)
        reranked_results = self._rerank(
            boosted_results[:top_k * 2],
            original_query,
            top_k
        )

        # 5. Calculate confidence
        confidence = self._calculate_confidence(reranked_results)

        # 6. Format chunks
        chunks = [
            {
                "id": r.id,
                "text": r.payload.get("text", ""),
                "metadata": r.payload.get("metadata", {}),
                "score": r.score,
                "boosted": r.payload.get("boosted", False)
            }
            for r in reranked_results
        ]

        logger.info(f"Retrieved {len(chunks)} chunks, confidence={confidence:.3f}")
        return chunks, confidence

    def _apply_score_boosting(
        self,
        results: List[SearchResult],
        query: str,
        query_vector: np.ndarray
    ) -> List[SearchResult]:
        """
        CRITICAL: Boost scores to avoid rejecting near-correct answers
        """
        query_keywords = set(query.lower().split())

        for result in results:
            original_score = result.score
            boost = 0.0

            # Boost 1: Skip vector-based boost (no vectors available)

            # Boost 2: Exact keyword match
            text = result.payload.get("text", "").lower()
            keyword_matches = sum(1 for kw in query_keywords if kw in text)
            if len(query_keywords) > 0 and keyword_matches >= len(query_keywords) * 0.7:
                boost += 0.1
                logger.debug(f"Applied keyword boost +0.1 ({keyword_matches} matches)")

            # Boost 3: Source credibility (official documents)
            info_type = result.payload.get("metadata", {}).get("info_type", "")
            if info_type == "van_ban_phap_ly":
                boost += 0.05
                logger.debug(f"Applied source boost +0.05 (official doc)")

            # Apply boost
            if boost > 0:
                result.score = min(original_score + boost, 1.0)
                result.payload["boosted"] = True
                result.payload["original_score"] = original_score
                logger.info(f"Score boosted: {original_score:.3f} → {result.score:.3f}")

        # Re-sort by boosted scores
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _rerank(
        self,
        results: List[SearchResult],
        query: str,
        top_k: int
    ) -> List[SearchResult]:
        """
        Rerank (skipped if reranker not available)
        """
        if not results:
            return []

        # If reranker not available, just sort and return
        if self.reranker is None:
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

        # Prepare pairs for reranking
        pairs = [(query, r.payload.get("text", "")) for r in results]

        # Rerank scores
        rerank_scores = self.reranker.predict(pairs)

        # Combine with original scores (weighted)
        for i, result in enumerate(results):
            original = result.score
            rerank = float(rerank_scores[i])

            # Weighted combination
            result.score = 0.6 * original + 0.4 * rerank

        # Sort and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _calculate_confidence(self, results: List[SearchResult]) -> float:
        """
        Calculate ensemble confidence score
        """
        if not results:
            return 0.0

        # Weighted average of top 3 scores
        top_scores = [r.score for r in results[:3]]

        if len(top_scores) == 1:
            return top_scores[0]
        elif len(top_scores) == 2:
            return 0.7 * top_scores[0] + 0.3 * top_scores[1]
        else:
            return 0.5 * top_scores[0] + 0.3 * top_scores[1] + 0.2 * top_scores[2]


# Global instance
bge_retriever = BGEMultiVectorRetriever()
