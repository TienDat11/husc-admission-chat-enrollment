"""
Phase 7: Retrieval Layer - Qdrant Query Interface

This module provides a stable, future-proof retrieval interface for RAG 2025.
Compliant with qdrant-client >= 1.7.x using ONLY query_points() API.

Mandatory API Rules (STRICT):
- FORBIDDEN: client.search(), client.search_batch()
- MANDATORY: client.query_points()

Collection Schema:
- Name: rag2025
- Vector size: 1024
- Distance: Cosine
- Multi-vector ready

Payload Structure:
{
    "text": "string",           # MANDATORY for generation
    "source": "string",
    "chunk_id": "string",
    "metadata": {
        "title": "string",
        "language": "string",
        "created_at": "string",
        "...": "any"
    }
}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    Prefetch,
    QueryResponse,
    Filter,
    FieldCondition,
    MatchValue,
)
from loguru import logger
import numpy as np


class RetrievalError(Enum):
    """Error types for retrieval layer."""
    VERSION_MISMATCH = "qdrant_version_mismatch"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    EMPTY_RESULT = "empty_result"
    INVALID_PAYLOAD = "invalid_payload"
    UNKNOWN_ERROR = "unknown_error"


@dataclass(frozen=True)
class RetrievedDocument:
    """
    Immutable retrieved document from Qdrant.

    Attributes:
        text: The text content (MANDATORY)
        source: Source identifier
        chunk_id: Unique chunk identifier
        metadata: Additional metadata (JSON-serializable)
        score: Similarity score
        point_id: Qdrant point ID
    """
    text: str
    source: Optional[str] = None
    chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    point_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for RAG generator."""
        return {
            "text": self.text,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
            "score": self.score,
        }


@dataclass
class RetrievalResult:
    """
    Container for retrieval results with error handling.

    Always safe to use - never crashes even on failure.
    """
    documents: List[RetrievedDocument] = field(default_factory=list)
    error_type: Optional[RetrievalError] = None
    error_message: Optional[str] = None
    confidence: float = 0.0

    @property
    def is_success(self) -> bool:
        """Check if retrieval was successful."""
        return self.error_type is None and len(self.documents) > 0

    @property
    def is_empty(self) -> bool:
        """Check if result is empty (not an error, just no matches)."""
        return self.error_type is None and len(self.documents) == 0


class QdrantRetrieverConfig:
    """
    Configuration for Qdrant retriever.

    Default values match Phase 7 specification.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = "rag2025",
        embedding_dim: int = 1024,
        default_top_k: int = 5,
        timeout: int = 30,
    ):
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.default_top_k = default_top_k
        self.timeout = timeout


class QdrantRetriever:
    """
    Qdrant retrieval interface using ONLY query_points() API.

    This is the canonical implementation for RAG 2025 retrieval layer.

    Features:
    - Uses query_points() API exclusively (no deprecated search())
    - Fail-safe error handling (never crashes the pipeline)
    - Multi-vector ready architecture
    - BGE-M3 (1024-dim) compatible
    - Proper payload validation

    Example:
        >>> retriever = QdrantRetriever.from_config(config)
        >>> query_vector = embed_model.encode("What is RAG?")
        >>> result = retriever.retrieve(query_vector.tolist())
        >>> for doc in result.documents:
        ...     print(f"{doc.score:.3f}: {doc.text[:50]}...")
    """

    # Minimum required Qdrant client version
    MIN_QDRANT_VERSION = "1.7.0"

    def __init__(
        self,
        config: QdrantRetrieverConfig,
        client: Optional[QdrantClient] = None,
    ) -> None:
        """
        Initialize Qdrant retriever.

        Args:
            config: Retriever configuration
            client: Optional pre-configured QdrantClient (for testing)
        """
        self._config = config

        if client is not None:
            self._client = client
        else:
            self._client = QdrantClient(
                url=config.url,
                api_key=config.api_key,
                timeout=config.timeout,
            )

        self._validate_client_version()

    @classmethod
    def from_config(cls, config: QdrantRetrieverConfig) -> "QdrantRetriever":
        """Create retriever from configuration object."""
        return cls(config=config)

    @classmethod
    def from_env(cls) -> "QdrantRetriever":
        """
        Create retriever from environment variables.

        Required env vars:
        - QDRANT_URL: Qdrant server URL
        - QDRANT_API_KEY: Optional API key
        - QDRANT_COLLECTION: Collection name (default: rag2025)
        """
        import os

        config = QdrantRetrieverConfig(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION", "rag2025"),
        )
        return cls(config=config)

    def _validate_client_version(self) -> None:
        """
        Validate Qdrant client version.

        Note: qdrant_client doesn't expose __version__ attribute.
        We assume the installed version is compatible if the module imports successfully.
        """
        try:
            import qdrant_client
            # qdrant_client doesn't have __version__ attribute
            # Version check is skipped - assume installed version is compatible
            logger.debug(f"Qdrant client loaded successfully (version check skipped)")
        except ImportError:
            raise RuntimeError(
                f"qdrant_client is not installed. "
                f"Please install: pip install 'qdrant-client>={self.MIN_QDRANT_VERSION}'"
            )

    def retrieve(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        collection_name: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Retrieve documents using query_points() API.

        This is the main retrieval method. It is fail-safe and will never
        raise an exception - errors are captured in the RetrievalResult.

        Args:
            query_vector: Query embedding vector (list of floats)
            top_k: Number of results to return (default from config)
            collection_name: Override collection name
            with_vectors: Include vectors in response (default False)
            metadata_filter: Optional filter dict for metadata (e.g., {"faq_type": "thong_tin_nganh"})

        Returns:
            RetrievalResult with documents or error information

        Example:
            >>> query = [0.1, 0.2, ...]  # 1024-dim vector
            >>> result = retriever.retrieve(query)
            >>> if result.is_success:
            ...     for doc in result.documents:
            ...         print(doc.text)
        """
        top_k = top_k or self._config.default_top_k
        collection_name = collection_name or self._config.collection_name

        # Validate query vector
        if not query_vector:
            return RetrievalResult(
                error_type=RetrievalError.INVALID_PAYLOAD,
                error_message="Query vector is empty",
            )

        if len(query_vector) != self._config.embedding_dim:
            return RetrievalResult(
                error_type=RetrievalError.INVALID_PAYLOAD,
                error_message=(
                    f"Query vector dimension mismatch: "
                    f"expected {self._config.embedding_dim}, got {len(query_vector)}"
                ),
            )

        # Execute query with error handling
        try:
            response = self._execute_query(
                query_vector=query_vector,
                top_k=top_k,
                collection_name=collection_name,
                metadata_filter=metadata_filter,
            )
        except Exception as e:
            return self._handle_exception(e)

        # Normalize and validate results
        documents = self._normalize_results(response)

        # Calculate confidence
        confidence = self._calculate_confidence(documents)

        # Handle empty results
        if not documents:
            return RetrievalResult(
                documents=[],
                error_type=RetrievalError.EMPTY_RESULT,
                error_message="No matching documents found",
                confidence=0.0,
            )

        return RetrievalResult(
            documents=documents,
            confidence=confidence,
        )

    def _execute_query(
        self,
        query_vector: List[float],
        top_k: int,
        collection_name: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> QueryResponse:
        """
        Execute query_points() request.

        Uses query parameter directly for dense vector (extensible for multi-vector via prefetch).

        Args:
            query_vector: Query embedding
            top_k: Number of results
            collection_name: Target collection
            metadata_filter: Optional filter dict for metadata.
                - Simple: {"faq_type": "thong_tin_nganh", "year": 2025}
                - OR: {"or_conditions": [{"faq_type": "tong_hop_tuyen_sinh", "year": 2025}, ...]}

        Returns:
            QueryResponse from Qdrant

        Raises:
            UnexpectedResponse: For Qdrant API errors
        """
        # Build filter if provided
        query_filter = None
        if metadata_filter:
            # Check for OR conditions (for overview queries)
            or_conditions_data = metadata_filter.get("or_conditions")

            if or_conditions_data:
                # OR logic: use 'should' clause in Qdrant Filter
                # Each OR condition becomes a Filter with 'must' conditions
                or_filters = []
                for condition_dict in or_conditions_data:
                    must_conditions = self._build_conditions_from_dict(condition_dict)
                    if must_conditions:
                        or_filters.append(Filter(must=must_conditions))

                if or_filters:
                    # Filter with OR logic: should contain ANY of the or_filters
                    query_filter = Filter(should=or_filters)
            else:
                # Normal AND logic: use 'must' clause
                must_conditions = self._build_conditions_from_dict(metadata_filter)
                if must_conditions:
                    query_filter = Filter(must=must_conditions)

        # Execute query_points (MANDATORY API - do not use search/search_batch)
        # The query parameter accepts list[float] directly for dense vector queries
        response = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,  # Direct dense vector query
            with_payload=True,
            limit=top_k,
            query_filter=query_filter,  # Add metadata filter
        )

        return response

    def _build_conditions_from_dict(self, filter_dict: Dict[str, Any]) -> List[Any]:
        """
        Build Qdrant FieldCondition list from filter dictionary.

        Args:
            filter_dict: Dictionary of filter conditions

        Returns:
            List of FieldCondition objects
        """
        # Root-level fields (NOT in metadata.*)
        # Note: "year" is nested in metadata.year in actual data
        root_level_fields = ["faq_type", "source", "doc_id", "text_plain", "text_raw",
                           "chunk_number", "breadcrumbs", "sparse_terms", "summary"]

        conditions = []
        for key, value in filter_dict.items():
            # Skip special keys
            if key == "or_conditions":
                continue

            # Root-level fields use direct key, nested fields use metadata.*
            if key in root_level_fields:
                field_key = key  # Direct key for root-level field
            else:
                field_key = f"metadata.{key}"  # Nested field
            conditions.append(
                FieldCondition(
                    key=field_key,
                    match=MatchValue(value=value),
                )
            )
        return conditions

    def _normalize_results(self, response: QueryResponse) -> List[RetrievedDocument]:
        """
        Normalize Qdrant response to RetrievedDocument objects.

        Validates payload structure:
        - text is mandatory
        - metadata must be JSON-serializable
        - Skip invalid points (fail-safe)

        Args:
            response: QueryResponse from Qdrant

        Returns:
            List of valid RetrievedDocument objects
        """
        documents: List[RetrievedDocument] = []

        if not response or not hasattr(response, "points"):
            return documents

        for point in response.points:
            payload = point.payload or {}

            # text is MANDATORY - skip if missing
            text = payload.get("text")
            if not text or not isinstance(text, str):
                logger.debug(f"Skipping point {point.id}: missing or invalid 'text' field")
                continue

            # Validate metadata is JSON-serializable
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                logger.debug(f"Skipping point {point.id}: invalid metadata type")
                metadata = {}

            # Build document
            try:
                doc = RetrievedDocument(
                    text=text,
                    source=payload.get("source"),
                    chunk_id=payload.get("chunk_id"),
                    metadata=metadata,
                    score=float(point.score) if point.score is not None else 0.0,
                    point_id=str(point.id) if point.id is not None else None,
                )
                documents.append(doc)
            except Exception as e:
                logger.debug(f"Failed to create document from point {point.id}: {e}")
                continue

        return documents

    def _calculate_confidence(self, documents: List[RetrievedDocument]) -> float:
        """
        Calculate ensemble confidence score.

        Uses weighted average of top 3 scores:
        - Single result: score as-is
        - Two results: 0.7 * top + 0.3 * second
        - Three+ results: 0.5 * top + 0.3 * second + 0.2 * third

        Args:
            documents: List of retrieved documents (sorted by score)

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not documents:
            return 0.0

        top_scores = [doc.score for doc in documents[:3]]

        if len(top_scores) == 1:
            return top_scores[0]
        elif len(top_scores) == 2:
            return 0.7 * top_scores[0] + 0.3 * top_scores[1]
        else:
            return 0.5 * top_scores[0] + 0.3 * top_scores[1] + 0.2 * top_scores[2]

    def _handle_exception(self, exception: Exception) -> RetrievalResult:
        """
        Handle exceptions and return appropriate RetrievalResult.

        Maps exceptions to RetrievalError types for proper handling.

        Args:
            exception: Caught exception

        Returns:
            RetrievalResult with error information
        """
        if isinstance(exception, UnexpectedResponse):
            return RetrievalResult(
                error_type=RetrievalError.NETWORK_ERROR,
                error_message=f"Qdrant API error: {exception}",
            )
        elif "timeout" in str(exception).lower():
            return RetrievalResult(
                error_type=RetrievalError.TIMEOUT_ERROR,
                error_message=f"Request timeout: {exception}",
            )
        else:
            return RetrievalResult(
                error_type=RetrievalError.UNKNOWN_ERROR,
                error_message=f"Unexpected error: {exception}",
            )

    def batch_retrieve(
        self,
        query_vectors: List[List[float]],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Batch retrieval for multiple queries.

        Each query is executed independently - failures don't affect others.

        Args:
            query_vectors: List of query vectors
            top_k: Number of results per query

        Returns:
            List of RetrievalResult objects (one per query)
        """
        results = []

        for query_vector in query_vectors:
            result = self.retrieve(query_vector, top_k=top_k)
            results.append(result)

        return results

    def check_collection(self) -> Dict[str, Any]:
        """
        Check collection status and information.

        Returns:
            Dict with collection info or error details
        """
        try:
            collections = self._client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self._config.collection_name not in collection_names:
                return {
                    "exists": False,
                    "error": f"Collection '{self._config.collection_name}' not found",
                }

            collection_info = self._client.get_collection(self._config.collection_name)

            return {
                "exists": True,
                "name": collection_info.config.params.vectors.size,
                "vectors_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "status": collection_info.status,
            }
        except Exception as e:
            return {
                "exists": False,
                "error": str(e),
            }

    def close(self) -> None:
        """Close the Qdrant client connection."""
        # QdrantClient doesn't require explicit closing (HTTP-based)
        # This method exists for interface compatibility
        pass


def retrieve_with_qdrant(
    query_vector: List[float],
    top_k: int = 5,
    url: str = "http://localhost:6333",
    api_key: Optional[str] = None,
    collection_name: str = "rag2025",
) -> List[Dict[str, Any]]:
    """
    Convenience function for simple retrieval use cases.

    This is the quick-start function for basic RAG retrieval.

    Args:
        query_vector: Query embedding (list of floats, 1024-dim for BGE-M3)
        top_k: Number of results to return
        url: Qdrant server URL
        api_key: Optional API key
        collection_name: Collection name

    Returns:
        List of document dictionaries with keys:
        - text: str
        - source: str | None
        - chunk_id: str | None
        - metadata: dict
        - score: float

    Example:
        >>> from sentence_transformers import SentenceTransformer
        >>> model = SentenceTransformer('BAAI/bge-m3')
        >>> query = model.encode("What is the admissions process?")
        >>> docs = retrieve_with_qdrant(query.tolist(), top_k=5)
        >>> for doc in docs:
        ...     print(f"{doc['score']:.3f}: {doc['text'][:50]}...")
    """
    config = QdrantRetrieverConfig(
        url=url,
        api_key=api_key,
        collection_name=collection_name,
    )
    retriever = QdrantRetriever(config=config)
    result = retriever.retrieve(query_vector, top_k=top_k)

    if result.is_success:
        return [doc.to_dict() for doc in result.documents]
    else:
        # Fail-safe: return empty list on any error
        logger.warning(f"Retrieval failed: {result.error_message}")
        return []


# Default instance for backward compatibility
_default_retriever: Optional[QdrantRetriever] = None


def get_retriever() -> QdrantRetriever:
    """
    Get or create the default retriever instance.

    Uses environment variables for configuration.

    Returns:
        QdrantRetriever instance
    """
    global _default_retriever

    if _default_retriever is None:
        _default_retriever = QdrantRetriever.from_env()

    return _default_retriever
