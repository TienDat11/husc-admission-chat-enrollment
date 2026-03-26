"""
Vector Store Module (Epic 2.2)

Abstract interface + implementations for vector storage and retrieval.
Supports:
- NumPy/FAISS in-memory store (low-data mode)
- Optional Qdrant integration (high-scale mode)
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Search result model."""

    doc_id: str = Field(description="Document ID")
    chunk_id: int = Field(description="Chunk ID within document")
    score: float = Field(description="Similarity score")
    text: str = Field(description="Chunk text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")


class VectorStore(ABC):
    """
    Abstract base class for vector stores.
    
    All implementations must support:
    - Adding vectors with metadata
    - Dense similarity search
    - Persistence (save/load)
    """

    @abstractmethod
    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Add vectors to store.

        Args:
            vectors: numpy array of shape (n, dim)
            ids: List of unique IDs
            metadatas: List of metadata dicts
        """
        pass

    @abstractmethod
    def search(
        self, query_vector: np.ndarray, top_k: int = 20
    ) -> List[SearchResult]:
        """
        Dense similarity search.

        Args:
            query_vector: Query embedding of shape (dim,)
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save store to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load store from disk."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return number of vectors in store."""
        pass


class NumpyVectorStore(VectorStore):
    """
    In-memory vector store using NumPy.
    
    Features:
    - Cosine similarity search
    - File-backed persistence (.npz format)
    - Memory-efficient for <10k vectors
    - Auto-scaling top_k based on corpus size
    """

    def __init__(self, dim: int = 768):
        """
        Initialize NumPy vector store.

        Args:
            dim: Embedding dimension
        """
        self.dim = dim
        self.vectors: np.ndarray | None = None
        self.ids: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

        logger.info(f"NumpyVectorStore initialized with dim={dim}")

    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Add vectors to store.

        Args:
            vectors: numpy array of shape (n, dim)
            ids: List of unique IDs
            metadatas: List of metadata dicts
        """
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dim}, "
                f"got {vectors.shape[1]}"
            )

        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"IDs length mismatch: {len(ids)} IDs for {vectors.shape[0]} vectors"
            )

        if len(metadatas) != vectors.shape[0]:
            raise ValueError(
                f"Metadatas length mismatch: {len(metadatas)} metadatas for "
                f"{vectors.shape[0]} vectors"
            )

        # Initialize or concatenate
        if self.vectors is None:
            self.vectors = vectors.astype(np.float32)
            self.ids = ids
            self.metadatas = metadatas
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
            self.ids.extend(ids)
            self.metadatas.extend(metadatas)

        logger.info(f"Added {len(ids)} vectors (total: {len(self.ids)})")

    def search(
        self, query_vector: np.ndarray, top_k: int = 20
    ) -> List[SearchResult]:
        """
        Dense cosine similarity search.

        Args:
            query_vector: Query embedding of shape (dim,)
            top_k: Number of results to return

        Returns:
            List of SearchResult objects sorted by score (descending)
        """
        if self.vectors is None or len(self.ids) == 0:
            logger.warning("Empty vector store, returning no results")
            return []

        # Ensure query vector shape
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Normalize query vector
        query_norm = query_vector / (
            np.linalg.norm(query_vector, axis=1, keepdims=True) + 1e-8
        )

        # Normalize corpus vectors (cached)
        if not hasattr(self, "_vectors_norm"):
            self._vectors_norm = self.vectors / (
                np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8
            )

        # Compute cosine similarity
        scores = np.dot(self._vectors_norm, query_norm.T).flatten()

        # Auto-scale top_k
        top_k = min(top_k, len(self.ids))

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results
        results: List[SearchResult] = []
        for idx in top_indices:
            idx_int = int(idx)
            metadata = self.metadatas[idx_int]

            result = SearchResult(
                doc_id=metadata.get("doc_id", "unknown"),
                chunk_id=metadata.get("chunk_id", 0),
                score=float(scores[idx_int]),
                text=metadata.get("text", ""),
                metadata=metadata,
            )
            results.append(result)

        logger.debug(
            f"Search returned {len(results)} results (top score: "
            f"{results[0].score:.3f})"
        )
        return results

    def save(self, path: Path) -> None:
        """
        Save store to .npz file.

        Args:
            path: Path to save .npz file
        """
        if self.vectors is None:
            logger.warning("Empty vector store, nothing to save")
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        # Save vectors, ids, and metadatas
        np.savez_compressed(
            path,
            vectors=self.vectors,
            ids=np.array(self.ids, dtype=object),
            metadatas=np.array(self.metadatas, dtype=object),
            dim=self.dim,
        )

        logger.info(
            f"Saved NumpyVectorStore: {len(self.ids)} vectors to {path}"
        )

    def load(self, path: Path) -> None:
        """
        Load store from .npz file.

        Args:
            path: Path to .npz file
        """
        if not path.exists():
            raise FileNotFoundError(f"Vector store not found: {path}")

        data = np.load(path, allow_pickle=True)

        self.vectors = data["vectors"].astype(np.float32)
        self.ids = data["ids"].tolist()
        self.metadatas = data["metadatas"].tolist()
        self.dim = int(data["dim"])

        # Clear cached normalized vectors
        if hasattr(self, "_vectors_norm"):
            delattr(self, "_vectors_norm")

        logger.info(
            f"Loaded NumpyVectorStore: {len(self.ids)} vectors from {path}"
        )

    def count(self) -> int:
        """Return number of vectors in store."""
        return len(self.ids)

    def get_by_id(self, doc_id: str) -> List[SearchResult]:
        """
        Retrieve all chunks for a given document ID.

        Args:
            doc_id: Document ID

        Returns:
            List of SearchResult objects
        """
        results: List[SearchResult] = []

        for i, metadata in enumerate(self.metadatas):
            if metadata.get("doc_id") == doc_id:
                result = SearchResult(
                    doc_id=metadata.get("doc_id", "unknown"),
                    chunk_id=metadata.get("chunk_id", 0),
                    score=1.0,  # No scoring for direct retrieval
                    text=metadata.get("text", ""),
                    metadata=metadata,
                )
                results.append(result)

        return results


class QdrantVectorStore(VectorStore):
    """
    Qdrant vector store implementation using Phase 7 retrieval interface.

    Compliant with qdrant-client >= 1.7.x using ONLY query_points() API.

    Features:
    - BGE-M3 (1024-dim) support
    - Fail-safe error handling
    - Multi-vector ready architecture
    - Proper payload validation per Phase 7 spec

    Payload structure required:
    {
        "text": "string",           # MANDATORY
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

    # Minimum required Qdrant version
    MIN_QDRANT_VERSION = "1.7.0"

    def __init__(
        self,
        url: str,
        api_key: str | None,
        collection_name: str,
        embedding_dim: int = 1024,
        create_collection: bool = False,
    ):
        """
        Initialize Qdrant vector store.

        Args:
            url: Qdrant server URL
            api_key: Optional API key for Qdrant Cloud
            collection_name: Name of the collection
            embedding_dim: Vector dimension (default 1024 for BGE-M3)
            create_collection: Create collection if not exists
        """
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams, CreateCollection
        import qdrant_client

        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        # Validate Qdrant client (check if module is available)
        # qdrant_client doesn't expose __version__ attribute
        # We assume the installed version is compatible if imports work
        logger.debug(f"Qdrant client loaded successfully")

        self._client = QdrantClient(url=url, api_key=api_key)

        # Optionally create collection
        if create_collection:
            self._ensure_collection()

        logger.info(
            f"QdrantVectorStore initialized: collection={collection_name}, "
            f"url={url}, dim={embedding_dim}"
        )

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        from qdrant_client.http.models import CreateCollection, Distance, VectorParams

        collections = self._client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Collection created: {self.collection_name}")
        else:
            logger.info(f"Using existing collection: {self.collection_name}")

    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Add vectors to Qdrant collection.

        Uses upsert_batch for efficient batch insertion.

        Args:
            vectors: numpy array of shape (n, dim)
            ids: List of unique IDs
            metadatas: List of metadata dicts with required structure
        """
        from qdrant_client.http.models import PointStruct

        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.embedding_dim}, "
                f"got {vectors.shape[1]}"
            )

        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"IDs length mismatch: {len(ids)} IDs for {vectors.shape[0]} vectors"
            )

        if len(metadatas) != vectors.shape[0]:
            raise ValueError(
                f"Metadatas length mismatch: {len(metadatas)} metadatas for "
                f"{vectors.shape[0]} vectors"
            )

        # Prepare points
        points = []
        for i, (id_, vector, metadata) in enumerate(zip(ids, vectors, metadatas)):
            # Validate mandatory 'text' field
            if "text" not in metadata:
                logger.warning(f"Skipping point {id_}: missing mandatory 'text' field")
                continue

            points.append(
                PointStruct(
                    id=id_,
                    vector=vector.tolist(),
                    payload=metadata,
                )
            )

        if not points:
            logger.warning("No valid points to insert (check payload structure)")
            return

        # Batch upsert
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        logger.info(f"Added {len(points)} vectors to Qdrant collection")

    def search(
        self, query_vector: np.ndarray, top_k: int = 20
    ) -> List[SearchResult]:
        """
        Search Qdrant using query_points() API (Phase 7 compliant).

        Args:
            query_vector: Query embedding of shape (dim,)
            top_k: Number of results to return

        Returns:
            List of SearchResult objects sorted by score (descending)
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Normalize query vector
        query_vector = query_vector / (
            np.linalg.norm(query_vector, axis=1, keepdims=True) + 1e-8
        )

        # Use query_points API (MANDATORY - not search/search_batch)
        # The query parameter accepts list[float] directly for dense vector queries
        try:
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector[0].tolist(),  # Direct dense vector query
                with_payload=True,
                with_vector=False,
                limit=top_k,
            )
        except Exception as e:
            logger.error(f"Qdrant query_points failed: {e}")
            return []

        # Normalize results
        results: List[SearchResult] = []

        if not response or not hasattr(response, "points"):
            return results

        for point in response.points:
            payload = point.payload or {}

            # text is MANDATORY
            text = payload.get("text", "")
            if not text:
                continue

            result = SearchResult(
                doc_id=str(point.id),
                chunk_id=payload.get("chunk_id", 0),
                score=float(point.score) if point.score is not None else 0.0,
                text=text,
                metadata=payload.get("metadata", {}),
            )
            results.append(result)

        logger.debug(
            f"Qdrant search returned {len(results)} results "
            f"(top score: {results[0].score:.3f if results else 0:.3f})"
        )
        return results

    def save(self, path: Path) -> None:
        """Qdrant persistence is handled server-side."""
        logger.info("Qdrant persistence is server-side, no local save needed")

    def load(self, path: Path) -> None:
        """Qdrant persistence is handled server-side."""
        logger.info("Qdrant persistence is server-side, no local load needed")

    def count(self) -> int:
        """Return number of vectors in store."""
        try:
            collection_info = self._client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return 0

    def delete_by_id(self, doc_id: str) -> bool:
        """
        Delete points by document ID.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            # Note: This requires payload index on doc_id for efficiency
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=[doc_id],
            )
            logger.info(f"Deleted point {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete point {doc_id}: {e}")
            return False
