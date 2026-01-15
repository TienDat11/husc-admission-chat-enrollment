"""
Embedding Service (Epic 2.1)

Manages embedding generation with:
- Batch encoding for efficiency
- Dimension validation (768-dim enforcement)
- Model lifecycle management
- Optional caching
"""
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import RAGSettings


class EmbeddingService:
    """
    Embedding service with batching and validation.
    
    Supports:
    - intfloat/e5-small-v2 (768-dim)
    - intfloat/multilingual-e5-base (768-dim)
    - Batch encoding with configurable batch size
    - L2 normalization
    - Dimension validation
    """

    def __init__(self, settings: RAGSettings):
        """
        Initialize embedding service.

        Args:
            settings: RAGSettings instance with model config
        """
        self.settings = settings
        self.model_name = settings.EMBEDDING_MODEL
        self.expected_dim = settings.EMBEDDING_DIM
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self.normalize = settings.EMBEDDING_NORMALIZE

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        # Validate model dimension
        self._validate_model_dimension()

        logger.info(
            f"EmbeddingService initialized: model={self.model_name}, "
            f"dim={self.expected_dim}, batch_size={self.batch_size}"
        )

    def _validate_model_dimension(self) -> None:
        """
        Validate that model outputs expected dimension.
        
        Raises:
            ValueError: If dimension mismatch
        """
        # Encode test string
        test_embedding = self.model.encode("test", normalize_embeddings=False)
        actual_dim = len(test_embedding)

        if actual_dim != self.expected_dim:
            raise ValueError(
                f"Model dimension mismatch: expected {self.expected_dim}, "
                f"got {actual_dim} for model {self.model_name}"
            )

        logger.info(f"Model dimension validated: {actual_dim}")

    def encode_batch(
        self, texts: List[str], show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode batch of texts to embeddings.

        Args:
            texts: List of text strings to encode
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.empty((0, self.expected_dim), dtype=np.float32)

        logger.debug(f"Encoding batch of {len(texts)} texts")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        # Ensure float32 for memory efficiency
        embeddings = embeddings.astype(np.float32)

        # Validate dimension
        if embeddings.shape[1] != self.expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.expected_dim}, "
                f"got {embeddings.shape[1]}"
            )

        logger.debug(f"Encoded {len(texts)} texts to shape {embeddings.shape}")
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode single text to embedding.

        Args:
            text: Text string to encode

        Returns:
            numpy array of shape (embedding_dim,)
        """
        embeddings = self.encode_batch([text], show_progress=False)
        return embeddings[0]

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode query with optional query prefix (for e5 models).

        Args:
            query: Query string

        Returns:
            numpy array of shape (embedding_dim,)
        """
        # E5 models benefit from "query:" prefix
        if "e5" in self.model_name.lower():
            query = f"query: {query}"

        return self.encode_single(query)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode documents with optional document prefix (for e5 models).

        Args:
            documents: List of document strings

        Returns:
            numpy array of shape (len(documents), embedding_dim)
        """
        # E5 models benefit from "passage:" prefix
        if "e5" in self.model_name.lower():
            documents = [f"passage: {doc}" for doc in documents]

        return self.encode_batch(documents, show_progress=True)

    def save_embeddings(
        self, embeddings: np.ndarray, output_path: Path
    ) -> None:
        """
        Save embeddings to .npy file.

        Args:
            embeddings: numpy array of embeddings
            output_path: Path to save .npy file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embeddings)
        logger.info(
            f"Saved embeddings: shape={embeddings.shape} to {output_path}"
        )

    def load_embeddings(self, input_path: Path) -> np.ndarray:
        """
        Load embeddings from .npy file.

        Args:
            input_path: Path to .npy file

        Returns:
            numpy array of embeddings
        """
        embeddings = np.load(input_path)
        logger.info(
            f"Loaded embeddings: shape={embeddings.shape} from {input_path}"
        )

        # Validate dimension
        if embeddings.shape[1] != self.expected_dim:
            raise ValueError(
                f"Loaded embeddings dimension mismatch: expected "
                f"{self.expected_dim}, got {embeddings.shape[1]}"
            )

        return embeddings
