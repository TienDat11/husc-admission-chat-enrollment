"""
Embedding Service (Qwen3 Embedding)

Manages embedding generation with:
- Batch encoding for efficiency
- Dimension validation
- Qwen3 instruction-aware query/document prompting
"""
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import RAGSettings


class EmbeddingService:
    """Embedding service for Qwen3-Embedding models."""

    def __init__(self, settings: RAGSettings):
        self.settings = settings
        self.model_name = settings.EMBEDDING_MODEL
        self.expected_dim = settings.EMBEDDING_DIM
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self.normalize = settings.EMBEDDING_NORMALIZE

        logger.info(f"Loading embedding model: {self.model_name}")

        # Load model with CPU optimization for Qwen3
        self.model = SentenceTransformer(
            self.model_name,
            device="cpu",
            model_kwargs={"torch_dtype": "auto"}
        )

        self._validate_model_dimension()

    def _validate_model_dimension(self) -> None:
        test_embedding = self.model.encode(
            "test",
            normalize_embeddings=False,
        )
        actual_dim = len(test_embedding)
        if actual_dim != self.expected_dim:
            raise ValueError(
                f"Model dimension mismatch: expected {self.expected_dim}, got {actual_dim}"
            )

    def encode_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.expected_dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ).astype(np.float32)

        if embeddings.shape[1] != self.expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.expected_dim}, got {embeddings.shape[1]}"
            )

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode_batch([text], show_progress=False)[0]

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode query with Qwen3 instruction prompt.

        Uses built-in "query" prompt which applies:
        "Instruct: {task}\nQuery: {query}"

        This is CRITICAL for Qwen3 retrieval quality.
        """
        embedding = self.model.encode(
            query,
            prompt_name="query",  # CRITICAL for Qwen3
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        ).astype(np.float32)

        return embedding

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode documents WITHOUT instruction (Qwen3 recommendation).

        No prompt_name = default encoding (no instruction prefix).
        """
        if not documents:
            return np.empty((0, self.expected_dim), dtype=np.float32)

        embeddings = self.model.encode(
            documents,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        if embeddings.shape[1] != self.expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.expected_dim}, got {embeddings.shape[1]}"
            )

        return embeddings

    def save_embeddings(self, embeddings: np.ndarray, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embeddings)

    def load_embeddings(self, input_path: Path) -> np.ndarray:
        embeddings = np.load(input_path)
        if embeddings.shape[1] != self.expected_dim:
            raise ValueError(
                f"Loaded embeddings dimension mismatch: expected {self.expected_dim}, got {embeddings.shape[1]}"
            )
        return embeddings
