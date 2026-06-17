"""
Embedding Service

Supports multiple embedding families:
- Qwen3
- Harrier
- BGE

@spec(S10.B2) — Lazy loading: model loaded on first encode call, not at import/init.
"""
import threading
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger

from config.settings import RAGSettings


class EmbeddingService:
    """Embedding service for Qwen3/Harrier/BGE models.
    @spec(S10.B2)

    Model is loaded lazily on first encode call to keep startup < 5s.
    Thread-safe via threading.Lock (works with run_in_threadpool callers).
    """

    def __init__(self, settings: RAGSettings):
        self.settings = settings
        self.model_name = settings.EMBEDDING_MODEL
        self.expected_dim = settings.EMBEDDING_DIM
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self.normalize = settings.EMBEDDING_NORMALIZE
        self.provider = settings.EMBEDDING_PROVIDER

        # @spec(S10.B2): lazy — model loaded on first use, not at init
        self._model = None
        self._lock = threading.Lock()

        logger.info(
            f"EmbeddingService configured (lazy): {self.model_name} "
            f"(provider={self.provider}, dim={self.expected_dim})"
        )

    def _ensure_loaded(self) -> None:
        """Load model on first call. Thread-safe double-check locking."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:  # double-check after acquiring lock
                return
            logger.info(f"Lazy-loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device="cpu",
                model_kwargs={"torch_dtype": "auto"},
            )
            self._validate_model_dimension()
            logger.info(f"Embedding model loaded: {self.model_name}")

    def _validate_model_dimension(self) -> None:
        test_embedding = self._model.encode(
            "test",
            normalize_embeddings=False,
        )
        actual_dim = len(test_embedding)
        if actual_dim != self.expected_dim:
            raise ValueError(
                f"Model dimension mismatch: expected {self.expected_dim}, got {actual_dim}"
            )

    @property
    def is_loaded(self) -> bool:
        """Whether the embedding model has been loaded."""
        return self._model is not None

    def encode_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        self._ensure_loaded()
        if not texts:
            return np.empty((0, self.expected_dim), dtype=np.float32)

        embeddings = self._model.encode(
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

    def _query_prompt_name(self) -> str | None:
        provider = (self.provider or "").lower()
        model = (self.model_name or "").lower()

        if provider == "qwen" or "qwen" in model:
            return "query"

        if provider == "harrier" or "harrier" in model:
            return "web_search_query"

        return None

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode query with provider-specific prompt strategy.

        - Qwen: prompt_name="query"
        - Harrier: prompt_name="web_search_query"
        - BGE/others: no prompt_name
        """
        self._ensure_loaded()
        prompt_name = self._query_prompt_name()

        encode_kwargs = {
            "normalize_embeddings": self.normalize,
            "convert_to_numpy": True,
        }
        if prompt_name:
            encode_kwargs["prompt_name"] = prompt_name

        embedding = self._model.encode(
            query,
            **encode_kwargs,
        ).astype(np.float32)

        if embedding.shape[0] != self.expected_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.expected_dim}, got {embedding.shape[0]}"
            )

        return embedding

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode documents without instruction prompts.
        """
        self._ensure_loaded()
        if not documents:
            return np.empty((0, self.expected_dim), dtype=np.float32)

        embeddings = self._model.encode(
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
