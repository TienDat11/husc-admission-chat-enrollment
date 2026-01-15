"""
Configuration settings using Pydantic BaseSettings.
All environment variables are loaded here with strict typing.
"""
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class RAGSettings(BaseSettings):
    """
    2025 RAG System Configuration.
    Loads from .env file and environment variables.
    """

    # Pydantic v2 config
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    # ========== Paths ==========
    INDEX_DIR: Path = Field(default=Path("./index"), description="Vector index storage directory")
    DATA_DIR: Path = Field(default=Path("./data"), description="Raw data directory")
    CACHE_DIR: Path = Field(default=Path("./cache"), description="Cache directory")

    # ========== Embedding Model ==========
    EMBEDDING_MODEL: str = Field(
        default="intfloat/e5-small-v2",
        description="HuggingFace model name for embeddings (e5-small-v2: 384-dim, multilingual-e5: 768-dim, bge-m3: 1024-dim)",
    )
    EMBEDDING_DIM: int = Field(default=768, description="Embedding dimension (strict: 384/768/1024)")
    EMBEDDING_BATCH_SIZE: int = Field(default=32, description="Batch size for encoding")
    EMBEDDING_NORMALIZE: bool = Field(default=True, description="L2-normalize embeddings")

    # ========== BGE-M3 Configuration (Phase 7) ==========
    BGE_MODEL: str = Field(
        default="BAAI/bge-m3",
        description="BGE-M3 model for multilingual embeddings (1024-dim)"
    )
    BGE_DIM: int = Field(default=1024, description="BGE-M3 embedding dimension")

    # ========== Reranker Model ==========
    RERANKER_MODEL: str = Field(
        default="BAAI/bge-reranker-base", description="Cross-encoder reranker model"
    )

    # ========== Chunking Parameters ==========
    CHUNK_SIZE_TOKENS: int = Field(default=350, description="Default chunk size in tokens")
    CHUNK_OVERLAP: int = Field(default=70, description="Default chunk overlap in tokens")
    CHUNK_PROFILE: Literal["auto", "faq", "policy"] = Field(
        default="auto", description="Chunking profile"
    )

    # ========== Retrieval Parameters ==========
    RAG_CONF_THRESHOLD: float = Field(
        default=0.35, description="Confidence threshold for RAG answer (adaptive)"
    )
    MAX_RERANK: int = Field(
        default=50, description="Max candidates to rerank with cross-encoder"
    )
    SEMANTIC_COMPRESSION_TOPK: int = Field(
        default=3, description="Top-K hits to compress with LLM summarizer"
    )
    TOP_K_DENSE: int = Field(default=20, description="Top-K for dense retrieval")
    TOP_K_SPARSE: int = Field(default=20, description="Top-K for BM25 sparse retrieval")

    # ========== Qdrant Configuration ==========
    QDRANT_URL: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant API key (optional)")
    QDRANT_COLLECTION: str = Field(
        default="rag2025", description="Qdrant collection name (Phase 7 compliant)"
    )
    QDRANT_MIN_VERSION: str = Field(
        default="1.7.0", description="Minimum qdrant-client version"
    )
    USE_QDRANT: bool = Field(
        default=False, description="Enable Qdrant (False=use NumPy local store)"
    )
    QDRANT_TIMEOUT: int = Field(
        default=30, description="Qdrant request timeout in seconds"
    )

    # ========== LLM Configuration ==========
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Gemini API key")
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    ZAI_API_KEY: Optional[str] = Field(default=None, description="Z.AI API key (GLM-4.5)")
    LLM_MODEL: str = Field(default="gemini-2.0-flash-exp", description="LLM model name")
    LLM_TEMPERATURE: float = Field(default=0.1, description="LLM temperature")
    FORCE_RAG_ONLY: bool = Field(
        default=False, description="Disable LLM fallback (RAG-only mode)"
    )

    # ========== Cache Configuration ==========
    CACHE_TTL: int = Field(default=900, description="Cache TTL in seconds (15 min)")
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis URL")
    USE_REDIS_CACHE: bool = Field(default=False, description="Enable Redis caching")

    # ========== Monitoring & Logging ==========
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    ENABLE_OTEL: bool = Field(default=False, description="Enable OpenTelemetry tracing")

    # ========== Validation ==========
    @field_validator("EMBEDDING_DIM", mode="before")
    @classmethod
    def validate_embedding_dim(cls, v):
        """Validate embedding dimension."""
        try:
            v = int(v)
        except (ValueError, TypeError):
            pass
        # Support: 384 (e5-small-v2), 768 (multilingual-e5), 1024 (bge-m3)
        if v not in [384, 768, 1024]:
            raise ValueError(
                "EMBEDDING_DIM must be 384 (e5-small-v2), 768 (multilingual-e5), or 1024 (bge-m3)"
            )
        return v

    @field_validator("CHUNK_SIZE_TOKENS", mode="before")
    @classmethod
    def validate_chunk_size(cls, v):
        """Enforce chunk size within 300-500 token range."""
        try:
            v = int(v)
        except (ValueError, TypeError):
            pass
        if not (300 <= v <= 500):
            raise ValueError("CHUNK_SIZE_TOKENS must be between 300-500")
        return v

    @field_validator("RAG_CONF_THRESHOLD", mode="before")
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence threshold is between 0-1."""
        try:
            v = float(v)
        except (ValueError, TypeError):
            pass
        if not (0.0 <= v <= 1.0):
            raise ValueError("RAG_CONF_THRESHOLD must be between 0.0 and 1.0")
        return v

    def get_adaptive_threshold(self, doc_count: int) -> float:
        """
        Adaptive confidence threshold based on corpus size.

        Args:
            doc_count: Number of documents in corpus

        Returns:
            Adaptive threshold (0.35 for <500, 0.45 for 500-5k, 0.55 beyond)
        """
        if doc_count < 500:
            return 0.35
        elif doc_count < 5000:
            return 0.45
        else:
            return 0.55


# Global settings instance
settings = RAGSettings()
