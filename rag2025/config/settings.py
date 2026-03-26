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

    # ========== Runtime ==========
    LOG_LEVEL: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Application log level for loguru/uvicorn"
    )

    # ========== Paths ==========
    INDEX_DIR: Path = Field(default=Path("./index"), description="Vector index storage directory")
    DATA_DIR: Path = Field(default=Path("./data"), description="Raw data directory")
    CACHE_DIR: Path = Field(default=Path("./cache"), description="Cache directory")

    # ========== Embedding Model ==========
    EMBEDDING_MODEL: str = Field(
        default="Qwen/Qwen3-Embedding-4B",
        description="HuggingFace model name for embeddings",
    )
    EMBEDDING_DIM: int = Field(default=2560, description="Embedding dimension")
    EMBEDDING_BATCH_SIZE: int = Field(default=8, description="Batch size for encoding")
    EMBEDDING_NORMALIZE: bool = Field(default=True, description="L2-normalize embeddings")

    # ========== Qwen Embedding Configuration ==========
    QWEN_EMBEDDING_MODEL: str = Field(
        default="Qwen/Qwen3-Embedding-4B",
        description="Qwen3 embedding model for multilingual retrieval",
    )
    QWEN_EMBEDDING_DIM: int = Field(default=2560, description="Qwen3 embedding dimension")

    # ========== Reranker Configuration ==========
    RERANKER_MODEL: str = Field(
        default="Qwen/Qwen3-Reranker-8B", description="Cross-encoder reranker model"
    )
    RERANKER_ENABLED: bool = Field(default=True, description="Enable reranker layer")
    RERANKER_WEIGHT: float = Field(default=0.35, description="Weight of reranker score in fusion")

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

    # ========== Hybrid Retrieval Configuration ==========
    USE_HYBRID_RETRIEVAL: bool = Field(
        default=False,
        description="Enable hybrid retrieval (dense + BM25 sparse with RRF fusion)"
    )

    HYBRID_FUSION_DENSE_WEIGHT: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for dense retrieval in RRF fusion (0.0-1.0)"
    )

    HYBRID_FUSION_SPARSE_WEIGHT: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for sparse (BM25) retrieval in RRF fusion (0.0-1.0)"
    )

    BM25_INDEX_PATH: Optional[str] = Field(
        default=None,
        description="Reserved for v2 BM25 persistence. None = in-memory only (v1)."
    )

    # ========== LanceDB Configuration ==========
    LANCEDB_URI: str = Field(
        default="./data/lancedb", description="LanceDB local URI (embedded)"
    )
    LANCEDB_TABLE: str = Field(
        default="rag2025", description="LanceDB table name for chunks"
    )
    LANCEDB_ENTITY_TABLE: str = Field(
        default="husc_entities", description="LanceDB table name for entities"
    )
    USE_LANCEDB: bool = Field(
        default=True, description="Enable LanceDB embedded vector store"
    )

    # ========== LLM Configuration ==========
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Gemini API key")
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI-compat API key")
    ZAI_API_KEY: Optional[str] = Field(default=None, description="Z.AI API key (GLM-4.5)")
    LLM_MODEL: str = Field(default="gemini-2.5-flash", description="LLM model name")
    LLM_TEMPERATURE: float = Field(default=0.1, description="LLM temperature")
    FORCE_RAG_ONLY: bool = Field(
        default=False, description="Disable LLM fallback (RAG-only mode)"
    )

    # ========== RamClouds / OpenAI-compat Primary Provider ==========
    RAMCLOUDS_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for ramclouds.me (primary LLM provider, gemini-2.5-flash)"
    )
    RAMCLOUDS_BASE_URL: str = Field(
        default="https://ramclouds.me/v1",
        description="Base URL for OpenAI-compatible primary provider"
    )
    RAMCLOUDS_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Model name at the primary provider"
    )
    GROQ_API_KEY: Optional[str] = Field(default=None, description="Groq API key (LLM fallback)")
    QWEN_API_KEY: Optional[str] = Field(default=None, description="Qwen API key (NER fallback)")

    # ========== GraphRAG Configuration ==========
    GRAPHRAG_ALPHA: float = Field(
        default=0.6,
        description="Fusion weight for vector/RRF score in GraphRAG (1-alpha = PPR weight)"
    )
    GRAPHRAG_PPR_ALPHA: float = Field(
        default=0.85,
        description="Damping factor for Personalized PageRank"
    )
    GRAPHRAG_SIMPLE_THRESHOLD: int = Field(
        default=2,
        description="Complexity score ≤ this routes to PaddedRAG; > this routes to GraphRAG"
    )

    # ========== Cache Configuration ==========
    CACHE_TTL: int = Field(default=900, description="Cache TTL in seconds (15 min)")
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis URL")
    USE_REDIS_CACHE: bool = Field(default=False, description="Enable Redis caching")

    # ========== Guardrail & Error Exposure ==========
    GUARDRAIL_ENABLED: bool = Field(default=True, description="Enable out-of-scope guardrail")
    GUARDRAIL_MODEL: str = Field(
        default="llama-3.1-8b-instant",
        description="Small Groq model used for scope/no-result classification",
    )
    ERROR_EXPOSURE_MODE: Literal["dev", "prod"] = Field(
        default="dev",
        description="dev: expose detailed internal error codes; prod: expose only one public code",
    )

    # ========== Validation ==========
    @field_validator("EMBEDDING_DIM", mode="before")
    @classmethod
    def validate_embedding_dim(cls, v):
        """Validate embedding dimension."""
        try:
            v = int(v)
        except (ValueError, TypeError):
            pass
        if v not in [1024, 2560, 4096]:
            raise ValueError(
                "EMBEDDING_DIM must be one of 1024, 2560, 4096 (Qwen3-Embedding variants)"
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
