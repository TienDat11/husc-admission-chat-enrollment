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

    # ========== Embedding Runtime Selection ==========
    EMBEDDING_PROVIDER: Literal["qwen", "harrier", "bge"] = Field(
        default="qwen",
        description="Embedding provider to control query prompting behavior",
    )

    EMBEDDING_MODEL: str = Field(
        default="Qwen/Qwen3-Embedding-4B",
        description="Active HuggingFace model name used at runtime",
    )
    EMBEDDING_DIM: int = Field(default=2560, description="Active embedding dimension used at runtime")
    EMBEDDING_BATCH_SIZE: int = Field(default=8, description="Batch size for encoding")
    EMBEDDING_NORMALIZE: bool = Field(default=True, description="L2-normalize embeddings")

    # ========== Embedding Presets (for easy switching in .env) ==========
    QWEN_EMBEDDING_MODEL: str = Field(
        default="Qwen/Qwen3-Embedding-4B",
        description="Qwen3 embedding model preset",
    )
    QWEN_EMBEDDING_DIM: int = Field(default=2560, description="Qwen3 embedding dimension preset")

    HARRIER_EMBEDDING_MODEL: str = Field(
        default="microsoft/harrier-oss-v1-0.6b",
        description="Harrier embedding model preset",
    )
    HARRIER_EMBEDDING_DIM: int = Field(default=1024, description="Harrier embedding dimension preset")

    BGE_EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-m3",
        description="BGE embedding model preset",
    )
    BGE_EMBEDDING_DIM: int = Field(default=1024, description="BGE embedding dimension preset")

    # ========== Reranker Configuration ==========
    RERANKER_MODEL: str = Field(
        default="Qwen/Qwen3-Reranker-8B", description="Cross-encoder reranker model"
    )
    RERANKER_ENABLED: bool = Field(default=True, description="Enable reranker layer")
    RERANKER_WEIGHT: float = Field(default=0.35, description="Weight of reranker score in fusion")
    RERANKER_MAX_LENGTH: int = Field(
        default=128, description="Max tokens fed to cross-encoder for scoring (truncation is scoring-only, not answer text)"
    )
    RERANKER_POOL: int = Field(
        default=5, description="Number of top candidates to rerank in adaptive mode"
    )
    RERANKER_ADAPTIVE_CONFIDENCE: float = Field(
        default=0.5, description="Confidence threshold below which adaptive reranker activates"
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
    @property
    def LLM_MODEL(self) -> str:
        """Backward-compatible alias that always tracks RAMCLOUDS_MODEL."""
        return self.RAMCLOUDS_MODEL

    LLM_TEMPERATURE: float = Field(default=0.1, description="LLM temperature")
    FORCE_RAG_ONLY: bool = Field(
        default=False, description="Disable LLM fallback (RAG-only mode)"
    )

    # ========== AI Box / OpenAI-compatible Primary Provider ==========
    RAMCLOUDS_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for AI Box (primary OpenAI-compatible LLM provider)"
    )
    RAMCLOUDS_BASE_URL: str = Field(
        default="https://api.ai-box.vn/v1",
        description="Base URL for OpenAI-compatible primary provider"
    )
    RAMCLOUDS_MODEL: str = Field(
        default="deepseek-v4-flash",
        description="Model name at the primary provider (NER + fallback)"
    )
    RAMCLOUDS_NER_MODEL: str = Field(
        default="deepseek-v4-flash",
        description="Model for NER/graph extraction at ramclouds.me (needs speed + JSON)"
    )
    RAMCLOUDS_HYDE_MODEL: str = Field(
        default="deepseek-v4-flash",
        description="Model for HYDE query enhancement at AI Box (needs creativity)"
    )
    RAMCLOUDS_GEN_MODEL: str = Field(
        default="deepseek-v4-pro",
        description="Model for final answer generation at AI Box (needs quality)"
    )
    RAMCLOUDS_GEN_FALLBACK_MODEL: str = Field(
        default="deepseek-v4-flash",
        description="Fallback generation model at AI Box"
    )
    RAMCLOUDS_ROUTER_MODEL: str = Field(
        default="",
        description=(
            "Model for the routing/HyDE classifier client (S15.4 / ADR-D, "
            "MUST-FIX #3). Empty (default) resolves to RAMCLOUDS_HYDE_MODEL "
            "via the `effective_router_model` property — a true no-op vs the "
            "pre-S15 wiring at graphrag_retriever.py:259. Set this to swap the "
            "router LLM without touching the HYDE pipeline used elsewhere."
        ),
    )
    GROQ_API_KEY: Optional[str] = Field(default=None, description="Groq API key (LLM fallback)")
    GROQ_ROUTER_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model for the router classify/step-back calls (fast path)",
    )
    GROQ_NER_MODEL: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model for NER extraction (needs speed, free-tier)",
    )
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

    # ========== Answer Generation Parameters ==========
    MAX_CONTEXT_CHARS: int = Field(
        default=12000, description="Max characters of chunk context fed to generation"
    )
    MAX_GEN_TOKENS: int = Field(
        default=1200, description="Max output tokens for normal answers"
    )
    MAX_GEN_TOKENS_ENUM: int = Field(
        default=4000, description="Max output tokens for program list answers"
    )

    # ========== Guardrail & Error Exposure ==========
    GUARDRAIL_ENABLED: bool = Field(default=True, description="Enable out-of-scope guardrail")
    GUARDRAIL_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model used for scope/no-result classification",
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
        if v not in [640, 1024, 2560, 4096, 5376]:
            raise ValueError(
                "EMBEDDING_DIM must be one of 640, 1024, 2560, 4096, 5376 (Qwen/Harrier/BGE variants)"
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

    @property
    def effective_router_model(self) -> str:
        """Resolved model name for the router/classify LLM client (S15.4 / ADR-D).

        Returns ``RAMCLOUDS_ROUTER_MODEL`` when explicitly set; otherwise falls
        through to ``RAMCLOUDS_HYDE_MODEL``. An empty string is treated as
        unset so operators can "clear" an override without pinning the router
        to a blank model name.

        Wire-up: ``graphrag_retriever.from_disk`` (currently ``:259`` —
        ``get_llm_client(force_model=settings.RAMCLOUDS_HYDE_MODEL)``) must
        switch to ``get_llm_client(force_model=settings.effective_router_model)``
        for the override to take effect end-to-end. That edit is owned by
        Lane A; this property exposes the resolved value safely so the swap
        is a one-line change.
        """
        override = (self.RAMCLOUDS_ROUTER_MODEL or "").strip()
        return override or self.RAMCLOUDS_HYDE_MODEL


# Global settings instance
settings = RAGSettings()
