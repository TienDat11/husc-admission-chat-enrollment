"""
Configuration settings using Pydantic BaseSettings.
All environment variables are loaded here with strict typing.
"""
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

try:
    from rag2025.config.ns24_models import ALLOWED_MODELS, ROLE_MODELS
except ModuleNotFoundError:
    from config.ns24_models import ALLOWED_MODELS, ROLE_MODELS


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
    CANONICAL_STORE_ROOT: Path = Field(
        default=Path("./data/canonical_store"),
        description=(
            "Root of the versioned canonical admissions store (ticket 12 admin "
            "lifecycle writes here). Anchored to rag2025/ like LANCEDB_URI — "
            "this is source-of-truth admin data, so it must not depend on the "
            "process CWD."
        ),
    )

    @field_validator("CANONICAL_STORE_ROOT", mode="after")
    @classmethod
    def resolve_canonical_store_root(cls, v: Path) -> Path:
        """Relative path → absolute (rag2025/ as anchor). Absolute giữ nguyên."""
        text = str(v)
        if text.startswith("/") or (len(text) > 1 and text[1] == ":"):
            return v
        return (Path(__file__).resolve().parent.parent / v).resolve()

    # ========== Embedding Runtime Selection ==========
    EMBEDDING_PROVIDER: Literal["qwen", "harrier", "bge"] = Field(
        default="qwen",
        description="Embedding provider to control query prompting behavior",
    )

    EMBEDDING_MODEL: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description="Active HuggingFace model name used at runtime",
    )
    EMBEDDING_DIM: int = Field(default=1024, description="Active embedding dimension used at runtime")
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
    RERANKER_ENABLED: bool = Field(default=False, description="Enable reranker layer")
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
    # T1.8 chunk size gates — hard cap (split above) + soft floor (merge below).
    # These are DIFFERENT knobs from CHUNK_SIZE_TOKENS (target budget): a single
    # chunk is never allowed above HARD_MAX_TOKENS, and chunks below
    # SOFT_MIN_TOKENS are merged into a neighbor (best-effort).
    HARD_MAX_TOKENS: int = Field(
        default=512, ge=1, description="Absolute hard cap for any single chunk (tokens)"
    )
    SOFT_MIN_TOKENS: int = Field(
        default=50, ge=0, description="Soft floor below which chunks are merged (tokens)"
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
    @field_validator("LANCEDB_URI", mode="after")
    @classmethod
    def resolve_lancedb_uri(cls, v):
        """Relative ./data/lancedb → absolute (rag2025/ as anchor). Schemed/absolute giữ nguyên."""
        if "://" in v or v.startswith("/") or (len(v) > 1 and v[1] == ":"):
            return v
        return str((Path(__file__).resolve().parent.parent / v).resolve())
    LANCEDB_TABLE: str = Field(
        default="husc_chunks", description="LanceDB table name for chunks"
    )
    LANCEDB_ENTITY_TABLE: str = Field(
        default="husc_entities", description="LanceDB table name for entities"
    )
    USE_LANCEDB: bool = Field(
        default=True, description="Enable LanceDB embedded vector store"
    )

    # ========== LLM Configuration (ns24 gateway) ==========
    # One gateway, one key. Every call site names a role; the role resolves to a
    # primary model plus an optional fallback model ON THE SAME GATEWAY, so a
    # vendor problem surfaces as an error instead of a silent vendor swap.
    # Allowed model names live in config/ns24_models.py (ALLOWED_MODELS) and are
    # enforced here at startup — a typo fails the boot, not the first question.
    NS24_API_KEY: Optional[str] = Field(
        default=None, description="ns24 gateway API key (the only LLM credential)"
    )
    NS24_BASE_URL: str = Field(
        default="https://api.ns24.io/v1", description="ns24 OpenAI-compatible base URL"
    )
    NS24_STREAM: bool = Field(
        default=True,
        description="Stream gateway responses internally (accumulated before parse)",
    )

    NS24_MODEL: str = Field(
        default=ROLE_MODELS["default"][0], description="Default role primary model"
    )
    NS24_FALLBACK_MODEL: str = Field(
        default=ROLE_MODELS["default"][1], description="Default role fallback model"
    )
    NS24_ROUTER_MODEL: str = Field(
        default=ROLE_MODELS["router"][0], description="Router/classify primary model"
    )
    NS24_ROUTER_FALLBACK_MODEL: str = Field(
        default=ROLE_MODELS["router"][1], description="Router/classify fallback model"
    )
    NS24_NER_MODEL: str = Field(
        default=ROLE_MODELS["ner"][0], description="NER/graph-extraction primary model"
    )
    NS24_NER_FALLBACK_MODEL: str = Field(
        default=ROLE_MODELS["ner"][1], description="NER/graph-extraction fallback model"
    )
    NS24_HYDE_MODEL: str = Field(
        default=ROLE_MODELS["hyde"][0], description="HYDE query-enhancement primary model"
    )
    NS24_HYDE_FALLBACK_MODEL: str = Field(
        default=ROLE_MODELS["hyde"][1], description="HYDE query-enhancement fallback model"
    )
    NS24_GUARDRAIL_MODEL: str = Field(
        default=ROLE_MODELS["guardrail"][0], description="Guardrail classification primary model"
    )
    NS24_GUARDRAIL_FALLBACK_MODEL: str = Field(
        default=ROLE_MODELS["guardrail"][1], description="Guardrail classification fallback model"
    )
    NS24_GEN_MODEL: str = Field(
        default=ROLE_MODELS["gen"][0], description="Final answer generation primary model"
    )
    NS24_GEN_FALLBACK_MODEL: str = Field(
        default=ROLE_MODELS["gen"][1], description="Final answer generation fallback model"
    )

    LLM_TEMPERATURE: float = Field(default=0.1, description="LLM temperature")
    FORCE_RAG_ONLY: bool = Field(
        default=False, description="Disable LLM fallback (RAG-only mode)"
    )

    @field_validator(
        "NS24_MODEL",
        "NS24_FALLBACK_MODEL",
        "NS24_ROUTER_MODEL",
        "NS24_ROUTER_FALLBACK_MODEL",
        "NS24_NER_MODEL",
        "NS24_NER_FALLBACK_MODEL",
        "NS24_HYDE_MODEL",
        "NS24_HYDE_FALLBACK_MODEL",
        "NS24_GUARDRAIL_MODEL",
        "NS24_GUARDRAIL_FALLBACK_MODEL",
        "NS24_GEN_MODEL",
        "NS24_GEN_FALLBACK_MODEL",
        mode="after",
    )
    @classmethod
    def validate_ns24_model(cls, v: str, info) -> str:
        """Every configured model must be in the gateway whitelist.

        An empty fallback is allowed and means "no fallback for this role".
        """
        text = (v or "").strip()
        if text and text not in ALLOWED_MODELS:
            raise ValueError(
                f"{info.field_name}={text!r} is not an allowed ns24 model. "
                f"Allowed: {', '.join(ALLOWED_MODELS)}"
            )
        return text

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

    @field_validator("HARD_MAX_TOKENS", "SOFT_MIN_TOKENS", mode="before")
    @classmethod
    def validate_chunk_gate_ints(cls, v):
        """Coerce chunk gate values to int so env strings are accepted."""
        try:
            v = int(v)
        except (ValueError, TypeError):
            pass
        return v

    @model_validator(mode="after")
    def validate_chunk_gate_order(self):
        """Fail fast if the hard cap does not strictly exceed the soft floor."""
        if self.HARD_MAX_TOKENS <= self.SOFT_MIN_TOKENS:
            raise ValueError(
                "HARD_MAX_TOKENS must be strictly greater than SOFT_MIN_TOKENS"
                f" (got {self.HARD_MAX_TOKENS} <= {self.SOFT_MIN_TOKENS})"
            )
        return self

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
