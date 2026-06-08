"""
FastAPI Application - RAG 2025 (Updated)

Main API service with:
- HYDE query enhancement (Input Layer)
- Qwen3 embedding retrieval with LanceDB (Retrieval Layer)
- LLM answer generation (Generation Layer)
- CORS support for UI connection
"""
import sys
import os
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# CRITICAL: Load .env BEFORE any imports that use env vars
import config.env_loader  # noqa: F401

from typing import Any, Dict, List, Literal, Optional
from collections import defaultdict, deque
from asyncio import Lock
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

# Import new services
from services.query_enhancer import query_enhancer
from services.lancedb_retrieval import get_retriever, LanceDBRetrieverConfig
from services.llm_generator import get_llm_generator
from services.reranker import RerankerService
from services.query_cache import QueryCache
from services.guardrail import GuardrailService

# Keep existing imports for backward compatibility
from config.settings import RAGSettings
from chunker import ChunkConfig, Chunker

# Embedding service for provider-aware query/document encoding
from services.embedding import EmbeddingService

# Initialize settings
settings = RAGSettings()

# Security/limits config
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
MAX_GRAPH_UPDATE_CHUNKS = int(os.getenv("MAX_GRAPH_UPDATE_CHUNKS", "100"))
MAX_GRAPH_CHUNK_TEXT_LENGTH = int(os.getenv("MAX_GRAPH_CHUNK_TEXT_LENGTH", "5000"))
ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN", "")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
GROUNDING_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", "0.18"))

# Simple in-memory rate limiter
_rate_limit_lock = Lock()
_rate_limit_buckets: dict[str, deque[float]] = defaultdict(deque)

# Lightweight in-memory metrics (no external dependencies)
_metrics_lock = Lock()
_metrics = {
    "query_total": 0,
    "query_errors": 0,
    "query_cache_hits": 0,
    "query_guardrail_blocks": 0,
    "query_pii_blocks": 0,
    "query_low_groundedness": 0,
    "query_total_latency_ms": 0.0,
    "query_count_latency": 0,
    "unified_query_total": 0,
    "unified_query_errors": 0,
    "unified_query_cache_hits": 0,
    "unified_query_low_groundedness": 0,
    "unified_query_total_latency_ms": 0.0,
    "unified_query_count_latency": 0,
}


async def _metric_inc(key: str, amount: int = 1) -> None:
    async with _metrics_lock:
        _metrics[key] = int(_metrics.get(key, 0)) + amount


async def _metric_add_float(key: str, value: float) -> None:
    async with _metrics_lock:
        _metrics[key] = float(_metrics.get(key, 0.0)) + value


def _safe_groundedness_score(answer: str, chunks: List[Dict[str, Any]]) -> float:
    source_texts = [
        chunk.get("text") or chunk.get("text_plain") or chunk.get("summary") or ""
        for chunk in chunks
        if isinstance(chunk, dict)
    ]
    source_texts = [text for text in source_texts if text]
    if not answer or not source_texts:
        return 0.0
    try:
        from results.metrics import faithfulness_score
        return float(faithfulness_score(answer, source_texts))
    except Exception as exc:
        logger.warning(f"Groundedness scoring fallback: {exc}")
        answer_tokens = set(answer.lower().split())
        source_tokens = set(" ".join(source_texts).lower().split())
        if not answer_tokens:
            return 0.0
        overlap = len(answer_tokens & source_tokens)
        return overlap / len(answer_tokens)


@asynccontextmanager
async def _track_latency(total_key: str, count_key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        latency_ms = (time.perf_counter() - t0) * 1000
        await _metric_add_float(total_key, latency_ms)
        await _metric_inc(count_key)


def _metrics_snapshot() -> Dict[str, Any]:
    query_count = int(_metrics.get("query_count_latency", 0))
    unified_count = int(_metrics.get("unified_query_count_latency", 0))
    return {
        "query_total": int(_metrics.get("query_total", 0)),
        "query_errors": int(_metrics.get("query_errors", 0)),
        "query_cache_hits": int(_metrics.get("query_cache_hits", 0)),
        "query_guardrail_blocks": int(_metrics.get("query_guardrail_blocks", 0)),
        "query_pii_blocks": int(_metrics.get("query_pii_blocks", 0)),
        "query_low_groundedness": int(_metrics.get("query_low_groundedness", 0)),
        "query_avg_latency_ms": (
            float(_metrics.get("query_total_latency_ms", 0.0)) / query_count
            if query_count else 0.0
        ),
        "unified_query_total": int(_metrics.get("unified_query_total", 0)),
        "unified_query_errors": int(_metrics.get("unified_query_errors", 0)),
        "unified_query_cache_hits": int(_metrics.get("unified_query_cache_hits", 0)),
        "unified_query_low_groundedness": int(_metrics.get("unified_query_low_groundedness", 0)),
        "unified_query_avg_latency_ms": (
            float(_metrics.get("unified_query_total_latency_ms", 0.0)) / unified_count
            if unified_count else 0.0
        ),
    }

def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


async def _enforce_rate_limit(request: Request) -> None:
    if RATE_LIMIT_PER_MINUTE <= 0:
        return
    ip = _client_ip(request)
    now = time.time()
    window_start = now - 60.0
    async with _rate_limit_lock:
        bucket = _rate_limit_buckets[ip]
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Too many requests")
        bucket.append(now)


def require_admin_token(x_admin_token: Optional[str] = Header(default=None)) -> None:
    if not ADMIN_API_TOKEN:
        raise HTTPException(status_code=503, detail="Admin token is not configured")
    if x_admin_token != ADMIN_API_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")


# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
)

# Initialize FastAPI app
app = FastAPI(
    title="RAG API 2025 - HYDE + Multi-Embedding + LanceDB",
    description="Advanced RAG with provider-aware query enhancement and embedded vector retrieval",
    version="2.0.0"
)

# CORS middleware - restricted allowlist
raw_allowed_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
if raw_allowed_origins:
    allowed_origins = [origin.strip() for origin in raw_allowed_origins.split(",") if origin.strip()]
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "x-admin-token"],
)


# ========== API Models ==========


class SimpleQueryRequest(BaseModel):
    """Simple query request - user only needs to send a string"""
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH, description="User's question")
    force_rag_only: Optional[bool] = Field(default=False, description="Force RAG only mode")


class SourceChip(BaseModel):
    """Structured source citation for FE chip rendering (rich-markdown T4)."""
    id: str = Field(..., description="Unique chunk or source identifier")
    title: str = Field(..., description="Human-readable source title (first 80 chars)")
    url: Optional[str] = Field(default=None, description="Canonical source URL when available")
    snippet: str = Field(default="", description="Short excerpt from the chunk (first 120 chars)")
    data_year: str = Field(default="N/A", description="Effective year of the data (e.g. 2025, 2026)")


class QueryResponse(BaseModel):
    """Enhanced query response with all fields"""
    original_query: str
    enhanced_query: str
    query_type: str

    answer: str
    sources: List[SourceChip]
    confidence: float
    groundedness_score: float = 0.0

    top_k_used: int
    chunks_used: int
    provider: str
    trace_id: str

    status_code: str = "SUCCESS"
    status_reason: str = ""
    data_gap_hints: List[str] = Field(default_factory=list)
    internal_status_code: Optional[str] = None
    pii_detected: bool = False

    chunks: Optional[List[Dict[str, Any]]] = None


# Legacy models for backward compatibility
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    lancedb_connected: bool = False
    vectors_count: int = 0
    collection: str = ""
    embedding_model: str = ""
    reranker_model: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ChunkPreviewRequest(BaseModel):
    """Chunk preview request model"""
    text: str = Field(
        ...,
        min_length=10,
        description="Text to chunk (minimum 10 characters)",
        example="Trong công tác tuyển sinh năm 2025, Bộ GDĐT quy định tất cả thí sinh phải đăng ký xét tuyển trực tuyến."
    )
    profile: str = Field(
        default="auto",
        description="Chunking profile to use",
        pattern="^(auto|faq|policy)$",
        example="auto"
    )


class ChunkPreview(BaseModel):
    """Individual chunk preview"""
    chunk_id: int = Field(description="Chunk index")
    text: str = Field(description="Chunk text")
    token_count: int = Field(description="Token count for this chunk")
    char_count: int = Field(description="Character count")
    sparse_terms_count: int = Field(description="Number of BM25 sparse terms extracted")


class ChunkPreviewResponse(BaseModel):
    """Chunk preview response model"""
    text_length: int = Field(description="Original text length in characters")
    profile_used: str = Field(description="Chunking profile used")
    profile_config: Dict[str, Any] = Field(description="Profile configuration details")
    chunks: List[ChunkPreview] = Field(description="List of chunk previews")
    total_chunks: int = Field(description="Total number of chunks generated")
    total_tokens: int = Field(description="Sum of tokens across all chunks")
    avg_tokens_per_chunk: float = Field(description="Average tokens per chunk")


# ========== Global Services ==========

query_enhancer_service = None
lancedb_retriever_service = None
llm_generator_service = None
embedding_service: EmbeddingService | None = None
reranker_service: RerankerService | None = None
query_cache: QueryCache | None = None
guardrail_service: GuardrailService | None = None

# GraphRAG unified pipeline
unified_pipeline = None

# Hybrid search service (dense + BM25 + RRF)
hybrid_search_service: Optional["HybridSearchService"] = None

# Debug tools
chunk_config: ChunkConfig | None = None
chunker: Chunker | None = None


# ========== Startup/Shutdown ==========


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global query_enhancer_service, lancedb_retriever_service, llm_generator_service, embedding_service, unified_pipeline, reranker_service, query_cache, guardrail_service, hybrid_search_service

    logger.info("=" * 60)
    logger.info("Starting RAG API 2025 v3.0 - Unified PaddedRAG + GraphRAG")
    logger.info("=" * 60)

    try:
        # Legacy HYDE enhancer
        logger.info("Initializing HYDE Query Enhancer...")
        query_enhancer_service = query_enhancer

        logger.info("Initializing Embedding Service...")
        embedding_service = EmbeddingService(settings)

        logger.info("Initializing LanceDB Retriever...")
        import concurrent.futures

        def init_lancedb():
            try:
                return get_retriever()
            except Exception as e:
                logger.error(f"LanceDB initialization failed: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(init_lancedb)
            try:
                lancedb_retriever_service = future.result(timeout=10.0)
                if lancedb_retriever_service:
                    logger.info("LanceDB Retriever initialized successfully")
                else:
                    logger.warning("LanceDB not available – degraded mode")
            except concurrent.futures.TimeoutError:
                logger.warning("LanceDB initialization timeout – degraded mode")
                lancedb_retriever_service = None

        logger.info("Initializing LLM Generator...")
        llm_generator_service = get_llm_generator()

        logger.info("Initializing Reranker Service...")
        reranker_service = RerankerService(settings)

        logger.info("Initializing Query Cache...")
        query_cache = QueryCache(ttl_seconds=settings.CACHE_TTL)

        logger.info("Initializing Guardrail Service...")
        guardrail_service = GuardrailService(settings)

        # Initialize Unified GraphRAG Pipeline
        logger.info("Initializing Unified RAG Pipeline (PaddedRAG + GraphRAG)...")
        try:
            from services.graphrag_retriever import UnifiedRAGPipeline
            unified_pipeline = UnifiedRAGPipeline.from_disk(
                alpha=settings.GRAPHRAG_ALPHA,
                beta=1.0 - settings.GRAPHRAG_ALPHA,
            )
            graph_stats = unified_pipeline._graphrag.graph_stats
            logger.info(
                f"GraphRAG Pipeline ready: "
                f"nodes={graph_stats['nodes']}, edges={graph_stats['edges']}"
            )
        except Exception as e:
            logger.warning(f"GraphRAG pipeline init failed: {e} – graph routing disabled")
            unified_pipeline = None

        # ── Hybrid Search (optional) ──────────────────────────────────────
        if settings.USE_HYBRID_RETRIEVAL and lancedb_retriever_service:
            try:
                from services.hybrid_search import HybridSearchService as _HybridSearchService
                hybrid_search_service = _HybridSearchService(
                    lancedb_retriever=lancedb_retriever_service,
                    settings=settings,
                )
                if not hybrid_search_service.build_bm25_index():
                    logger.warning(
                        "HybridSearchService: BM25 index build failed — disabling hybrid search"
                    )
                    hybrid_search_service = None
                else:
                    logger.info("HybridSearchService: BM25 index built successfully")
            except Exception as e:
                logger.error(f"HybridSearchService: startup init failed: {e}", exc_info=True)
                hybrid_search_service = None

        logger.info("=" * 60)
        logger.info("API Ready! PaddedRAG + GraphRAG + SmartRouter")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down RAG API 2025")


# ========== API Endpoints ==========


@app.get("/", response_model=Dict[str, Any])
async def root():
    """API info"""
    return {
        "name": "RAG API 2025",
        "version": "2.0.0",
        "features": [
            "HYDE query enhancement",
            "Provider-aware multilingual retrieval (Qwen/Harrier/BGE)",
            "Score boosting for near-matches",
            "LanceDB embedded vector store",
            "Multi-LLM fallback (ramclouds/gemini → Groq → compat)"
        ],
        "endpoints": {
            "POST /query": "Main RAG query endpoint (simple string input)",
            "GET /health": "Health check with LanceDB status",
            "GET /docs": "Swagger API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with LanceDB connection"""
    response = HealthResponse(status="unknown")

    if lancedb_retriever_service:
        try:
            collection_info = lancedb_retriever_service.check_collection()
            if collection_info.get("exists"):
                response.lancedb_connected = True
                response.vectors_count = collection_info.get("vectors_count", 0)
                response.collection = settings.LANCEDB_TABLE
                response.status = "healthy"
            else:
                response.status = f"table_not_found: {collection_info.get('error', 'unknown')}"
        except Exception as e:
            response.status = f"lancedb_error: {str(e)[:50]}"
    else:
        response.status = "lancedb_not_initialized"

    response.embedding_model = settings.EMBEDDING_MODEL
    response.reranker_model = settings.RERANKER_MODEL
    response.metrics = _metrics_snapshot()

    return response


@app.get("/metrics", response_model=Dict[str, Any])
async def metrics_endpoint():
    """Lightweight internal metrics endpoint."""
    return {
        "status": "ok",
        "metrics": _metrics_snapshot(),
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: SimpleQueryRequest, raw_request: Request):
    """
    Main RAG endpoint with HYDE multi-variant enhancement

    Flow:
    1. HYDE: user query → 3-5 query variants
    2. Embedding: encode ALL variants to vectors
    3. LanceDB retrieval: retrieve chunks for ALL variants
    4. Merge & Deduplicate: combine results from all variants
    5. LLM: generate answer from merged chunks
    """
    if query_enhancer_service is None or lancedb_retriever_service is None or llm_generator_service is None or embedding_service is None:
        raise HTTPException(
            status_code=503,
            detail="Services not initialized. Please check server logs."
        )

    logger.info(f"Received query: {request.query[:100]}...")
    trace_id = str(uuid.uuid4())

    await _metric_inc("query_total")

    await _enforce_rate_limit(raw_request)

    try:
        async with _track_latency("query_total_latency_ms", "query_count_latency"):
            cache_key = f"query:{request.query.strip().lower()}|force:{request.force_rag_only}"
            if query_cache:
                cached_response = query_cache.get(cache_key)
                if cached_response is not None:
                    await _metric_inc("query_cache_hits")
                    logger.info(f"Query cache hit trace_id={trace_id}")
                    payload = dict(cached_response)
                    payload["trace_id"] = trace_id
                    payload.setdefault("pii_detected", False)
                    return QueryResponse(**payload)

            precheck = await guardrail_service.precheck(request.query) if guardrail_service else None
            if precheck and not precheck.is_in_scope:
                await _metric_inc("query_guardrail_blocks")
                if precheck.pii_detected:
                    await _metric_inc("query_pii_blocks")
                public_code = guardrail_service.public_status(precheck.internal_code)
                internal_code = precheck.internal_code if guardrail_service.expose_internal() else None
                logger.info(
                    f"guardrail_block trace_id={trace_id} query='{request.query[:80]}' internal_code={precheck.internal_code} reason={precheck.reason}"
                )
                response_payload = {
                    "original_query": request.query,
                    "enhanced_query": request.query,
                    "query_type": "out_of_scope",
                    "answer": precheck.short_answer,
                    "sources": [],
                    "confidence": 0.0,
                    "top_k_used": 0,
                    "chunks_used": 0,
                    "provider": "Guardrail",
                    "trace_id": trace_id,
                    "status_code": public_code,
                    "status_reason": precheck.reason,
                    "data_gap_hints": precheck.data_gap_hints,
                    "internal_status_code": internal_code,
                    "pii_detected": precheck.pii_detected,
                    "chunks": [],
                }
                if query_cache:
                    query_cache.set(cache_key, response_payload)
                return QueryResponse(**response_payload)

            # Step 1: HYDE Enhancement - get query variants
        enhanced_request = await query_enhancer_service.enhance_query(
            user_query=request.query,
            force_rag_only=request.force_rag_only
        )

        variants = enhanced_request.get("variants", [])
        original_query = enhanced_request["original_query"]
        detected_intent = enhanced_request["detected_intent"]
        query_type = enhanced_request["query_type"]
        top_k = enhanced_request["top_k"]

        # ========== INTENT → FAQ_TYPE MAPPING ==========
        # Dynamic faq_type selection based on query intent
        INTENT_TO_FAQ_TYPES = {
            "overview": ["tong_hop_tuyen_sinh", "tong_hop_nhom_nganh", "thong_tin_nganh"],
            "specific_program": ["thong_tin_nganh"],
            "tuition": ["hoc_phi"],
        }

        # Keyword sets for intent detection
        OVERVIEW_KEYWORDS = [
            "bao nhiêu ngành", "tổng cộng", "có mấy ngành",
            "tổng số", "tất cả ngành", "các ngành", "tuyển sinh mấy ngành"
        ]
        TUITION_KEYWORDS = ["học phí", "tiền học", "đóng phí", "tín chỉ", "tín", "bao nhiêu tiền", "1 kỳ", "một kỳ", "học kỳ"]

        def get_faq_type_filter(query: str) -> dict | None:
            """Build metadata filter based on query intent."""
            query_lower = query.lower()

            # Overview queries - allow summary and program chunks
            if any(kw in query_lower for kw in OVERVIEW_KEYWORDS):
                return {"or_conditions": [{"faq_type": t} for t in INTENT_TO_FAQ_TYPES["overview"]]}

            # Tuition queries - only hoc_phi chunks
            if any(kw in query_lower for kw in TUITION_KEYWORDS):
                return {"or_conditions": [{"faq_type": t} for t in INTENT_TO_FAQ_TYPES["tuition"]]}

            # Specific program queries (list only, not overview)
            # Detect program list queries for metadata filtering
            is_program_list = any(
                keyword in query_lower
                for keyword in ["các ngành", "danh sách ngành", "có ngành nào", "ngành nào"]
            )
            if is_program_list:
                return {"faq_type": "thong_tin_nganh"}

            # Default: no filter, pure semantic retrieval
            return None

        # Build metadata filter for this query
        metadata_filter = get_faq_type_filter(original_query)

        logger.info(
            f"Query type: {query_type}, intent: {detected_intent}, "
            f"variants: {len(variants)}, top_k={top_k}, filter: {metadata_filter}"
        )
        logger.debug(f"Variants: {variants}")

        # DEBUG: Log variant details
        for i, variant in enumerate(variants):
            logger.info(f"  Variant {i+1}: {variant[:80]}...")

        # Step 2: Embedding Encoding - encode ALL variants to vectors
        all_chunks = []
        all_scores = []

        # For program/overview queries, use higher top_k to get more results
        is_program_query = metadata_filter is not None
        retrieval_top_k = top_k * 3 if is_program_query else top_k

        for i, variant in enumerate(variants):
            query_vector = await run_in_threadpool(
                lambda: embedding_service.encode_query(variant).tolist()
            )

            logger.debug(f"Encoded variant {i+1}/{len(variants)}: {variant[:50]}...")

            if hybrid_search_service:
                retrieval_result = await hybrid_search_service.retrieve(
                    query=variant,
                    query_vector=query_vector,
                    top_k=retrieval_top_k,
                    metadata_filter=metadata_filter,
                )
            else:
                retrieval_result = lancedb_retriever_service.retrieve(
                    query_vector=query_vector,
                    top_k=retrieval_top_k,
                    metadata_filter=metadata_filter,
                    query=variant,
                )

            if retrieval_result.error_type is None and len(retrieval_result.documents) == 0 and metadata_filter is not None:
                logger.warning(f"Variant {i+1}: filtered retrieval returned 0 docs, falling back to no-filter")
                if hybrid_search_service:
                    retrieval_result = await hybrid_search_service.retrieve(
                        query=variant,
                        query_vector=query_vector,
                        top_k=retrieval_top_k,
                        metadata_filter=None,
                    )
                else:
                    retrieval_result = lancedb_retriever_service.retrieve(
                        query_vector=query_vector,
                        top_k=retrieval_top_k,
                        metadata_filter=None,
                        query=variant,
                    )

            if retrieval_result.error_type is not None:
                logger.warning(f"Variant {i+1} retrieval error: {retrieval_result.error_message}")
            else:
                for doc in retrieval_result.documents:
                    all_chunks.append(doc.to_dict())
                    all_scores.append(doc.score)

        logger.info(f"Retrieved {len(all_chunks)} total chunks from {len(variants)} variants")

        # Step 4: Merge & Deduplicate chunks based on unique IDs
        if all_chunks:
            # Deduplicate by chunk ID, keeping highest score
            unique_chunks = {}
            for chunk, score in zip(all_chunks, all_scores):
                chunk_id = (
                    chunk.get("chunk_id")
                    or chunk.get("point_id")
                    or chunk.get("source")
                    or chunk.get("text", "")[:120]
                )
                if chunk_id not in unique_chunks or score > unique_chunks[chunk_id]["score"]:
                    chunk["score"] = score  # Add score for tracking
                    unique_chunks[chunk_id] = chunk

            # Sort by score and take top_k
            chunks = sorted(
                unique_chunks.values(),
                key=lambda x: x.get("score", 0),
                reverse=True
            )[:top_k]

            # Calculate average confidence
            confidence = sum(all_scores) / len(all_scores) if all_scores else 0.0
        else:
            chunks = []
            confidence = 0.0

        if chunks and reranker_service:
            chunks = reranker_service.rerank(
                query=original_query,
                chunks=chunks,
                top_k=top_k,
                apply_lost_in_middle=True,
            )

        status_code = "SUCCESS"
        status_reason = ""
        data_gap_hints: List[str] = []
        internal_status_code: Optional[str] = None

        if len(chunks) == 0 and guardrail_service:
            no_result = await guardrail_service.classify_no_result(original_query)
            status_code = guardrail_service.public_status(no_result.internal_code)
            status_reason = no_result.reason
            data_gap_hints = no_result.data_gap_hints
            internal_status_code = (
                no_result.internal_code if guardrail_service.expose_internal() else None
            )
            logger.info(
                f"no_result_classification query='{original_query[:80]}' internal_code={no_result.internal_code} reason={no_result.reason}"
            )

        logger.info(f"After deduplication: {len(chunks)} unique chunks, avg_confidence={confidence:.3f}")

        # Step 5: Limit context size for program/overview queries
        # For program/overview queries, use fewer chunks to avoid token limit
        max_chunks_for_generation = 15 if is_program_query else 30
        if len(chunks) > max_chunks_for_generation:
            logger.info(f"Limiting chunks for generation: {len(chunks)} -> {max_chunks_for_generation}")
            chunks = chunks[:max_chunks_for_generation]

        # Step 6: LLM Answer Generation
        result = await llm_generator_service.generate_answer(
            query=original_query,
            chunks=chunks,
            confidence=confidence,
            is_program_list_query=is_program_query,  # Pass for higher token limit on list/overview queries
        )

        logger.info(f"Generated answer using {result['provider']}")

        # Step 7: Groundedness scoring
        groundedness = _safe_groundedness_score(result["answer"], chunks)
        if groundedness < GROUNDING_THRESHOLD and chunks:
            await _metric_inc("query_low_groundedness")
            logger.warning(
                f"LOW_GROUNDEDNESS trace_id={trace_id} score={groundedness:.3f} "
                f"threshold={GROUNDING_THRESHOLD}"
            )
            result["answer"] = (
                "⚠️ " + result["answer"] + "\n\n"
                "(Lưu ý: Câu trả lời này có mức bám sát tài liệu nguồn thấp. "
                "Vui lòng kiểm tra lại với tài liệu chính thức.)"
            )

        # Step 8: Build response
        response_payload = {
            "original_query": original_query,
            "enhanced_query": f"({len(variants)} variants) " + "; ".join([v[:30] for v in variants[:2]]),
            "query_type": query_type,
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": confidence,
            "groundedness_score": groundedness,
            "top_k_used": top_k,
            "chunks_used": result["chunks_used"],
            "provider": result["provider"],
            "trace_id": trace_id,
            "status_code": status_code,
            "status_reason": status_reason,
            "data_gap_hints": data_gap_hints,
            "internal_status_code": internal_status_code,
            "pii_detected": False,
            "chunks": chunks,
        }

        if query_cache:
            query_cache.set(cache_key, response_payload)

        return QueryResponse(**response_payload)

    except HTTPException:
        await _metric_inc("query_errors")
        raise
    except Exception as e:
        await _metric_inc("query_errors")
        logger.exception(f"Query failed trace_id={trace_id}: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")


# ========== Debug/Tools Endpoints ==========


@app.post("/debug/preview-chunks", response_model=ChunkPreviewResponse, tags=["Debug/Tools"])
async def preview_chunks(request: ChunkPreviewRequest):
    """
    Preview chunking results without saving to disk

    This endpoint is useful for UAT and testing chunking quality
    """
    global chunk_config, chunker

    try:
        # Lazy load chunker on first use
        if chunk_config is None or chunker is None:
            config_path = Path(__file__).parent / "config" / "chunk_profiles.yaml"
            if not config_path.exists():
                config_path = Path(__file__).parent.parent / "config" / "chunk_profiles.yaml"

            if not config_path.exists():
                raise HTTPException(
                    status_code=500,
                    detail="Chunk profiles config not found"
                )

            logger.info(f"Loading chunk config from: {config_path}")
            chunk_config = ChunkConfig(config_path)
            chunker = Chunker(chunk_config)

        # Validate profile
        profile_name = request.profile.lower()
        if profile_name not in chunk_config.profiles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid profile '{profile_name}'. Must be one of: {list(chunk_config.profiles.keys())}"
            )

        # Get profile config
        profile_config = chunk_config.get_profile(profile_name)

        # Create mock document
        mock_doc = {
            "id": "preview_doc",
            "text": request.text,
            "metadata": {"source": "preview_endpoint"}
        }

        # Chunk the document
        logger.info(f"Chunking text ({len(request.text)} chars) with profile: {profile_name}")
        chunks = chunker.chunk_document(mock_doc, profile_name=profile_name)

        # Build preview response
        chunk_previews: List[ChunkPreview] = []
        total_tokens = 0

        for chunk in chunks:
            token_count = len(chunker.tokenizer.encode(chunk.text))
            total_tokens += token_count

            preview = ChunkPreview(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                token_count=token_count,
                char_count=len(chunk.text),
                sparse_terms_count=len(chunk.sparse_terms)
            )
            chunk_previews.append(preview)

        # Calculate average
        avg_tokens = total_tokens / len(chunks) if chunks else 0

        return ChunkPreviewResponse(
            text_length=len(request.text),
            profile_used=profile_name,
            profile_config={
                "description": profile_config.description,
                "chunk_size": profile_config.chunk_size,
                "overlap": profile_config.overlap,
                "min_tokens": profile_config.min_tokens,
                "separator_priority": profile_config.separator_priority,
            },
            chunks=chunk_previews,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            avg_tokens_per_chunk=avg_tokens
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Chunk preview failed: {e}")
        raise HTTPException(status_code=500, detail="Chunk preview failed")


# ========== Development Server ==========


# ========== GraphRAG v2 Endpoint ==========


class UnifiedQueryRequest(BaseModel):
    """Unified query request for PaddedRAG + GraphRAG pipeline."""
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH, description="User question (Vietnamese)")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    force_route: Optional[Literal["padded_rag", "graph_rag"]] = Field(
        default=None,
        description="Force routing: 'padded_rag' or 'graph_rag' (overrides auto-router)"
    )


class UnifiedQueryResponse(BaseModel):
    """Response from unified pipeline with routing metadata."""
    query: str
    route: str
    answer: str
    sources: List[str]
    confidence: float
    groundedness_score: float = 0.0
    router_info: Dict[str, Any]
    graph_stats: Optional[Dict[str, Any]] = None
    latency_ms: float
    trace_id: str

    # G2 contract — mirror /query's status surface so the FE ChatLayout
    # banner + gap-hint UX behaves identically on /v2. All Optional with
    # safe defaults; existing /v2 consumers are unaffected.
    status_code: str = "SUCCESS"
    status_reason: Optional[str] = None
    data_gap_hints: List[str] = Field(default_factory=list)
    internal_status_code: Optional[str] = None
    pii_detected: bool = False


# ───────────────────────────────────────────────────────────────────────────
# /v2 answer-synthesis helper (S16.x / PROD-path correctness fix).
#
# The previous /v2 handler called `llm.chat(...)` INLINE, which bypassed:
#   * S16.5 URL-faithfulness post-guard (ungrounded URL strip)
#   * Season-aware GAP_DISCLAIMER + RISKY_INTENTS graceful fallback
#   * S15.6 contact-keyword abstain guard
#
# It also SWALLOWED `RouterResult.auto_answer` (CONTACT_BLOCK / vague_reject)
# by emitting a generic "không có thông tin" string for skip_retrieval
# routes. This helper is the single seam used by the /v2 endpoint so the
# fixes are testable in isolation (no GraphRAG or LanceDB needed).
#
# Returns a dict shaped like `LLMGenerator.generate_answer()` so the
# downstream `UnifiedQueryResponse(**payload)` builder is unchanged.
# ───────────────────────────────────────────────────────────────────────────
def _v2_program_list_heuristic(query: str) -> bool:
    """Mirror the /v3 `is_enum` heuristic so /v2 honors the larger token
    budget for enumeration/overview queries (max_tokens_enum)."""
    if not query:
        return False
    q = query.lower()
    return any(k in q for k in [
        "bao nhiêu", "danh sách", "liệt kê", "tất cả",
        "các ngành", "tổng cộng",
    ])


def _build_v2_chunks(documents: List[Any]) -> List[Dict[str, Any]]:
    """Map `RAGResult.documents` (RetrievedDocument objects) into the
    dict shape `LLMGenerator.generate_answer()` expects. Mirrors the
    /v3 endpoint's `chunks_for_rerank` construction so the two
    endpoints feed the generator identically.
    """
    chunks: List[Dict[str, Any]] = []
    for d in documents or []:
        meta = d.metadata if isinstance(d.metadata, dict) else {}
        chunks.append({
            "text": getattr(d, "text", "") or "",
            "summary": meta.get("summary", "") if isinstance(meta, dict) else "",
            "metadata": meta or {},
            "score": float(getattr(d, "score", 0.0) or 0.0),
            "chunk_id": getattr(d, "chunk_id", None),
            "source": getattr(d, "source", None),
        })
    return chunks


async def _synthesize_v2_answer(
    rag_result: Any,
    query: str,
    generator: Any,
) -> Dict[str, Any]:
    """Build the (answer, sources, confidence, provider) dict for the
    /v2 endpoint.

    Behavior:
      1. If `rag_result.router_result.skip_retrieval` AND
         `router_result.auto_answer` is set (e.g. CONTACT_BLOCK /
         HYDE_REJECT_VAGUE) → return the auto_answer string VERBATIM
         (do NOT call the generator; do NOT emit a generic fallback).
         This is the regression fix for the swallowed CONTACT_BLOCK.
      2. Otherwise, build the chunks list from `rag_result.documents`
         and call `generator.generate_answer(...)`. This puts /v2 on
         the same S16.5 URL-guard + season-gap + S15.6 contact-abstain
         path as /v3 and /query.

    Args:
        rag_result: The `RAGResult` returned by
            `UnifiedRAGPipeline.query()`. Must expose `.documents`,
            `.router_result` (with `skip_retrieval`/`auto_answer`),
            and `.confidence`.
        query: Raw user query.
        generator: An `LLMGenerator`-like object exposing
            `async generate_answer(query, chunks, confidence, is_program_list_query)`.

    Returns:
        Dict with keys: `answer`, `sources`, `confidence`, `provider`.
        The shape matches `LLMGenerator.generate_answer()`'s return
        so the /v2 response builder is unchanged.
    """
    router_result = getattr(rag_result, "router_result", None)

    # 1) Honor auto_answer / CONTACT_BLOCK — never swallow it.
    if (
        router_result is not None
        and getattr(router_result, "skip_retrieval", False)
        and getattr(router_result, "auto_answer", None)
    ):
        return {
            "answer": router_result.auto_answer,
            "sources": [],
            "confidence": float(getattr(rag_result, "confidence", 1.0) or 1.0),
            "provider": "auto_answer",
        }

    # 2) Normal path: build chunks and route through LLMGenerator so the
    # S16.5 URL-guard, season-aware GAP_DISCLAIMER, and S15.6 contact-
    # abstain guard all run.
    chunks = _build_v2_chunks(getattr(rag_result, "documents", []) or [])
    is_program_list = _v2_program_list_heuristic(query)
    gen_result = await generator.generate_answer(
        query=query,
        chunks=chunks,
        confidence=float(getattr(rag_result, "confidence", 0.0) or 0.0),
        is_program_list_query=is_program_list,
    )
    # Ensure the contract keys the /v2 response builder reads are present.
    return {
        "answer": gen_result.get("answer", ""),
        "sources": list(gen_result.get("sources", [])),
        "confidence": float(getattr(rag_result, "confidence", 0.0) or 0.0),
        "provider": gen_result.get("provider", "unknown"),
    }


@app.post("/v2/query", response_model=UnifiedQueryResponse)
async def unified_query(request: UnifiedQueryRequest, raw_request: Request):
    """Unified RAG query: auto-routes to PaddedRAG or GraphRAG.

    - Simple 1-hop queries → PaddedRAG (faster, ~150ms)
    - Multi-hop / comparative → GraphRAG (+PPR, ~400ms)

    Routing is performed by HyDE + Step-Back classification using gemini-2.5-flash.
    """
    if not unified_pipeline:
        raise HTTPException(
            status_code=503,
            detail="Unified pipeline not initialized"
        )

    await _enforce_rate_limit(raw_request)
    trace_id = str(uuid.uuid4())
    await _metric_inc("unified_query_total")

    cache_key = f"v2:{request.query.strip().lower()}|top_k:{request.top_k}|route:{request.force_route or 'auto'}"
    if request.force_route == "padded_rag":
        cache_key += "|effective:padded_rag"
    elif request.force_route == "graph_rag":
        cache_key += "|effective:graph_rag"

    if query_cache:
        cached_response = query_cache.get(cache_key)
        if cached_response is not None:
            await _metric_inc("unified_query_cache_hits")
            logger.info(f"Unified query cache hit trace_id={trace_id}")
            payload = dict(cached_response)
            payload["trace_id"] = trace_id
            return UnifiedQueryResponse(**payload)

    # G2-T0: guardrail precheck (mirror /query:533-563) so out-of-scope
    # queries short-circuit with the same status_code / data_gap_hints /
    # pii_detected surface that /query already returns. Without this, /v2
    # silently returns SUCCESS for OOS queries, breaking the FE ChatLayout
    # banner + gap-hint UX.
    if guardrail_service is not None:
        try:
            precheck = await guardrail_service.precheck(request.query)
            if precheck and not precheck.is_in_scope:
                await _metric_inc("query_guardrail_blocks")
                if precheck.pii_detected:
                    await _metric_inc("query_pii_blocks")
                public_code = guardrail_service.public_status(precheck.internal_code)
                internal_code = (
                    precheck.internal_code if guardrail_service.expose_internal() else None
                )
                logger.info(
                    f"v2_guardrail_block trace_id={trace_id} query='{request.query[:80]}' "
                    f"internal_code={precheck.internal_code} reason={precheck.reason}"
                )
                oos_payload = {
                    "query": request.query,
                    "route": "guardrail",
                    "answer": precheck.short_answer,
                    "sources": [],
                    "confidence": 0.0,
                    "groundedness_score": 0.0,
                    "router_info": {
                        "step_back": None,
                        "intent": "guardrail_block",
                        "complexity": 0,
                        "reasoning": precheck.reason,
                        "hyde_variants": [],
                    },
                    "graph_stats": None,
                    "latency_ms": 0.0,
                    "trace_id": trace_id,
                    "status_code": public_code,
                    "status_reason": precheck.reason,
                    "data_gap_hints": list(precheck.data_gap_hints or []),
                    "internal_status_code": internal_code,
                    "pii_detected": precheck.pii_detected,
                }
                if query_cache:
                    query_cache.set(cache_key, oos_payload)
                return UnifiedQueryResponse(**oos_payload)
        except Exception as _guard_exc:
            # Never let a guardrail outage break the /v2 pipeline — fall through
            # and serve the normal path with default SUCCESS status.
            logger.warning(f"v2_guardrail_precheck_error: {_guard_exc}")

    # 1. Get PaddedRAG baseline documents first (from LanceDB)
    baseline_docs = []
    if lancedb_retriever_service and embedding_service:
        try:
            query_vec = await run_in_threadpool(
                lambda: embedding_service.encode_query(request.query).tolist()
            )
            result = lancedb_retriever_service.retrieve(
                query_vector=query_vec,
                top_k=request.top_k * 3,
                query=request.query,
            )
            if result.is_success:
                baseline_docs = result.documents
        except Exception as exc:
            logger.warning(f"LanceDB retrieval failed: {exc}")

    # 2. Run unified pipeline (router + optional graph fusion)
    # G2-T0: status fields default to SUCCESS; populated below when the
    # pipeline returns zero documents (mirror /query:726-736).
    status_code = "SUCCESS"
    status_reason: Optional[str] = None
    data_gap_hints: List[str] = []
    internal_status_code: Optional[str] = None
    pii_detected = False

    try:
        async with _track_latency("unified_query_total_latency_ms", "unified_query_count_latency"):
            rag_result = await unified_pipeline.query(
                user_query=request.query,
                baseline_docs=baseline_docs,
                top_k=request.top_k,
            )

            # G2-T0: classify no-result (mirror /query:726-736) so the FE
            # banner + gap-hint path lights up when retrieval is empty.
            if (not rag_result.documents) and guardrail_service is not None:
                try:
                    no_result = await guardrail_service.classify_no_result(request.query)
                    # Only surface the classification when the guardrail
                    # actually transitioned the status away from SUCCESS —
                    # mirrors /query:728 which reassigns unconditionally
                    # only inside the same `if len(chunks) == 0` branch.
                    if no_result.internal_code != "SUCCESS":
                        status_code = guardrail_service.public_status(no_result.internal_code)
                        status_reason = no_result.reason
                        data_gap_hints = list(no_result.data_gap_hints or [])
                        internal_status_code = (
                            no_result.internal_code if guardrail_service.expose_internal() else None
                        )
                        logger.info(
                            f"v2_no_result_classification trace_id={trace_id} "
                            f"internal_code={no_result.internal_code} reason={no_result.reason}"
                        )
                except Exception as _nr_exc:
                    logger.warning(f"v2_classify_no_result_error: {_nr_exc}")

            if request.force_route == "padded_rag":
                rag_result.route = "padded_rag"
                rag_result.documents = baseline_docs[:request.top_k]
                rag_result.ppr_scores = {}
                rag_result.confidence = rag_result.documents[0].score if rag_result.documents else 0.0
                logger.info("Route override applied: padded_rag")
            elif request.force_route == "graph_rag" and baseline_docs:
                docs, ppr_scores = await unified_pipeline._graphrag.retrieve(
                    query=request.query,
                    router_result=rag_result.router_result,
                    baseline_docs=baseline_docs,
                    top_k=request.top_k,
                )
                rag_result.route = "graph_rag"
                rag_result.documents = docs
                rag_result.ppr_scores = ppr_scores
                rag_result.confidence = docs[0].score if docs else 0.0
                logger.info("Route override applied: graph_rag")

            # 3. Generate answer — route through LLMGenerator (NOT inline llm.chat).
            #    This wires /v2 into the SAME S16.5 URL-faithfulness guard,
            #    season-aware GAP_DISCLAIMER, and S15.6 contact-keyword abstain
            #    guard that /v3 and /query already use. The helper also honors
            #    RouterResult.auto_answer (CONTACT_BLOCK / vague_reject) so the
            #    skip_retrieval path is no longer swallowed by a generic fallback.
            if llm_generator_service is None:
                raise HTTPException(
                    status_code=503,
                    detail="LLM generator not initialized",
                )
            synthesis = await _synthesize_v2_answer(
                rag_result=rag_result,
                query=request.query,
                generator=llm_generator_service,
            )
            answer = synthesis["answer"]
            sources = synthesis["sources"]

            groundedness = _safe_groundedness_score(answer, [
                {"text": d.text, "summary": d.text, "text_plain": d.text}
                for d in rag_result.documents
            ])
            if groundedness < GROUNDING_THRESHOLD and rag_result.documents:
                await _metric_inc("unified_query_low_groundedness")
                logger.warning(
                    f"LOW_GROUNDEDNESS_UNIFIED trace_id={trace_id} score={groundedness:.3f} "
                    f"threshold={GROUNDING_THRESHOLD}"
                )
                answer = (
                    "⚠️ " + answer + "\n\n"
                    "(Lưu ý: Câu trả lời này có mức bám sát tài liệu nguồn thấp. "
                    "Vui lòng kiểm tra lại với tài liệu chính thức.)"
                )

            response_payload = {
                "query": request.query,
                "route": rag_result.route,
                "answer": answer,
                "sources": sources,
                "confidence": rag_result.confidence,
                "groundedness_score": 0.0,
                "router_info": {
                    "step_back": rag_result.router_result.step_back_query,
                    "intent": rag_result.router_result.intent,
                    "complexity": rag_result.router_result.complexity,
                    "reasoning": rag_result.router_result.reasoning,
                    "hyde_variants": rag_result.router_result.hyde_variants,
                },
                "graph_stats": unified_pipeline._graphrag.graph_stats,
                "latency_ms": rag_result.latency_ms,
                "trace_id": trace_id,
                "status_code": status_code,
                "status_reason": status_reason,
                "data_gap_hints": data_gap_hints,
                "internal_status_code": internal_status_code,
                "pii_detected": pii_detected,
            }

            if query_cache:
                query_cache.set(cache_key, response_payload)

            return UnifiedQueryResponse(**response_payload)

    except Exception as exc:
        await _metric_inc("unified_query_errors")
        logger.exception(f"Unified query failed trace_id={trace_id}: {exc}")
        raise HTTPException(status_code=500, detail="Unified query processing failed")


# ========================================================================
# /v3/query — Hybrid (Dense+BM25 RRF) + Aggregation Booster + Reranker
# ------------------------------------------------------------------------
# Khác /v2/query:
#   - /v2 chỉ dùng dense LanceDB → SmartRouter (HyDE+Step-back) → GraphRAG
#   - /v3 dùng HybridSearchService (RRF k=60) → Aggregation Booster
#     → Reranker (Qwen3-Reranker-8B + lost-in-middle) → LLMGenerator
# Mục tiêu: tăng has_answer rate cho corpus tuyển sinh HUSC, giảm
# hallucination khi câu trả lời bị cắt ngắn. Giữ /v2/query nguyên cho
# rollback và so sánh AB.
# ========================================================================


class V3QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_pool: int = Field(default=20, ge=5, le=50,
                                description="Số doc lấy từ Hybrid trước khi rerank")
    apply_lost_in_middle: bool = Field(default=True)


class V3QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]
    confidence: float
    chunks_used: int
    pipeline_stages: Dict[str, Any]
    latency_ms: float
    provider: str
    trace_id: str


@app.post("/v3/query", response_model=V3QueryResponse)
async def v3_query(request: V3QueryRequest, raw_request: Request):
    """V3 query: Hybrid (Dense+BM25 RRF) + Booster + Reranker + Generator.

    Flow:
      1. Embedding service: encode query
      2. HybridSearchService.retrieve(top_k=candidate_pool) → fused docs
         (fallback dense-only nếu hybrid_search_service is None)
      3. boost_with_aggregation(query, docs, max_inject=2)
      4. reranker.rerank(query, chunks, top_k, apply_lost_in_middle)
      5. llm_generator.generate_answer(query, chunks, confidence, is_enum)
    """
    if (lancedb_retriever_service is None
            or embedding_service is None
            or llm_generator_service is None):
        raise HTTPException(status_code=503,
                            detail="Core services not initialized")

    await _enforce_rate_limit(raw_request)
    trace_id = str(uuid.uuid4())
    t_start = time.perf_counter()

    stages: Dict[str, Any] = {
        "hybrid_used": False,
        "booster_injected": 0,
        "reranker_used": False,
        "lost_in_middle": False,
        "candidate_pool": request.candidate_pool,
        "final_top_k": request.top_k,
    }

    cache_key = (
        f"v3:{request.query.strip().lower()}|tk:{request.top_k}|cp:{request.candidate_pool}"
    )
    if query_cache:
        cached_response = query_cache.get(cache_key)
        if cached_response is not None:
            payload = dict(cached_response)
            payload["trace_id"] = trace_id
            return V3QueryResponse(**payload)

    try:
        # Stage 1: Embedding
        query_vec = await run_in_threadpool(
            lambda: embedding_service.encode_query(request.query).tolist()
        )

        # Stage 2: Hybrid retrieval (Dense + BM25 RRF) or fallback dense-only
        if hybrid_search_service is not None:
            retrieval = await hybrid_search_service.retrieve(
                query=request.query,
                query_vector=query_vec,
                top_k=request.candidate_pool,
            )
            stages["hybrid_used"] = True
        else:
            retrieval = await run_in_threadpool(
                lambda: lancedb_retriever_service.retrieve(
                    query_vector=query_vec,
                    top_k=request.candidate_pool,
                    query=request.query,
                )
            )
            stages["hybrid_used"] = False

        if not retrieval.is_success:
            raise HTTPException(status_code=502, detail="Retrieval failed")

        baseline_docs = list(retrieval.documents)

        # Stage 3: Aggregation Booster
        try:
            from services.aggregation_booster import boost_with_aggregation
            before_n = len(baseline_docs)
            baseline_docs = boost_with_aggregation(
                query=request.query,
                baseline_docs=baseline_docs,
                lancedb_retriever=lancedb_retriever_service,
                top_k=request.candidate_pool + 2,
                max_inject=2,
            )
            stages["booster_injected"] = max(0, len(baseline_docs) - before_n)
        except Exception as exc:
            logger.warning(f"Booster failed: {exc}")

        # Convert RetrievedDocument -> dict for reranker
        chunks_for_rerank = [
            {
                "text": d.text,
                "summary": (d.metadata or {}).get("summary", ""),
                "metadata": d.metadata or {},
                "score": float(getattr(d, "score", 0.0) or 0.0),
                "chunk_id": d.chunk_id,
                "source": d.source,
            }
            for d in baseline_docs
        ]

        # Stage 4: Reranker
        if reranker_service is not None and reranker_service.enabled:
            chunks_top = await run_in_threadpool(
                lambda: reranker_service.rerank(
                    query=request.query,
                    chunks=chunks_for_rerank,
                    top_k=request.top_k,
                    apply_lost_in_middle=request.apply_lost_in_middle,
                )
            )
            stages["reranker_used"] = True
            stages["lost_in_middle"] = request.apply_lost_in_middle
        else:
            chunks_top = chunks_for_rerank[: request.top_k]

        confidence = float(chunks_top[0].get("score", 0.0)) if chunks_top else 0.0

        # Stage 5: Generation
        is_enum = any(k in request.query.lower()
                      for k in ["bao nhiêu", "danh sách", "liệt kê", "tất cả",
                                "các ngành", "tổng cộng"])
        gen_result = await llm_generator_service.generate_answer(
            query=request.query,
            chunks=chunks_top,
            confidence=confidence,
            is_program_list_query=is_enum,
        )
        answer = gen_result.get("answer", "")
        provider = gen_result.get("provider", "unknown")

        sources = list({
            (c.get("source") or (c.get("metadata") or {}).get("source", "unknown"))
            for c in chunks_top
        })

        latency_ms = (time.perf_counter() - t_start) * 1000

        response_payload = {
            "query": request.query,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "chunks_used": len(chunks_top),
            "pipeline_stages": stages,
            "latency_ms": round(latency_ms, 1),
            "provider": provider,
            "trace_id": trace_id,
        }

        if query_cache:
            query_cache.set(cache_key, response_payload)

        return V3QueryResponse(**response_payload)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"v3_query failed trace_id={trace_id}: {exc}")
        raise HTTPException(status_code=500, detail="V3 query processing failed")


@app.post("/v2/graph/update")
async def incremental_graph_update(
    request: Dict[str, Any],
    raw_request: Request,
    _: None = Depends(require_admin_token),
):
    """Incrementally add new chunks to the knowledge graph.

    Request body: {"chunks": [{"id": "...", "text": "...", "faq_type": "..."}]}

    Enables 10-20 year scalability: add new admission data without full rebuild.
    """
    await _enforce_rate_limit(raw_request)

    if not unified_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    raw_chunks = request.get("chunks", [])
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise HTTPException(status_code=400, detail="No chunks provided")
    if len(raw_chunks) > MAX_GRAPH_UPDATE_CHUNKS:
        raise HTTPException(status_code=400, detail=f"Too many chunks (max={MAX_GRAPH_UPDATE_CHUNKS})")

    from src.domain.entities import Chunk as DomainChunk

    new_chunks = []
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("id", "")).strip()
        text = str(chunk.get("text", "")).strip()
        if not chunk_id or not text:
            continue
        if len(text) > MAX_GRAPH_CHUNK_TEXT_LENGTH:
            raise HTTPException(status_code=400, detail=f"Chunk text too long (max={MAX_GRAPH_CHUNK_TEXT_LENGTH})")
        metadata = chunk.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail="Invalid metadata")

        new_chunks.append(
            DomainChunk(
                chunk_id=chunk_id,
                text=text,
                faq_type=str(chunk.get("faq_type", "")),
                metadata=metadata,
            )
        )

    if not new_chunks:
        raise HTTPException(status_code=400, detail="No valid chunks provided")

    async with _graph_update_lock:
        await unified_pipeline.incremental_update(new_chunks)

    return {
        "status": "ok",
        "chunks_added": len(new_chunks),
        "graph_stats": unified_pipeline._graphrag.graph_stats,
    }


@app.get("/v2/graph/stats")
async def graph_stats():
    """Return current knowledge graph statistics."""
    if not unified_pipeline:
        return {"error": "Pipeline not initialized"}
    return unified_pipeline._graphrag.graph_stats


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting development server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
