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

import asyncio
from typing import Any, Callable, Dict, List, Literal, Optional
from dataclasses import replace
from asyncio import FIRST_COMPLETED, Lock, create_task
import time
import uuid
import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
import numpy as np
from pydantic import BaseModel, Field

# Import new services
from src.services.query_enhancer import query_enhancer
from src.services.lancedb_retrieval import (
    get_retriever,
    LanceDBRetriever,
    LanceDBRetrieverConfig,
)
from src.app.request_tracker import RequestTracker
from src.services.llm_generator import get_llm_generator
from src.services.reranker import (
    LOAD_STATE_DISABLED_BY_CONFIG,
    LOAD_STATE_LOAD_FAILED,
    RerankerService,
)
from src.services.query_cache import QueryCache
from src.services.guardrail import GuardrailService
from src.services.llm_generator import _chunk_text_for_stream as _chunk_answer_for_stream

# FIX-05 (ticket 44): `is_generation_failure` derives the degraded-generation
# signal from the generator's `provider` tag as a safety net, so a caller that
# forgets the explicit `generation_failed` flag still cannot publish a
# fallback answer as SUCCESS.
from src.services.llm_generator import is_generation_failure, is_policy_refusal
from src.services.abstention_policy import refusal_footer  # T10: refusal trailer for student abstentions
# Ticket 7: unanswered-questions queue and feedback log.
from src.services.unanswered_queue import UnansweredQueueService
from src.services.log_redaction import safe_query, redact_pii
from src.services import serve_probe as _serve_probe
from src.services.rate_limit import FixedWindowLimiter

# Keep existing imports for backward compatibility
from config.settings import RAGSettings
from config.request_context import set_request_id, reset_request_id
from chunker import ChunkConfig, Chunker

# Embedding service for provider-aware query/document encoding
from services.embedding import EmbeddingService

# Ticket 10: claim-level citation contract (services/citation.py). The model
# is defined in the service module and flows into this schema so the FE type
# is GENERATED from it (export_openapi.py -> api/openapi.json ->
# gen-api-types.mjs), never hand-copied.
from services.citation import ClaimCitation

# Initialize settings
settings = RAGSettings()

# Security/limits config
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
MAX_GRAPH_UPDATE_CHUNKS = int(os.getenv("MAX_GRAPH_UPDATE_CHUNKS", "100"))
MAX_GRAPH_CHUNK_TEXT_LENGTH = int(os.getenv("MAX_GRAPH_CHUNK_TEXT_LENGTH", "5000"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
GROUNDING_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", "0.18"))

_graph_update_lock = Lock()

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
    # Ticket 40: bumped once per rerank bypass caused by a dead model,
    # counted at every call seam (legacy /query, /v2 gate, /v3 gate).
    "reranker_bypass_load_failed": 0,
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
        "reranker_bypass_load_failed": int(_metrics.get("reranker_bypass_load_failed", 0)),
    }


# Ticket 40: the only trace values a caller may ever see. Anything else
# (MagicMock attribute objects, unknown strings) normalizes to None so a
# duck-typed test double can neither forge nor suppress a bypass trace.
_RERANK_TRACE_REASONS = (LOAD_STATE_LOAD_FAILED, LOAD_STATE_DISABLED_BY_CONFIG)


def _rerank_trace_reason(reranker_service: Any) -> Optional[str]:
    """Normalized ``rerank_skipped_reason`` for a (possibly fake) service."""
    reason = getattr(reranker_service, "skip_reason", None)
    return reason if reason in _RERANK_TRACE_REASONS else None


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


async def _enforce_rate_limit(request: Request) -> None:
    """Limit anonymous query traffic per client IP.

    The counter itself is :class:`FixedWindowLimiter`; this wrapper only binds
    the IP key. Rejection surfaces as 429 (the dependency raises it) — the
    previous inline version raised an undefined ``_RateLimited``, so the limit
    never applied to anyone.
    """
    await _query_rate_limiter(request)


# Anonymous query limiter, keyed by client IP. The budget is read through a
# callable rather than captured so a test can change it without reimporting
# this module. Admin traffic is deliberately *not* metered here: it has its own
# limiter keyed by actor (``src/routers/admin_rate_limit.py``), so one admin
# cannot spend another's budget or the anonymous pool's.
_query_rate_limiter = FixedWindowLimiter(
    limit=lambda: RATE_LIMIT_PER_MINUTE,
    key_fn=_client_ip,
)


# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
)

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown.

    Phase-1 (SLICE-WARMUP-UX.PHASE1): core warmups run in PARALLEL via
    `asyncio.gather` + `loop.run_in_executor` under an `asyncio.Semaphore(2)`
    to bound RAM pressure. Stage machine on `app.state.warmup_stage`:
    'warming' → 'ready' on success, 'failed' on any exception.
    """
    # ---- startup ----
    global query_enhancer_service, lancedb_retriever_service, llm_generator_service, embedding_service, unified_pipeline, reranker_service, query_cache, guardrail_service, hybrid_search_service

    logger.info("=" * 60)
    logger.info("Starting RAG API 2025 v3.0 - Unified PaddedRAG + GraphRAG")
    logger.info("=" * 60)

    # Initialize the warmup stage BEFORE any task starts so /health can
    # report 'warming' immediately.
    app.state.warmup_stage = "warming"

    # Canonical admissions store (ticket 12). File-backed under
    # settings.CANONICAL_STORE_ROOT; the admin lifecycle endpoints read it from
    # app.state. Fully guarded — a bad root or an unresolvable import must
    # degrade those endpoints to their 503 branch, never take down the API.
    try:
        try:
            from app.lifecycle import configure_canonical_record_store
        except ModuleNotFoundError:
            from rag2025.src.app.lifecycle import configure_canonical_record_store  # type: ignore
        configure_canonical_record_store(app)
    except Exception as exc:
        logger.error(f"Canonical record store wiring failed: {exc}", exc_info=True)
    # Ticket 7: the unanswered-questions queue. File-backed beneath the
    # canonical-store root; the admin surface and the /v2 hook read it from
    # app.state. Fail-soft: a queue that cannot be attached never takes down
    # startup, and the admin routes degrade to their 503 branch while /v2
    # keeps serving answers (the hook swallows queue write failures).
    try:
        app.state.unanswered_queue = UnansweredQueueService(
            data_dir=settings.CANONICAL_STORE_ROOT
        )
        logger.info(
            "unanswered queue attached at %s",
            app.state.unanswered_queue.queue_path,
        )
    except Exception as exc:
        logger.warning(f"unanswered queue wiring failed: {exc}")

    # ── Fast, synchronous warmups (no executor needed) ──
    try:
        logger.info("Initializing HYDE Query Enhancer...")
        query_enhancer_service = query_enhancer

        logger.info("Initializing LLM Generator...")
        llm_generator_service = get_llm_generator()

        logger.info("Initializing Query Cache...")
        query_cache = QueryCache(ttl_seconds=settings.CACHE_TTL)

        logger.info("Initializing Guardrail Service...")
        guardrail_service = GuardrailService(settings)
    except Exception as e:
        logger.error(f"Fast-init service failed: {e}", exc_info=True)
        app.state.warmup_stage = "failed"

    # ── Heavy warmups in parallel (Semaphore=2 caps concurrent model load) ──
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(2)

    async def _run(name: str, fn):
        t0 = time.perf_counter()
        try:
            async with sem:
                await loop.run_in_executor(None, fn)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(f"[warmup] {name} OK in {elapsed_ms}ms")
            return name, True, None
        except BaseException as exc:  # noqa: BLE001 — surface every failure for visibility
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(f"[warmup] {name} FAILED after {elapsed_ms}ms: {exc}", exc_info=True)
            return name, False, exc

    def _init_embedding():
        global embedding_service
        logger.info("Initializing Embedding Service...")
        embedding_service = EmbeddingService(settings)

    def _init_reranker():
        global reranker_service
        logger.info("Initializing Reranker Service...")
        reranker_service = RerankerService(settings)

    def _init_graphrag():
        global unified_pipeline
        logger.info("Initializing Unified RAG Pipeline (PaddedRAG + GraphRAG)...")
        from services.graphrag_retriever import UnifiedRAGPipeline
        unified_pipeline = UnifiedRAGPipeline.from_disk(
            alpha=settings.GRAPHRAG_ALPHA,
            beta=1.0 - settings.GRAPHRAG_ALPHA,
        )

    def _init_hybrid():
        global hybrid_search_service
        hybrid_search_service = build_hybrid_service(lancedb_retriever_service)
        if hybrid_search_service is not None:
            logger.info("HybridSearchService: BM25 index built successfully")

    def _init_lancedb():
        global lancedb_retriever_service
        try:
            lancedb_retriever_service = get_retriever()
            if lancedb_retriever_service:
                logger.info("LanceDB Retriever initialized successfully")
            else:
                logger.warning("LanceDB not available – degraded mode")
        except Exception as e:
            logger.error(f"LanceDB initialization failed: {e}")
            lancedb_retriever_service = None

    # Launch all heavy warmups in parallel (semaphore caps concurrency at 2).
    gather_results = await asyncio.gather(
        _run("lancedb", _init_lancedb),
        _run("embedding", _init_embedding),
        _run("reranker", _init_reranker),
        _run("graphrag", _init_graphrag),
        return_exceptions=False,
    )

    # HybridSearch depends on lancedb retriever → run AFTER lancedb init.
    # Run it as a follow-up parallel task under the same semaphore.
    if settings.USE_HYBRID_RETRIEVAL and lancedb_retriever_service:
        await asyncio.gather(_run("hybrid_bm25", _init_hybrid))

    # Stage flip: any failure → 'failed'; else 'ready'.
    failed_names = [name for (name, ok, _err) in gather_results if not ok]
    if any(not ok for (_n, ok, _e) in gather_results) or hybrid_search_service is None and settings.USE_HYBRID_RETRIEVAL:
        # Note: hybrid failure alone is non-fatal (logged) — only the core
        # services flip the stage to 'failed'.
        core_failed = [n for (n, ok, _e) in gather_results if not ok and n in {"lancedb", "embedding", "reranker", "graphrag"}]
        if core_failed:
            app.state.warmup_stage = "failed"
            logger.error(f"Warmup FAILED: {core_failed}")
        else:
            app.state.warmup_stage = "ready"
    else:
        app.state.warmup_stage = "ready"

    logger.info("=" * 60)
    logger.info(f"API Ready (stage={app.state.warmup_stage})")
    logger.info("=" * 60)

    # Share service references on app.state so extracted routers can
    # read them via request.app.state (avoiding module-name coupling).
    # The retrieval aliases go through the same publisher a runtime swap uses,
    # so startup and swap cannot disagree about which attributes exist.
    app.state.embedding_service = embedding_service
    app.state.reranker_service = reranker_service
    app.state._metrics_snapshot = _metrics_snapshot
    _publish_retrieval_aliases(app.state)

    yield

    # ---- shutdown ----
    logger.info("Shutting down RAG API 2025")


app = FastAPI(
    title="RAG API 2025 - HYDE + Multi-Embedding + LanceDB",
    description="Advanced RAG with provider-aware query enhancement and embedded vector retrieval",
    version="2.0.0",
    lifespan=lifespan
)
# Store lazy accessors on app so extracted routers can read module-level
# globals via request.app._health_*() — works regardless of the module's
# import name (since lambdas close over THIS module's scope).
app._health_lancedb = lambda: lancedb_retriever_service
app._health_embedding = lambda: embedding_service
app._health_reranker = lambda: reranker_service
app._health_pipeline = lambda: unified_pipeline
app._health_hybrid = lambda: hybrid_search_service
app._health_metrics = lambda: _metrics_snapshot


# ============================================================================
# /admin/reload-table runtime dependencies (RequestTracker + retriever builder)
#
# The route requires ``app.state.request_tracker`` and
# ``app.state.build_retriever`` and 503s without them. Neither was ever
# assigned in src/ — only tests assigned them — so the route was dead in
# production while every test passed: each test built the binding the code
# reads, which is exactly how the P1 "swap is decorative" bug survived before.
# Both are wired here, at import time, so a real app has them before any
# request arrives.
# ============================================================================


async def build_retriever_for_table(table_name: str) -> LanceDBRetriever:
    """Build a retriever for ``table_name`` — the reload route's candidate.

    Same construction path as startup (``LanceDBRetriever`` over
    ``settings.LANCEDB_URI``), differing only in which table it opens, so a
    reload and a restart cannot produce differently-configured retrievers.

    Raises whatever construction raises — notably ``FileNotFoundError`` from
    ``ensure_table`` when the table does not exist — because the route turns
    that into a 503 naming the failure. A retriever that silently fell back to
    the current table would make a typo'd ``table_name`` look like a successful
    reload that changed nothing.

    Blocking LanceDB work runs in a thread: the route awaits this on the event
    loop, and opening a table must not stall every other in-flight request.
    """
    config = LanceDBRetrieverConfig(
        uri=settings.LANCEDB_URI,
        table_name=table_name,
        embedding_dim=settings.EMBEDDING_DIM,
        default_top_k=5,
    )
    return await asyncio.to_thread(lambda: LanceDBRetriever(config=config))


app.state.request_tracker = RequestTracker()
app.state.build_retriever = build_retriever_for_table


@app.middleware("http")
async def _track_in_flight_requests(request: Request, call_next):
    """Count in-flight requests so reload/rollback can drain before swapping.

    Without this, ``tracker.drain()`` observes a counter that nothing
    increments: it returns "drained" instantly and the route closes the old
    retriever while queries are still reading it.

    ``/admin/*`` is skipped: the reload route itself holds ``_reload_lock``
    while draining, so counting it would make the drain wait on the very
    request performing it, and it could only ever time out.
    """
    if request.url.path.startswith("/admin/"):
        return await call_next(request)
    async with app.state.request_tracker.track():
        return await call_next(request)


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):  # noqa: D401
    """Phase-1 (SLICE-WARMUP-UX): enrich the 503 warming gate with a
    `retry_after_ms` field in the body so the FE retry layer can use
    sub-second backoff without parsing just the header."""
    from fastapi.responses import JSONResponse

    detail = exc.detail
    retry_after_seconds: int = 0
    extra_headers = dict(getattr(exc, "headers", None) or {})

    if isinstance(extra_headers, dict):
        ra = extra_headers.get("Retry-After") or extra_headers.get("retry-after")
        if ra is not None:
            try:
                retry_after_seconds = int(ra)
            except (TypeError, ValueError):
                retry_after_seconds = 0

    if exc.status_code == 503:
        body: Dict[str, Any] = {
            "detail": detail,
            "retry_after_ms": retry_after_seconds * 1000 if retry_after_seconds else 2000,
        }
    else:
        body = {"detail": detail}

    return JSONResponse(status_code=exc.status_code, content=body, headers=extra_headers)

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

# Any localhost/127.0.0.1 port is allowed in dev so the Vite dev server works
# regardless of which port it lands on (8080 default, auto-increments to 8081+
# when busy). Production should set ALLOWED_ORIGINS explicitly; when set, the
# regex is disabled so only the configured origins are accepted.
_localhost_origin_regex = (
    None if raw_allowed_origins
    else r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=_localhost_origin_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "x-admin-token", "Idempotency-Key"],
)

# Mount the /api/meta router (admission-year banner + data-freshness for the FE
# YearBanner). Without this the FE's GET /api/meta returns 404 and the banner
# silently renders its error state.
from routers.meta import router as meta_router
app.include_router(meta_router)

# Mount the /v2 deterministic recommender router (PHASE-A4). Pure-arithmetic;
# NO LLM, NO network, single-digit ms — see routers/recommender.py.
from routers.recommender import router as recommender_router
app.include_router(recommender_router)

# Mount the extracted /health router
from routers.health import router as health_router
app.include_router(health_router)

# Mount the extracted /admin router. It contributes POST /admin/warmup,
# POST /admin/reload-table (S13.4) and the ticket-12 canonical lifecycle
# endpoints under /admin/records. Before this mount those endpoints existed but
# were unreachable in the running app: nothing else in this module served an
# /admin/* path (the admin-token guard on POST /v2/graph/update is a separate
# route), so no path+method pair is registered twice.
from routers.admin import router as admin_router
app.include_router(admin_router)

# Mount the password-login session router (ticket T08). This router is
# mounted *before* the admin router so its lockout counter is reachable even
# when the admin rate limiter (keyed by actor) has not yet been satisfied —
# the login route itself has no authenticated-actor key.
from routers.admin_session import router as admin_session_router
app.include_router(admin_session_router)

# Mount the read-only /admin/audit router (ticket 14). It contributes only
# GET /admin/audit/events, GET /admin/audit/events/{event_id} and
# GET /admin/audit/verify — all behind require_action(RECORDS_READ). The
# ledger is append-only by construction and this router deliberately has no
# mutating route at all, so "no endpoint may delete or edit an event" holds
# by absence rather than by a guard that could be removed.
from routers.admin_audit import router as admin_audit_router
app.include_router(admin_audit_router)

# Mount the /v2/compare router (ticket 18). It contributes POST /v2/compare,
# the multi-major comparison contract: only majors sharing an effective year
# are grouped, every high-impact cell carries its own provenance, and a missing
# cell is reported as missing rather than inferred. The router is thin — the
# contract lives in src/domain/comparison.py and the service in
# src/services/comparison_service.py — and it declares its own /v2 prefix, so
# nothing else in this module serves that path+method pair.
from routers.comparison import router as comparison_router
app.include_router(comparison_router)

# Mount the review diff + preview router (ticket 25). It contributes only
# GET /admin/records/{record_id}/review-diff and
# POST /admin/records/{record_id}/review-preview — both read-only behind
# require_action(RECORDS_READ), and neither mutates a record: submitting for
# review from the preview screen goes through the existing lifecycle
# transitions route (ticket 12), so there is no second submit endpoint to
# drift from it. The preview itself builds a throwaway index and runs the
# real answer path; it never writes the production table, graph, or cache.
from routers.admin_preview import router as admin_preview_router
app.include_router(admin_preview_router)

# Identity + RBAC (ticket 13). One dependency for every admin-authenticated
# route in the process, including POST /v2/graph/update below, which used to
# carry its own divergent token check (503 when unconfigured, optional header).
# That route mutates the knowledge graph, so it keeps the machine credential:
# role 'ops' is granted graph.update and no records.* action, which is what
# makes "the shared token cannot publish admissions data" structural.
from routers.admin_auth import require_action as require_admin_action
from src.domain.admin_identity import AdminAction

# Re-export HealthResponse for backward compat (extracted to routers/health)
from routers.health import HealthResponse  # noqa: F401


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


# ============================================================================
# Runtime service swap seam (P1 fix, ticket 22 Part 0)
#
# The bug this closes: /admin/reload-table used to assign ``state.retriever``,
# but every query path reads the module globals above — so the swap was
# invisible to /query, /v2 and /v3 and the blue-green swap was decorative.
# ``swap_runtime_services`` is now the ONE place that rebinds a retriever for a
# running process, and both the reload route and the two-level rollback go
# through it.
# ============================================================================


def build_hybrid_service(retriever: Any) -> Optional["HybridSearchService"]:
    """The BM25/dense hybrid service over ``retriever``, or ``None``.

    The single construction path: app startup and every runtime swap both call
    this, so "which retriever does the sparse arm index" has exactly one answer
    and a failed BM25 build disables the sparse arm instead of leaving one
    indexing a table nobody serves.
    """
    if not (settings.USE_HYBRID_RETRIEVAL and retriever):
        return None
    from services.hybrid_search import HybridSearchService as _HybridSearchService

    svc = _HybridSearchService(lancedb_retriever=retriever, settings=settings)
    if not svc.build_bm25_index():
        logger.warning(
            "HybridSearchService: BM25 index build failed — disabling hybrid search"
        )
        return None
    return svc


def _publish_retrieval_aliases(app_state: Any) -> None:
    """Point every ``app.state`` retrieval alias at the current globals.

    ``state.retriever`` is the alias ``routers/meta.py`` reads for the
    data-freshness banner; before this it was written only by
    /admin/reload-table, so the banner kept describing the startup table while
    queries served another one.
    """
    app_state.lancedb_retriever_service = lancedb_retriever_service
    app_state.hybrid_search_service = hybrid_search_service
    app_state.unified_pipeline = unified_pipeline
    app_state.retriever = lancedb_retriever_service


def swap_runtime_services(retriever: Any, *, app_state: Any = None) -> Dict[str, Any]:
    """Rebind every place a query can read a retriever — the one swap seam.

    Rebinds, in one call:

    * ``lancedb_retriever_service`` — the global /query, /v2 and /v3 read;
    * ``hybrid_search_service`` — rebuilt over ``retriever`` through
      :func:`build_hybrid_service`, or ``None`` when hybrid retrieval is off or
      its BM25 index cannot be built (a stale index over the *old* table would
      otherwise keep serving mixed-release answers);
    * ``unified_pipeline``'s chunk fetchers — the GraphRAG expander admits
      PPR-only chunks through them, so they must come from the same table;
    * the ``app.state`` aliases (via :func:`_publish_retrieval_aliases`).

    The pipeline is **rebound, not rebuilt**: ``UnifiedRAGPipeline.from_disk``
    re-reads the graph from disk and builds fresh LLM clients, which would
    discard the in-memory graph updates applied through /v2/graph/update — and
    the only retriever-bound part of the pipeline is the expander's fetcher
    pair, so rebinding them moves the whole pipeline to the new table.

    Ordering is the point: the slow part (the BM25 build) happens **before**
    anything is rebound, and the rebind itself is a synchronous block with no
    ``await`` — so a caller that holds the reload drain lock cannot be running
    a query handler in the middle of it, and no answer can mix two releases.

    Returns ``{"previous_retriever", "hybrid_rebuilt", "pipeline_rebound"}``.
    The previous retriever is **returned, not closed**: the reload route closes
    it after the swap, while a rollback keeps it as the compensation target.
    """
    global lancedb_retriever_service, hybrid_search_service

    previous = lancedb_retriever_service
    new_hybrid = build_hybrid_service(retriever)

    lancedb_retriever_service = retriever
    hybrid_search_service = new_hybrid

    pipeline_rebound = False
    if unified_pipeline is not None:
        graphrag = getattr(unified_pipeline, "_graphrag", None)
        if graphrag is not None:
            graphrag._chunk_fetcher = getattr(retriever, "fetch_by_id", None)
            graphrag._chunk_batch_fetcher = getattr(retriever, "fetch_by_ids", None)
            pipeline_rebound = True

    if app_state is not None:
        _publish_retrieval_aliases(app_state)

    logger.info(
        "Runtime swap: retriever {} -> {} (hybrid_rebuilt={}, pipeline_rebound={})".format(
            getattr(previous, "table_name", previous),
            getattr(retriever, "table_name", retriever),
            new_hybrid is not None,
            pipeline_rebound,
        )
    )
    return {
        "previous_retriever": previous,
        "hybrid_rebuilt": new_hybrid is not None,
        "pipeline_rebound": pipeline_rebound,
    }


# The swap seam, reachable from a router that must not import this module by
# name (main.py runs as ``main`` under uvicorn and as a loaded module in tests).
# Same idiom as the ``_health_*`` accessors: a closure over THIS module's scope.
app._swap_runtime_services = lambda retriever: swap_runtime_services(
    retriever, app_state=app.state
)


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
            "Ns24 gateway LLM with same-gateway fallback per role"
        ],
        "endpoints": {
            "POST /query": "Main RAG query endpoint (simple string input)",
            "GET /health": "Health check with LanceDB status",
            "GET /docs": "Swagger API documentation"
        }
    }


@app.get("/metrics", response_model=Dict[str, Any])
async def metrics_endpoint():
    """Lightweight internal metrics endpoint."""
    return {
        "status": "ok",
        "metrics": _metrics_snapshot(),
    }


# @spec(SLICE-A.LATENCY) — Fire-and-forget model preloader. The Qwen3
# embedding model is lazy-loaded on first encode (~15s). When the FE
# opens the chat screen it calls GET /warmup so the first /v2/query
# does not pay the cold-load. Idempotent: the underlying _ensure_loaded
# double-checked lock makes repeat calls cheap.
@app.get("/warmup", response_model=Dict[str, Any])
async def warmup_endpoint():
    """Preload the embedding model to remove the cold-start penalty from
    the user's first real query. Returns 503 if the service is not yet
    initialized; otherwise 200 with the warmup latency in ms."""
    if embedding_service is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service not initialized. Please check server logs.",
        )
    t0 = time.perf_counter()
    await run_in_threadpool(lambda: embedding_service.encode_query("warmup"))
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {"warmed": True, "ms": elapsed_ms}


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

    logger.info(f"Received query: {safe_query(request.query, max_chars=100)}...")
    trace_id = str(uuid.uuid4())
    # Attribute every ns24 call made while answering this query to its trace_id
    # in the token ledger (same pattern as /v2's wrapper).
    request_id_token = set_request_id(trace_id)

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
                    f"guardrail_block trace_id={trace_id} "
                    f"query='{safe_query(request.query, max_chars=80)}' "
                    f"internal_code={precheck.internal_code} reason={precheck.reason}"
                )
                response_payload = {
                    "original_query": request.query,
                    "enhanced_query": request.query,
                    "query_type": "out_of_scope",
                    "answer": (
                        precheck.short_answer + refusal_footer()
                        if (not precheck.pii_detected and "📞 Hotline:" not in precheck.short_answer)
                        else precheck.short_answer
                    ),
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
                # V5-R013 / FIX 2: liet_ke (enumerate / list majors) must
                # pull from BOTH the aggregate chunk types
                # (tong_hop_tuyen_sinh, tong_hop_nhom_nganh) AND the
                # per-major detail chunks (thong_tin_nganh). The previous
                # omission made the liet_ke intent fall through to a
                # narrower "thong_tin_nganh" filter that excluded the
                # listing-style chunks (liet_ke_28_nganh_2026_v2,
                # bang_xep_hang_chi_tieu_2026_v2, tuyensinh_overview_2026_v2),
                # which is why the generator only had the COUNT signal and
                # produced "29 ngành" without listing names.
                "liet_ke": ["tong_hop_tuyen_sinh", "tong_hop_nhom_nganh", "thong_tin_nganh"],
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
                    # V5-R013 / FIX 2: pull from BOTH aggregate listing chunks
                    # (tong_hop_*) AND per-major detail chunks (thong_tin_nganh)
                    # so the liet_ke intent can list the 29 tên ngành instead of
                    # only emitting a count.
                    return {
                        "or_conditions": [
                            {"faq_type": t}
                            for t in INTENT_TO_FAQ_TYPES["liet_ke"]
                        ]
                    }

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

            # Step 2: Embedding Encoding - encode ALL variants to vectors in ONE batch call
            all_chunks = []
            all_scores = []

            # For program/overview queries, use higher top_k to get more results
            is_program_query = metadata_filter is not None
            retrieval_top_k = top_k * 3 if is_program_query else top_k

            # Sửa #6: 1 call encode_batch thay vì N×encode_query.
            # query_vectors is List[List[float]] matching variants order.
            query_vectors = await run_in_threadpool(
                lambda: embedding_service.encode_batch(variants).astype(np.float32).tolist()
            )

            for i, (variant, query_vector) in enumerate(zip(variants, query_vectors)):
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
                # Ticket 40: rerank() silently returns the input order when
                # the model is dead. Count and log the bypass at this seam —
                # a dead reranker must not degrade answer quality in silence.
                _rr_reason = _rerank_trace_reason(reranker_service)
                if _rr_reason == LOAD_STATE_LOAD_FAILED:
                    await _metric_inc("reranker_bypass_load_failed")
                    logger.warning(
                        f"query_rerank_bypass trace_id={trace_id} "
                        "reason=load_failed"
                    )
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
                    f"no_result_classification "
                    f"query='{safe_query(original_query, max_chars=80)}' "
                    f"internal_code={no_result.internal_code} reason={no_result.reason}"
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

            # FIX-05 (ticket 44): the same invariant holds on the legacy
            # /query surface — a generator fallback must not be published as
            # SUCCESS. Guarded on the untouched default so a data-gap
            # classification (no chunks) keeps its more specific status.
            generation_failed = bool(result.get("generation_failed")) or is_generation_failure(
                result.get("provider")
            )
            if generation_failed and status_code == "SUCCESS":
                status_code = "GENERATION_FAILED"
                status_reason = "generation_failed"
                internal_status_code = (
                    "GENERATION_FAILED" if guardrail_service.expose_internal() else None
                )
                await _metric_inc("query_generation_failed")
                logger.warning(
                    f"query_generation_failed trace_id={trace_id} provider={result['provider']}"
                )

            # Step 7: Groundedness scoring
            groundedness = await asyncio.to_thread(
                _safe_groundedness_score, result["answer"], chunks
            )
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
            # T16 (K3): stamp when a returned chunk belongs to a probed release.
            _stamp_probe_served(raw_request.app, chunks)

            return QueryResponse(**response_payload)

    except HTTPException:
        await _metric_inc("query_errors")
        raise
    except Exception as e:
        await _metric_inc("query_errors")
        logger.exception(f"Query failed trace_id={trace_id}: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")
    finally:
        reset_request_id(request_id_token)


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
    # Multi-turn conversation context (bounded). Backward-compatible: clients
    # that omit this field keep the original single-turn behavior unchanged.
    # The cap (4 messages = 2 user + 2 assistant) is enforced at the FE slice
    # site; the BE tolerates a small overage by re-slicing in llm_generator.
    history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional bounded chat history, e.g. [{'role':'user','content':'...'}, ...] (max ~4 msgs)",
    )


class UnifiedQueryResponse(BaseModel):
    """Response from unified pipeline with routing metadata."""
    query: str
    route: str
    answer: str
    # /v2 routes through generate_answer (S16.x), which now emits enriched
    # SourceChip dicts (id/title/url/snippet/data_year) — same as /query.
    # Was List[str]; mismatch caused a 500 ValidationError on every real
    # query. Aligned to List[SourceChip] so the FE chip UX works on /v2.
    sources: List[SourceChip]
    # Ticket 10: per-claim provenance for every high-impact number in the
    # answer (cutoff / timeline / quota / tuition). Optional-by-default: a
    # response without citations (canned auto-answer, guardrail block) is
    # valid; a missing page inside a citation is valid too (acceptance #3).
    citations: List[ClaimCitation] = Field(default_factory=list)
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

    # FIX-05 (ticket 44): answer-level confidence, SEPARATE from the
    # retrieval-level `confidence` above. `None` whenever no model produced
    # the answer (generation failed, or the answer was a canned auto-answer)
    # — never a number, so the FE cannot mistake retrieval quality for
    # answer quality. `confidence` keeps its retrieval meaning only.
    answer_confidence: Optional[float] = None

    # Ticket 40: why rerank was skipped, visible on BOTH /v2 paths:
    # "load_failed" (model dead) | "disabled_by_config" (intentional) |
    # None (rerank ran, or was outranked by confidence). Additive Optional
    # with a safe default — existing consumers unaffected.
    rerank_skipped_reason: Optional[str] = None

    # Ticket 17: how the answer was produced, so the FE can tell an
    # evidence-policy abstention apart from a model answer without
    # classifying prose. "evidence-policy:{reason_code}" means the model
    # was never called (the policy refusal is the whole answer);
    # "evidence-policy:unresolved-conflict" is the conflict case;
    # None means a model wrote it. Additive Optional — unaffected consumers
    # keep working, and the FE keys its abstain/conflict states on this.
    provider: Optional[str] = None


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
    history: Optional[List[Dict[str, str]]] = None,
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
        Dict with keys: `answer`, `sources`, `confidence`, `provider`, plus the
        FIX-05 signals `generation_failed` (the generator produced no model
        output) and `model_answer` (`False` for both a failed generation and a
        canned router auto-answer — it is what gates `answer_confidence` on the
        response payload). The shape is a superset of
        `LLMGenerator.generate_answer()`'s return so the /v2 response builder
        is unchanged for existing consumers.
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
            # FIX-05: no generator round-trip happened, so nothing failed —
            # but no model wrote this answer either (`model_answer=False`),
            # which keeps `answer_confidence` null on the payload.
            "generation_failed": False,
            "model_answer": False,
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
        history=history,
    )
    # Ensure the contract keys the /v2 response builder reads are present.
    #
    # FIX-05 (ticket 44): `provider` is a generator-internal diagnostic. The
    # public signal is `generation_failed`, taken from the explicit flag the
    # generator now returns, with `is_generation_failure(provider)` as a
    # derived safety net (covers "Fallback(Error) + season-fallback" style
    # suffixed tags and any caller path that predates the flag).
    provider = gen_result.get("provider", "unknown")
    generation_failed = bool(gen_result.get("generation_failed")) or is_generation_failure(provider)
    return {
        "answer": gen_result.get("answer", ""),
        "sources": list(gen_result.get("sources", [])),
        # Ticket 10: pass the generator's claim-level citations through to
        # the /v2 payload builder. Without this line the response field would
        # exist but always be empty (Main's FIX-05 condition #3).
        "citations": list(gen_result.get("citations", [])),
        "confidence": float(getattr(rag_result, "confidence", 0.0) or 0.0),
        "provider": provider,
        "generation_failed": generation_failed,
        # FIX-05: `model_answer` gates `answer_confidence` on the payload, so
        # it must mean "a model wrote this" — not merely "nothing failed".
        # A T11 policy refusal is intentional (generation_failed=False) but
        # model-free, and `not generation_failed` alone marked it True, which
        # published a groundedness number as the refusal's answer confidence:
        # measured live, a year-mismatch refusal shipped
        # answer_confidence=0.8696876764297485 for text no model wrote, and the
        # FE renders that number (useChatStream.ts reads answer_confidence).
        # A fabricated confidence on an abstention is exactly the
        # "unsafe confident-answer" the abstention policy exists to prevent.
        "model_answer": not generation_failed and not is_policy_refusal(provider),
    }


# ───────────────────────────────────────────────────────────────────────────
# /v2 shared core — extracted so /v2/query AND /v2/query/stream call the
# IDENTICAL pipeline (router → booster → graph → _synthesize_v2_answer with
# all post-guards). The stream endpoint must NEVER re-implement the pipeline;
# it must reuse this helper so the streamed text is the post-guard answer.
# Returns a tuple (response_payload, rag_result) where rag_result is needed
# for the route label in the meta event.
# ───────────────────────────────────────────────────────────────────────────
async def _build_v2_query_payload(
    request: UnifiedQueryRequest,
    raw_request: Request,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run the /v2 pipeline end-to-end and return the response_payload dict
    shaped for `UnifiedQueryResponse(**payload)`. Raises HTTPException on
    guardrail precheck / 503 (caller decides whether to short-circuit
    streaming or surface the JSON error).

    The request's trace_id is bound as the LLM request id, so every ns24 call
    made while answering this query is attributed to it in the token ledger.
    """
    await _enforce_rate_limit(raw_request)
    trace_id = str(uuid.uuid4())
    token = set_request_id(trace_id)
    try:
        return await _build_v2_query_payload_inner(
            request, raw_request, progress_cb, trace_id
        )
    finally:
        reset_request_id(token)


async def _build_v2_query_payload_inner(
    request: UnifiedQueryRequest,
    raw_request: Request,
    progress_cb: Optional[Callable[[str], None]],
    trace_id: str,
) -> Dict[str, Any]:
    """Body of `_build_v2_query_payload`; trace_id is supplied by the wrapper."""
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
            await _queue_unanswered(
                getattr(raw_request.app.state, "unanswered_queue", None),
                trace_id, request.query,
                payload["route"], payload["status_code"],
            )
            return payload

    if guardrail_service is not None:
        try:
            # L3.1 (merged guard): scope check moved into the router's
            # merged LLM call (_classify_guard_route). Only PII regex
            # pre-check remains here — it is pure regex (no LLM round-trip)
            # and must fire BEFORE the router so PII never leaks into
            # retrieval or generation.
            if guardrail_service._contains_sensitive_pii(request.query):
                await _metric_inc("query_guardrail_blocks")
                await _metric_inc("query_pii_blocks")
                pii_answer = guardrail_service._pii_warning_answer()
                logger.info(
                    f"v2_pii_block trace_id={trace_id} "
                    f"query='{safe_query(request.query, max_chars=80)}'"
                )
                oos_payload = {
                    "query": request.query,
                    "route": "guardrail",
                    "answer": pii_answer,
                    "sources": [],
                    "confidence": 0.0,
                    "groundedness_score": 0.0,
                    "router_info": {
                        "step_back": None,
                        "intent": "guardrail_block",
                        "complexity": 0,
                        "reasoning": "sensitive_pii_detected",
                        "hyde_variants": [],
                    },
                    "graph_stats": None,
                    "latency_ms": 0.0,
                    "trace_id": trace_id,
                    "status_code": "SENSITIVE_PII_DETECTED",
                    "status_reason": "sensitive_pii_detected",
                    "data_gap_hints": [
                        "Ẩn bớt thông tin định danh trước khi gửi câu hỏi",
                        "Chỉ giữ lại phần nội dung liên quan tuyển sinh",
                    ],
                    "internal_status_code": (
                        "SENSITIVE_PII_DETECTED"
                        if guardrail_service.expose_internal()
                        else None
                    ),
                    "pii_detected": True,
                }
                if query_cache:
                    query_cache.set(cache_key, oos_payload)
                return oos_payload
        except Exception as _guard_exc:
            logger.warning(f"v2_pii_precheck_error: {_guard_exc}")

    # ── Ticket 35 / resolution (A): calibrated retrieval confidence ───────
    # `retrieval_confidence` is the CALIBRATED (dense-cosine scale) value the
    # adaptive rerank gate consumes. Hybrid documents carry RRF fusion scores
    # (max 0.6/61 + 0.4/61 = 0.01639) — a ranking scale, not a relevance
    # scale — while the gate compares against
    # RERANKER_ADAPTIVE_CONFIDENCE=0.5, so feeding it an RRF score would make
    # the comparison always true and rerank nearly every query (~+1.8s),
    # destroying the fast path. It stays None on the dense path, which
    # already produces this value, so flag-off behavior is unchanged (C2).
    retrieval_confidence: Optional[float] = None

    async def _compute_baseline() -> list:
        """Compute the PaddedRAG baseline_docs for the /v2 hot path.

        Wraps the original sequential block (embed + lancedb retrieve +
        aggregation-booster) in a coroutine so it can run CONCURRENTLY with
        the router's LLM round-trip. The router depends only on the raw
        query string, so today the two were strictly sequential (sum of
        latencies); running them in parallel turns that into max(). All
        existing try/except fallbacks, the booster call, and the
        `request.top_k * 3` slicing are preserved EXACTLY.

        Ticket 35: when `hybrid_search_service` is wired (USE_HYBRID_RETRIEVAL
        = true and the BM25 index built at startup), the fused dense+BM25
        list supplies the RANKING while `retrieval_confidence` carries the
        calibrated confidence for the rerank gate. When it is None the dense
        call below is the single retrieval, exactly as before.
        """
        nonlocal retrieval_confidence
        baseline_docs: list = []
        if lancedb_retriever_service and embedding_service:
            try:
                query_vec = await run_in_threadpool(
                    lambda: embedding_service.encode_query(request.query).tolist()
                )
                if hybrid_search_service:
                    # Hybrid arm — reuses the query_vec encoded above (never
                    # re-encode: that is ~1.1s). `retrieve()` has no separate
                    # `query` plumbing to the dense arm; it injects the query
                    # text into `metadata_filter` itself so the temporal
                    # classifier still sees it, hence metadata_filter=None.
                    result = await hybrid_search_service.retrieve(
                        query=request.query,
                        query_vector=query_vec,
                        top_k=request.top_k * 3,
                        metadata_filter=None,
                    )
                    # Resolution (A): `result.documents[*].score` stays the RRF
                    # fusion score — that IS the ranking. The gate's input is
                    # re-derived from the dense arm on the cosine scale: the
                    # same quantity, from the same call parameters, that the
                    # dense path feeds the gate. Measured at ~52-63ms, and it
                    # runs inside max() with the ~5s router leg.
                    probe = lancedb_retriever_service.retrieve(
                        query_vector=query_vec,
                        top_k=request.top_k * 3,
                        query=request.query,
                    )
                    if probe.is_success and probe.documents:
                        retrieval_confidence = float(probe.documents[0].score)
                    else:
                        # No dense candidate at all → no calibrated evidence.
                        # Use the cosine-scale FLOOR (0.0) rather than letting
                        # `rag_result.confidence` fall through: that value is an
                        # RRF score (max 0.0164) which is *always* below 0.5, so
                        # the gate would be driven by a scale-mismatched number
                        # again — the exact defect this resolution removes. 0.0
                        # is a real cosine value and keeps the invariant "the
                        # gate only ever compares cosine-scale confidences".
                        # The gate then reranks, which is the conservative
                        # choice: a sparse-only pool is the least reliable
                        # ordering and the most valuable to rerank.
                        retrieval_confidence = 0.0
                        logger.info(
                            "v2_hybrid_confidence_unavailable: dense probe "
                            "returned no documents; using cosine floor 0.0 "
                            "so the rerank gate never sees an RRF score"
                        )
                else:
                    result = lancedb_retriever_service.retrieve(
                        query_vector=query_vec,
                        top_k=request.top_k * 3,
                        query=request.query,
                    )
                if result.is_success:
                    baseline_docs = result.documents
            except Exception as exc:
                logger.warning(f"LanceDB retrieval failed: {exc}")

        # 1b. Aggregation Booster — fires even when vector retrieval returns
        # zero docs (BUG-D fix). The booster injects curated chunks via
        # aggregation rules, giving the LLM something to work with even when
        # the vector index has no matching content for the query.
        if lancedb_retriever_service is not None:
            try:
                from services.aggregation_booster import boost_with_aggregation
                baseline_docs = boost_with_aggregation(
                    query=request.query,
                    baseline_docs=baseline_docs,
                    lancedb_retriever=lancedb_retriever_service,
                    top_k=request.top_k * 3,
                    max_inject=2,
                )
            except Exception as exc:
                logger.warning(f"v2_booster_failed: {exc}")

        return baseline_docs

    # Kick off the router LLM round-trip and the baseline retrieval
    # CONCURRENTLY. The router depends only on the raw query string — it
    # does NOT need baseline_docs — so the ~5s router LLM call and the
    # embedding+retrieve+booster block are independent and can overlap.
    # total_latency: max(router, baseline) instead of sum(router, baseline).
    # The router already has its own internal try/except fallback (returns
    # L3.1 (merged guard): kick off the router LLM call in parallel with
    # baseline embedding/retrieval. The router's merged LLM call
    # (_classify_guard_route) handles BOTH scope check AND route/step_back,
    # so we no longer need a separate precheck. The OOS short-circuit now
    # happens AFTER the router returns (cheap, just a discard of the
    # baseline_docs we already computed).
    router_task = create_task(unified_pipeline._router.route(request.query))
    if progress_cb:
        progress_cb("Đang phân tích câu hỏi…")
    baseline_docs = await _compute_baseline()
    router_result = await router_task
    if progress_cb:
        progress_cb("Đang truy xuất tài liệu…")

    # L3.1 (merged guard): scope check now lives inside the router's merged
    # LLM call (_classify_guard_route). If the router says out-of-scope,
    # discard baseline_docs (one cheap wasted vector lookup, no extra LLM)
    # and return the guardrail block payload.
    if not router_result.is_in_scope:
        await _metric_inc("query_guardrail_blocks")
        oos_answer = (
            "Xin lỗi, câu hỏi này nằm ngoài phạm vi tư vấn tuyển sinh HUSC. "
            "Bạn vui lòng hỏi về ngành, điểm chuẩn, học phí, tổ hợp hoặc "
            "phương thức xét tuyển của Trường Đại học Khoa học - Đại học Huế."
            + refusal_footer()
        )
        logger.info(
            f"v2_router_scope_block trace_id={trace_id} "
            f"query='{safe_query(request.query, max_chars=80)}' "
            f"reason={router_result.reasoning}"
        )
        oos_payload = {
            "query": request.query,
            "route": "guardrail",
            "answer": oos_answer,
            "sources": [],
            "confidence": 0.0,
            "groundedness_score": 0.0,
            "router_info": {
                "step_back": None,
                "intent": "guardrail_block",
                "complexity": 0,
                "reasoning": router_result.reasoning,
                "hyde_variants": [],
            },
            "graph_stats": None,
            "latency_ms": 0.0,
            "trace_id": trace_id,
            "status_code": "NOT_IN_HUSC_SCOPE",
            "status_reason": router_result.reasoning,
            "data_gap_hints": [
                "Câu hỏi nằm ngoài phạm vi tư vấn tuyển sinh HUSC",
                "HUSC chỉ tư vấn về ngành, điểm chuẩn, học phí, tổ hợp và phương thức xét tuyển của Trường ĐH Khoa học - ĐH Huế",
            ],
            "internal_status_code": (
                "NOT_IN_HUSC_SCOPE"
                if guardrail_service is not None and guardrail_service.expose_internal()
                else None
            ),
            "pii_detected": False,
        }
        if query_cache:
            query_cache.set(cache_key, oos_payload)
        await _queue_unanswered(
            getattr(raw_request.app.state, "unanswered_queue", None),
            trace_id, request.query,
            oos_payload["route"], oos_payload["status_code"],
        )
        return oos_payload

    status_code = "SUCCESS"
    status_reason: Optional[str] = None
    data_gap_hints: List[str] = []
    internal_status_code: Optional[str] = None
    pii_detected = False

    async with _track_latency("unified_query_total_latency_ms", "unified_query_count_latency"):
        rag_result = await unified_pipeline.query(
            user_query=request.query,
            baseline_docs=baseline_docs,
            top_k=request.top_k,
            router_result=router_result,
        )

        if (not rag_result.documents) and guardrail_service is not None:
            try:
                no_result = await guardrail_service.classify_no_result(request.query)
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
            rag_result = replace(
                rag_result,
                route="padded_rag",
                documents=baseline_docs[:request.top_k],
                ppr_scores={},
                confidence=(baseline_docs[0].score if baseline_docs else 0.0),
            )
            logger.info("Route override applied: padded_rag")
        elif request.force_route == "graph_rag" and baseline_docs:
            docs, ppr_scores = await unified_pipeline._graphrag.retrieve(
                query=request.query,
                router_result=rag_result.router_result,
                baseline_docs=baseline_docs,
                top_k=request.top_k,
            )
            rag_result = replace(
                rag_result,
                route="graph_rag",
                documents=docs,
                ppr_scores=ppr_scores,
                confidence=(docs[0].score if docs else 0.0),
            )
            logger.info("Route override applied: graph_rag")

        # ── Ticket 35 / resolution (A): calibrated retrieval confidence ──
        # Every route above derives `confidence` from the ranking scores,
        # which are RRF fusion scores whenever hybrid supplies the ranking
        # (max 0.6/61 + 0.4/61 = 0.01639). The adaptive gate below compares
        # that value against RERANKER_ADAPTIVE_CONFIDENCE=0.5 — a cosine-scale
        # threshold — so an RRF value would fire rerank on essentially every
        # query and destroy the fast path. Restore the calibrated value
        # measured in `_compute_baseline`; the ranking order is untouched.
        # `None` means the dense path already produced a calibrated value and
        # must be left alone (C2: flag-off is byte-identical to before).
        # Guarded on `documents` so the auto-answer / empty-retrieval paths,
        # which never consult the gate, keep their existing confidence.
        if retrieval_confidence is not None and rag_result.documents:
            rag_result.confidence = retrieval_confidence
        # Ticket 40: pipeline trace the caller reads to learn rerank was
        # skipped and why. Set in the dead/off branch below; stays None when
        # the reranker ran or was legitimately outranked by confidence.
        rerank_skipped_reason: Optional[str] = None

        # ── Adaptive reranker gate ────────────────────────────────────────
        # When the reranker is enabled AND the retrieval confidence is below
        # the adaptive threshold, rerank the top RERANKER_POOL candidates to
        # improve ordering for hard/ambiguous queries. High-confidence (easy)
        # queries skip rerank entirely to preserve the fast path.
        #
        # CRITICAL: rerank() truncates chunk text ONLY inside the
        # cross-encoder scoring (max_length tokens). The FULL original chunk
        # text is always preserved in rag_result.documents — the generator
        # receives untruncated text.
        # ──────────────────────────────────────────────────────────────────
        if (
            rag_result.documents
            and reranker_service is not None
            and reranker_service.enabled
            and rag_result.confidence < reranker_service.adaptive_confidence
        ):
            pool_n = min(reranker_service.pool, len(rag_result.documents))
            # Convert RetrievedDocument list → dict list for reranker
            docs_for_rerank = [
                {
                    "text": d.text,
                    "score": d.score,
                    "chunk_id": d.chunk_id,
                    "source": d.source,
                    "metadata": d.metadata if isinstance(d.metadata, dict) else {},
                    "_idx": idx,
                }
                for idx, d in enumerate(rag_result.documents[:pool_n])
            ]
            reranked = await run_in_threadpool(
                lambda: reranker_service.rerank(
                    query=request.query,
                    chunks=docs_for_rerank,
                    top_k=pool_n,
                    apply_lost_in_middle=True,
                )
            )
            # Reorder rag_result.documents to match the reranked order
            reranked_indices = [r["_idx"] for r in reranked]
            remaining_docs = rag_result.documents[pool_n:]
            rag_result.documents = [
                rag_result.documents[i] for i in reranked_indices
            ] + remaining_docs
            if rag_result.documents:
                rag_result.confidence = rag_result.documents[0].score
            logger.info(
                f"v2_adaptive_rerank trace_id={trace_id} pool={pool_n} "
                f"conf={rag_result.confidence:.3f} thr={reranker_service.adaptive_confidence}"
            )
        elif rag_result.documents and reranker_service is not None and reranker_service.enabled:
            logger.info(
                f"v2_rerank_skipped trace_id={trace_id} "
                f"conf={rag_result.confidence:.3f} >= thr={reranker_service.adaptive_confidence}"
            )
        elif rag_result.documents and reranker_service is not None:
            # Ticket 40: documents exist and the reranker exists as an object
            # but cannot run — either the model is dead (load failed) or it
            # was turned off on purpose. Both must be visible; only the dead
            # model bumps the bypass counter.
            rerank_skipped_reason = _rerank_trace_reason(reranker_service)
            if rerank_skipped_reason == LOAD_STATE_LOAD_FAILED:
                await _metric_inc("reranker_bypass_load_failed")
                logger.warning(
                    f"v2_rerank_bypass trace_id={trace_id} reason=load_failed"
                )
            elif rerank_skipped_reason is not None:
                logger.info(
                    f"v2_rerank_skipped trace_id={trace_id} "
                    f"reason={rerank_skipped_reason}"
                )

        # Ticket 35 / resolution (A): the rerank branch above rewrites
        # `confidence` from the reordered top document, whose score is an RRF
        # fusion score while hybrid supplies the ranking. Re-apply the
        # calibrated value so the response payload — and the tier the
        # `v2_adaptive_rerank` log line reports — never carry an RRF number.
        # The gate's decision above already used the calibrated value.
        if retrieval_confidence is not None and rag_result.documents:
            rag_result.confidence = retrieval_confidence

        if llm_generator_service is None:
            raise HTTPException(
                status_code=503,
                detail="LLM generator not initialized",
            )
        if progress_cb:
            progress_cb("Đang tổng hợp câu trả lời…")
        synthesis = await _synthesize_v2_answer(
            rag_result=rag_result,
            query=request.query,
            generator=llm_generator_service,
            history=request.history,
        )
        answer = synthesis["answer"]
        sources = synthesis["sources"]

        # ── FIX-05 (ticket 44): degraded generation is NOT SUCCESS ────────
        # A fallback answer (no provider configured, or every provider
        # raising) is a fixed Vietnamese string written by the generator,
        # not a model completion. It used to reach the student labelled
        # SUCCESS with a retrieval confidence attached, which reads as a
        # real answer. Publish the degraded status instead.
        #
        # Precedence: the empty-retrieval classification above (INSUFFICIENT_DATA
        # / HUSC_ENTITY_NOT_FOUND / …) wins when it already fired, because a
        # data-gap diagnosis is the actionable one; the override applies while
        # the status is still the untouched default, i.e. exactly the
        # "documents exist but the LLM is dead" case this fix targets, plus the
        # "no documents and the guardrail had nothing to classify" case which
        # would otherwise leak SUCCESS.
        generation_failed = bool(synthesis.get("generation_failed"))
        if generation_failed and status_code == "SUCCESS":
            status_code = "GENERATION_FAILED"
            status_reason = "generation_failed"
            internal_status_code = (
                "GENERATION_FAILED"
                if guardrail_service is not None and guardrail_service.expose_internal()
                else None
            )
            await _metric_inc("unified_query_generation_failed")
            logger.warning(
                f"v2_generation_failed trace_id={trace_id} "
                f"provider={synthesis.get('provider')} "
                f"documents={len(rag_result.documents)}"
            )

        groundedness = await asyncio.to_thread(
            _safe_groundedness_score,
            answer,
            [
                {"text": d.text, "summary": d.text, "text_plain": d.text}
                for d in rag_result.documents
            ],
        )
        if groundedness < GROUNDING_THRESHOLD and rag_result.documents and not generation_failed:
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
            # Ticket 10: claim -> chunk -> record -> source trace, same dict
            # on the JSON and SSE paths (synthesis covers both endpoints).
            "citations": synthesis.get("citations", []),
            "confidence": rag_result.confidence,
            "groundedness_score": groundedness,
            # FIX-05 (ticket 44): answer-level confidence. Null whenever no
            # model wrote the answer — a failed generation or a canned
            # router auto-answer. It is never a substitute for the retrieval
            # `confidence` above, and never a fabricated default.
            "answer_confidence": (
                groundedness if synthesis.get("model_answer") else None
            ),
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
            # Ticket 17: the generator already distinguishes an
            # evidence-policy abstention from a model answer via
            # `provider` ("evidence-policy:{reason_code}"), but the value
            # was dropped here — so an abstention reached the FE as
            # status_code SUCCESS with ordinary prose, indistinguishable
            # from a real answer. Additive Optional; None on the model path.
            "provider": synthesis.get("provider"),
            # Ticket 40: "load_failed" | "disabled_by_config" | None.
            "rerank_skipped_reason": rerank_skipped_reason,
        }

        if query_cache:
            query_cache.set(cache_key, response_payload)
        # T16 (K3): stamp when a returned chunk belongs to a probed release.
        _stamp_probe_served(raw_request.app, rag_result.documents)
        await _queue_unanswered(
            getattr(raw_request.app.state, "unanswered_queue", None),
            trace_id, request.query,
            response_payload["route"], response_payload["status_code"],
        )
    return response_payload

# Ticket 7: hook into the /v2 answer path. When the pipeline returns an
# answer that is NOT an answer (out of scope, no basis, provider failure),
# push the redacted question into the unanswered queue. Fail-soft: a failed
# queue must never disturb the answer being returned to the student.


def _is_unanswerable(status_code: str) -> bool:
    """Whether ``status_code`` means the question was not answered.

    Three categories, exactly what the ticket wants queued: out of scope
    (NOT_IN_HUSC_SCOPE), missing basis (the no_result classification:
    INSUFFICIENT_DATA, HUSC_ENTITY_NOT_FOUND, ENTITY_PARSING_GAP, and any
    future code the guardrail publishes), and provider failure
    (GENERATION_FAILED). The sensitive-PII block is deliberately excluded:
    the student received a hint answer ("Ẩn bớt thông tin định danh trước khi
    gửi câu hỏi") rather than an unanswered question, so it is not queued.
    """
    return status_code not in ("SUCCESS", "SENSITIVE_PII_DETECTED")


async def _queue_unanswered(svc: Any, trace_id: str, query: str, route: str,
                            status_code: str) -> None:
    """Push an unanswered question into the queue, fail-soft."""
    # Only genuinely unanswered answers are queued: a SUCCESS answer and the
    # sensitive-PII hint are not unanswered questions.
    if not _is_unanswerable(status_code):
        return
    # A bare test app has no queue attached (lifespan never ran): skip
    # instead of raising, so the answer path never depends on the queue.
    if svc is None:
        return
    try:
        await asyncio.to_thread(
            svc.push, redact_pii(query), trace_id, route, status_code
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"unanswered_queue push failed: {exc}")


def _stamp_probe_served(app: Any, chunks: Any) -> None:
    """Stamp the T16 probe log's ``served_at`` when a returned chunk belongs
    to a probed release (K3). Sync call inside query handlers: the stamping
    itself is a small append-only file write, and it runs after the answer
    is already built, so it never delays generation. Fail-soft, like
    ``_queue_unanswered``: a missing data dir or IO error only logs."""
    try:
        queue = getattr(getattr(app, "state", None), "unanswered_queue", None)
        queue_path = getattr(queue, "queue_path", None)
        if queue_path is not None:
            data_dir = queue_path.parent.parent
        else:
            data_dir = settings.CANONICAL_STORE_ROOT
        probe_path = _serve_probe.probe_log_path(data_dir)
        if not probe_path.is_file():
            return
        stamped = _serve_probe.stamp_served_from_chunks(data_dir, chunks or [])
        if stamped:
            logger.info(f"T16 probe: stamped served_at for {stamped} release(s)")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"T16 probe stamp failed: {exc}")

def _format_sse(event: str, data: Dict[str, Any]) -> bytes:
    """Render a single SSE frame as bytes: `event: <name>\\ndata: <json>\\n\\n`.

    The payload is sanitized via `repr()` fallback so a stray MagicMock
    in the meta dict (e.g. from a unit-test fake) does not crash the
    streaming response with a 500. Real /v2 payloads contain only
    JSON-serializable primitives (str, int, float, bool, list, dict)."""
    safe = json.loads(json.dumps(data, default=lambda o: repr(o), ensure_ascii=False))
    return f"event: {event}\ndata: {json.dumps(safe, ensure_ascii=False)}\n\n".encode("utf-8")


@app.post("/v2/query", response_model=UnifiedQueryResponse)
async def unified_query(request: UnifiedQueryRequest, raw_request: Request):
    """Unified RAG query: auto-routes to PaddedRAG or GraphRAG.

    - Simple 1-hop queries → PaddedRAG (faster, ~150ms)
    - Multi-hop / comparative → GraphRAG (+PPR, ~400ms)

    Routing is performed by HyDE + Step-Back classification using gemini-3.5-flash-lite.
    """
    if not unified_pipeline:
        # Phase-1 contract: tell the FE to back off with both the RFC 7231
        # `Retry-After` header AND a body field (`retry_after_ms`) so naive
        # fetch wrappers and the FE retry layer can both back off safely.
        raise HTTPException(
            status_code=503,
            detail="Service is warming up. Please retry shortly.",
            headers={"Retry-After": "2"},
        )

    try:
        response_payload = await _build_v2_query_payload(request, raw_request)
        return UnifiedQueryResponse(**response_payload)
    except Exception as exc:
        await _metric_inc("unified_query_errors")
        logger.exception(f"Unified query failed: {exc}")
        raise HTTPException(status_code=500, detail="Unified query processing failed")


# ───────────────────────────────────────────────────────────────────────────
# /v2/query/stream — true token-stream SSE.
#
# This is a TRUE TOKEN-STREAM, not a fake-stream. Progress stage events are
# emitted via a progress callback inside _build_v2_query_payload(). The
# generation phase uses LLMGenerator.generate_answer_stream() to yield raw
# token deltas as they arrive from the LLM.
#
# After all tokens arrive, post-guards (URL strip, contact-abstain, etc.) run
# on the FULL assembled answer. The `done` event carries the guarded version,
# which may differ from the streamed delta concatenation. The FE reconciles
# on `done`.
# ───────────────────────────────────────────────────────────────────────────
@app.post("/v2/query/stream")
async def unified_query_stream(request: UnifiedQueryRequest, raw_request: Request):
    """SSE true-token-stream variant of /v2/query.

    Stage events → meta → token deltas → done (guarded). FE reconciles
    on done if deltas differ from guarded answer.
    """
    if not unified_pipeline:
        raise HTTPException(
            status_code=503,
            detail="Unified pipeline not initialized"
        )

    async def _generator():
        # Stage events arrive via a progress callback wired into
        # _build_v2_query_payload. We use an asyncio.Queue to shuttle
        # them from the callback into the generator.

        stage_queue: asyncio.Queue = asyncio.Queue(maxsize=10)

        def _progress(stage: str):
            """Synchronous callback — puts stage label into the queue.
            Use put_nowait() because Queue has maxsize=10 and we must
            never block the pipeline."""
            stage_queue.put_nowait(stage)

        # Run the heavy pipeline (with progress callback) in a task so
        # we can interleave stage events from the queue.
        pipeline_task = asyncio.create_task(
            _build_v2_query_payload(request, raw_request, progress_cb=_progress)
        )

        answer: str = ""
        meta: Optional[Dict[str, Any]] = None
        response_payload: Optional[Dict[str, Any]] = None
        get_queue_task: Optional[asyncio.Task] = None

        try:
            while True:
                if get_queue_task is None:
                    get_queue_task = asyncio.create_task(stage_queue.get())
                done, pending = await asyncio.wait(
                    [pipeline_task, get_queue_task],
                    return_when=FIRST_COMPLETED,
                )
                for d in done:
                    result = d.result()
                    if isinstance(result, dict):
                        response_payload = result
                        answer = response_payload.get("answer", "") or ""
                        meta = {
                            "route": response_payload.get("route"),
                            "sources": response_payload.get("sources", []),
                            "status_code": response_payload.get("status_code"),
                            # FIX-05 (ticket 44): the stream must carry the
                            # SAME status surface as the non-stream response,
                            # otherwise a degraded generation renders as a
                            # normal answer on the SSE path only.
                            "status_reason": response_payload.get("status_reason"),
                            "internal_status_code": response_payload.get("internal_status_code"),
                            "answer_confidence": response_payload.get("answer_confidence"),
                            # Ticket 17: retrieval confidence. The FE renders
                            # one confidence per source chip, so a meta frame
                            # without this field forces it to invent a default
                            # and every streamed chip carries a fabricated
                            # number while the non-stream path carries the
                            # real one. Same value, both paths.
                            "confidence": response_payload.get("confidence"),
                            # Ticket 10: the streamed answer must carry the
                            # same per-claim citations as the JSON response;
                            # the FE reconciles them on the `done` frame.
                            "citations": response_payload.get("citations", []),
                            "trace_id": response_payload.get("trace_id"),
                            # Ticket 17: the streamed surface must classify an
                            # evidence-policy abstention exactly like the JSON
                            # one. Without this the FE cannot tell the streamed
                            # refusal from a normal answer.
                            "provider": response_payload.get("provider"),
                            # Ticket 40: the stream must carry the same
                            # rerank-skipped trace as the non-stream response.
                            "rerank_skipped_reason": response_payload.get(
                                "rerank_skipped_reason"
                            ),
                        }
                    elif isinstance(result, str):
                        yield _format_sse("stage", {"stage": result})
                        get_queue_task = None

                if response_payload is not None:
                    break

            # Cancel any pending tasks
            for p in pending:
                p.cancel()

            # Emit meta event
            if meta:
                yield _format_sse("meta", meta)

            # Stream token deltas
            for piece in _chunk_answer_for_stream(answer):
                yield _format_sse("delta", {"text": piece})

            # Emit done with the full guarded answer
            yield _format_sse("done", {"answer": answer})
        finally:
            # Ensure pipeline task is cancelled on client disconnect
            pipeline_task.cancel()
            if get_queue_task is not None and not get_queue_task.done():
                get_queue_task.cancel()

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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

    # Attribute every ns24 call made while answering this query to its trace_id
    # in the token ledger (same pattern as /v2's wrapper).
    request_id_token = set_request_id(trace_id)
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

        # Convert RetrievedDocument -> dict for reranker (parallel-ready)
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

        # Sửa #7: parallel booster + rerank. Booster depends on baseline_docs
        # only; reranker depends on chunks_for_rerank (also derived from
        # baseline_docs) but does NOT depend on booster output — safe to
        # overlap with asyncio.gather.

        def _booster_task():
            from src.services.aggregation_booster import boost_with_aggregation
            return boost_with_aggregation(
                query=request.query,
                baseline_docs=baseline_docs,
                lancedb_retriever=lancedb_retriever_service,
                top_k=request.candidate_pool + 2,
                max_inject=2,
            )

        rerank_executed = False
        rerank_skipped_reason: Optional[str] = None

        async def _rerank_task():
            nonlocal rerank_executed, rerank_skipped_reason
            if reranker_service is not None and reranker_service.enabled:
                rerank_executed = True
                return await run_in_threadpool(
                    lambda: reranker_service.rerank(
                        query=request.query,
                        chunks=chunks_for_rerank,
                        top_k=request.top_k,
                        apply_lost_in_middle=request.apply_lost_in_middle,
                    )
                )
            # Ticket 40: the static gate is closed — expose WHY, and count
            # only the silent-quality-collapse case (dead model). A duck-typed
            # fake without a known skip_reason normalizes to None: no bump,
            # no forged trace.
            if chunks_for_rerank and reranker_service is not None:
                rerank_skipped_reason = _rerank_trace_reason(reranker_service)
                if rerank_skipped_reason == LOAD_STATE_LOAD_FAILED:
                    await _metric_inc("reranker_bypass_load_failed")
                    logger.warning(
                        f"v3_rerank_bypass trace_id={trace_id} reason=load_failed"
                    )
            return list(chunks_for_rerank[: request.top_k])

        booster_result, rerank_result = await asyncio.gather(
            run_in_threadpool(_booster_task),
            _rerank_task(),
            return_exceptions=True,
        )

        # Handle exceptions explicitly — không silent drop
        if isinstance(booster_result, Exception):
            logger.warning(f"Booster failed: {booster_result}")
            baseline_docs_after_booster = baseline_docs
        else:
            baseline_docs_after_booster = booster_result
            stages["booster_injected"] = max(
                0, len(baseline_docs_after_booster) - len(baseline_docs)
            )

        if isinstance(rerank_result, Exception):
            logger.warning(f"Reranker failed: {rerank_result}")
            chunks_top = list(chunks_for_rerank[: request.top_k])
        else:
            chunks_top = list(rerank_result)
            # Ticket 40: `reranker_used` must reflect ACTUAL execution.
            # Before this fix a dead reranker still flipped it to True,
            # because the fallback list arrived without an exception.
            stages["reranker_used"] = rerank_executed
            if rerank_executed:
                stages["lost_in_middle"] = request.apply_lost_in_middle
        stages["rerank_skipped_reason"] = rerank_skipped_reason

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
        # T16 (K3): a chunk of a probed release was just served.
        _stamp_probe_served(raw_request.app, chunks_top)

        return V3QueryResponse(**response_payload)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"v3_query failed trace_id={trace_id}: {exc}")
        raise HTTPException(status_code=500, detail="V3 query processing failed")
    finally:
        reset_request_id(request_id_token)


@app.post("/v2/graph/update")
async def incremental_graph_update(
    request: Dict[str, Any],
    raw_request: Request,
    _: None = Depends(require_admin_action(AdminAction.GRAPH_UPDATE)),
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


class FeedbackRequest(BaseModel):
    """Feedback on a served answer (public endpoint for the student)."""

    trace_id: str = Field(..., description="trace_id of the answer being rated")
    helpful: bool = Field(..., description="True for 👍, False for 👎")
    reason: str | None = Field(default=None, description="Optional free-text reason")
    # T07: the question that produced this answer, when the client has
    # it. A negative rating on a SUCCESS trace is promoted into the
    # unanswered queue under this question (redacted by the service) so
    # officer triage sees the actual question, not an empty marker.
    question: str | None = Field(
        default=None,
        description="The question that produced this answer, when the client has it",
    )


class FeedbackResponse(BaseModel):
    """What the feedback call recorded."""

    trace_id: str
    helpful: bool
    recorded: bool
    reason: str | None = None


@app.post("/v2/feedback", response_model=FeedbackResponse, tags=["feedback"])
async def feedback(
    request: FeedbackRequest,
    raw_request: Request,
) -> FeedbackResponse:
    """Student 👍/👎 feedback on a served answer.

    Public endpoint: no authentication — it is the student talking to the
    system, not the office. Idempotent by ``trace_id``: a second rating for
    the same answer returns the existing record instead of appending a
    duplicate. The feedback is anchored to the unanswered queue when a queued
    item exists for the same ``trace_id``.

    Malformed input (empty/oversized ``trace_id``, oversized ``reason``) is
    refused with 400 — the queue service's ``ValueError`` contract — rather
    than surfacing as an unhandled 500. A malformed request must never
    register as a server fault.
    """
    svc = getattr(raw_request.app.state, "unanswered_queue", None)
    if svc is None:
        return FeedbackResponse(
            trace_id=request.trace_id,
            helpful=request.helpful,
            recorded=False,
            reason=request.reason,
        )
    try:
        stored = svc.feedback(
            request.trace_id,
            request.helpful,
            request.reason,
            question=request.question,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FeedbackResponse(
        trace_id=stored["trace_id"],
        helpful=stored["helpful"],
        recorded=True,
        reason=stored["reason"],
    )


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
