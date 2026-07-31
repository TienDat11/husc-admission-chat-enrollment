"""Extracted GET /health endpoint with warmup readiness contract.

Extracted from rag2025/src/main.py to its own router module.
Mounts as GET /health on the root FastAPI app.

@spec(SLICE-WARMUP-UX.PHASE1)
"""
from __future__ import annotations

from typing import Any, Dict, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from config.settings import settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response.

    Backward-compatible: legacy fields remain so older clients (FE
    pre-Phase-1) keep working. New fields drive the `ready`/`stage`
    contract that the FE polls to gate Send-button activation.
    """

    status: str
    lancedb_connected: bool = False
    vectors_count: int = 0
    collection: str = ""
    embedding_model: str = ""
    reranker_model: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # New: warmup readiness contract (SLICE-WARMUP-UX.PHASE1)
    ready: bool = False
    stage: Literal["warming", "ready", "failed"] = "warming"
    services: Dict[str, bool] = Field(default_factory=dict)


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Health check with LanceDB connection + warmup readiness contract.

    New fields (Phase 1 of warmup-ux):
      - `ready: bool`  — True when all core subsystems are wired up.
      - `stage: 'warming'|'ready'|'failed'` — set by the lifespan state machine.
      - `services: dict` — per-subsystem boolean map for diagnostics.

    Legacy fields (`status`, `lancedb_connected`, `embedding_model`,
    `reranker_model`, `metrics`) are preserved for backward compat.
    """
    # Read service references from module-level lambdas stashed on the
    # app object by main.py. Closures capture THIS module's scope so they
    # see updated values whether set by lifespan or by test (importlib
    # import under any module name).
    _metrics_snapshot = getattr(request.app, '_health_metrics', lambda: lambda: {})()
    lancedb_retriever_service = getattr(request.app, '_health_lancedb', lambda: None)()
    embedding_service = getattr(request.app, '_health_embedding', lambda: None)()
    reranker_service = getattr(request.app, '_health_reranker', lambda: None)()
    unified_pipeline = getattr(request.app, '_health_pipeline', lambda: None)()
    hybrid_search_service = getattr(request.app, '_health_hybrid', lambda: None)()

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

    # Warmup readiness (SLICE-WARMUP-UX.PHASE1)
    stage_value: Literal["warming", "ready", "failed"] = "warming"
    raw_stage = getattr(request.app.state, "warmup_stage", None)
    if raw_stage in ("warming", "ready", "failed"):
        stage_value = raw_stage  # type: ignore[assignment]

    graphrag_ok = (
        unified_pipeline is not None
        and getattr(unified_pipeline, "_graphrag", None) is not None
    )
    services_map: Dict[str, bool] = {
        "embedding": embedding_service is not None,
        "reranker": reranker_service is not None,
        "graphrag": graphrag_ok,
        "hybrid_bm25": hybrid_search_service is not None,
        "lancedb": response.lancedb_connected,
    }

    response.services = services_map
    response.stage = stage_value
    response.ready = (
        unified_pipeline is not None
        and reranker_service is not None
        and embedding_service is not None
        and graphrag_ok
    )
    # Stage "failed" must always surface as not-ready, even if some
    # subsystems are technically up.
    if stage_value == "failed":
        response.ready = False

    return response
