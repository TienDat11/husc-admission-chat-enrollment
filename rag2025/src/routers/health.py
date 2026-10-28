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


# Ticket 40: the reranker's three-state load vocabulary. The literal is kept
# local so the health router does not import sentence_transformers; the
# values are asserted against services/reranker.py in tests.
_LOAD_STATE_LOAD_FAILED = "load_failed"


def _reranker_load_ok(reranker_service: Any) -> bool:
    """True ⟺ the reranker is exactly as configured: loaded, or turned off
    on purpose. False ⟺ config enables it but the model failed to load, or
    the service was never initialized.

    Reads the real load state, NOT object existence: after ``_init_reranker``
    the object always exists even when the model is dead. Duck-typed legacy
    fakes without a ``load_state`` attribute are treated as loaded.
    """
    if reranker_service is None:
        return False
    return getattr(reranker_service, "load_state", None) != _LOAD_STATE_LOAD_FAILED

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
    # T11: embedding dimension of the LOADED embedder (attribute on the
    # service, not a settings constant) + whether the stored LanceDB table's
    # vector column actually matches it. Default -1 = "unknown / no table".
    embedding_dim: int = -1
    vector_dim_match: bool = False
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # New: warmup readiness contract (SLICE-WARMUP-UX.PHASE1)
    ready: bool = False
    stage: Literal["warming", "ready", "failed"] = "warming"
    services: Dict[str, bool] = Field(default_factory=dict)

    # Ticket 40: true ⟺ the reranker is exactly as configured — loaded or
    # intentionally off. False ⟺ config enables it but the model failed to
    # load (or it was never initialized). `services["reranker"]` and `ready`
    # are derived from this same value, not from object existence.
    reranker_load_ok: bool = False


def _vector_dim_match(collection_info: Dict[str, Any], embedding_service: Any) -> bool:
    """T11: does the table's vector column match the loaded embedder's dim?

    ``vector_dim`` comes from the table schema (``LanceDBAdapter.vector_dim``);
    the embedder's dim is the service's ``expected_dim`` — an attribute on the
    loaded service, not a settings constant. -1 on either side means "unknown"
    and must not compare as a match.
    """
    table_dim = int(collection_info.get("vector_dim", -1))
    embedder_dim = int(getattr(embedding_service, "expected_dim", -1))
    return table_dim != -1 and embedder_dim != -1 and table_dim == embedder_dim



@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Health check with LanceDB connection + warmup readiness contract.

    New fields (Phase 1 of warmup-ux):
      - `ready: bool`  — True when all core subsystems are wired up.
      - `stage: 'warming'|'ready'|'failed'`  — set by the lifespan state machine.
      - `services: dict`  — per-subsystem boolean map for diagnostics.
      - `reranker_load_ok: bool`  — ticket 40: the reranker is exactly as
        configured (loaded or intentionally off); False only when the model
        is dead or was never initialized.

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
                response.vector_dim_match = _vector_dim_match(collection_info, embedding_service)
            else:
                response.status = f"table_not_found: {collection_info.get('error', 'unknown')}"
        except Exception as e:
            response.status = f"lancedb_error: {str(e)[:50]}"
    else:
        response.status = "lancedb_not_initialized"

    if embedding_service is not None:
        response.embedding_dim = int(getattr(embedding_service, "expected_dim", -1))


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
    # Ticket 40: the reranker is healthy iff it is exactly as configured —
    # loaded or intentionally off. A dead model (config on, load failed) is
    # NOT healthy, even though the service object always exists after
    # _init_reranker(). `services["reranker"]`, `reranker_load_ok` and
    # `ready` all read this one value, never object existence.
    reranker_ok = _reranker_load_ok(reranker_service)
    services_map: Dict[str, bool] = {
        "embedding": embedding_service is not None,
        "reranker": reranker_ok,
        "graphrag": graphrag_ok,
        "hybrid_bm25": hybrid_search_service is not None,
        "lancedb": response.lancedb_connected,
    }

    response.services = services_map
    response.stage = stage_value
    response.reranker_load_ok = reranker_ok
    response.ready = (
        unified_pipeline is not None
        and reranker_ok
        and embedding_service is not None
        and graphrag_ok
    )
    # Stage "failed" must always surface as not-ready, even if some
    # subsystems are technically up.
    if stage_value == "failed":
        response.ready = False

    return response
