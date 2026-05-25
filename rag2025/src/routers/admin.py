"""Admin endpoints: warmup post-deploy + atomic reload-table for blue-green swap.

@spec(S10.B7) /admin/warmup
@spec(S13.4) /admin/reload-table — atomic retriever swap with drain (RequestTracker).
SLA: <30s total = drain ≤25s + swap ≤1s + healthcheck ≤4s (per plan F4).
"""
import asyncio
import os
import time
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException, Header, Request
from loguru import logger
from pydantic import BaseModel, Field

router = APIRouter(prefix="/admin", tags=["admin"])

ADMIN_TOKEN_HEADER = "X-Admin-Token"

# F4 budget — drain bounded so total stays under 30s SLA.
DEFAULT_DRAIN_TIMEOUT_S = 25.0
ABS_MAX_DRAIN_TIMEOUT_S = 30.0

_reload_lock = asyncio.Lock()


def require_admin_token(token: str = Header(..., alias=ADMIN_TOKEN_HEADER)) -> None:
    """Verify admin token from env (ADMIN_API_TOKEN).
    @spec(S10.B7)
    """
    expected = os.environ.get("ADMIN_API_TOKEN")
    if not expected or token != expected:
        raise HTTPException(status_code=403, detail="forbidden")


@router.post("/warmup", dependencies=[Depends(require_admin_token)])
async def warmup() -> dict:
    """Pre-warm heavy services (embedding model, LLM connection).
    Call after deploy before serving production traffic.
    @spec(S10.B7)
    """
    from rag2025.src.app.lifecycle import AppServices

    services = AppServices()
    results: dict = {}

    # Warm embedding (~5s for Qwen3 model load)
    t0 = time.perf_counter()
    embedding = services.get_embedding()
    embedding.encode_query("warmup ping")
    results["embedding_ms"] = round((time.perf_counter() - t0) * 1000)

    # LLM warmup (connection check) — skip if no provider key
    ramclouds_key = os.environ.get("RAMCLOUDS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if ramclouds_key:
        t1 = time.perf_counter()
        llm = services.get_llm_client()
        try:
            await llm.chat(
                user_message="ping",
                max_tokens=5,
            )
        except Exception:
            pass  # warmup is best-effort

        results["llm_ms"] = round((time.perf_counter() - t1) * 1000)

    return {"status": "warmed", "timings_ms": results}


# ============================================================================
# /admin/reload-table — @spec(S13.4) blue-green LanceDB atomic swap
# ============================================================================


class ReloadTableRequest(BaseModel):
    """Body for POST /admin/reload-table."""

    table_name: str = Field(..., min_length=1, max_length=128)
    drain_timeout_s: float = Field(
        default=DEFAULT_DRAIN_TIMEOUT_S,
        gt=0.0,
        le=ABS_MAX_DRAIN_TIMEOUT_S,
    )


class ReloadTableResponse(BaseModel):
    """Response for POST /admin/reload-table."""

    active_table: str
    drained: bool
    drain_ms: float
    total_ms: float
    in_flight_at_swap: int


RetrieverBuilder = Callable[[str], Awaitable[Any]]


async def _safe_close(retriever: Any) -> None:
    """Close a retriever, swallowing exceptions so cleanup never raises."""
    close = getattr(retriever, "close", None)
    if close is None:
        return
    try:
        result = close()
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        logger.exception("reload-table: error closing old retriever (ignored)")


@router.post(
    "/reload-table",
    response_model=ReloadTableResponse,
    dependencies=[Depends(require_admin_token)],
)
async def reload_table(payload: ReloadTableRequest, request: Request) -> ReloadTableResponse:
    """Drain → build new retriever → healthcheck → atomic ref swap → close old.

    @spec(S13.4) atomic blue-green swap (M1+F1+F4 from plan v2.1).
    """
    state = request.app.state
    tracker = getattr(state, "request_tracker", None)
    builder: RetrieverBuilder | None = getattr(state, "build_retriever", None)
    if tracker is None or builder is None:
        raise HTTPException(
            status_code=503,
            detail="App state missing request_tracker or build_retriever",
        )

    start = time.monotonic()
    async with _reload_lock:
        # 1. Drain in-flight requests (bounded timeout per F4).
        drained = await tracker.drain(timeout=payload.drain_timeout_s)
        drain_ms = (time.monotonic() - start) * 1000.0
        if not drained:
            logger.warning(
                f"reload-table: drain timeout after {payload.drain_timeout_s}s; "
                f"in_flight={tracker.in_flight}; proceeding with swap"
            )

        # 2. Build new retriever.
        try:
            new_retriever = await builder(payload.table_name)
        except Exception as exc:
            logger.exception(f"reload-table: builder failed for {payload.table_name}")
            raise HTTPException(status_code=503, detail=f"Builder error: {exc}") from exc

        # 3. Healthcheck.
        try:
            healthy = await new_retriever.healthcheck()
        except Exception as exc:
            await _safe_close(new_retriever)
            logger.exception("reload-table: healthcheck raised")
            raise HTTPException(status_code=503, detail=f"Healthcheck raised: {exc}") from exc
        if not healthy:
            await _safe_close(new_retriever)
            raise HTTPException(status_code=503, detail="New retriever failed healthcheck")

        # 4. Atomic reference swap (atomic in CPython for object reassignment).
        in_flight_at_swap = tracker.in_flight
        old_retriever = getattr(state, "retriever", None)
        state.retriever = new_retriever

        # 5. Async cleanup of the old retriever — do not block the response.
        if old_retriever is not None:
            asyncio.create_task(_safe_close(old_retriever))

        total_ms = (time.monotonic() - start) * 1000.0
        return ReloadTableResponse(
            active_table=payload.table_name,
            drained=drained,
            drain_ms=drain_ms,
            total_ms=total_ms,
            in_flight_at_swap=in_flight_at_swap,
        )

