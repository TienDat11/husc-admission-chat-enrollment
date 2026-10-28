"""Comparison router (ticket 18).

Exposes the comparison contract as ``/v2/compare`` so the contract models land
in the OpenAPI schema and are code-generated for the frontend (criterion 5).
The endpoint is intentionally thin: the contract logic lives in
:mod:`src.services.comparison_service`; this router only parses a request,
delegates, and returns the response model.

The router is mounted by ``rag2025/src/main.py``. Main owns the mount line:

    from routers.comparison import router as comparison_router
    app.include_router(comparison_router)

until Main adds that, ``export_openapi.py`` includes the router itself before
dumping the schema (idempotent: it checks the path is not already registered),
so ``api/openapi.json`` advertises the contract regardless of whether the
runtime mount has happened yet. Once Main mounts, the runtime export is
byte-identical to the one the script produces here.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.domain.comparison import ComparisonResponse, RefusedMajor
from src.services.comparison_service import ComparisonService, ComparisonRoutingError

router = APIRouter(prefix="/v2", tags=["v2"])


class CompareRequest(BaseModel):
    """A multi-major same-year comparison question."""

    query: str = Field(..., min_length=1, max_length=500)
    effective_year: int = Field(..., ge=2000, le=2100)
    record_ids: List[str] = Field(
        default_factory=list, description="Candidate major record_ids to compare"
    )
    # Echoed back; the contract still validates route == graph_rag regardless.
    route: str = Field(default="graph_rag")
    intent: str = Field(default="so_sanh")


def _service(request: Optional[Request] = None) -> ComparisonService:
    from src.infrastructure.canonical_record_store import CanonicalRecordStore
    from src.services.source_document_service import SourceDocumentService

    if request is not None:
        store = getattr(request.app.state, "canonical_record_store", None)
        source_document = getattr(request.app.state, "source_document_service", None)
        if store is None:
            raise HTTPException(503, "comparison router is not bound to a store")
        return ComparisonService(store=store, source_document=source_document)
    # Test / export path: callers pass a Request they constructed, or build a
    # service from the default production store on disk.
    store = CanonicalRecordStore(__import__("os").environ.get(
        "CANONICAL_RECORD_STORE_ROOT",
        "rag2025/data/canonical_records/records",
    ))
    return ComparisonService(store=store)


@router.post("/compare", response_model=ComparisonResponse, tags=["v2"])
async def compare(request: Request, payload: CompareRequest) -> ComparisonResponse:
    """Build a same-year comparison table for several majors.

    The comparison contract lives in :class:`ComparisonResponse`; read its
    docstring for the three guarantees (same effective year, per-cell
    provenance, missing cells marked not inferred).
    """
    service = _service(request)
    try:
        return service.build(
            query=payload.query,
            route=payload.route,
            intent=payload.intent,
            effective_year=payload.effective_year,
            record_ids=payload.record_ids,
        )
    except ComparisonRoutingError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - router maps service errors to 4xx
        raise HTTPException(422, str(exc)) from exc
