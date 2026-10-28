"""Review-diff and preview endpoints (ticket 25).

Mounted alongside ``/admin/records/...`` (the lifecycle routes in
``src/routers/admin.py``); these are read-only preview routes so they take
``RECORDS_READ`` — the reviewer sees the diff and the preview answer with the
same permission they already need to open the record. Submitting for review
stays the existing ``POST /admin/records/{id}/transitions`` call (RECORDS_SUBMIT
+ four-eyes), which the preview screen invokes directly: criterion 4 is
"submit from the preview screen", not "a new submit endpoint".

The service comes from ``app.state.review_preview_service`` when startup
wiring provides one; otherwise it is built lazily from the same store and
startup services the rest of the admin surface uses. The lazy path mirrors
what main.py's lifespan assigns (``embedding_service`` / ``unified_llm``) and
falls back to the process-wide generator singleton — reading a state
attribute nobody sets would make the endpoint permanently 503.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from config.settings import settings
from src.domain.admin_identity import AdminAction
from src.infrastructure.canonical_record_store import (
    CanonicalRecordStore,
    RecordNotFoundError,
)
from src.routers.admin_auth import require_action
from src.routers.admin_rate_limit import admin_rate_limit
from src.services.review_preview import ReviewPreviewService

router = APIRouter(
    prefix="/admin",
    tags=["admin-preview"],
    # Ticket 27: every admin route spends one unit of the calling actor's
    # budget, attached at ROUTER level exactly like admin.py / admin_audit.py
    # so a route added later cannot ship unthrottled. The preview is the most
    # expensive read on the surface (it builds a throwaway index), so missing
    # it here is the case the rate-limit guard test exists to catch.
    dependencies=[Depends(admin_rate_limit)],
)


class PreviewRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    version: Optional[int] = Field(default=None, ge=1)
    top_k: int = Field(default=5, ge=1, le=20)


def _service(request: Request) -> ReviewPreviewService:
    service = getattr(request.app.state, "review_preview_service", None)
    if service is not None:
        return service
    store = getattr(request.app.state, "canonical_record_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="canonical record store is not configured",
        )
    if not isinstance(store, CanonicalRecordStore):
        raise HTTPException(
            status_code=503,
            detail="canonical record store has an unexpected type",
        )
    embedder = getattr(request.app.state, "embedding_service", None)
    if embedder is None:
        raise HTTPException(
            status_code=503,
            detail="preview requires the startup embedding service",
        )
    # The app's generator singleton when lifespan stored it, else the
    # process-wide one main.py itself uses. Never a hard 503 on an attribute
    # name the lifespan does not define.
    from src.services.llm_generator import get_llm_generator

    generator = getattr(request.app.state, "llm_generator_service", None)
    if generator is None:
        generator = get_llm_generator()
    # Serving vectors (read-only) so the preview embeds only the draft's
    # chunks; the canonical corpus lives next to the LanceDB dir.
    vectors_path = Path(settings.LANCEDB_URI).parent / "canonical_corpus" / "vectors.jsonl"
    service = ReviewPreviewService(
        store,
        embedder=embedder,
        embedding_dim=int(getattr(embedder, "expected_dim", settings.EMBEDDING_DIM)),
        generator=generator,
        llm=getattr(request.app.state, "unified_llm", None),
        vectors_path=vectors_path if vectors_path.is_file() else None,
    )
    request.app.state.review_preview_service = service
    return service


@router.get(
    "/records/{record_id}/review-diff",
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def review_diff(
    record_id: str,
    request: Request,
    version: Optional[int] = None,
) -> dict:
    """Diff of the draft (or a given version) vs the published version."""
    service = _service(request)
    try:
        diff, draft, issues, _status = service.diff_for(record_id, version)
    except RecordNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"record '{record_id}' not found"
        ) from None
    return {
        "record_id": record_id,
        "draft_version": draft.version,
        "high_impact_version": diff.high_impact_version,
        "diff": diff.to_dict(),
        "validation_errors": [v.to_dict() for v in issues],
    }


@router.post(
    "/records/{record_id}/review-preview",
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def review_preview(
    record_id: str,
    body: PreviewRequest,
    request: Request,
) -> dict:
    """The answer a student would receive if this draft were published."""
    service = _service(request)
    try:
        result = await service.preview(
            record_id, body.query, version=body.version, top_k=body.top_k
        )
    except RecordNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"record '{record_id}' not found"
        ) from None
    return result.to_dict()
