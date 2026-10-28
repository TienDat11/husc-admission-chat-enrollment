"""Conflict-queue endpoints — ticket 38.

The HTTP half of "two official sources disagree, a human decides":

- ``GET  /admin/conflicts`` — refresh the queue (detect from the seed report
  and from live published records, upsert, persist) and list every item with
  the detection accounting the response contract requires.
- ``POST /admin/conflicts/{conflict_id}/resolve`` — record the override:
  which source document's statement wins, and why.

Both are mounted under the admin router's ``/admin`` prefix, so the paths the
ticket names are what an officer's UI calls.

The error mapping is the same table shape as ``admin.py``'s lifecycle errors
(ticket 38 note 4 — "đừng invent envelope thứ hai"): a missing item is 404,
a decided item asked again is 409, a blank reason or a winner that states
nothing is 422, a winner with no record to point at is 404. RBAC and the
missing-credential 422 come from the shared dependencies, not from here.

The reason check appears twice on purpose: this route refuses a blank/whitespace
``reason`` with 422 before touching the service, and the audit model refuses an
override event without one. Removing either alone still rejects the request —
the two enforcement points are independent (ticket 12's pattern).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.domain.admin_identity import AdminAction, Identity
from src.routers.admin_auth import current_identity, require_action
from src.services.conflict_queue import (
    ConflictAlreadyResolvedError,
    ConflictError,
    ConflictItem,
    ConflictItemNotFoundError,
    ConflictQueueService,
    ConflictWinnerNotInStatementsError,
    ConflictWinnerRecordNotFoundError,
    DetectionResult,
    ORIGIN_SEED_REPORT,
    STATUS_OPEN,
    STATUS_RESOLVED,
)


router = APIRouter(prefix="/conflicts", tags=["admin-conflicts"])
# Same code map the lifecycle routes use, for the same reason: one envelope.
_CONFLICT_ERRORS: dict[type[ConflictError], int] = {
    ConflictItemNotFoundError: 404,
    ConflictAlreadyResolvedError: 409,
    ConflictWinnerNotInStatementsError: 422,
    ConflictWinnerRecordNotFoundError: 404,
}


class ConflictItemView(BaseModel):
    """One queued conflict, as an officer reads it before deciding."""

    conflict_id: str
    record_type: str
    ma_nganh: str
    admission_year: int
    field: str
    statements: dict[str, Any] = Field(default_factory=dict)
    origin: str
    status: str
    resolution: dict[str, Any] | None = None


class ConflictQueueResponse(BaseModel):
    """The queue plus the accounting the ticket's criteria measure.

    ``seed_reported_items`` counts items whose origin is the seed report, so
    "all 74 entered the queue" is checkable without parsing the item list;
    ``deduped`` / ``dedup_reason`` state, per criterion 9, what the detector
    collapsed and why instead of hiding it.
    """

    items: list[ConflictItemView]
    total: int
    open: int
    resolved: int
    seed_reported_items: int
    deduped: int
    dedup_reason: str | None = None


class ResolveConflictRequest(BaseModel):
    """The decision body: the winning source and the reason.

    ``actor`` is deliberately absent — the event's actor comes from the
    credential, never from the body.
    """

    winner_source_document_id: str = Field(min_length=1)
    reason: str | None = None


def _conflict_service(request: Request) -> ConflictQueueService:
    """The app's conflict service, or a lazy build from what the app exposes.

    Mirrors the lifecycle router's accessor: a service mounted on
    ``app.state.conflict_queue_service`` wins (that is the main.py hook);
    otherwise the store and ledger already configured by
    ``configure_canonical_record_store`` are enough to build one. An app with
    no canonical store answers 503 rather than inventing an empty queue —
    the same "refuse, don't fake" rule the audit router uses for its ledger.
    """
    service = getattr(request.app.state, "conflict_queue_service", None)
    if isinstance(service, ConflictQueueService):
        return service
    store = getattr(request.app.state, "canonical_record_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="canonical record store not configured",
        )
    ledger = getattr(request.app.state, "audit_ledger", None)
    seed_path = getattr(request.app.state, "conflict_seed_report_path", None)
    queue_path = getattr(request.app.state, "conflict_queue_path", None)
    service = ConflictQueueService(
        store,
        ledger,
        seed_report_path=seed_path,
        queue_path=queue_path,
    )
    # Memoize: the queue's read-modify-write (resolve, refresh) is serialized
    # by one threading.Lock per instance, so a fresh service per request would
    # let two concurrent resolves last-writer-win — losing a resolution whose
    # audit event is already in the ledger. Same one-service-per-app pattern
    # the lifecycle accessor uses.
    request.app.state.conflict_queue_service = service
    return service


def _item_view(item: ConflictItem) -> ConflictItemView:
    return ConflictItemView(
        conflict_id=item.conflict_id,
        record_type=item.record_type,
        ma_nganh=item.ma_nganh,
        admission_year=item.admission_year,
        field=item.field,
        statements=dict(item.statements),
        origin=item.origin,
        status=item.status,
        resolution=dict(item.resolution) if item.resolution else None,
    )


def _queue_response(
    items: list[ConflictItem], detection: DetectionResult | None = None
) -> ConflictQueueResponse:
    views = [_item_view(item) for item in items]
    return ConflictQueueResponse(
        items=views,
        total=len(views),
        open=sum(1 for item in items if item.status == STATUS_OPEN),
        resolved=sum(1 for item in items if item.status == STATUS_RESOLVED),
        seed_reported_items=sum(1 for item in items if item.origin == ORIGIN_SEED_REPORT),
        deduped=detection.deduped if detection else 0,
        dedup_reason=detection.dedup_reason if detection else None,
    )


@router.get(
    "",
    response_model=ConflictQueueResponse,
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def list_conflicts(request: Request) -> ConflictQueueResponse:
    """Refresh the queue from current data and return every item.

    Detection runs on read so the queue reflects the corpus without a manual
    command (criterion 1); it never selects a winner, so refreshing is safe
    in front of a live officer mid-decision.
    """
    service = _conflict_service(request)
    try:
        items, detection = service.refresh()
    except ConflictError as exc:
        raise HTTPException(
            status_code=_CONFLICT_ERRORS.get(type(exc), 500), detail=str(exc)
        ) from exc
    return _queue_response(items, detection)


@router.post(
    "/{conflict_id}/resolve",
    response_model=ConflictItemView,
    dependencies=[Depends(require_action(AdminAction.RECORDS_OVERRIDE))],
)
async def resolve_conflict(
    conflict_id: str,
    payload: ResolveConflictRequest,
    request: Request,
    identity: Identity = Depends(current_identity),
) -> ConflictItemView:
    """Record the override decision for one conflict.

    The reason must be non-blank *here* (422) — the first of the two
    enforcement points. The actor and role handed to the service are the
    identity the credential resolved to, so a body claiming to be somebody
    else has nothing to claim with.
    """
    if not (payload.reason or "").strip():
        raise HTTPException(
            status_code=422,
            detail="resolving a conflict requires a non-blank reason",
        )
    service = _conflict_service(request)
    try:
        item = service.resolve(
            conflict_id,
            winner_source_document_id=payload.winner_source_document_id,
            reason=payload.reason,
            actor_id=identity.actor_id,
            role=identity.role.value,
        )
    except ConflictError as exc:
        status = _CONFLICT_ERRORS.get(type(exc))
        if status is None:
            raise
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    return _item_view(item)
