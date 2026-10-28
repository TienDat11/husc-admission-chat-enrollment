"""Read-only audit ledger endpoints — ticket 14.

Three routes, all GET, all through ticket 13's ``require_action``:

- ``GET /admin/audit/events``   — the ledger, newest first, filterable by
  target / actor / action.
- ``GET /admin/audit/events/{event_id}`` — one event, with its chain position.
- ``GET /admin/audit/verify``   — walk the hash chain and report the first break.

**There is no POST, PUT, PATCH or DELETE here, and there is no handler behind
one.** The ticket's acceptance is "không tồn tại endpoint xóa hoặc sửa event",
so the read-only property is the absence of a mutation route rather than a
guard on one — a guard can be removed by the next edit, an absent route cannot
be reached at all. An event is undone by appending the undo, which is what the
ledger's own append-only write is for.

Authorization is ``records.read``: the spec's matrix gives ``viewer`` "read
admin surface", and the ledger *is* the admin surface's history. An admin
action with no matching read grant (there is none today) would be a new row in
``src/domain/admin_identity.py``, not a second policy in this file.

The ledger itself is injected on ``app.state.audit_ledger``, the same pattern
the lifecycle routes use for their store. Absent wiring is a 503 — the same
"genuinely absent configuration" the ticket-12 router reserves 503 for — rather
than an empty list, which would read as "nothing has ever happened".
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from src.domain.admin_identity import AdminAction
from src.domain.audit import AuditAction
from src.infrastructure.audit_ledger import AuditEntry, AuditLedger
from src.routers.admin_auth import require_action
from src.routers.admin_rate_limit import admin_rate_limit

router = APIRouter(
    prefix="/admin/audit",
    tags=["admin"],
    # Ticket 27: same per-actor budget as the rest of /admin. The ledger is
    # append-only and read-only, but an unbounded reader still costs CPU, and
    # this router serves an audit chain whose verification walks every entry.
    dependencies=[Depends(admin_rate_limit)],
)

DEFAULT_LIMIT = 100
MAX_LIMIT = 1000


class AuditEventRecord(BaseModel):
    """One stored event plus its position in the hash chain.

    The hashes are part of the payload on purpose: ``verify`` is only meaningful
    to a caller who can also see what it verified, and an auditor comparing this
    listing against an archived copy needs the links, not just the events.
    """

    seq: int
    prev_hash: str
    entry_hash: str
    event: dict


class AuditEventList(BaseModel):
    """Newest-first page of the ledger."""

    events: list[AuditEventRecord]
    count: int


class ChainVerificationResponse(BaseModel):
    """The result of walking the chain."""

    ok: bool
    checked: int
    broken_index: int | None = None
    reason: str | None = None
    last_hash: str | None = None


def _ledger(request: Request) -> AuditLedger:
    """The app's audit ledger, or 503 when the app was never wired to one."""
    ledger = getattr(request.app.state, "audit_ledger", None)
    if ledger is None:
        raise HTTPException(status_code=503, detail="audit ledger is not configured")
    return ledger


def _record(entry: AuditEntry) -> AuditEventRecord:
    return AuditEventRecord(
        seq=entry.seq,
        prev_hash=entry.prev_hash,
        entry_hash=entry.entry_hash,
        event=entry.event.to_dict(),
    )


@router.get(
    "/events",
    response_model=AuditEventList,
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def list_events(
    request: Request,
    target: str | None = Query(default=None, description="record id / release id / subject id"),
    actor_id: str | None = Query(default=None, description="who performed the action"),
    action: AuditAction | None = Query(default=None, description="event kind"),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
) -> AuditEventList:
    """The most recent events, newest first, matching every supplied filter.

    Newest first because the question being asked is almost always "what just
    changed"; the chain order is still visible in ``seq``, which descends down
    the page and never skips a number within a page.
    """
    matched = _ledger(request).find(target=target, actor_id=actor_id, action=action)
    page = matched[-limit:]
    return AuditEventList(
        events=[_record(entry) for entry in reversed(page)],
        count=len(matched),
    )


@router.get(
    "/events/{event_id}",
    response_model=AuditEventRecord,
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def get_event(event_id: str, request: Request) -> AuditEventRecord:
    """One event by its id, with the chain position it was stored at."""
    entry = _ledger(request).get(event_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"no such audit event: {event_id}")
    return _record(entry)


@router.get(
    "/verify",
    response_model=ChainVerificationResponse,
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def verify_chain(request: Request) -> ChainVerificationResponse:
    """Whether the ledger is intact, and where it first is not.

    A 200 with ``ok: false`` rather than a 5xx: a broken chain is an answer, not
    a server failure, and the caller needs ``broken_index`` to investigate.
    """
    verification = _ledger(request).verify()
    return ChainVerificationResponse(
        ok=verification.ok,
        checked=verification.checked,
        broken_index=verification.broken_index,
        reason=verification.reason,
        last_hash=verification.last_hash,
    )
