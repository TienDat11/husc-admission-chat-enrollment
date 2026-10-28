"""Unanswered-question queue surface — ticket 7.

Read-only view of questions the system could not answer, with resolution
markers so the admission team can triage batches (T15). All routes go through
ticket 13's ``require_action`` with ``records.read``.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from src.domain.admin_identity import AdminAction
from src.routers.admin_auth import require_action

router = APIRouter(
    prefix="/unanswered",
    tags=["admin"],
    # Ticket 27: the unanswered surface is read-only, but a viewer still
    # spends one unit of the *calling actor's* budget, because reviewing
    # hundreds of items is a real cost.
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)


class UnansweredItem(BaseModel):
    """One unanswered question, already redacted, with its resolution state."""

    question: str
    trace_id: str
    asked_at: str
    route: str
    status_code: str
    resolved: bool
    resolved_at: str | None = None
    resolved_by_record_id: str | None = None
    out_of_scope: bool = False


class UnansweredList(BaseModel):
    """A page of unanswered items and the total count."""

    items: list[UnansweredItem]
    count: int


def _service(request: Request):
    """The app's unanswered queue, or 503 when the app was never wired to one.

    Absent wiring is a 503 — the same "genuinely absent configuration" the
    other ticket-14 ticket-12 admin routers reserve for that case — rather
    than an empty list, which would read as "nothing has ever happened".
    """
    svc = getattr(request.app.state, "unanswered_queue", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="unanswered queue is not configured")
    return svc


@router.get(
    "/",
    response_model=UnansweredList,
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def list_unanswered(
    request: Request,
    resolved: bool | None = Query(default=None, description="Include resolved items too"),
) -> UnansweredList:
    """Open unanswered questions, newest first, with the resolution markers.

    By default only open items are returned (the ones the team still needs to
    review); pass ``resolved=true`` to see everything, including items T15
    already resolved.
    """
    svc = _service(request)
    items = svc.unanswered() if resolved is None or resolved is False else svc.items()
    # Newest first so the team always sees what just arrived.
    return UnansweredList(
        items=[
            UnansweredItem(
                question=item.question,
                trace_id=item.trace_id,
                asked_at=item.asked_at,
                route=item.route,
                status_code=item.status_code,
                resolved=item.resolved,
                resolved_at=item.resolved_at,
                resolved_by_record_id=item.resolved_by_record_id,
                out_of_scope=item.out_of_scope,
            )
            for item in reversed(items)
        ],
        count=len(items),
    )
