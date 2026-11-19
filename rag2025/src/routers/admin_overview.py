"""Admin overview counts — U6 (Frame 5 "Tổng quan").

``GET /admin/overview`` answers the sidebar badges and the overview cards in one
read: how many records await review, how many open blockers exist, how many
unanswered questions are still open and how many of those are past their SLA,
and which release is active. It is read-only and gated by ``RECORDS_READ``.

Two backends answer it with the same definitions:

- PostgreSQL (``app.state.db_engine`` is wired): one query over
  ``canonical_records``, ``conflict_entries``, ``unanswered_questions`` and
  ``releases``. Overdue means ``sla_due_at < now()``.
- File stores (no engine): the canonical record store, the conflict queue and
  the unanswered queue on ``app.state``, and the release pointer on disk. The
  SLA is 48 hours from ``asked_at``, the window the unanswered queue documents.

"Open blocker" is an open conflict entry. The schema has no severity column,
so every open conflict is treated as blocking until an officer resolves it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import text

from src.domain.admin_identity import AdminAction
from src.domain.canonical.record import RecordStatus
from src.routers.admin_auth import require_action
from src.routers.admin_conflicts import _conflict_service
from src.services.conflict_queue import STATUS_OPEN
from src.services.release_status import ReleaseStatusService

router = APIRouter(
    prefix="/overview",
    tags=["admin-overview"],
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)

UNANSWERED_SLA = timedelta(hours=48)

_OVERVIEW_SQL = text(
    """
    SELECT
      (SELECT count(*) FROM canonical_records WHERE status = 'review')
        AS review_count,
      (SELECT count(*) FROM conflict_entries WHERE status = 'open')
        AS open_blocker_count,
      (SELECT count(*) FROM unanswered_questions
        WHERE resolved_at IS NULL AND out_of_scope = false)
        AS unanswered_open_count,
      (SELECT count(*) FROM unanswered_questions
        WHERE resolved_at IS NULL AND out_of_scope = false
          AND sla_due_at < now())
        AS unanswered_overdue_count,
      (SELECT release_id FROM releases WHERE status = 'active')
        AS active_release_id
    """
)


class OverviewResponse(BaseModel):
    """Counts behind the sidebar badges and the Frame 5 cards."""

    review_count: int
    open_blocker_count: int
    unanswered_open_count: int
    unanswered_overdue_count: int
    active_release_id: str | None


async def _postgres_overview(engine) -> OverviewResponse:
    async with engine.connect() as conn:
        row = (await conn.execute(_OVERVIEW_SQL)).one()
    return OverviewResponse(**dict(row._mapping))


async def _file_overview(request: Request) -> OverviewResponse:
    store = getattr(request.app.state, "canonical_record_store", None)
    queue = getattr(request.app.state, "unanswered_queue", None)
    if store is None or queue is None:
        raise HTTPException(status_code=503, detail="overview is not configured")

    review_count = sum(
        1
        for record_id in store.record_ids()
        if store.get_record(record_id).latest.status is RecordStatus.REVIEW
    )
    open_blocker_count = sum(
        1 for item in _conflict_service(request).items() if item.status == STATUS_OPEN
    )
    now = datetime.now(timezone.utc)
    open_items = [item for item in await queue.items() if not item.resolved and not item.out_of_scope]
    overdue = sum(
        1
        for item in open_items
        if datetime.fromisoformat(item.asked_at) + UNANSWERED_SLA < now
    )
    releases_root = request.app.state.admin_rollback_service.releases_root
    active_release_id = ReleaseStatusService(releases_root=releases_root).status().active_release_id
    return OverviewResponse(
        review_count=review_count,
        open_blocker_count=open_blocker_count,
        unanswered_open_count=len(open_items),
        unanswered_overdue_count=overdue,
        active_release_id=active_release_id,
    )


@router.get("", response_model=OverviewResponse)
async def get_overview(request: Request) -> OverviewResponse:
    """The overview counts. Absent wiring is a 503, never a zeroed dashboard."""
    engine = getattr(request.app.state, "db_engine", None)
    if engine is not None:
        return await _postgres_overview(engine)
    return await _file_overview(request)
