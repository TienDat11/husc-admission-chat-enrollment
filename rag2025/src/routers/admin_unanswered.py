"""Unanswered-question queue surface — ticket 7 / A12.

Read-only view plus the two triage mutations:

* ``POST /admin/unanswered/{trace_id}/out-of-scope`` (``RECORDS_EDIT``) — marks
  the question ``out_of_scope = True`` without deleting the row. A second call
  while already out of scope answers ``409 VERSION_CONFLICT``.
* ``POST /admin/unanswered/{trace_id}/promote-golden`` (``RECORDS_CREATE``) — one
  atomic SQL transaction: creates a ``draft`` canonical record, inserts
  ``golden_queries(origin='promoted')``, stamps ``resolved_by_record_id`` on the
  question, and appends an audit event. Never auto-publishes. ``viewer`` (only
  ``RECORDS_READ``) receives ``403``.
* ``GET /admin/feedback`` (``RECORDS_READ``) — recent rows from ``chat_feedback``.

SQL mode is used when ``app.state.db_engine`` is wired; file mode falls back to
the queue service on ``app.state.unanswered_queue``.
"""
from __future__ import annotations

from datetime import datetime, date, timezone, timedelta
from hashlib import sha256
from json import dumps
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.domain.admin_identity import AdminAction
from src.infrastructure.sql_audit_log import canonical_payload, compute_entry_hash
from src.routers.admin_auth import current_identity, require_action
from src.services.pii_redactor import redact_pii as _redact_pii
from src.services.unanswered_queue import UnansweredQueueService

router = APIRouter(
    prefix="/unanswered",
    tags=["admin-unanswered"],
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
    occurrence_count: int = 1
    sla_due_at: str | None = None
    is_overdue: bool = False


class UnansweredList(BaseModel):
    """A page of unanswered items and the total count."""

    items: list[UnansweredItem]
    count: int


class OutOfScopeResult(BaseModel):
    trace_id: str
    out_of_scope: bool


class PromoteGoldenBody(BaseModel):
    answer: str = Field(..., min_length=1, description="The officer's answer text.")
    coordinate_key: str = Field(
        ..., min_length=1, description="Stable coordinate key for the new draft record."
    )
    subject: str = Field(default="cau_hoi_chua_tra_loi")
    scope: str = Field(default="tuyen_sinh")
    admission_year: int = Field(default=2026)


class PromoteGoldenResult(BaseModel):
    trace_id: str
    record_id: str
    status: str
    golden_query_id: str


class FeedbackItem(BaseModel):
    feedback_id: str
    trace_id: str
    helpful: bool
    reason: str | None
    answered_at: str
    linked_item: bool
    created_at: str


class FeedbackList(BaseModel):
    items: list[FeedbackItem]
    count: int


def _service(request: Request) -> UnansweredQueueService:
    svc = getattr(request.app.state, "unanswered_queue", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="unanswered queue is not configured")
    return svc


def _engine(request: Request):
    return getattr(request.app.state, "db_engine", None)


def _item_to_response(item) -> UnansweredItem:
    return UnansweredItem(
        question=item.question,
        trace_id=item.trace_id,
        asked_at=item.asked_at,
        route=item.route,
        status_code=item.status_code,
        resolved=item.resolved,
        resolved_at=item.resolved_at,
        resolved_by_record_id=item.resolved_by_record_id,
        out_of_scope=item.out_of_scope,
        occurrence_count=item.occurrence_count,
        sla_due_at=item.sla_due_at,
        is_overdue=item.is_overdue,
    )


@router.get(
    "/",
    response_model=UnansweredList,
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def list_unanswered(
    request: Request,
    resolved: Optional[bool] = Query(
        default=None, description="Include resolved items too when true"
    ),
) -> UnansweredList:
    """Open unanswered questions, newest first.

    Default excludes resolved items and out-of-scope items. Pass
    ``resolved=true`` to also surface resolved rows.
    """
    svc = _service(request)
    engine = _engine(request)
    if engine is not None:
        from src.infrastructure.db import make_session_factory

        factory = make_session_factory(engine)
        conditions = ["resolved_at IS NULL", "out_of_scope = false"]
        params: dict[str, object] = {}
        if resolved is True:
            conditions = ["out_of_scope = false"]
        sql = (
            "SELECT trace_id, question, asked_at, route, status_code, resolved_at, "
            "resolved_by_record_id, out_of_scope, occurrence_count, sla_due_at "
            "FROM unanswered_questions WHERE " + " AND ".join(conditions) +
            " ORDER BY asked_at DESC"
        )
        async with factory() as conn:
            result = await conn.execute(text(sql), params)
            rows = [dict(r._mapping) for r in result]
        now = datetime.now(timezone.utc)
        items = []
        for row in rows:
            asked = row["asked_at"]
            asked_iso = asked.isoformat() if hasattr(asked, "isoformat") else str(asked)
            sla = row["sla_due_at"]
            sla_iso = sla.isoformat() if hasattr(sla, "isoformat") else str(sla) if sla else None
            overdue = bool(sla_iso) and not row["resolved_at"] and not row["out_of_scope"] and datetime.fromisoformat(sla_iso) < now
            items.append(
                UnansweredItem(
                    question=str(row["question"]),
                    trace_id=str(row["trace_id"]),
                    asked_at=asked_iso,
                    route=str(row.get("route") or ""),
                    status_code=str(row["status_code"]),
                    resolved=row["resolved_at"] is not None,
                    resolved_at=(
                        row["resolved_at"].isoformat()
                        if hasattr(row.get("resolved_at"), "isoformat")
                        else (str(row["resolved_at"]) if row.get("resolved_at") else None)
                    ),
                    resolved_by_record_id=(
                        str(row["resolved_by_record_id"])
                        if row.get("resolved_by_record_id")
                        else None
                    ),
                    out_of_scope=bool(row.get("out_of_scope", False)),
                    occurrence_count=int(row.get("occurrence_count") or 1),
                    sla_due_at=sla_iso,
                    is_overdue=overdue,
                )
            )
        return UnansweredList(items=items, count=len(items))
    items = (
        await svc.items() if resolved is True else await svc.unanswered()
    )
    return UnansweredList(
        items=[_item_to_response(item) for item in items],
        count=len(items),
    )


@router.post(
    "/{trace_id}/out-of-scope",
    response_model=OutOfScopeResult,
    dependencies=[Depends(require_action(AdminAction.RECORDS_EDIT))],
)
async def mark_out_of_scope(trace_id: str, request: Request) -> OutOfScopeResult:
    """Mark a question out of scope without deleting the row.

    Returns ``409 VERSION_CONFLICT`` when the question is already out of scope.
    """
    svc = _service(request)
    engine = _engine(request)
    if engine is not None:
        from src.infrastructure.db import make_session_factory

        factory = make_session_factory(engine)
        async with factory() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT trace_id, out_of_scope FROM unanswered_questions WHERE trace_id = :tid",
                    ),
                    {"tid": trace_id},
                )
            ).mappings().first()
            if row is None:
                raise HTTPException(status_code=404, detail="unanswered question not found")
            if bool(row["out_of_scope"]):
                raise HTTPException(
                    status_code=409,
                    detail={"code": "VERSION_CONFLICT", "message": "question is already out of scope"},
                )
            await conn.execute(
                text(
                    "UPDATE unanswered_questions SET out_of_scope = true WHERE trace_id = :tid",
                ),
                {"tid": trace_id},
            )
            await conn.commit()
        return OutOfScopeResult(trace_id=trace_id, out_of_scope=True)
    current = await svc.mark_out_of_scope(trace_id, True)
    # File mode has no notion of "already out of scope" conflict; re-read to be safe.
    current = await svc.mark_out_of_scope(trace_id, True)
    return OutOfScopeResult(trace_id=trace_id, out_of_scope=current.out_of_scope)


@router.post(
    "/{trace_id}/promote-golden",
    response_model=PromoteGoldenResult,
    dependencies=[Depends(require_action(AdminAction.RECORDS_CREATE))],
)
async def promote_golden(
    trace_id: str,
    body: PromoteGoldenBody,
    request: Request,
    identity=Depends(current_identity),
) -> PromoteGoldenResult:
    """Promote one unanswered question to a draft record + golden query.

    Runs in one atomic SQL transaction:
      1. insert a ``draft`` canonical record (never auto-publishes),
      2. insert ``golden_queries(origin='promoted')``,
      3. stamp ``resolved_by_record_id`` + ``resolved_at`` on the question,
      4. append an audit event.
    """
    engine = _engine(request)
    if engine is None:
        raise HTTPException(status_code=503, detail="postgres is not configured")
    actor = identity.actor_id
    record_id = body.coordinate_key
    now = datetime.now(timezone.utc)
    effective_from = date.today()
    redacted_question = None

    golden_query_id: Optional[str] = None

    async def _work(conn):
        nonlocal redacted_question, golden_query_id
        # Load the question (redacted) and confirm it is open.
        row = (
            await conn.execute(
                text(
                    "SELECT question, out_of_scope, resolved_at FROM unanswered_questions WHERE trace_id = :tid",
                ),
                {"tid": trace_id},
            )
        ).mappings().first()
        if row is None:
            raise HTTPException(status_code=404, detail="unanswered question not found")
        if row["resolved_at"] is not None:
            raise HTTPException(
                status_code=409,
                detail={"code": "VERSION_CONFLICT", "message": "question is already resolved"},
            )
        redacted_question = _redact_pii(str(row["question"]))

        # Advisory-locked audit chain, all inside this transaction.
        await conn.execute(text("SELECT pg_advisory_xact_lock(hashtext('audit_chain'))"))
        prev = (
            await conn.execute(text("SELECT entry_hash FROM audit_events ORDER BY seq DESC LIMIT 1"))
        ).scalar_one_or_none()

        # 1. draft canonical record
        before = {"trace_id": trace_id, "question": redacted_question}
        after = {
            "record_id": record_id,
            "version": 1,
            "coordinate_key": body.coordinate_key,
            "status": "draft",
            "authored_by": actor,
        }
        event_id_audit = uuid4()
        audit_payload = canonical_payload(
            {
                "event_id": event_id_audit,
                "aggregate_type": "unanswered_question",
                "aggregate_id": trace_id,
                "actor_id": actor,
                "role": identity.role.value,
                "action": "promote_golden",
                "before": before,
                "after": after,
                "reason": "promoted to draft record and golden query",
                "before_version": None,
                "after_version": 1,
                "release_id": None,
                "request_id": None,
                "created_at": now,
            }
        )
        entry_hash = compute_entry_hash(prev, audit_payload)
        await conn.execute(
            text(
                "INSERT INTO audit_events "
                "(event_id, aggregate_type, aggregate_id, actor_id, role, action, before, after, "
                "reason, before_version, after_version, release_id, request_id, created_at, "
                "prev_hash, entry_hash) "
                "VALUES (:event_id, :aggregate_type, :aggregate_id, :actor_id, :role, :action, "
                "CAST(:before AS jsonb), CAST(:after AS jsonb), :reason, :before_version, "
                ":after_version, :release_id, :request_id, :created_at, :prev_hash, :entry_hash)"
            ),
            {
                "event_id": event_id_audit,
                "aggregate_type": "unanswered_question",
                "aggregate_id": trace_id,
                "actor_id": actor,
                "role": identity.role.value,
                "action": "promote_golden",
                "before": dumps(before, ensure_ascii=False),
                "after": dumps(after, ensure_ascii=False),
                "reason": "promoted to draft record and golden query",
                "before_version": None,
                "after_version": 1,
                "release_id": None,
                "request_id": None,
                "created_at": now,
                "prev_hash": prev,
                "entry_hash": entry_hash,
            },
        )

        await conn.execute(
            text(
                "INSERT INTO canonical_records "
                "(record_id, version, coordinate_key, record_type, scope, subject, "
                "admission_year, admission_round, methods_hash, status, authority_level, "
                "effective_from, text, structured_payload, major_codes, admission_methods, "
                "authored_by, created_at, updated_at) "
                "VALUES (:record_id, 1, :coordinate_key, 'HOI_DAP', :scope, :subject, "
                ":admission_year, 'all', 'all', 'draft', 'husc_official', :effective_from, "
                ":text, '{}', '[]', '[]', :authored_by, :created_at, :updated_at)"
            ),
            {
                "record_id": record_id,
                "coordinate_key": body.coordinate_key,
                "scope": body.scope,
                "subject": body.subject,
                "admission_year": body.admission_year,
                "effective_from": effective_from,
                "text": body.answer,
                "authored_by": actor,
                "created_at": now,
                "updated_at": now,
            },
        )

        # 2. golden_queries row
        golden_query_id = str(uuid4())
        await conn.execute(
            text(
                "INSERT INTO golden_queries "
                "(query_id, text, expected_coordinate_keys, expected_pages, record_type, "
                "major_codes, admission_year, kind, origin, source_trace_id, created_by, created_at) "
                "VALUES (:query_id, :text, CAST(:expected_coordinate_keys AS jsonb), '[]', 'HOI_DAP', "
                "'[]', :admission_year, 'in_scope', 'promoted', :source_trace_id, :created_by, :created_at)"
            ),
            {
                "query_id": golden_query_id,
                "text": redacted_question,
                "expected_coordinate_keys": dumps([body.coordinate_key], ensure_ascii=False),
                "admission_year": body.admission_year,
                "source_trace_id": trace_id,
                "created_by": actor,
                "created_at": now,
            },
        )

        # 3. stamp resolved_by_record_id + resolved_at
        await conn.execute(
            text(
                "UPDATE unanswered_questions "
                "SET resolved_by_record_id = :rid, resolved_at = :now "
                "WHERE trace_id = :tid"
            ),
            {"rid": record_id, "now": now, "tid": trace_id},
        )

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as conn:
        await _work(conn)
        await conn.commit()

    return PromoteGoldenResult(
        trace_id=trace_id,
        record_id=record_id,
        status="draft",
        golden_query_id=golden_query_id,  # type: ignore[name-defined]
    )


@router.get(
    "/feedback",
    response_model=FeedbackList,
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def list_feedback(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> FeedbackList:
    """Recent chat_feedback rows, newest first."""
    engine = _engine(request)
    if engine is not None:
        from src.infrastructure.db import make_session_factory

        factory = make_session_factory(engine)
        async with factory() as conn:
            result = await conn.execute(
                text(
                    "SELECT feedback_id, trace_id, helpful, reason, answered_at, linked_item, created_at "
                    "FROM chat_feedback ORDER BY created_at DESC LIMIT :lim OFFSET :off"
                ),
                {"lim": limit, "off": offset},
            )
            rows = [dict(r._mapping) for r in result]
            count_row = (
                await conn.execute(text("SELECT count(*) AS c FROM chat_feedback"))
            ).scalar_one()
        items = []
        for row in rows:
            answered = row["answered_at"]
            created = row["created_at"]
            items.append(
                FeedbackItem(
                    feedback_id=str(row["feedback_id"]),
                    trace_id=str(row["trace_id"]),
                    helpful=bool(row["helpful"]),
                    reason=(str(row["reason"]) if row.get("reason") else None),
                    answered_at=answered.isoformat() if hasattr(answered, "isoformat") else str(answered),
                    linked_item=bool(row["linked_item"]),
                    created_at=created.isoformat() if hasattr(created, "isoformat") else str(created),
                )
            )
        return FeedbackList(items=items, count=int(count_row))
    svc = _service(request)
    rows = await svc.items()  # feedback is not in file service.items(); keep minimal
    # File mode has no chat_feedback table; return empty with 0 count.
    return FeedbackList(items=[], count=0)
