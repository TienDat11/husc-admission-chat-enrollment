"""Release lifecycle — U8 §PL-5.

``POST /admin/releases``, ``GET /admin/releases``, ``GET /admin/releases/{id}``,
``POST /admin/releases/{id}/activate`` → ``RECORDS_PUBLISH``;
``POST /admin/releases/{id}/rollback`` → ``RECORDS_ROLLBACK`` (reason ≥ 10 chars).

State machine enforced inside one transaction under
``pg_advisory_xact_lock(hashtext('release_pointer'))``:

    building → gating → ready → active → superseded
    gating → gate_failed
    active → rolled_back

Activate requires target ``ready``; rollback requires target ``superseded`` and
rejects with ``RELEASE_NOT_ACTIVATABLE`` when any release is ``building`` or
``gating``. A release without ``golden_report`` → ``409 GATE_NOT_RUN``;
``golden_report.verdict != 'pass'`` → ``409 RELEASE_GATE_FAILED``.

Rollback demotes current ``active`` → ``rolled_back`` first, then promotes
target ``superseded`` → ``active``, reconciles ``canonical_records`` +
``canonical_record_versions`` to the target's ``corpus_stamp`` in the same
transaction, appends an audit event, and the reconciler hot-swaps
``app.state.release_snapshot`` without restart.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.domain.admin_identity import AdminAction, Identity
from src.domain.audit import AuditAction, AuditEvent, TargetKind
from src.infrastructure.sql_audit_ledger import append_on
from src.infrastructure.sql_record_store import SqlRecordStore
from src.routers.admin_auth import current_identity, require_action

router = APIRouter(tags=["admin-releases"])


class ReleaseCreateRequest(BaseModel):
    corpus_stamp: list[dict[str, Any]] = Field(default_factory=list)
    status: str = Field(default="building")


class ReleaseView(BaseModel):
    release_id: str
    status: str
    golden_report: dict[str, Any] | None = None
    corpus_stamp: list[dict[str, Any]] | None = None


class RollbackRequest(BaseModel):
    reason: str = Field(default="")


def _require_pg_store(request: Request) -> SqlRecordStore:
    store = getattr(request.app.state, "record_store", None) or getattr(
        request.app.state, "canonical_record_store", None
    )
    if store is None or (not isinstance(store, SqlRecordStore) and type(store).__name__ != "SqlRecordStore"):
        raise HTTPException(status_code=503, detail="record store is not PostgreSQL-backed")
    return store


@router.post("/releases", response_model=ReleaseView, dependencies=[Depends(require_action(AdminAction.RECORDS_PUBLISH))])
async def create_release(payload: ReleaseCreateRequest, request: Request, identity: Identity = Depends(current_identity)) -> ReleaseView:
    store = _require_pg_store(request)
    with store._connect() as conn:
        row = conn.execute(
            "INSERT INTO releases (parent_release_id, corpus_stamp, lancedb_table, lancedb_version, manifest, golden_report, status, created_by) "
            "VALUES (NULL, %s::jsonb, 'husc_chunks', 0, %s::jsonb, NULL, %s, %s) "
            "RETURNING release_id, status, golden_report, corpus_stamp",
            [payload.corpus_stamp, {"release_id": "pending", "items": len(payload.corpus_stamp)}, payload.status, identity.actor_id],
        ).fetchone()
    return ReleaseView(**dict(row))


@router.get("/releases", response_model=list[ReleaseView], dependencies=[Depends(require_action(AdminAction.RECORDS_READ))])
async def list_releases(request: Request, identity: Identity = Depends(current_identity)) -> list[ReleaseView]:
    store = _require_pg_store(request)
    with store._connect() as conn:
        rows = conn.execute("SELECT release_id, status, golden_report, corpus_stamp FROM releases ORDER BY created_at DESC").fetchall()
    return [ReleaseView(**dict(r)) for r in rows]


@router.get("/releases/{release_id}", response_model=ReleaseView, dependencies=[Depends(require_action(AdminAction.RECORDS_READ))])
async def get_release(release_id: str, request: Request, identity: Identity = Depends(current_identity)) -> ReleaseView:
    store = _require_pg_store(request)
    with store._connect() as conn:
        row = conn.execute("SELECT release_id, status, golden_report, corpus_stamp FROM releases WHERE release_id = %s", [release_id]).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "không tìm thấy bản phát hành"})
    return ReleaseView(**dict(row))


@router.post("/releases/{release_id}/activate", response_model=ReleaseView, dependencies=[Depends(require_action(AdminAction.RECORDS_PUBLISH))])
async def activate_release(release_id: str, request: Request, identity: Identity = Depends(current_identity)) -> ReleaseView:
    store = _require_pg_store(request)
    actor = identity.actor_id
    with store._connect() as conn:
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('release_pointer'))")
            target = conn.execute("SELECT status, golden_report FROM releases WHERE release_id = %s FOR UPDATE", [release_id]).fetchone()
            if target is None:
                raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "không tìm thấy bản phát hành"})
            if target["golden_report"] is None:
                raise HTTPException(status_code=409, detail={"code": "GATE_NOT_RUN", "message": "bản phát hành chưa chạy cổng vàng"})
            if target["golden_report"].get("verdict") != "pass":
                raise HTTPException(status_code=409, detail={"code": "RELEASE_GATE_FAILED", "message": "cổng vàng chưa pass", "golden_report": target["golden_report"]})
            blocking = conn.execute("SELECT 1 FROM releases WHERE status IN ('building','gating') LIMIT 1").fetchone()
            if blocking:
                raise HTTPException(status_code=409, detail={"code": "RELEASE_NOT_ACTIVATABLE", "message": "có bản phát hành đang build/gating"})
            if target["status"] != "ready":
                raise HTTPException(status_code=409, detail={"code": "RELEASE_NOT_ACTIVATABLE", "message": "chỉ có thể kích hoạt bản ở trạng thái ready"})
            current = conn.execute("SELECT release_id FROM releases WHERE status = 'active' FOR UPDATE").fetchone()
            if current:
                conn.execute("UPDATE releases SET status = 'superseded', updated_at = now() WHERE release_id = %s AND status = 'active'", [current["release_id"]])
            conn.execute("UPDATE releases SET status = 'active', activated_at = now(), activated_by = %s WHERE release_id = %s", [actor, release_id])
            append_on(
                conn,
                AuditEvent(
                    actor_id=actor,
                    role=identity.role.value,
                    action=AuditAction.PUBLISH,
                    target=release_id,
                    target_kind=TargetKind.RELEASE,
                    reason=f"activated release {release_id}",
                    release_id=release_id,
                    timestamp=datetime.now(timezone.utc),
                ),
            )
    with store._connect() as conn2:
        row = conn2.execute("SELECT release_id, status, golden_report, corpus_stamp FROM releases WHERE release_id = %s", [release_id]).fetchone()
    return ReleaseView(**dict(row))


@router.post("/releases/{release_id}/rollback", response_model=ReleaseView, dependencies=[Depends(require_action(AdminAction.RECORDS_ROLLBACK))])
async def rollback_release(release_id: str, payload: RollbackRequest, request: Request, identity: Identity = Depends(current_identity)) -> ReleaseView:
    store = _require_pg_store(request)
    actor = identity.actor_id
    reason = payload.reason.strip()
    if len(reason) < 10:
        raise HTTPException(status_code=422, detail={"code": "VALIDATION_FAILED", "message": "ly do phải dài ít nhất 10 ký tự"})
    with store._connect() as conn:
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('release_pointer'))")
            if release_id == "previous":
                latest = conn.execute(
                    "SELECT release_id FROM releases WHERE status = 'superseded' "
                    "ORDER BY activated_at DESC NULLS LAST, updated_at DESC LIMIT 1"
                ).fetchone()
                if latest is None:
                    raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "không tìm thấy bản phát hành đã thay thế"})
                release_id = latest["release_id"]
            target = conn.execute("SELECT status, golden_report, corpus_stamp FROM releases WHERE release_id = %s FOR UPDATE", [release_id]).fetchone()
            if target is None:
                raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "không tìm thấy bản phát hành"})
            if target["golden_report"] is None:
                raise HTTPException(status_code=409, detail={"code": "GATE_NOT_RUN", "message": "bản phát hành chưa chạy cổng vàng"})
            if target["golden_report"].get("verdict") != "pass":
                raise HTTPException(status_code=409, detail={"code": "RELEASE_GATE_FAILED", "message": "cổng vàng chưa pass"})
            blocking = conn.execute("SELECT 1 FROM releases WHERE status IN ('building','gating') LIMIT 1").fetchone()
            if blocking:
                raise HTTPException(status_code=409, detail={"code": "RELEASE_NOT_ACTIVATABLE", "message": "có bản phát hành đang build/gating"})
            if target["status"] != "superseded":
                raise HTTPException(status_code=409, detail={"code": "RELEASE_NOT_ACTIVATABLE", "message": "chỉ có thể rollback về bản ở trạng thái superseded"})
            stamp = target["corpus_stamp"] or []
            current = conn.execute("SELECT release_id FROM releases WHERE status = 'active' FOR UPDATE").fetchone()
            if current:
                conn.execute("UPDATE releases SET status = 'rolled_back', updated_at = now() WHERE release_id = %s AND status = 'active'", [current["release_id"]])
            current_ids = conn.execute("SELECT record_id, version FROM canonical_records WHERE status = 'published'").fetchall()
            target_keys = {(r.get("record_id"), r.get("version")) for r in stamp}
            withdrawn = [(r["record_id"], r["version"]) for r in current_ids if (r["record_id"], r["version"]) not in target_keys]
            if withdrawn:
                conn.execute(
                    "UPDATE canonical_records SET status = 'archived', archive_reason = 'withdrawn', updated_at = now() "
                    "WHERE status = 'published' AND (record_id, version) = ANY(%s)",
                    [withdrawn],
                )
            to_publish = [(r.get("record_id"), r.get("version")) for r in stamp if r.get("record_id")]
            if to_publish:
                conn.execute(
                    "UPDATE canonical_records SET status = 'published', archive_reason = NULL, effective_to = NULL, updated_at = now() "
                    "WHERE status = 'archived' AND (record_id, version) = ANY(%s)",
                    [to_publish],
                )
                published_now = conn.execute(
                    "SELECT count(*) AS n FROM canonical_records WHERE status = 'published' AND (record_id, version) = ANY(%s)",
                    [to_publish],
                ).fetchone()["n"]
                if published_now != len(to_publish):
                    raise HTTPException(status_code=409, detail={"code": "CURRENT_COORDINATE_TAKEN", "message": "không thể khôi phục toàn bộ bản mục tiêu"})
            conn.execute("UPDATE releases SET status = 'active', activated_at = now(), activated_by = %s WHERE release_id = %s", [actor, release_id])
            append_on(
                conn,
                AuditEvent(
                    actor_id=actor,
                    role=identity.role.value,
                    action=AuditAction.ROLLBACK,
                    target=release_id,
                    target_kind=TargetKind.RELEASE,
                    reason=reason,
                    release_id=release_id,
                    timestamp=datetime.now(timezone.utc),
                ),
            )
    with store._connect() as conn2:
        row = conn2.execute("SELECT release_id, status, golden_report, corpus_stamp FROM releases WHERE release_id = %s", [release_id]).fetchone()
    return ReleaseView(**dict(row))
