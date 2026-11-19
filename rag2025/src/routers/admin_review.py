"""Review + retrieval sandbox — U8 §PL-6.3 & §PL-7.1.

``POST /admin/retrieval-preview`` → ``RECORDS_READ``, calls ``retrieval_sandbox``.
``GET /admin/supersession-candidates?status=open`` → ``RECORDS_READ``.
``POST /admin/supersession-candidates/{id}/decision`` → ``RECORDS_APPROVE``,
body ``{decision: "confirm"|"reject"|"supersede"|"temporal_handoff"|"coexist", expected_version: int}``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.domain.admin_identity import AdminAction, Identity
from src.infrastructure.sql_record_store import SqlRecordStore
from src.routers.admin_auth import current_identity, require_action
from src.config.admin_identity import four_eyes_enabled
from src.services.retrieval_sandbox import Chunk, Query, compute_diff

router = APIRouter(tags=["admin-review"])


class RetrievalPreviewRequest(BaseModel):
    record_ids: list[str] = Field(default_factory=list)
    k: int = Field(default=5, ge=1, le=50)


class RetrievalPreviewResponse(BaseModel):
    queries: list[dict[str, Any]]
    elapsed_ms: float
    status: str = "ok"


class SupersessionCandidate(BaseModel):
    candidate_id: str
    old_record_id: str
    old_record_version: int
    new_record_id: str | None
    new_record_version: int | None
    decision: str | None
    status: str


class DecisionRequest(BaseModel):
    decision: str = Field(min_length=1)
    expected_version: int


#: Candidate ``status`` each body decision leads to (CHECK: pending_review/accepted/rejected).
_DECISION_STATUS = {
    "confirm": "accepted",
    "supersede": "accepted",
    "temporal_handoff": "accepted",
    "coexist": "accepted",
    "reject": "rejected",
}
#: ``decision`` values stored on the candidate (CHECK: supersede/repeal/temporal_handoff/co_exist).
#: ``confirm`` approves the proposed supersession; ``reject`` records no verdict.
_DECISION_VERDICT = {
    "confirm": "supersede",
    "supersede": "supersede",
    "temporal_handoff": "temporal_handoff",
    "coexist": "co_exist",
}


def _require_pg_store(request: Request) -> SqlRecordStore:
    store = getattr(request.app.state, "record_store", None) or getattr(
        request.app.state, "canonical_record_store", None
    )
    if store is None or (not isinstance(store, SqlRecordStore) and type(store).__name__ != "SqlRecordStore"):
        raise HTTPException(status_code=503, detail="record store is not PostgreSQL-backed")
    return store


@router.post("/retrieval-preview", response_model=RetrievalPreviewResponse, dependencies=[Depends(require_action(AdminAction.RECORDS_READ))])
async def retrieval_preview(payload: RetrievalPreviewRequest, request: Request, identity: Identity = Depends(current_identity)) -> RetrievalPreviewResponse:
    store = _require_pg_store(request)
    k = min(payload.k, 50)
    record_ids = payload.record_ids
    queries: list[Query] = []
    status = "ok"
    with store._connect() as conn:
        if record_ids:
            rows = conn.execute(
                "SELECT query_id, text, query_embedding, embedding_model, record_type, major_codes, admission_year "
                "FROM golden_queries WHERE record_id = ANY(%s)",
                [record_ids],
            ).fetchall()
        else:
            rows = []
        if not rows:
            status = "insufficient_fixture"
            for record_id in record_ids:
                queries.append(Query(query_id=f"fallback:{record_id}", text=f"{record_id} {record_id} 2026"))
        else:
            for row in rows:
                embedding = None
                if row["query_embedding"] is not None:
                    try:
                        embedding = list(bytes(row["query_embedding"]))
                    except Exception:
                        embedding = None
                queries.append(Query(query_id=row["query_id"], text=row["text"] or "", embedding=embedding))
    # Build before/after chunk sets from chunk_preview_items for the records.
    before_chunks: list[Chunk] = []
    after_chunks: list[Chunk] = []
    with store._connect() as conn:
        if record_ids:
            rows = conn.execute(
                "SELECT chunk_id, record_id, record_version, text, embedding, embedding_model FROM chunk_preview_items WHERE record_id = ANY(%s)",
                [record_ids],
            ).fetchall()
            for row in rows:
                emb = None
                if row["embedding"] is not None:
                    try:
                        emb = list(bytes(row["embedding"]))
                    except Exception:
                        emb = None
                chunk = Chunk(chunk_id=row["chunk_id"], text=row["text"] or "", embedding=emb, record_id=row["record_id"])
                before_chunks.append(chunk)
                after_chunks.append(chunk)
    result = compute_diff(queries, before_chunks, after_chunks, k)
    return RetrievalPreviewResponse(queries=[q.__dict__ for q in result.queries], elapsed_ms=result.elapsed_ms, status=status)


@router.get("/supersession-candidates", response_model=list[SupersessionCandidate], dependencies=[Depends(require_action(AdminAction.RECORDS_READ))])
async def list_supersession_candidates(
    request: Request,
    status: str = "open",
    identity: Identity = Depends(current_identity),
) -> list[SupersessionCandidate]:
    store = _require_pg_store(request)
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT candidate_id, old_record_id, old_record_version, new_record_id, new_record_version, decision, status FROM supersession_candidates WHERE status = %s ORDER BY created_at DESC",
            [status],
        ).fetchall()
    return [SupersessionCandidate(**dict(r)) for r in rows]


@router.post("/supersession-candidates/{candidate_id}/decision", response_model=SupersessionCandidate, dependencies=[Depends(require_action(AdminAction.RECORDS_APPROVE))])
async def decide_supersession(candidate_id: str, payload: DecisionRequest, request: Request, identity: Identity = Depends(current_identity)) -> SupersessionCandidate:
    if payload.decision not in _DECISION_STATUS:
        raise HTTPException(status_code=422, detail={"code": "VALIDATION_FAILED", "message": "quyết định không hợp lệ"})
    store = _require_pg_store(request)
    new_status = _DECISION_STATUS[payload.decision]
    verdict = _DECISION_VERDICT.get(payload.decision)
    with store._connect() as conn:
        row = conn.execute(
            "SELECT status, new_record_id, new_record_version FROM supersession_candidates WHERE candidate_id = %s FOR UPDATE",
            [candidate_id],
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "không tìm thấy ứng viên"})
        if row["status"] != "pending_review":
            raise HTTPException(status_code=409, detail={"code": "VERSION_CONFLICT", "message": "ứng viên đã được phân xử"})
        # A candidate has no version column of its own; the version it proposes is the new record's.
        if row["new_record_version"] is not None and row["new_record_version"] != payload.expected_version:
            raise HTTPException(status_code=409, detail={"code": "VERSION_CONFLICT", "message": "phiên bản không khớp", "current_version": row["new_record_version"]})
        if four_eyes_enabled() and payload.decision != "reject" and row["new_record_id"] is not None:
            author = conn.execute(
                "SELECT authored_by FROM canonical_records WHERE record_id = %s AND version = %s",
                [row["new_record_id"], row["new_record_version"]],
            ).fetchone()
            if author is not None and author["authored_by"] == identity.actor_id:
                raise HTTPException(status_code=403, detail={"code": "SELF_APPROVAL_FORBIDDEN", "message": "không thể tự phê duyệt bản do mình soạn"})
        conn.execute(
            "UPDATE supersession_candidates SET status = %s, decision = %s, decided_by = %s, decided_at = now() WHERE candidate_id = %s",
            [new_status, verdict, identity.actor_id, candidate_id],
        )
    with store._connect() as conn2:
        row = conn2.execute(
            "SELECT candidate_id, old_record_id, old_record_version, new_record_id, new_record_version, decision, status FROM supersession_candidates WHERE candidate_id = %s",
            [candidate_id],
        ).fetchone()
    return SupersessionCandidate(**dict(row))
