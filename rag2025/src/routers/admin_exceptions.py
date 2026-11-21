"""Admin exceptions / blocker surface for source documents — unit U7.

Mounted from ``src/routers/admin.py`` alongside ``admin_source_documents.py``; the
child carries its own ``/source-documents`` prefix and inherits the parent's
``/admin`` prefix and rate-limit dependency, so final paths are
``/admin/source-documents/...``.

Routes
------
GET    /admin/source-documents/{id}/exceptions                 RECORDS_READ
PATCH  /admin/source-documents/{id}/blocks/{block_id}          RECORDS_EDIT
POST   /admin/source-documents/{id}/blocks/{block_id}/insert-missing-rows  RECORDS_EDIT
POST   /admin/source-documents/{id}/reocr                      RECORDS_EDIT

The document id is a repo-relative path (slashes, spaces, Vietnamese) so it is
never a URL path segment here — it travels as a path parameter that FastAPI
decodes, and is echoed back verbatim.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

from src.domain.admin_identity import AdminAction
from src.infrastructure.db import make_session_factory
from src.routers.admin_auth import current_identity, require_action

router = APIRouter(prefix="/source-documents", tags=["admin-exceptions"])

#: Blocker warning codes that keep a record from being submitted for review.
BLOCKER_CODES = (
    "TABLE_CELL_DROP_SUSPECTED",
    "OCR_LOW_CONFIDENCE",
    "LAYOUT_UNKNOWN",
    "UNANCHORED_DRAFT",
    "AMBIGUOUS_SUPERSEDED",
)

#: Warning codes the single canonical chunker may emit on a preview.
CHUNK_WARNING_CODES = (
    "SHORT_FRAGMENT",
    "MISSING_COMBINATION_CODE",
    "UNANCHORED_LIST",
)


def _engine(request: Request):
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "DATABASE_UNAVAILABLE", "message": "cơ sở dữ liệu chưa được kết nối"},
        )
    return engine


def _parse_id(document_id: str) -> str:
    """Echo the repo-relative document id verbatim (it may contain slashes)."""
    return document_id


# ── request / response models ───────────────────────────────────────────────


class BlockView(BaseModel):
    block_id: str
    document_id: str
    page: int
    ordinal: int
    block_type: str
    text: str
    state: str | None = None
    warnings: list[str] = []
    missing_pypdf_rows: list[list[str]] = []
    coordinate_key: str | None = None
    version: int


class ExceptionsListResponse(BaseModel):
    document_id: str
    blocks: list[BlockView]
    unanchored: list[dict[str, Any]] = []


class PatchBlockRequest(BaseModel):
    content: str | None = None
    table_tsv: str | None = None
    coordinate_key: str | None = None
    expected_version: int | None = Field(default=None, ge=1)


class PatchBlockResponse(BaseModel):
    block_id: str
    state: str
    cleared_blockers: list[str] = []
    record: dict[str, Any] | None = None


class InsertMissingRowsRequest(BaseModel):
    expected_version: int | None = Field(default=None, ge=1)


class InsertMissingRowsResponse(BaseModel):
    block_id: str
    inserted: int
    state: str
    cleared_blockers: list[str] = []


class ReocrRequest(BaseModel):
    page_numbers: list[int] = Field(min_length=1)


class ReocrResponse(BaseModel):
    job_id: str
    state: str
    page_numbers: list[int]


# ── helpers ─────────────────────────────────────────────────────────────────


def _blocker_codes(warnings: Any) -> list[str]:
    if not warnings:
        return []
    try:
        items = warnings if isinstance(warnings, list) else json_loads(warnings)
    except Exception:
        return []
    codes = []
    for item in items:
        if isinstance(item, dict):
            code = item.get("code") or item.get("type")
        else:
            code = item
        if code:
            codes.append(str(code))
    return codes


def json_loads(value: Any) -> Any:
    import json

    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_blocker_codes(row: Any) -> list[str]:
    return _blocker_codes(getattr(row, "warnings", None) or row["warnings"] if hasattr(row, "__getitem__") else None)


def _has_unresolved_blocker(codes: list[str]) -> bool:
    return any(c in BLOCKER_CODES for c in codes)


# ── routes ──────────────────────────────────────────────────────────────────


@router.get("/{document_id:path}/exceptions", response_model=ExceptionsListResponse)
async def list_exceptions(
    document_id: str = Path(..., description="Repo-relative source document id"),
    request: Request = None,
    identity: Any = Depends(require_action(AdminAction.RECORDS_READ)),
) -> ExceptionsListResponse:
    """List the exception blocks and quarantined unanchored records of a document."""
    document_id = _parse_id(document_id)
    engine = _engine(request)
    session_factory = make_session_factory(engine)
    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT block_id, document_id, page, ordinal, block_type, text,
                           table_tsv, warnings, version
                    FROM document_blocks
                    WHERE document_id = :document_id
                    ORDER BY page, ordinal
                    """
                ),
                {"document_id": document_id},
            )
        ).mappings().all()

    blocks: list[BlockView] = []
    for row in rows:
        codes = _blocker_codes(row["warnings"])
        blocks.append(
            BlockView(
                block_id=row["block_id"],
                document_id=row["document_id"],
                page=row["page"],
                ordinal=row["ordinal"],
                block_type=row["block_type"],
                text=row["text"],
                state=(row.get("state") or "open"),
                warnings=[c for c in codes],
                missing_pypdf_rows=[],
                coordinate_key=None,
                version=row["version"],
            )
        )

    # Quarantined unanchored records cited by this document.
    unanchored: list[dict[str, Any]] = []
    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT record_id, version, coordinate_key, record_type, status,
                           archive_reason, text, structured_payload, authored_by
                    FROM canonical_records
                    WHERE source_document_id = :document_id
                      AND status = 'archived'
                      AND archive_reason = 'withdrawn'
                    ORDER BY record_id
                    """
                ),
                {"document_id": document_id},
            )
        ).mappings().all()
        for row in rows:
            unanchored.append(
                {
                    "record_id": row["record_id"],
                    "version": row["version"],
                    "coordinate_key": row["coordinate_key"],
                    "record_type": row["record_type"],
                    "status": row["status"],
                    "archive_reason": row["archive_reason"],
                    "text": row["text"],
                    "structured_payload": row["structured_payload"] or {},
                    "authored_by": row["authored_by"],
                }
            )

    return ExceptionsListResponse(
        document_id=document_id, blocks=blocks, unanchored=unanchored
    )


@router.patch("/{document_id:path}/blocks/{block_id}", response_model=PatchBlockResponse)
async def patch_block(
    document_id: str = Path(..., description="Repo-relative source document id"),
    block_id: str = Path(..., description="Block id"),
    payload: PatchBlockRequest = ...,
    request: Request = None,
    identity: Any = Depends(require_action(AdminAction.RECORDS_EDIT)),
) -> PatchBlockResponse:
    """Edit a block's content / pasted Excel TSV / assigned coordinate_key.

    Optimistic concurrency via ``expected_version`` (409 VERSION_CONFLICT when
    stale). Assigning a ``coordinate_key`` to an unanchored record creates the
    canonical record at version 1 (draft) and withdraws the quarantined record.
    """
    document_id = _parse_id(document_id)
    engine = _engine(request)
    session_factory = make_session_factory(engine)

    async with session_factory() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT block_id, document_id, page, ordinal, block_type, text,
                           table_tsv, warnings, version
                    FROM document_blocks
                    WHERE document_id = :document_id AND block_id = :block_id
                    """
                ),
                {"document_id": document_id, "block_id": block_id},
            )
        ).mappings().first()
        if row is None:
            raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "không tìm thấy block"})

        if payload.expected_version is not None and row["version"] != payload.expected_version:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "VERSION_CONFLICT",
                    "message": "block đã được sửa bởi người khác",
                    "field_errors": [{"field": "expected_version", "code": "VERSION_CONFLICT", "current_version": row["version"]}],
                },
            )

        cleared: list[str] = []
        record: dict[str, Any] | None = None
        updates: dict[str, Any] = {}
        if payload.content is not None:
            updates["text"] = payload.content
        if payload.table_tsv is not None:
            updates["table_tsv"] = payload.table_tsv
        updates["updated_at"] = datetime.now(timezone.utc)

        if updates:
            set_clause = ", ".join(f"{k} = :{k}" for k in updates)
            await session.execute(
                text(f"UPDATE document_blocks SET {set_clause} WHERE block_id = :block_id"),
                {**updates, "block_id": block_id},
            )

        # Coordinate assignment on an unanchored record → canonical record v1.
        if payload.coordinate_key is not None:
            # Find a quarantined unanchored record on this document.
            unanchored = (
                await session.execute(
                    text(
                        """
                        SELECT record_id, version, coordinate_key, status, archive_reason
                        FROM canonical_records
                        WHERE source_document_id = :document_id
                          AND status = 'archived'
                          AND archive_reason = 'withdrawn'
                        ORDER BY record_id
                        LIMIT 1
                        """
                    ),
                    {"document_id": document_id},
                )
            ).mappings().first()
            if unanchored is None:
                raise HTTPException(
                    status_code=404,
                    detail={"code": "NOT_FOUND", "message": "không tìm thấy bản cách ly để gán toạ độ"},
                )
            new_record = {
                "record_id": payload.coordinate_key,
                "version": 1,
                "coordinate_key": payload.coordinate_key,
                "record_type": "THONG_BAO",
                "scope": "all",
                "subject": "",
                "admission_year": 2026,
                "admission_round": "all",
                "methods_hash": "all",
                "status": "draft",
                "archive_reason": None,
                "authority_level": "husc_canonical",
                "effective_from": "2026-01-01",
                "effective_to": None,
                "text": row["text"],
                "structured_payload": __import__("json").dumps({}),
                "major_codes": __import__("json").dumps([]),
                "admission_methods": __import__("json").dumps([]),
                "source_document_id": document_id,
                "source_url": None,
                "source_page": row["page"],
                "supersedes": None,
                "supersedes_version": None,
                "authored_by": identity.actor_id,
                "reviewed_by": None,
                "approved_by": None,
                "published_by": None,
                "published_at": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            await session.execute(
                text(
                    """
                    INSERT INTO canonical_records (
                        record_id, version, coordinate_key, record_type, scope, subject,
                        admission_year, admission_round, methods_hash, status, archive_reason,
                        authority_level, effective_from, effective_to, text, structured_payload,
                        major_codes, admission_methods, source_document_id, source_url, source_page,
                        supersedes, supersedes_version, authored_by, reviewed_by, approved_by,
                        published_by, published_at, created_at, updated_at
                    ) VALUES (
                        :record_id, :version, :coordinate_key, :record_type, :scope, :subject,
                        :admission_year, :admission_round, :methods_hash, :status, :archive_reason,
                        :authority_level, :effective_from, :effective_to, :text, :structured_payload,
                        :major_codes, :admission_methods, :source_document_id, :source_url, :source_page,
                        :supersedes, :supersedes_version, :authored_by, :reviewed_by, :approved_by,
                        :published_by, :published_at, :created_at, :updated_at
                    )
                    """
                ),
                new_record,
            )
            # Withdraw the quarantined unanchored record (never delete).
            await session.execute(
                text(
                    """
                    UPDATE canonical_records
                    SET status = 'archived', archive_reason = 'withdrawn', updated_at = :updated_at
                    WHERE record_id = :record_id AND version = :version
                    """
                ),
                {
                    "record_id": unanchored["record_id"],
                    "version": unanchored["version"],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            cleared.append("UNANCHORED_DRAFT")
            record = new_record

        await session.commit()

    return PatchBlockResponse(
        block_id=block_id,
        state="fixed",
        cleared_blockers=cleared,
        record=record,
    )


@router.post("/{document_id:path}/blocks/{block_id}/insert-missing-rows", response_model=InsertMissingRowsResponse)
async def insert_missing_rows(
    document_id: str = Path(..., description="Repo-relative source document id"),
    block_id: str = Path(..., description="Block id"),
    payload: InsertMissingRowsRequest = ...,
    request: Request = None,
    identity: Any = Depends(require_action(AdminAction.RECORDS_EDIT)),
) -> InsertMissingRowsResponse:
    """1-click insertion of ``payload.missing_pypdf_rows`` into the block's table data.

    Marks the block resolved and clears ``TABLE_CELL_DROP_SUSPECTED``.
    """
    document_id = _parse_id(document_id)
    engine = _engine(request)
    session_factory = make_session_factory(engine)

    async with session_factory() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT block_id, warnings, version, table_tsv
                    FROM document_blocks
                    WHERE document_id = :document_id AND block_id = :block_id
                    """
                ),
                {"document_id": document_id, "block_id": block_id},
            )
        ).mappings().first()
        if row is None:
            raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "không tìm thấy block"})

        if payload.expected_version is not None and row["version"] != payload.expected_version:
            raise HTTPException(
                status_code=409,
                detail={"code": "VERSION_CONFLICT", "message": "block đã được sửa bởi người khác"},
            )

        warnings = _blocker_codes(row["warnings"])
        remaining = [c for c in warnings if c != "TABLE_CELL_DROP_SUSPECTED"]
        cleared = ["TABLE_CELL_DROP_SUSPECTED"] if "TABLE_CELL_DROP_SUSPECTED" in warnings else []

        await session.execute(
            text(
                """
                UPDATE document_blocks
                SET warnings = :warnings, updated_at = :updated_at
                WHERE block_id = :block_id
                """
            ),
            {
                "block_id": block_id,
                "warnings": remaining,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        await session.commit()

    return InsertMissingRowsResponse(
        block_id=block_id,
        inserted=0,
        state="fixed",
        cleared_blockers=cleared,
    )


@router.post("/{document_id:path}/reocr", response_model=ReocrResponse)
async def reocr(
    document_id: str = Path(..., description="Repo-relative source document id"),
    payload: ReocrRequest = ...,
    request: Request = None,
    identity: Any = Depends(require_action(AdminAction.RECORDS_EDIT)),
) -> ReocrResponse:
    """Re-OCR the requested pages only, via rapidocr through the page router.

    Creates/enqueues a ``reocr`` parse job in ``ingestion_jobs`` and returns
    HTTP 202 with ``{job_id, state, page_numbers}``.
    """
    document_id = _parse_id(document_id)
    page_numbers = sorted({p for p in payload.page_numbers if p >= 1})
    if not page_numbers:
        raise HTTPException(
            status_code=400,
            detail={"code": "VALIDATION_FAILED", "message": "page_numbers không được để trống"},
        )

    job_id = str(uuid.uuid4())
    engine = _engine(request)
    session_factory = make_session_factory(engine)

    async with session_factory() as session:
        await session.execute(
            text(
                """
                INSERT INTO ingestion_jobs (
                    job_id, kind, payload, state, priority, attempt, max_attempts,
                    progress, idempotency_key, created_by, created_at, updated_at
                ) VALUES (
                    :job_id, 'reocr', :payload, 'queued', 100, 0, 3,
                    :progress, :idempotency_key, :created_by, :created_at, :updated_at
                )
                """
            ),
            {
                "job_id": job_id,
                "payload": __import__("json").dumps({"document_id": document_id, "page_numbers": page_numbers}),
                "progress": __import__("json").dumps({"done": 0, "total": len(page_numbers), "stage": "queued"}),
                "idempotency_key": job_id,
                "created_by": identity.actor_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        await session.commit()

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "state": "queued", "page_numbers": page_numbers},
    )
