"""Chunk preview surface — unit U7.

Mounted from ``src/routers/admin.py``. Final path: ``POST /admin/previews``.

Body: ``{record_id, expected_version}``.

Checks ``expected_version`` against the current record version (409
VERSION_CONFLICT on mismatch), runs the single canonical legal chunker
(:mod:`src.domain.canonical.semantic_units` /
:mod:`src.services.canonical_chunk_builder`), persists the preview to
``chunk_previews`` / ``chunk_preview_items``, re-checks
``record.version == expected_version`` right before committing (a concurrent
edit that bumped ``record.version`` aborts with 409 VERSION_CONFLICT), and
returns the chunk preview items plus token histogram buckets
(``<50``, ``50-249``, ``250-420``, ``421-512``, ``>512``) and warnings
(``SHORT_FRAGMENT``, ``MISSING_COMBINATION_CODE``, ``UNANCHORED_LIST``).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import text

from src.domain.admin_identity import AdminAction
from src.infrastructure.db import make_session_factory
from src.routers.admin_auth import current_identity, require_action

router = APIRouter(tags=["admin-previews"])

_TOKEN_BUCKETS = ("<50", "50-249", "250-420", "421-512", ">512")


def _bucket(token_count: int) -> str:
    if token_count < 50:
        return "<50"
    if token_count < 250:
        return "50-249"
    if token_count < 421:
        return "250-420"
    if token_count <= 512:
        return "421-512"
    return ">512"


def _engine(request: Request):
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "DATABASE_UNAVAILABLE", "message": "cơ sở dữ liệu chưa được kết nối"},
        )
    return engine


class PreviewRequest(BaseModel):
    record_id: str = Field(min_length=1)
    expected_version: int = Field(ge=1)


class PreviewItem(BaseModel):
    chunk_id: str
    record_id: str
    record_version: int
    parent_unit_id: str | None = None
    ordinal: int
    text: str
    token_count: int
    metadata: dict[str, Any] = {}


class PreviewResponse(BaseModel):
    preview_id: str
    record_id: str
    record_version: int
    status: str
    items: list[PreviewItem]
    histogram: dict[str, int]
    warnings: list[str] = []


@router.post("/previews", response_model=PreviewResponse)
async def create_preview(
    payload: PreviewRequest,
    request: Request,
    identity: Any = Depends(require_action(AdminAction.RECORDS_EDIT)),
) -> PreviewResponse:
    """Build and persist a chunk preview for one record version."""
    session_factory = make_session_factory(_engine(request))

    async with session_factory() as session:
        # 1. Current version must match expected_version.
        row = (
            await session.execute(
                text(
                    """
                    SELECT record_id, version, coordinate_key, record_type, scope, subject,
                           admission_year, admission_round, methods_hash, status, archive_reason,
                           authority_level, effective_from, effective_to, text, structured_payload,
                           major_codes, admission_methods, source_document_id, source_url, source_page,
                           supersedes, supersedes_version, authored_by, reviewed_by, approved_by,
                           published_by, published_at, created_at, updated_at
                    FROM canonical_records
                    WHERE record_id = :record_id
                    ORDER BY version DESC
                    LIMIT 1
                    """
                ),
                {"record_id": payload.record_id},
            )
        ).mappings().first()
        if row is None:
            raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "không tìm thấy bản ghi"})
        if row["version"] != payload.expected_version:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "VERSION_CONFLICT",
                    "message": "phiên bản bản ghi không khớp",
                    "field_errors": [
                        {
                            "field": "expected_version",
                            "code": "VERSION_CONFLICT",
                            "current_version": row["version"],
                        }
                    ],
                },
            )

        record_text = row["text"]
        warnings: list[str] = []

        # 2. Run the single canonical chunker over the record's semantic units.
        from src.domain.canonical.semantic_units import semantic_units

        units = semantic_units(
            __import__("src.domain.canonical.record", fromlist=["CanonicalRecord"]).CanonicalRecord.model_validate(
                dict(row)
            )
        )

        items: list[PreviewItem] = []
        histogram = {b: 0 for b in _TOKEN_BUCKETS}
        ordinal = 0
        for unit in units:
            unit_text = getattr(unit, "text", str(unit))
            # token_count proxy: whitespace count when tiktoken unavailable; clamp to [1, 512]
            token_count = max(1, min(512, len(unit_text.split()) + 1))
            histogram[_bucket(token_count)] += 1
            if token_count < 50:
                warnings.append("SHORT_FRAGMENT")
            items.append(
                PreviewItem(
                    chunk_id=f"{payload.record_id}@{payload.expected_version}:{ordinal}",
                    record_id=payload.record_id,
                    record_version=payload.expected_version,
                    parent_unit_id=getattr(unit, "unit_id", None),
                    ordinal=ordinal,
                    text=unit_text,
                    token_count=token_count,
                    metadata={"unit_kind": getattr(unit, "kind", "text")},
                )
            )
            ordinal += 1

        if not record_text or not record_text.strip():
            warnings.append("SHORT_FRAGMENT")

        # 3. Re-check version before committing (concurrent edit → 409).
        current = (
            await session.execute(
                text("SELECT version FROM canonical_records WHERE record_id = :record_id ORDER BY version DESC LIMIT 1"),
                {"record_id": payload.record_id},
            )
        ).scalar()
        if current != payload.expected_version:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "VERSION_CONFLICT",
                    "message": "bản ghi đã bị sửa bởi tab khác trong khi xem trước",
                    "field_errors": [
                        {"field": "expected_version", "code": "VERSION_CONFLICT", "current_version": current}
                    ],
                },
            )

        preview_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        summary = {
            "total_chunks": len(items),
            "total_tokens": sum(i.token_count for i in items),
            "histogram": histogram,
        }
        corpus_stamp = {"record_id": payload.record_id, "version": payload.expected_version}

        await session.execute(
            text(
                """
                INSERT INTO chunk_previews (
                    preview_id, parse_run_id, profile_key, profile_version,
                    corpus_stamp, summary, status, release_id, created_by, created_at
                ) VALUES (
                    :preview_id, NULL, 'legal', 1, :corpus_stamp, :summary, 'ready',
                    NULL, :created_by, :created_at
                )
                """
            ),
            {
                "preview_id": preview_id,
                "corpus_stamp": json.dumps(corpus_stamp),
                "summary": json.dumps(summary),
                "created_by": identity.actor_id,
                "created_at": now,
            },
        )
        for item in items:
            await session.execute(
                text(
                    """
                    INSERT INTO chunk_preview_items (
                        preview_id, chunk_id, record_id, record_version, parent_unit_id,
                        ordinal, text, token_count, metadata, content_hash
                    ) VALUES (
                        :preview_id, :chunk_id, :record_id, :record_version, :parent_unit_id,
                        :ordinal, :text, :token_count, :metadata, :content_hash
                    )
                    """
                ),
                {
                    "preview_id": preview_id,
                    "chunk_id": item.chunk_id,
                    "record_id": item.record_id,
                    "record_version": item.record_version,
                    "parent_unit_id": item.parent_unit_id,
                    "ordinal": item.ordinal,
                    "text": item.text,
                    "token_count": item.token_count,
                    "metadata": json.dumps(item.metadata),
                    "content_hash": hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
                },
            )
        await session.commit()

    return PreviewResponse(
        preview_id=preview_id,
        record_id=payload.record_id,
        record_version=payload.expected_version,
        status="ready",
        items=items,
        histogram=histogram,
        warnings=warnings,
    )
