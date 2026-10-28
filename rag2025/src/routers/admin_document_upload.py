"""Document upload endpoint — ticket T04.

Admins upload a PDF/DOCX/XLSX and get a read report plus draft records.
The magic-byte check, the size cap, and the parse isolation all live
down the line (:mod:`src.services.source_document_service` →
:mod:`src.services.document_parser`); this module is the HTTP seam:
multipart in, the report out, and the typed 4xx/415/409 mapping.
"""
from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, computed_field

from src.domain.admin_identity import AdminAction, Identity
from src.routers.admin_auth import current_identity, require_action
from src.routers.admin_rate_limit import upload_rate_limit
from src.services.source_document_service import (
    DocumentAlreadyRegistered,
    SourceDocumentError,
    UploadPageLimit,
    UploadParseFailed,
    UploadRejected,
    UploadResult,
)

# The same service accessor the registry routes use: the app's configured
# service wins, otherwise the store + ledger on ``app.state`` are enough.
# Reused rather than reimplemented — an upload that reaches the registry
# must obey the same ledger-mandatory rule as a register.
from src.routers.admin_source_documents import _writable_service  # noqa: E402

router = APIRouter(prefix="/source-documents", tags=["admin-document-upload"])

#: Route budget: 20 MB per upload. Above it the request is 400 — the
#: operator splits the document or re-scans smaller, never silently drops.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
#: Pages the parser will read before it stops and asks for a smaller file.
MAX_PAGES = 300

_UPLOAD_ERRORS: dict[type[SourceDocumentError], int] = {
    UploadRejected: 415,
    DocumentAlreadyRegistered: 409,
    UploadPageLimit: 400,
    UploadParseFailed: 400,
}


class UploadReport(BaseModel):
    """What the upload read, and what it produced.

    ``report`` is the read report the ticket asks for (page count,
    characters per page, scan pages, PII flag, table count); ``warnings``
    is the Vietnamese operator-facing explanation of anything odd — a
    scan-only document, a PII flag, a read failure. ``draft_record_ids``
    is empty exactly when no draft was created: an image-only or
    unreadable document reports rather than drafts an empty record. A
    PDF/DOCX drafts at most one record; an XLSX drafts one per data row
    (ticket T17).
    """

    document_id: str
    sha256: str
    kind: str
    report: UploadParseReport
    draft_record_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def draft_record_id(self) -> str | None:
        """The primary draft id — the first row's for a spreadsheet, else None."""
        return self.draft_record_ids[0] if self.draft_record_ids else None


class UploadParseReport(BaseModel):
    """The read report itself: what the parser saw, page by page."""

    page_count: int
    chars_per_page: float
    scan_pages: list[int]
    pii_detected: bool
    table_count: int


def _view(result: UploadResult) -> UploadReport:
    return UploadReport(
        document_id=result.document_id,
        sha256=result.sha256,
        kind=result.kind,
        report=UploadParseReport(
            page_count=result.page_count,
            chars_per_page=result.chars_per_page,
            scan_pages=list(result.scan_pages),
            pii_detected=result.pii_detected,
            table_count=result.table_count,
        ),
        draft_record_ids=list(result.draft_record_ids),
        warnings=list(result.warnings),
    )


@router.post(
    "/upload",
    response_model=UploadReport,
    status_code=201,
    dependencies=[
        Depends(require_action(AdminAction.RECORDS_CREATE)),
        Depends(upload_rate_limit),
    ],
)
async def upload_source_document(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    identity: Identity = Depends(current_identity),
) -> UploadReport:
    """Upload a PDF/DOCX/XLSX, read it in a subprocess, and draft records.

    - magic bytes decide the kind; the extension is not read at all, so
      a ``.exe`` renamed ``.pdf`` is 415;
    - the same bytes again are 409 (the registry's content-hash key);
    - 20 MB cap and 300-page cap are 400;
    - a decompression bomb is refused *in the parser subprocess* and the
      route answers 400 while ``/v2/query`` keeps serving — the
      isolation is the whole point of the ticket;
    - an XLSX drafts one record per data row (ticket T17).
    """
    raw: bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"file is {len(raw)} bytes; the limit is {MAX_UPLOAD_BYTES}",
        )

    service = _writable_service(request)
    filename = file.filename or "upload"
    try:
        result = await asyncio.to_thread(
            service.upload_document,
            raw,
            filename,
            actor_id=identity.actor_id,
            role=identity.role.value,
            max_pages=MAX_PAGES,
        )
    except SourceDocumentError as exc:
        status = _UPLOAD_ERRORS.get(type(exc), 400)
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    return _view(result)
