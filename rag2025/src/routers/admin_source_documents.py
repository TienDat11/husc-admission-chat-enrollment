"""Source-document registry endpoints — ticket 39.

Mounted from ``src/routers/admin.py`` the way ticket 38's conflict queue is:
the child carries its own ``/source-documents`` prefix and inherits the parent's
``/admin`` prefix and rate-limit dependency, so ``main.py`` never learns the
ticket exists. Final paths are ``/admin/source-documents/...``.

``document_id`` is a repo-relative path (``new_data/THÔNG TIN…pdf``) — it
contains slashes, spaces, and Vietnamese, so it is never a URL path segment
here. Every route takes it as a query parameter or a body field and answers
with it verbatim; that is also the criterion-1 check, since a route that
decoded it into something else would show up as a mismatch between the id sent
and the id returned.

RBAC reuses the record grants rather than declaring new actions: reading the
registry is ``records.read``, registering is ``records.create``, refreshing an
entry is ``records.edit``, retiring is ``records.rollback`` (the only existing
grant whose meaning is "undo something already relied on"). Declaring
``sources.*`` would mean editing ``src/domain/admin_identity.py`` and its role
matrix, which other tickets are editing; the mapping is stated here so the next
reader sees a decision rather than an oversight.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote_from_bytes
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from src.domain.admin_identity import AdminAction, Identity
from src.domain.canonical.record import RecordStatus
from src.domain.source_document import classify_document_id
from src.infrastructure.canonical_record_store import CanonicalRecordStore
from src.infrastructure.source_document_registry import (
    RegistryPathError,
    SourceDocumentRegistry,
)
from src.routers.admin_auth import current_identity, require_action
from src.services.source_document_service import (
    BackfillReport,
    BlindCitationReport,
    DocumentAlreadyRegistered,
    DocumentNotRegisterable,
    DocumentNotRegistered,
    DocumentRetired,
    RegistryResult,
    SourceDocumentError,
    SourceDocumentFileNotFound,
    SourceDocumentNotServable,
    SourceDocumentService,
)

router = APIRouter(prefix="/source-documents", tags=["admin-source-documents"])

# Suffix → media type for GET /content. Keyed by the file's own extension so a
# source of any registered kind streams with a type the browser can render;
# unknown suffixes fall back to application/octet-stream.
_MEDIA_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".jsonl": "application/x-ndjson",
}

# One envelope for the service's refusals, same idiom as the conflict routes:
# a client error is the service saying no, not the server failing.
_REGISTRY_ERRORS: dict[type[SourceDocumentError], int] = {
    DocumentNotRegisterable: 422,
    DocumentAlreadyRegistered: 409,
    DocumentRetired: 409,
    DocumentNotRegistered: 404,
    SourceDocumentFileNotFound: 404,
    SourceDocumentNotServable: 404,
}

#: Repo root, the anchor a ``document_id`` resolves against. Derived from this
#: file's location rather than from ``CANONICAL_STORE_ROOT`` because the store
#: root is overridable by env and the document ids are not: ``new_data/…`` is
#: relative to the checkout, exactly as ``seed_canonical_records.py:47-48``
#: computes it (``parents[1]`` of that script, then ``.parent``).
REPO_ROOT = Path(__file__).resolve().parents[3]


class SourceDocumentView(BaseModel):
    """One registry entry, as an operator reads it."""

    document_id: str
    kind: str
    path: str
    sha256: str | None
    file_present: bool
    state: str
    revision: int
    registered_at: str
    registered_by: str
    updated_at: str
    updated_by: str
    notes: str | None


class SourceDocumentListView(BaseModel):
    """The registry plus the counts that make it checkable at a glance.

    ``total`` and ``registered`` differ only when something was retired;
    ``counts_by_kind`` is ``{"pdf_document": n}`` today, because the registry
    holds documents only (Ghi chú 4) — the field exists so a reader confirms
    that rather than assuming it.
    """

    entries: list[SourceDocumentView]
    total: int
    registered: int
    counts_by_kind: dict[str, int]


class RegistrationRequest(BaseModel):
    """The body of a register/update call: the id and an optional note."""

    document_id: str = Field(min_length=1)
    notes: str | None = None


class RetireRequest(BaseModel):
    """Retirement names a reason: it un-vouches for live citations."""

    document_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class RegisteredCheckView(BaseModel):
    """The answer to "is this document id in the registry?" plus why.

    ``kind`` is returned even when unregistered, so a caller sees whether the
    id was a corpus pointer (never registerable here) or a document nobody
    registered yet — the two read very differently to an admissions officer.
    """

    document_id: str
    registered: bool
    kind: str
    entry: SourceDocumentView | None = None


class BlindCitationItemView(BaseModel):
    record_id: str
    version: int
    status: str
    source_document_id: str
    kind: str
    record_version_key: str


class BlindCitationListView(BaseModel):
    """The named state: record versions citing something unregistered.

    ``items`` being non-empty is the expected reading today (the store's
    ``*.jsonl#slug`` pointers are here by design); the observable the ticket
    asks for is that the list exists and enumerates, not that it is clean.
    """

    statuses: list[str]
    scanned_versions: int
    cited_versions: int
    unregistered_count: int
    counts_by_kind: dict[str, int]
    counts_by_document_id: dict[str, int]
    items: list[BlindCitationItemView]


class BackfillView(BaseModel):
    """What a backfill over the record store did, with the counts measured."""

    scanned_versions: int
    citations: int
    registered: list[str]
    already_registered: list[str]
    skipped_kinds: dict[str, int]
    file_absent: list[str]


def _source_document_service(request: Request) -> SourceDocumentService:
    """The app's registry service, or a lazy build from what the app exposes.

    Mirrors the conflict-queue accessor: a service on
    ``app.state.source_document_service`` wins; otherwise the store and ledger
    ``configure_canonical_record_store`` already put on state are enough. The
    registry lives under the *same root* as the ledger (``<root>/audit/``) and
    the outbox (``<root>/outbox/``) — ``<root>/documents/`` — so a store copy
    never arrives without its registry. An app with no canonical store answers
    503 rather than inventing an empty registry.

    The ledger is deliberately not passed to the read accessor: see
    :func:`_writable_service` for why writes refuse a missing one.
    """
    service = getattr(request.app.state, "source_document_service", None)
    if isinstance(service, SourceDocumentService):
        return service
    store = getattr(request.app.state, "canonical_record_store", None)
    if not isinstance(store, CanonicalRecordStore):
        raise HTTPException(
            status_code=503,
            detail="canonical record store not configured",
        )
    ledger = getattr(request.app.state, "audit_ledger", None)
    repo_root = getattr(request.app.state, "source_document_repo_root", None)
    registry = SourceDocumentRegistry(
        store.root, Path(repo_root) if repo_root else REPO_ROOT
    )
    return SourceDocumentService(registry, store, ledger)


def _writable_service(request: Request) -> SourceDocumentService:
    """The registry service for a write, refusing one with no audit ledger.

    The service type allows ``ledger=None`` so a read-only wiring can exist, but
    an HTTP write must not: a registry change the ledger never heard about is
    precisely the state the ticket's audit criterion forbids, and answering 201
    while recording nothing would hide it. Same 503 the lifecycle accessor
    gives for the same condition (``src/routers/admin.py``), so the two admin
    surfaces fail alike.
    """
    service = _source_document_service(request)
    if service.ledger is None:
        raise HTTPException(
            status_code=503,
            detail="audit ledger is not configured",
        )
    return service


def _view(entry: Any) -> SourceDocumentView:
    return SourceDocumentView(
        document_id=entry.document_id,
        kind=entry.kind.value,
        path=entry.path,
        sha256=entry.sha256,
        file_present=entry.file_present,
        state=entry.state.value,
        revision=entry.revision,
        registered_at=entry.registered_at.isoformat(),
        registered_by=entry.registered_by,
        updated_at=entry.updated_at.isoformat(),
        updated_by=entry.updated_by,
        notes=entry.notes,
    )


def _raise_for(exc: SourceDocumentError) -> None:
    status = _REGISTRY_ERRORS.get(type(exc), 400)
    raise HTTPException(status_code=status, detail=str(exc)) from exc


def _blind_view(report: BlindCitationReport) -> BlindCitationListView:
    return BlindCitationListView(
        statuses=list(report.statuses),
        scanned_versions=report.scanned_versions,
        cited_versions=report.cited_versions,
        unregistered_count=len(report.items),
        counts_by_kind=dict(report.counts_by_kind),
        counts_by_document_id=dict(report.counts_by_document_id),
        items=[
            BlindCitationItemView(
                record_id=item.record_id,
                version=item.version,
                status=item.status,
                source_document_id=item.source_document_id,
                kind=item.kind,
                record_version_key=item.record_version_key,
            )
            for item in report.items
        ],
    )


@router.get(
    "",
    response_model=SourceDocumentListView,
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def list_source_documents(
    request: Request,
    include_retired: bool = Query(
        False, description="also list entries that were retired"
    ),
) -> SourceDocumentListView:
    """Every registered source document, sorted by id."""
    entries = _source_document_service(request).list_entries(
        include_retired=include_retired
    )
    by_kind: dict[str, int] = {}
    for entry in entries:
        by_kind[entry.kind.value] = by_kind.get(entry.kind.value, 0) + 1
    return SourceDocumentListView(
        entries=[_view(entry) for entry in entries],
        total=len(entries),
        registered=sum(1 for entry in entries if entry.is_registered),
        counts_by_kind=by_kind,
    )


@router.get(
    "/blind-citations",
    response_model=BlindCitationListView,
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def list_blind_citations(
    request: Request,
    scope: Literal["published", "all"] = Query(
        "published",
        description=(
            "which record versions to scan; 'published' is the criterion's "
            "population, 'all' measures every stored version"
        ),
    ),
) -> BlindCitationListView:
    """Record versions citing a document the registry does not hold.

    Read-only and non-blocking: nothing here stops a publish (Ghi chú 2 — the
    enforce step is a later ticket, once this list is clean for real).
    """
    statuses = None if scope == "all" else (RecordStatus.PUBLISHED,)
    report = _source_document_service(request).blind_citations(statuses)
    return _blind_view(report)


@router.get(
    "/registered",
    response_model=RegisteredCheckView,
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def check_registered(
    request: Request,
    document_id: str = Query(min_length=1, description="the citation, verbatim"),
) -> RegisteredCheckView:
    """Answer "is this document id in the registry?" for one id.

    The id is a query parameter, never a path segment: it carries slashes and
    Vietnamese, and the response echoes it back unchanged so a caller can see
    the verbatim round trip.
    """
    service = _source_document_service(request)
    entry = service.lookup(document_id)
    return RegisteredCheckView(
        document_id=document_id,
        registered=service.is_registered(document_id),
        kind=classify_document_id(document_id).value,
        entry=_view(entry) if entry is not None else None,
    )


@router.post(
    "/register",
    response_model=SourceDocumentView,
    status_code=201,
    dependencies=[Depends(require_action(AdminAction.RECORDS_CREATE))],
)
async def register_source_document(
    payload: RegistrationRequest,
    request: Request,
    identity: Identity = Depends(current_identity),
) -> SourceDocumentView:
    """Register a source document that already sits in the repo.

    Registration by reference, not upload: the file must be at
    ``repo_root / document_id``. Its sha256 is taken now; a missing file
    registers as ``file_present=false`` (an observation, not an error — the
    PDFs are gitignored).
    """
    try:
        result = _writable_service(request).register(
            payload.document_id,
            actor_id=identity.actor_id,
            role=identity.role.value,
            notes=payload.notes,
        )
    except SourceDocumentError as exc:
        _raise_for(exc)
        raise
    return _view(result.entry)


@router.post(
    "/update",
    response_model=SourceDocumentView,
    dependencies=[Depends(require_action(AdminAction.RECORDS_EDIT))],
)
async def update_source_document(
    payload: RegistrationRequest,
    request: Request,
    identity: Identity = Depends(current_identity),
) -> SourceDocumentView:
    """Re-observe a registered document (new hash and/or new note).

    This is how a file whose content changed after records cited it becomes
    visible: the response's ``sha256`` differs from the previous revision's,
    and the audit event names both revisions.
    """
    try:
        result = _writable_service(request).update(
            payload.document_id,
            actor_id=identity.actor_id,
            role=identity.role.value,
            notes=payload.notes,
        )
    except SourceDocumentError as exc:
        _raise_for(exc)
        raise
    return _view(result.entry)


@router.post(
    "/retire",
    response_model=SourceDocumentView,
    dependencies=[Depends(require_action(AdminAction.RECORDS_ROLLBACK))],
)
async def retire_source_document(
    payload: RetireRequest,
    request: Request,
    identity: Identity = Depends(current_identity),
) -> SourceDocumentView:
    """Stop vouching for a document; the entry stays as history."""
    try:
        result = _writable_service(request).retire(
            payload.document_id,
            actor_id=identity.actor_id,
            role=identity.role.value,
            reason=payload.reason,
        )
    except SourceDocumentError as exc:
        _raise_for(exc)
        raise
    return _view(result.entry)


@router.post(
    "/backfill",
    response_model=BackfillView,
    dependencies=[Depends(require_action(AdminAction.RECORDS_CREATE))],
)
async def backfill_source_documents(
    request: Request,
    identity: Identity = Depends(current_identity),
) -> BackfillView:
    """Register every file-backed document the record store already cites.

    Derived from the store, so a citation nobody thought to list is still
    registered; corpus pointers are counted and skipped by design. Safe to
    repeat — a second call reports ``already_registered`` and writes nothing.
    """
    report: BackfillReport = _writable_service(request).backfill(
        actor_id=f"backfill:{identity.actor_id}", role=identity.role.value
    )
    return BackfillView(
        scanned_versions=report.scanned_versions,
        citations=report.citations,
        registered=list(report.registered),
        already_registered=list(report.already_registered),
        skipped_kinds=dict(report.skipped_kinds),
        file_absent=list(report.file_absent),
    )


@router.get(
    "/content",
    dependencies=[Depends(require_action(AdminAction.RECORDS_READ))],
)
async def get_source_document_content(
    request: Request,
    document_id: str = Query(
        description="the registered document id; the file is served byte-for-byte"
    ),
) -> Response:
    """Serve the registered source file as raw bytes.

    ``document_id`` is looked up verbatim, resolved through the registry
    (so a ``..`` escape is refused), and read back raw: a returned body is
    byte-identical to the file on disk. Missing entry / path-escape / absent
    file all answer 404; an unauthenticated caller answers 422 per the
    admin surface's established convention.
    """
    try:
        content = _source_document_service(request).get_document_bytes(
            document_id
        )
    except (DocumentNotRegistered, RegistryPathError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SourceDocumentFileNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SourceDocumentNotServable as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # The registry accepts every supported ingestion extension (.jsonl, .txt,
    # .md, .pdf, .docx — see normalize_data.SUPPORTED_EXTENSIONS), so the media
    # type cannot assume "not pdf ⇒ docx": a .txt source would otherwise be
    # served as a Word document and the browser would fail to open it.
    media_type = _MEDIA_TYPES.get(
        Path(document_id).suffix.lower(), "application/octet-stream"
    )
    filename = Path(document_id).name
    # The ASCII `filename` fallback must encode in latin-1 (header encoding),
    # so: drop quotes/backslashes and control characters, replace every
    # remaining non-ASCII character with `_`, keep printable ASCII verbatim.
    # The true name lives only in the RFC 5987 `filename*` parameter, which
    # is percent-encoded UTF-8 and carries no such limit.
    _parts: list[str] = []
    for _ch in filename:
        _o = ord(_ch)
        if _ch in ('"', "'") or _o < 32 or _o == 127:
            continue
        _parts.append(_ch if _o <= 126 else "_")
    safe_filename = "".join(_parts)
    disposition = (
        f"inline; filename=\"{safe_filename}\"; "
        f"filename*=UTF-8''{quote_from_bytes(filename.encode('utf-8'), safe='')}"
    )
    return Response(content=content, media_type=media_type,
                    headers={"Content-Disposition": disposition})

