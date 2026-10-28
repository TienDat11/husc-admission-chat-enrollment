"""Source-document service — ticket 39.

The one place that decides what the registry records and what the ledger hears
about it. It joins three things that previously never met:

- :class:`src.infrastructure.source_document_registry.SourceDocumentRegistry`
  — the entries (id verbatim, resolved path, hash observed at write time),
- :class:`src.infrastructure.canonical_record_store.CanonicalRecordStore`
  — the records whose ``source_document_id`` needs something to point at,
- :class:`src.infrastructure.audit_ledger.AuditLedger` — the append-only trail.

**Audit goes through the road that already exists.** Ticket 12 already emits
``AuditAction.SOURCE_DOCUMENT_CHANGE`` when a record's provenance moves
(``src/services/admin_lifecycle.py:587``); this service emits the *same action*
when the document itself is registered, refreshed, or retired. No new
``AuditAction`` value exists or is needed (the ticket forbids inventing one),
and no second ledger is opened.

Because one action now serves two questions ("whose provenance moved?" and
"what happened to the document?"), the two are told apart by ``entity``:
record-provenance events carry the canonical-record entity, registry events
carry ``"source_document"`` (``src/domain/source_document.py``). ``target_kind``
is left at ``RECORD`` — the enum names three id spaces and none of them is the
document space; adding a fourth is ticket 14's surface, not this one's. A
reader that filters registry events by ``target_kind`` will get the wrong
answer, so filter by ``entity``. That is a documented limitation of the reuse,
not an accident, and it is the reason ``target`` for a registry event is the
verbatim ``document_id`` rather than anything composed.

**No publish-time enforcement, on purpose** (Ghi chú 2). ``is_registered``
answers the existence question; nothing in the lifecycle path calls it yet, so
the 188 records seeded before this ticket land keep publishing exactly as they
do today. The sequence the ticket fixes is (a) registry + backfill, (b) the
blind-citation report, (c) a later ticket flips enforcement once (b) is clean.
Flipping it here would fail closed on real admissions data.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from src.domain.audit import AuditAction, AuditEvent
from src.domain.canonical.record import (
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.domain.source_document import (
    SOURCE_DOCUMENT_ENTITY,
    SourceDocumentKind,
    classify_document_id,
)
from src.infrastructure.audit_ledger import AuditLedger
from src.infrastructure.canonical_record_store import CanonicalRecordStore
from src.infrastructure.source_document_registry import (
    SOURCE_DOCUMENTS_DIRNAME,
    RegistryPathError,
    SourceDocumentEntry,
    SourceDocumentRegistry,
)

#: Read size for the content hash; the PDFs are ~1-3 MB, so this is about not
#: holding a whole file in memory, not about throughput.
_HASH_CHUNK = 1 << 20


@contextmanager
def _acquire_upload_lock(lock_path: Path):
    """Hold an OS-released advisory file lock on ``lock_path``.

    A per-store-root lock (``<root>/documents/.upload-lock``) that any
    service instance sharing the same store root serializes through — no
    per-object lock survives two service objects on the same root, which is
    exactly what the route's ``asyncio.to_thread`` + default accessor
    produces.

    The lock is the OS's own advisory byte-range lock, never a lock file's
    mere existence:

    - Windows: ``msvcrt.locking`` (LK_NBLCK retried) on the persistent file.
    - POSIX: ``fcntl.flock`` on the same file.

    The kernel releases the lock when the holder dies, so a stale file never
    needs unlinking, and a live holder is never displaced. Waiting callers
    block (or, on Windows, retry) until the holder releases — a duplicate
    upload then reaches the commit window only after the first one has
    committed, reads the entry, and answers the 409 it is owed.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class SourceDocumentError(Exception):
    """Base for the service's refusals; the router maps these to 4xx."""


class UploadRejected(SourceDocumentError):
    """Uploaded bytes are not a document this surface will hold.

    Magic-byte detection said "not a PDF/DOCX" — a renamed ``.exe`` or
    anything else lands here. The router maps it to 415: the *type* of
    the file is wrong, so the client should change the file, not the call.
    """


class UploadParseFailed(SourceDocumentError):
    """The document is a PDF/DOCX, but the parser could not read it.

    A decompression-bomb refusal, a parse timeout, the memory cap, or an
    unreadable file lands here. 400 — the file itself is the problem, and
    retrying the same bytes fails the same way.
    """


class UploadPageLimit(SourceDocumentError):
    """More pages than the route will parse; the 400 above it.

    Split out so a 400's detail can say "too many pages" rather than a
    generic parse failure.
    """


class DocumentNotRegisterable(SourceDocumentError):
    """The id names something the registry will not hold.

    A ``*.jsonl#slug`` corpus pointer and a path that escapes the repo root
    both land here. The distinction matters to the caller: the first is a real
    citation of a different kind (it belongs in the blind-citation report), the
    second is an attack or a typo.
    """


class DocumentAlreadyRegistered(SourceDocumentError):
    """``register`` on an id the registry already vouches for."""

    def __init__(self, document_id: str, revision: int) -> None:
        super().__init__(f"{document_id!r} is already registered at revision {revision}")
        self.document_id = document_id
        self.revision = revision


class DocumentNotRegistered(SourceDocumentError):
    """``update``/``retire`` on an id with no entry at all."""

    def __init__(self, document_id: str) -> None:
        super().__init__(f"{document_id!r} is not in the registry")
        self.document_id = document_id


class DocumentRetired(SourceDocumentError):
    """A write aimed at an entry that exists but is retired.

    Separate from :class:`DocumentNotRegistered` because the remedy differs: a
    missing id is created, a retired id is *re-registered*, and the two answer
    different operator questions ("never vouched for" vs "vouched for, then
    stopped"). The router maps it to 409 — retrying the same call cannot
    succeed, changing the call to ``register`` can.
    """

    def __init__(self, document_id: str, revision: int) -> None:
        super().__init__(
            f"{document_id!r} is retired at revision {revision}; "
            "re-register it rather than updating or retiring again"
        )
        self.document_id = document_id
        self.revision = revision


class SourceDocumentFileNotFound(SourceDocumentError):
    """A registered document exists in the registry but the file is missing
    from disk.

    The id is a real document and is registered; only the file itself is gone,
    so the caller can re-register once the file is restored. The caller
    translates this to 404 — the citation is valid, the file is not there.
    """

    def __init__(self, document_id: str, path: Path) -> None:
        super().__init__(
            f"{document_id!r} is registered but the file is not at "
            f"{path!r}"
        )
        self.document_id = document_id
        self.path = path


class SourceDocumentNotServable(SourceDocumentError):
    """A registered document whose file the /content endpoint will not serve.

    Registration is deliberately broader than byte-serving: an editor can
    register any repo-relative path that passes the id-shape check, but
    serving byte-for-byte is restricted to real source-document locations
    (``uploads/…`` blobs and corpus-root files with a supported suffix).
    A registered id that escapes that policy (e.g. a dotfile such as
    ``rag2025/.env``) answers 404, not 200 and not 500.
    """

    def __init__(self, document_id: str) -> None:
        super().__init__(
            f"{document_id!r} is registered but is not a servable "
            "source-document location"
        )
        self.document_id = document_id

@dataclass(frozen=True)
class RegistryResult:
    """One registry write: what is stored now, and what the ledger heard.

    ``event`` is ``None`` only when the write changed nothing (an update whose
    hash and notes are already what was stored) — an event that records no
    change would be noise in the trail an auditor reads.
    """

    entry: SourceDocumentEntry
    event: AuditEvent | None
    created: bool


@dataclass(frozen=True)
class UploadResult:
    """One document upload: the stored file, the read report, and the drafts.

    ``draft_record_ids`` is empty exactly when no draft was created — a
    document whose pages are all scans (or none), or a spreadsheet with no
    data rows, carries the warnings instead: the ticket's "không tạo nháp
    rỗng" is the invariant, so an empty or image-only document is
    *reported* rather than drafted. A PDF/DOCX yields at most one draft; an
    XLSX yields one per data row (ticket T17).
    """

    document_id: str
    sha256: str
    kind: str
    parser: str
    page_count: int
    chars_per_page: float
    scan_pages: tuple[int, ...]
    pii_detected: bool
    table_count: int
    draft_record_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    entry: SourceDocumentEntry
    event: AuditEvent | None

    @property
    def draft_record_id(self) -> str | None:
        """The primary draft id — the first row's for a spreadsheet, else None."""
        return self.draft_record_ids[0] if self.draft_record_ids else None


@dataclass(frozen=True)
class BlindCitation:
    """One stored record version citing a document the registry does not hold.

    Carries the record identity *and* the cited id, because the report's whole
    purpose is naming which facts rest on an unregistered source. ``kind`` is
    the classification of the cited id, so a reader sees at once whether the
    citation is a corpus pointer (expected today) or a file path that simply
    was never registered (the actual gap).
    """

    record_id: str
    version: int
    status: str
    source_document_id: str
    kind: str
    record_version_key: str = ""

    def __post_init__(self) -> None:
        if not self.record_version_key:
            object.__setattr__(
                self, "record_version_key", f"{self.record_id}@{self.version}"
            )


@dataclass(frozen=True)
class BlindCitationReport:
    """The named state: "these citations resolve to nothing registered".

    ``counts_by_kind`` is the part an operator reads first — 299 corpus
    pointers and 0 unregistered file paths is a clean registry with a known
    backlog, while the reverse is a problem. ``statuses`` echoes what was
    scanned so a published-only report can never be mistaken for the whole
    store.
    """

    statuses: tuple[str, ...]
    scanned_versions: int
    cited_versions: int
    items: tuple[BlindCitation, ...]
    counts_by_kind: dict[str, int] = field(default_factory=dict)
    counts_by_document_id: dict[str, int] = field(default_factory=dict)

    @property
    def unregistered_document_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.counts_by_document_id))


@dataclass(frozen=True)
class BackfillReport:
    """What a backfill over the record store did.

    ``registered`` counts entries created (the three real PDFs, on the real
    store); ``already_registered`` counts ids found again on a second run —
    backfill is safe to repeat, and the audit trail does not fill with
    duplicate creates. ``citations`` counts record *versions* naming a
    file-backed document, which is not the number of entries a backfill
    creates: 188 versions cite only three distinct PDFs. The checkable
    invariant is therefore ``registered + already_registered == the distinct
    cited ids`` — every document any record cites ends up registered — and
    that is what the ticket's mutation ("bỏ sót một trong ba PDF") breaks.
    """

    scanned_versions: int
    citations: int
    registered: tuple[str, ...]
    already_registered: tuple[str, ...]
    skipped_kinds: dict[str, int]
    file_absent: tuple[str, ...]

    @property
    def registered_count(self) -> int:
        return len(self.registered)


class SourceDocumentService:
    """Registry writes, backfill, and the blind-citation report.

    ``registry`` is the store; ``store`` is the canonical record store backfill
    and the report read. ``ledger`` is optional for the same reason ticket 12
    makes it optional — a read-only or rehearsal wiring can build the service
    to answer questions without recording writes — and with no ledger every
    write still lands and ``event`` is ``None``. The admin router never takes
    that option: it refuses writes without a ledger (503), because an
    unrecorded registry change is the failure mode the ticket exists to remove.
    """

    def __init__(
        self,
        registry: SourceDocumentRegistry,
        store: CanonicalRecordStore,
        ledger: AuditLedger | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._registry = registry
        self._store = store
        self._ledger = ledger
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def registry(self) -> SourceDocumentRegistry:
        return self._registry

    @property
    def ledger(self) -> AuditLedger | None:
        """The ledger writes are recorded to, or ``None``.

        Exposed so the router can refuse a write that would land unrecorded —
        the same reason :class:`AdminLifecycleService` publishes its ledger.
        """
        return self._ledger

    @property
    def store(self) -> CanonicalRecordStore:
        """The canonical record store this service writes drafts into."""
        return self._store

    def is_registered(self, document_id: str | None) -> bool:
        """Whether ``document_id`` names a registered source document.

        **This is a measurement, not a gate.** Nothing in the publish path calls
        it: enforcing "published must cite a registered document" today would
        fail closed on the 188 records seeded before this ticket existed, so the
        enforce step is deliberately a later ticket (Ghi chú 2's sequence —
        registry, then report, then enforcement). Do not read a ``True`` here as
        "publish validated this".

        ``None``/blank is ``False``: a published record's schema validator
        already rejects a blank citation, so the only way to see one is in an
        unpublished version, and it names no document either way.
        """
        if not document_id or not document_id.strip():
            return False
        return self._registry.is_registered(document_id)

    def lookup(self, document_id: str) -> SourceDocumentEntry | None:
        """The stored entry for ``document_id``, retired ones included."""
        return self._registry.get(document_id)


    def get_document_bytes(self, document_id: str) -> bytes:
        """Read the registered source file as raw bytes.

        Resolves ``document_id`` through the registry, checks the serving
        policy, reads the file, and raises a typed refusal for each failure
        mode: not registered (``DocumentNotRegistered``), registered but not
        a servable location (``SourceDocumentNotServable``), id that escapes
        the repo root (``RegistryPathError``), or file absent on disk
        (``SourceDocumentFileNotFound``). The router translates all of these
        to HTTP 404.

        The serving check uses the registry's :meth:`is_servable` predicate,
        which is deliberately narrower than ``resolve_path`` — the parser's
        resolution stays unchanged so ingestion can still open every
        legitimately-registered id; only the byte-serving path is restricted.
        """
        entry = self.lookup(document_id)
        if entry is None:
            raise DocumentNotRegistered(document_id)
        if not self._registry.is_servable(document_id):
            raise SourceDocumentNotServable(document_id)
        path = self._registry.resolve_path(document_id)
        if not path.is_file():
            raise SourceDocumentFileNotFound(document_id, path)
        return path.read_bytes()

    def list_entries(self, *, include_retired: bool = False) -> list[SourceDocumentEntry]:
        return self._registry.list_entries(include_retired=include_retired)

    def resolve_path(self, document_id: str) -> Path:
        """Where the parser will find a registered document.

        Delegates to the registry, which evaluates ``repo_root / document_id`` —
        the same expression ``PdfText.__init__`` uses
        (``scripts/seed_canonical_records.py:251``). The registry does not own a
        second notion of where a document lives, and a test that compared this
        against the entry's stored ``path`` could not detect a change of meaning.
        """
        return self._registry.resolve_path(document_id)

    # -- writes ------------------------------------------------------------------

    def register(
        self,
        document_id: str,
        *,
        actor_id: str,
        role: str | None = None,
        notes: str | None = None,
    ) -> RegistryResult:
        """Add a document to the registry.

        Registration is by reference: the file must already be in the repo at
        ``document_id`` (there is no upload route — ``python-multipart`` is not
        installed and is not in ``requirements.txt``; see the ticket's Bối
        cảnh). The hash is computed **here and now**, from the file if it is
        present; a missing file registers as ``file_present=False, sha256=None``
        rather than failing, because the PDFs are gitignored and a deploy
        machine may legitimately not have them (Ghi chú 3).

        Raises :class:`DocumentNotRegisterable` for a corpus pointer or an id
        that escapes the repo root, and :class:`DocumentAlreadyRegistered` when
        the id is live — refreshing an existing entry is ``update``, so the
        audit trail distinguishes "this document started existing" from "this
        document changed".

        A *retired* id is revived here rather than refused: the event carries
        the tombstone's version key as ``before_version``, so the trail reads
        "retired at N, registered again at N+1" — the two facts a reader needs
        to tell a revival from a first registration, which ``revision == 1``
        alone cannot settle once tombstones keep the counter running.
        """
        self._require_registerable(document_id)
        existing = self._registry.get(document_id)
        if existing is not None and existing.is_registered:
            raise DocumentAlreadyRegistered(document_id, existing.revision)
        digest, present = self._hash_of(document_id)
        now = self._clock()
        entry = self._registry.create(
            document_id,
            sha256=digest,
            file_present=present,
            actor_id=actor_id,
            notes=notes,
            now=now,
        )
        event = self._emit(
            entry=entry,
            actor_id=actor_id,
            role=role,
            before_version=(
                None
                if existing is None
                else _document_version_key(document_id, existing.revision)
            ),
            reason=notes,
            timestamp=now,
        )
        return RegistryResult(entry=entry, event=event, created=True)

    def upload_document(
        self,
        file_bytes: bytes,
        filename: str,
        *,
        actor_id: str,
        role: str | None = None,
        notes: str | None = None,
        max_pages: int = 300,
        ram_limit_mb: int = 1024,
        timeout_s: float = 60.0,
        max_decompressed_mb: int = 512,
    ) -> UploadResult:
        """Store uploaded bytes, read them in a subprocess, and draft a record.

        The reverse of :meth:`register`: the document arrives as bytes
        instead of a path. Classification is by magic bytes, never by the
        extension — ``filename`` contributes only to the operator note. The
        content hash is the id's key (``uploads/<sha256>.<ext>``), so the
        same bytes again answer :class:`DocumentAlreadyRegistered` (409).

        The parse runs in a subprocess (:mod:`src.services.document_parser`):
        a hostile file that wedges the parser cannot take the chatbot down.
        Draft rules, per the ticket:

        - parse error / unreadable file -> no draft, :class:`UploadParseFailed`
          is raised only for *refusals* (bomb, timeout, cap); a plain
          extraction failure comes back with the error surfaced as a warning
          and no draft, so the report still reaches the operator;
        - all pages are scans (or no text at all) -> no draft (empty drafts
          are forbidden), the warnings name the scanned pages;
        - otherwise a draft ``THONG_BAO`` record is written, citing the
          stored document;
        - an XLSX (ticket T17) is a table: every data row becomes its own
          draft record, carrying the row as a structured ``values`` payload.

        Raises
        ------
        UploadRejected
            magic bytes are not PDF/DOCX/XLSX (415 at the route);
        DocumentAlreadyRegistered
            these bytes were already uploaded (409);
        UploadPageLimit
            more than ``max_pages`` pages (400, named);
        UploadParseFailed
            the parser refused the file (bomb/timeout/memory cap) (400).
        """
        return self._upload_document(
            file_bytes,
            filename,
            actor_id=actor_id,
            role=role,
            notes=notes,
            max_pages=max_pages,
            ram_limit_mb=ram_limit_mb,
            timeout_s=timeout_s,
            max_decompressed_mb=max_decompressed_mb,
        )

    def _upload_document(
        self,
        file_bytes: bytes,
        filename: str,
        *,
        actor_id: str,
        role: str | None,
        notes: str | None,
        max_pages: int,
        ram_limit_mb: int,
        timeout_s: float,
        max_decompressed_mb: int,
    ) -> UploadResult:
        """The upload work under a store-root lock; see :meth:`upload_document`."""
        from src.services.document_parser import parse_document
        from src.services.magic_bytes import DOCX_KIND, PDF_KIND, XLSX_KIND, detect_kind

        kind = detect_kind(file_bytes)
        if kind is None:
            raise UploadRejected(
                f"{filename!r} is not a PDF, DOCX or XLSX by its content; "
                "uploads are classified by magic bytes, not the extension"
            )
        extension = {
            PDF_KIND: "pdf",
            DOCX_KIND: "docx",
            XLSX_KIND: "xlsx",
        }[kind]
        digest = hashlib.sha256(file_bytes).hexdigest()
        document_id = f"uploads/{digest}.{extension}"

        # Serialize the whole commit window across instances of this store
        # root: the route runs uploads in worker threads (asyncio.to_thread)
        # and the default accessor builds a service per request, so two
        # identical uploads can reach the duplicate check at the same time.
        # A store-root file lock (msvcrt/POSIX advisory, released by the OS
        # on exit) is shared by every instance of this store root — no
        # per-object lock survives two service objects, and the lock never
        # needs manual cleanup: the kernel releases it when the holder dies.
        lock_path = self._registry.root / "documents" / ".upload-lock"
        with _acquire_upload_lock(lock_path):
            existing = self._registry.get(document_id)
            if existing is not None and existing.is_registered:
                raise DocumentAlreadyRegistered(document_id, existing.revision)

            # Parse before anything is written: a file the parser refuses or
            # cannot read must not leave an orphan blob under
            # source_documents/. The probe name carries a random suffix so
            # two concurrent parses of different contents never unlink each
            # other's file; identical contents are already serialized by the
            # lock above.
            probe_dir = self._store.root / SOURCE_DOCUMENTS_DIRNAME
            probe_dir.mkdir(parents=True, exist_ok=True)
            probe_path = probe_dir / f".probe-{digest}-{uuid.uuid4().hex}"
            try:
                probe_path.write_bytes(file_bytes)
                parse = parse_document(
                    probe_path,
                    kind,
                    ram_limit_mb=ram_limit_mb,
                    timeout_s=timeout_s,
                    max_decompressed_mb=max_decompressed_mb,
                )
            finally:
                probe_path.unlink(missing_ok=True)

            if parse.error:
                if parse.refused:
                    raise UploadParseFailed(parse.error)
                # Extraction failed but the file is structurally fine: report
                # it, still register the document, draft nothing (no empty
                # drafts).
                warnings: list[str] = [parse.error]
            else:
                warnings = list(parse.warnings)
            if parse.page_count > max_pages:
                raise UploadPageLimit(
                    f"document has {parse.page_count} pages; the limit is "
                    f"{max_pages}"
                )

            draft_record_ids: tuple[str, ...] = ()
            text = parse.text.strip()
            all_pages_scan = (
                parse.page_count > 0
                and len(parse.scan_pages) == parse.page_count
            )
            if kind == XLSX_KIND:
                # Ticket T17: one draft record per spreadsheet data row, each
                # carrying its ``{header: value}`` dict as the structured
                # payload. Drafts are written **before** the registry commit,
                # and on a retry that races an interrupted first attempt the
                # writers reuse any matching v1 record rather than appending
                # a second v1, so the upload is resumable.
                draft_record_ids, fell_back = self._write_xlsx_drafts(
                    document_id=document_id,
                    excel_rows=parse.excel_rows,
                    actor_id=actor_id,
                )
                if fell_back:
                    warnings.append(
                        "header không nêu rõ loại (điểm chuẩn / chỉ tiêu); "
                        "nháp dùng THONG_BAO"
                    )
            elif text and not all_pages_scan and not parse.error:
                draft_record_ids = (
                    self._write_upload_draft(
                        document_id=document_id,
                        digest=digest,
                        text=text,
                        actor_id=actor_id,
                        docx_headings=parse.headings,
                    ),
                )
            elif parse.error is None and all_pages_scan:
                warnings.append(
                    "all pages are scans (image only); no draft created"
                )

            target_path = probe_dir / f"{digest}.{extension}"
            target_path.write_bytes(file_bytes)

            now = self._clock()
            entry = self._registry.create(
                document_id,
                sha256=digest,
                file_present=True,
                actor_id=actor_id,
                notes=notes,
                now=now,
            )
        event = self._emit(
            entry=entry,
            actor_id=actor_id,
            role=role,
            before_version=None,
            reason=notes or "uploaded document",
            timestamp=now,
        )
        return UploadResult(
            document_id=document_id,
            sha256=digest,
            kind=kind,
            parser=parse.parser,
            page_count=parse.page_count,
            chars_per_page=parse.chars_per_page,
            scan_pages=parse.scan_pages,
            pii_detected=parse.pii_detected,
            table_count=parse.table_count,
            draft_record_ids=draft_record_ids,
            warnings=tuple(warnings),
            entry=entry,
            event=event,
        )

    def _write_upload_draft(
        self,
        *,
        document_id: str,
        digest: str,
        text: str,
        actor_id: str,
        docx_headings: Sequence[dict] | None = None,
    ) -> str:
        """The draft record an upload creates: one ``THONG_BAO`` version 1.

        The record cites the stored document by its registry id, so the
        provenance and the blind-citation report behave exactly as for a
        reference-registered document. ``record_id`` is stable per content
        hash: a different file, a different record; the same file can only
        ever draft once, because the registry refuses the second upload.

        A retry after an interrupted first attempt (the draft was written
        but the blob or registry entry was not committed) finds version 1
        already there; reusing it — rather than appending a second v1 — is
        what keeps the upload recoverable: the store is append-only, so a
        ``write_record`` of an existing version is a hard
        :class:`RecordVersionConflictError`.
        """
        record_id = f"upload:{digest[:16]}"
        existing = self._store.versions(record_id)
        if existing:
            stored = self._store.get_version(record_id, 1)
            if stored.source_document_id == document_id:
                return record_id
        now = self._clock()
        record = CanonicalRecord(
            record_id=record_id,
            record_type=RecordType.THONG_BAO,
            version=1,
            admission_year=now.year,
            effective_from=date(now.year, 1, 1),
            status=RecordStatus.DRAFT,
            authority_level=AuthorityLevel.HUSC_OFFICIAL,
            source_document_id=document_id,
            text=text,
            authored_by=actor_id,
            docx_headings=docx_headings or (),
        )
        self._store.write_record(record)
        return record.record_id

    def _write_xlsx_drafts(
        self,
        *,
        document_id: str,
        excel_rows: Sequence[dict],
        actor_id: str,
    ) -> tuple[tuple[str, ...], bool]:
        """One draft record per spreadsheet data row (ticket T17).

        Each row's ``{header: value}`` dict becomes the record's structured
        payload (the ``values`` extra ``structured_payload`` reads) and its
        rendered ``"<header>: <value>"`` lines become ``text``. The record
        kind is read off the header — a ``điểm chuẩn`` column names a
        ``DIEM_CHUAN`` row, a ``chỉ tiêu`` column a ``CHI_TIEU`` one, and a
        sheet that names neither stays ``THONG_BAO`` (the safe default the
        ticket prescribes when the kind is unclear).

        Returns ``(record_ids, fell_back)``: ``fell_back`` is ``True`` when
        no header named a kind, so the caller can note the ``THONG_BAO``
        fallback in the upload warnings. ``record_id`` is
        ``xlsx-<hash12>-row<n>`` where the hash is over the row's own content
        plus its 1-based number (the ticket's rule), so the id is
        deterministic and a re-upload of the same bytes — which the
        registry's content-hash key already refuses with 409 — could never
        mint a second record anyway.
        """
        now = self._clock()
        record_ids: list[str] = []
        fell_back = False
        for index, row in enumerate(excel_rows, start=1):
            if not row:
                continue
            record_type = _xlsx_record_type(row)
            fell_back = fell_back or record_type is RecordType.THONG_BAO
            record_id = _xlsx_row_id(row, index)
            # A retry of an interrupted upload finds the row's v1 already
            # written; reusing it keeps the upload resumable (the store is
            # append-only, so re-writing v1 is a hard conflict).
            if self._store.versions(record_id):
                stored = self._store.get_version(record_id, 1)
                if stored.source_document_id == document_id:
                    record_ids.append(record_id)
                    continue
            text = "\n".join(
                _xlsx_render(name, value) for name, value in row.items()
            )
            record = CanonicalRecord(
                record_id=record_id,
                record_type=record_type,
                version=1,
                admission_year=now.year,
                effective_from=date(now.year, 1, 1),
                status=RecordStatus.DRAFT,
                authority_level=AuthorityLevel.HUSC_OFFICIAL,
                source_document_id=document_id,
                text=text,
                authored_by=actor_id,
                values=row,
            )
            self._store.write_record(record)
            record_ids.append(record.record_id)
        return tuple(record_ids), fell_back

    def update(
        self,
        document_id: str,
        *,
        actor_id: str,
        role: str | None = None,
        notes: str | None = None,
    ) -> RegistryResult:
        """Re-observe a live document (new hash, new notes).

        This is how "the file's content changed after records already cited it"
        becomes visible: the entry carries the hash taken at the last
        observation, so a later ``update`` that records a different digest is
        the event that says so. A re-observation with nothing changed writes
        neither a new revision nor an audit line.

        A retired entry is refused with :class:`DocumentRetired` — an edit is
        not a way to un-retire. Bringing a retired id back is ``register``,
        which records the revival as a fresh registration.
        """
        current = self._require_live_entry(document_id)
        digest, present = self._hash_of(document_id)
        unchanged = (
            digest == current.sha256
            and present == current.file_present
            and (notes is None or notes == current.notes)
        )
        if unchanged:
            return RegistryResult(entry=current, event=None, created=False)
        now = self._clock()
        entry = self._registry.update(
            document_id,
            sha256=digest,
            file_present=present,
            actor_id=actor_id,
            notes=notes if notes is not None else current.notes,
            now=now,
        )
        event = self._emit(
            entry=entry,
            actor_id=actor_id,
            role=role,
            before_version=_document_version_key(document_id, current.revision),
            reason=notes,
            timestamp=now,
        )
        return RegistryResult(entry=entry, event=event, created=False)

    def retire(
        self,
        document_id: str,
        *,
        actor_id: str,
        reason: str,
        role: str | None = None,
    ) -> RegistryResult:
        """Stop vouching for a document.

        Soft: the entry stays (state ``retired``) so the registry can still
        answer "was this ever registered, and when did that stop?" — the
        question an audit reader asks after seeing a record cite a dead id. A
        reason is mandatory: retiring is the decision that un-vouches for
        citations other records still make, and a decision without one is not
        accountable (the same rule ``REASON_REQUIRED`` states for reject and
        rollback).

        Retiring an already-retired id is refused with
        :class:`DocumentRetired`, symmetric with ``update``: the second call
        would bump the revision and emit a second event for a state that has
        not changed, which an audit reader would misread as a new decision.
        """
        if not (reason or "").strip():
            raise SourceDocumentError(
                "retiring a source document requires a reason — it un-vouches "
                "for citations other records still make"
            )
        current = self._require_live_entry(document_id)
        now = self._clock()
        entry = self._registry.retire(document_id, actor_id=actor_id, now=now)
        event = self._emit(
            entry=entry,
            actor_id=actor_id,
            role=role,
            before_version=_document_version_key(document_id, current.revision),
            reason=reason,
            timestamp=now,
        )
        return RegistryResult(entry=entry, event=event, created=False)

    # -- backfill ----------------------------------------------------------------

    def backfill(
        self,
        *,
        actor_id: str = "system:backfill",
        role: str | None = None,
    ) -> BackfillReport:
        """Register every file-backed document the record store already cites.

        Reads the store, not a hand-written list: the point of the ticket's
        "188 records don't lose their resolution" criterion is that the
        registry is *derived from* the data, so a document a record cites but
        nobody thought to list still gets registered. Corpus pointers
        (``*.jsonl#slug``) are counted in ``skipped_kinds`` and deliberately
        not registered — they name a slice of a derived corpus, not a source
        document, and registering them would empty the blind-citation report of
        meaning (Ghi chú 4: choose one semantics; this is the chosen one).

        Idempotent: a second run reports the same ids as
        ``already_registered`` and emits no events. Each *new* entry emits one
        ``SOURCE_DOCUMENT_CHANGE`` (three PDFs, three events) — a per-record
        event for 188 records would bury the provenance-move events the same
        action already carries.
        """
        registered: list[str] = []
        already: list[str] = []
        absent: list[str] = []
        skipped: dict[str, int] = {}
        citations = 0
        scanned = 0
        seen: set[str] = set()
        for record_id in self._store.record_ids():
            for version in self._store.versions(record_id):
                record = self._store.get_version(record_id, version)
                scanned += 1
                document_id = (record.source_document_id or "").strip()
                if not document_id:
                    continue
                kind = classify_document_id(document_id)
                if kind is not SourceDocumentKind.PDF_DOCUMENT:
                    skipped[kind.value] = skipped.get(kind.value, 0) + 1
                    continue
                citations += 1
                if document_id in seen:
                    continue
                seen.add(document_id)
                outcome = self._backfill_one(
                    document_id, actor_id=actor_id, role=role
                )
                if outcome is None:
                    continue
                entry, created = outcome
                if created:
                    registered.append(entry.document_id)
                    if not entry.file_present:
                        absent.append(entry.document_id)
                else:
                    already.append(entry.document_id)
        return BackfillReport(
            scanned_versions=scanned,
            citations=citations,
            registered=tuple(registered),
            already_registered=tuple(already),
            skipped_kinds=skipped,
            file_absent=tuple(absent),
        )

    def _backfill_one(
        self, document_id: str, *, actor_id: str, role: str | None
    ) -> tuple[SourceDocumentEntry, bool] | None:
        """Ensure one cited document is registered; ``(entry, created)``.

        ``None`` when the id cannot be registered at all (an id the classifier
        accepted but the registry's own containment check rejects) — backfill
        must not die halfway because one citation is malformed; the
        blind-citation report is where such an id surfaces.
        """
        try:
            self._require_registerable(document_id)
        except DocumentNotRegisterable:
            return None
        existing = self._registry.get(document_id)
        if existing is not None and existing.is_registered:
            return existing, False
        digest, present = self._hash_of(document_id)
        now = self._clock()
        try:
            entry = self._registry.create(
                document_id,
                sha256=digest,
                file_present=present,
                actor_id=actor_id,
                notes="registered by backfill from the canonical record store",
                now=now,
            )
        except RegistryPathError:
            return None
        self._emit(
            entry=entry,
            actor_id=actor_id,
            role=role,
            before_version=None,
            reason="backfill from the canonical record store",
            timestamp=now,
        )
        return entry, True

    # -- blind-citation report -----------------------------------------------------

    def blind_citations(
        self, statuses: Iterable[RecordStatus] | None = (RecordStatus.PUBLISHED,)
    ) -> BlindCitationReport:
        """Record versions citing a document the registry does not hold.

        Default is **published only**, which is the only population the ticket's
        criterion is about: a draft's citation is a to-do, not a live fact
        resting on nothing. Pass ``None`` to scan every stored version (the
        numbers the ticket's acceptance measurement quotes for the real store
        are of that wider scan).

        The observable is *the named list*, not an empty one: today the real
        store's corpus pointers all appear here, and that is the expected
        reading of Ghi chú 4's chosen semantics, not a failure. A future ticket
        that flips publish-time enforcement (Ghi chú 2 step c) runs this to
        clean.
        """
        wanted = None if statuses is None else frozenset(
            status if isinstance(status, RecordStatus) else RecordStatus(status)
            for status in statuses
        )
        items: list[BlindCitation] = []
        by_kind: dict[str, int] = {}
        by_document: dict[str, int] = {}
        scanned = 0
        cited = 0
        for record_id in self._store.record_ids():
            for version in self._store.versions(record_id):
                record = self._store.get_version(record_id, version)
                if wanted is not None and record.status not in wanted:
                    continue
                scanned += 1
                document_id = (record.source_document_id or "").strip()
                if not document_id:
                    continue
                cited += 1
                if self.is_registered(document_id):
                    continue
                kind = classify_document_id(document_id).value
                items.append(
                    BlindCitation(
                        record_id=record_id,
                        version=version,
                        status=record.status.value,
                        source_document_id=document_id,
                        kind=kind,
                    )
                )
                by_kind[kind] = by_kind.get(kind, 0) + 1
                by_document[document_id] = by_document.get(document_id, 0) + 1
        return BlindCitationReport(
            statuses=tuple(status.value for status in (wanted or ())) or ("all",),
            scanned_versions=scanned,
            cited_versions=cited,
            items=tuple(items),
            counts_by_kind=by_kind,
            counts_by_document_id=by_document,
        )

    # -- internals ---------------------------------------------------------------

    def _require_registerable(self, document_id: str) -> None:
        """Refuse ids the registry can never hold, as a typed error.

        The registry's own ``_validate_document_id`` raises the plain
        :class:`RegistryPathError`; the service translates it so the router
        can map "this string is not a document id" (422) distinctly from
        "this document is not registered" (404).
        """
        text = (document_id or "").strip()
        if not text:
            raise DocumentNotRegisterable("document_id is blank")
        kind = classify_document_id(text)
        if kind is not SourceDocumentKind.PDF_DOCUMENT:
            raise DocumentNotRegisterable(
                f"{document_id!r} is a {kind.value}, not a file-backed document"
            )
        try:
            self._registry.resolve_path(text)
        except RegistryPathError as exc:
            raise DocumentNotRegisterable(str(exc)) from exc

    def _require_live_entry(self, document_id: str) -> SourceDocumentEntry:
        """The entry for ``document_id``, refusing three different situations.

        The registry's ``get`` is total — it answers ``None`` for an id it can
        never hold as well as for one nothing is stored at — so the write path
        re-checks the shape here and keeps the refusals distinguishable:
        an id that names a corpus pointer is :class:`DocumentNotRegisterable`
        (422: this string can never be a document), a valid id with no entry is
        :class:`DocumentNotRegistered` (404: nothing registered yet), and a
        retired entry is :class:`DocumentRetired` (409: registered, then
        stopped). Collapsing the first into the second would tell an operator
        to register an id that registration will refuse.
        """
        self._require_registerable(document_id)
        entry = self._registry.get(document_id)
        if entry is None:
            raise DocumentNotRegistered(document_id)
        if not entry.is_registered:
            raise DocumentRetired(document_id, entry.revision)
        return entry

    def _hash_of(self, document_id: str) -> tuple[str | None, bool]:
        """``(sha256, file_present)`` for a document id, read once.

        Hashing at write time only (Ghi chú 3): the registry never re-reads a
        file on the read path, so serving a citation costs no I/O against the
        PDF, and a hash that later stops matching is information an ``update``
        records rather than a request that fails.
        """
        path = self._registry.resolve_path(document_id)
        if not path.is_file():
            return None, False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
                digest.update(chunk)
        return digest.hexdigest(), True

    def _emit(
        self,
        *,
        entry: SourceDocumentEntry,
        actor_id: str,
        role: str | None,
        before_version: str | None,
        reason: str | None,
        timestamp: datetime,
    ) -> AuditEvent | None:
        """Append one registry event on the ticket-12 action, or ``None``.

        Same ledger, same action, same frame as the provenance-move emitter;
        ``entity`` is what says this one is about the document rather than about
        a record's citation (module docstring), and the version keys are
        revisions in the *document* id space — ``document_id@revision``, the
        shape ``_version_key`` gives record events, not a record key.
        """
        event = AuditEvent(
            action=AuditAction.SOURCE_DOCUMENT_CHANGE,
            actor_id=actor_id,
            role=role,
            target=entry.document_id,
            timestamp=timestamp,
            before_version=before_version,
            after_version=_document_version_key(entry.document_id, entry.revision),
            reason=reason,
            entity=SOURCE_DOCUMENT_ENTITY,
        )
        if self._ledger is None:
            return None
        self._ledger.append(event)
        return event


def _document_version_key(document_id: str, revision: int) -> str:
    """``document_id@revision`` — the identity of one registry entry state.

    ``None`` is not an option here (an ``after_version`` always names the state
    the write produced), and the prefix-free shape matches how record versions
    are keyed, so a reader who knows one convention knows both. The id space is
    told apart by ``entity``, not by the key's shape.
    """
    return f"{document_id}@{revision}"


def record_source_document_id(record: CanonicalRecord) -> str | None:
    """The document a record cites, blank-normalised to ``None``.

    One helper so ``strip()``-then-test happens in exactly one place across the
    service's scans.
    """
    text = (record.source_document_id or "").strip()
    return text or None


#: Header keywords that name a spreadsheet's fact kind, folded (lowercase, no
#: diacritics) so "Điểm chuẩn" and "diem chuan" match the same entry. First
#: match in declaration order wins; a sheet naming neither is a ``THONG_BAO``.
_XLSX_KIND_KEYWORDS: tuple[tuple[RecordType, str], ...] = (
    (RecordType.DIEM_CHUAN, "diem chuan"),
    (RecordType.CHI_TIEU, "chi tieu"),
)


def _xlsx_row_id(row: Mapping[str, object], index: int) -> str:
    """``xlsx-<hash12>-row<n>`` — a stable id for one spreadsheet data row.

    The hash is over the row's own content (canonical JSON: keys sorted, no
    whitespace) *plus* its 1-based number, exactly as the ticket prescribes,
    so two identical rows in one sheet still get distinct ids and re-reading
    the same bytes yields the same ids.
    """
    payload = json.dumps(
        {"row": index, "values": row}, sort_keys=True, ensure_ascii=False
    )
    row_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"xlsx-{row_hash}-row{index}"


def _xlsx_record_type(row: Mapping[str, object]) -> RecordType:
    """The record kind a spreadsheet row states, from its header names.

    Reads the ticket's two named shapes — ``điểm chuẩn`` → ``DIEM_CHUAN``,
    ``chỉ tiêu`` → ``CHI_TIEU`` — and falls back to ``THONG_BAO`` (the
    ticket's prescribed default) when the header names neither, so an
    unrecognised sheet still drafts rather than being dropped.
    """
    header = " ".join(_fold_header(name) for name in row)
    for record_type, keyword in _XLSX_KIND_KEYWORDS:
        if keyword in header:
            return record_type
    return RecordType.THONG_BAO


def _fold_header(text: str) -> str:
    """Lowercase + strip Vietnamese diacritics, so header matching is robust."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(
        "d" if char == "đ" else char
        for char in decomposed
        if char == "đ" or not unicodedata.combining(char)
    )
    return stripped


def _xlsx_render(name: str, value: object) -> str:
    """``"<header>: <value>"`` — the line ``semantic_units._render`` builds.

    The same two special cases apply: a bool renders as ``có``/``không``
    and an integer-valued float drops its trailing ``.0``, so a record's
    ``text`` and its ``values`` payload agree line for line.
    """
    if isinstance(value, bool):
        return f"{name}: {'có' if value else 'không'}"
    if isinstance(value, float) and value.is_integer():
        return f"{name}: {int(value)}"
    return f"{name}: {value}"
