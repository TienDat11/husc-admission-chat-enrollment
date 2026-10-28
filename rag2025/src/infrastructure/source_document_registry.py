"""Source-document registry — ticket 39.

Disk-backed, append-safe store of the documents that ``CanonicalRecord
.source_document_id`` is allowed to name. One JSON file per entry under
``<root>/documents/``, the same "travel with the records" placement the audit
ledger (``<root>/audit/``) and the re-index outbox (``<root>/outbox/``) use,
so a store copy never arrives without its registry.

The invariant this layer exists to hold:

    ``document_id`` is stored and read back **verbatim**.

It is the registry key, the parser's path segment
(``scripts/seed_canonical_records.py:251`` resolves
``repo_root / document_id``), and the string ticket 38's seed report keys its
conflict statements by. Encoding happens only when the id becomes a *filename*
— ``quote(document_id, safe="")`` under a ``doc-``/``_d`` component, exactly
the reversible scheme ``canonical_record_store`` uses for record ids, with the
same Windows load-bearing detail: ``quote`` leaves a trailing ``.`` literal and
Win32 silently strips trailing dots, so the non-dot terminator keeps the
mapping injective.

This module does no hashing and no auditing — those are the service's
(``src/services/source_document_service.py``). The store's job is that a write
either lands whole or not at all (``.tmp`` + ``os.replace``), and that reading
an entry returns the id the caller registered, byte for byte.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote

from src.domain.source_document import (
    SourceDocumentKind,
    SourceDocumentState,
    classify_document_id,
)

DOCUMENTS_DIRNAME = "documents"
SOURCE_DOCUMENTS_DIRNAME = "source_documents"
# Uploads are store-relative because the uploader writes them under the store,
# not the repo.
UPLOAD_ID_PREFIX = "uploads/"
#: Repo-relative roots a source-document id may sit under, for the /content
#: serving policy. Measured over the real store's 67 distinct ids (3 file paths
#: under ``new_data/`` and 64 ``*.jsonl#slug`` pointers under ``rag2025/data/``),
#: so the two roots cover every legitimate corpus a document may come from.
#: A repo-relative id is servable only when it resolves under one of these.
ALLOWED_CORPUS_ROOTS: tuple[str, ...] = ("new_data", "rag2025/data")
# Filename component scheme for a document_id; see the module docstring for
# why the prefix and the non-dot suffix are both required.
_DOC_DIR_PREFIX = "doc-"
_DOC_DIR_SUFFIX = "_d"
_ENTRY_FILE_RE = re.compile(r"^doc-(?P<encoded>.+)_d\.json$")

ENTRY_SCHEMA_VERSION = "1.0"


class RegistryPathError(ValueError):
    """A document_id cannot name a registry entry.

    Raised for blank ids, ids that try to escape the documents directory
    (``..`` segments, absolute paths), and ids whose kind the registry refuses
    to hold (``chunk_corpus`` pointers — see ``Ghi chú 4`` in the ticket: the
    registry vouches for real file-backed documents only, and the
    blind-citation report keeps exactly that meaning)."""


class RegistryStateError(ValueError):
    """The entry exists but its state forbids the write.

    Distinct from :class:`RegistryPathError`'s "this id can never be an entry"
    (blank, escaping, wrong kind): here the id is fine and the *history* is the
    problem — updating a retired document, say. The service maps it to a 409,
    because retrying the same call cannot succeed; re-registering can.
    """


@dataclass(frozen=True)
class SourceDocumentEntry:
    """One registered source document.

    ``document_id`` is the verbatim citation string (repo-relative,
    forward-slash). ``path`` is the absolute file it resolves to under
    ``repo_root`` — stored so a reader can see the containment decision that
    was made at registration time, and so ``resolve_path`` never re-derives a
    location from a mutable root. ``sha256`` is the file's content hash
    *observed at registration/update*; ``None`` with ``file_present=False`` is
    the ticket's ``registered, file_absent`` state (the PDFs are gitignored,
    so a machine may legitimately not have them) — it is recorded information,
    not an error. ``revision`` counts writes to this entry and is what an audit
    event's before/after version names.
    """

    document_id: str
    kind: SourceDocumentKind
    path: str
    sha256: str | None
    file_present: bool
    state: SourceDocumentState
    revision: int
    registered_at: datetime
    registered_by: str
    updated_at: datetime
    updated_by: str
    notes: str | None = None
    schema_version: str = ENTRY_SCHEMA_VERSION

    @property
    def is_registered(self) -> bool:
        """Whether this entry makes its id count as registered.

        A retired entry exists but vouches for nothing; ``file_present`` is
        *not* part of this — an entry whose file is missing is still the
        registry saying "this id is a real document" (Ghi chú 3: the point of
        the registry is the id's meaning, not the file's presence).
        """
        return self.state is SourceDocumentState.REGISTERED

    def to_json(self) -> bytes:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["state"] = self.state.value
        payload["registered_at"] = self.registered_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes | str) -> "SourceDocumentEntry":
        payload = json.loads(data)
        return cls(
            document_id=payload["document_id"],
            kind=SourceDocumentKind(payload["kind"]),
            path=payload["path"],
            sha256=payload["sha256"],
            file_present=payload["file_present"],
            state=SourceDocumentState(payload["state"]),
            revision=int(payload["revision"]),
            registered_at=datetime.fromisoformat(payload["registered_at"]),
            registered_by=payload["registered_by"],
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            updated_by=payload["updated_by"],
            notes=payload.get("notes"),
            schema_version=payload.get("schema_version", ENTRY_SCHEMA_VERSION),
        )


class SourceDocumentRegistry:
    """Disk-backed registry of source documents.

    ``root`` is the canonical-store root (the same directory
    ``CanonicalRecordStore`` takes); entries live under ``root/documents/``.
    Pass an explicit path — nothing here defaults to a production location, the
    same rule the record store states for itself. ``repo_root`` is the anchor
    a ``document_id`` resolves against; registration refuses an id that would
    land outside it.
    """

    def __init__(self, root: str | Path, repo_root: str | Path) -> None:
        self._root = Path(root)
        self._repo_root = Path(repo_root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    @property
    def documents_dir(self) -> Path:
        return self._root / DOCUMENTS_DIRNAME

    # -- write ---------------------------------------------------------------

    def put(self, entry: SourceDocumentEntry) -> SourceDocumentEntry:
        """Store ``entry`` atomically, keyed by its verbatim ``document_id``."""
        _validate_document_id(entry.document_id)
        _write_atomic(self._entry_path(entry.document_id), entry.to_json())
        return entry

    def create(
        self,
        document_id: str,
        *,
        sha256: str | None,
        file_present: bool,
        actor_id: str,
        notes: str | None = None,
        now: datetime | None = None,
    ) -> SourceDocumentEntry:
        """Register an id; raises ``FileExistsError`` when it is already live.

        The state machine, in one place:

        - nothing stored -> a first entry at revision 1;
        - an **active** entry -> ``FileExistsError``. Refreshing is ``update``,
          so the audit trail keeps "this document started existing" distinct
          from "this document changed";
        - a **retired** tombstone -> a fresh active entry at
          ``existing.revision + 1``. Revival is a new registration (new
          ``registered_at``/``registered_by``) that continues the same
          revision counter, so ``document_id@revision`` names exactly one entry
          state for the life of the id — a re-used ``@1`` would make an audit
          reader unable to tell which registration an event describes.

        The caller (service) decides ``file_present``/``sha256``; the store
        only persists what it is told, so the hashing policy lives in one
        place.
        """
        existing = self.get(document_id)
        if existing is not None and existing.is_registered:
            raise FileExistsError(document_id)
        stamp = now or datetime.now(timezone.utc)
        entry = self._build(
            document_id,
            sha256=sha256,
            file_present=file_present,
            state=SourceDocumentState.REGISTERED,
            revision=1 if existing is None else existing.revision + 1,
            registered_at=stamp,
            registered_by=actor_id,
            updated_at=stamp,
            updated_by=actor_id,
            notes=notes,
        )
        return self.put(entry)

    def update(
        self,
        document_id: str,
        *,
        sha256: str | None,
        file_present: bool,
        actor_id: str,
        notes: str | None = None,
        now: datetime | None = None,
    ) -> SourceDocumentEntry:
        """Rewrite a live entry at ``revision + 1``; state stays ``REGISTERED``.

        Two refusals, both about state rather than about the id:

        - nothing stored -> ``KeyError``. Update never creates; that is
          ``create``'s job, so the trail distinguishes "changed" from "started
          existing";
        - a retired entry -> :class:`RegistryStateError`. A note edit must not
          be a way to un-vouch-for-nothing-then-vouch-again; reviving goes
          through ``create``, which records it as a fresh registration.
        """
        current = self.get(document_id)
        if current is None:
            raise KeyError(document_id)
        if not current.is_registered:
            raise RegistryStateError(
                f"{document_id!r} is retired; re-register it rather than "
                "updating — retirement is undone by a new registration, "
                "not by an edit"
            )
        stamp = now or datetime.now(timezone.utc)
        entry = replace(
            current,
            sha256=sha256,
            file_present=file_present,
            state=SourceDocumentState.REGISTERED,
            revision=current.revision + 1,
            updated_at=stamp,
            updated_by=actor_id,
            notes=notes if notes is not None else current.notes,
        )
        return self.put(entry)

    def retire(
        self,
        document_id: str,
        *,
        actor_id: str,
        now: datetime | None = None,
    ) -> SourceDocumentEntry:
        """Mark the entry retired at ``revision + 1``; the file stays on disk.

        Soft removal keeps the registry honest about its own history: an id
        that was registered and then retired still has an entry saying when and
        by whom. Re-registering it (``create``) writes a fresh active entry over
        the tombstone at the next revision, so the revision counter is a
        per-document timeline that revival does not restart.
        """
        current = self.get(document_id)
        if current is None:
            raise KeyError(document_id)
        stamp = now or datetime.now(timezone.utc)
        entry = replace(
            current,
            state=SourceDocumentState.RETIRED,
            revision=current.revision + 1,
            updated_at=stamp,
            updated_by=actor_id,
        )
        return self.put(entry)

    # -- read ----------------------------------------------------------------

    def get(self, document_id: str) -> SourceDocumentEntry | None:
        """The entry stored under ``document_id`` (any state), or ``None``.

        ``None`` covers two cases on purpose: nothing is stored there, **and**
        the id is one the registry can never hold (a ``*.jsonl#slug`` corpus
        pointer, a blank, a path that escapes the repo root). Both answer the
        question a caller is asking — "is this vouched for?" — with "no", and
        an existence check that raised on the majority of real citations would
        make the blind-citation report unusable (the real store holds 299
        corpus pointers out of 487 versions). Writes stay strict: ``create``
        re-validates and still refuses such an id.

        Lookups are by exact key: an entry that *is* stored must carry the id
        the caller passed, byte for byte — a filename collision would surface
        as a different ``document_id`` here, which is why that check is part of
        ``get`` and still raises.
        """
        try:
            path = self._entry_path(document_id)
        except RegistryPathError:
            return None
        if not path.is_file():
            return None
        entry = SourceDocumentEntry.from_json(path.read_bytes())
        if entry.document_id != document_id:
            raise RegistryPathError(
                f"registry entry at {path} claims {entry.document_id!r} "
                f"for key {document_id!r}"
            )
        return entry

    def is_registered(self, document_id: str) -> bool:
        """Whether ``document_id`` is vouched for right now.

        True only for an active (non-retired) entry. This is the single
        existence question the whole ticket turns on — the publish-time check
        and the blind-citation report both ask exactly this — so it answers
        from the stored entry, never from a cached set.
        """
        entry = self.get(document_id)
        return entry is not None and entry.is_registered

    def resolve_path(self, document_id: str) -> Path:
        """The file a registered id names, as the parser would open it.

        Two id shapes:
        - ``uploads/<name>`` → store-relative: the blob lives under
          ``root/source_documents/`` because the uploader writes it there, not
          in the repo.
        - anything else → repo-relative: the same expression
          ``PdfText.__init__`` evaluates (``scripts/seed_canonical_records.py:251``),
          i.e. ``repo_root / document_id``.
        Neither is a re-derivation from the stored ``path`` — that is the point:
        criterion 5 is "the registry does not change what the id means", and a
        test that compared the stored ``path`` against itself could not fail.
        """
        _validate_document_id(document_id)
        if document_id.startswith(UPLOAD_ID_PREFIX):
            base = (self._root / SOURCE_DOCUMENTS_DIRNAME).resolve()
            candidate = (base / document_id[len(UPLOAD_ID_PREFIX):]).resolve()
            if not candidate.is_relative_to(base):
                raise RegistryPathError(
                    f"{document_id!r} resolves outside the source-documents store"
                )
            return candidate
        candidate = (self._repo_root / document_id).resolve()
        repo = self._repo_root.resolve()
        if not candidate.is_relative_to(repo):
            raise RegistryPathError(
                f"{document_id!r} resolves outside the repo root"
            )
        return candidate

    def is_servable(self, document_id: str) -> bool:
        """Whether the /content byte-serving policy will serve ``document_id``.

        A distinct, narrower question than "is this registered": a registered
        id still cannot be served if it is not a legitimate source-document
        location. The two rules:

        - ``uploads/<name>`` (store-relative, under ``<root>/source_documents/``)
          -> always servable: the uploader wrote the blob and its extension is
          validated by the parser before registration;
        - anything else (repo-relative) -> servable only when BOTH the resolved
          path sits under an allowed corpus root (:data:`ALLOWED_CORPUS_ROOTS`)
          AND its suffix is a supported ingestion extension.

        Everything else is not servable — e.g. a dotfile such as
        ``rag2025/.env`` that an editor could have registered.
        :meth:`resolve_path` is deliberately unchanged: ingestion still opens
        every legitimately-registered id; only the serving path applies this
        restriction, and a caller sees that by the distinct name.
        """
        from src.normalize_data import SUPPORTED_EXTENSIONS

        if document_id.startswith(UPLOAD_ID_PREFIX):
            return True
        text = (document_id or "").strip()
        if not text or text.startswith(("/", "\\")):
            return False
        if any(part == ".." for part in text.replace("\\", "/").split("/")):
            return False
        candidate = (self._repo_root / text).resolve()
        repo = self._repo_root.resolve()
        if not candidate.is_relative_to(repo):
            return False
        parts = candidate.relative_to(repo).parts
        for root in ALLOWED_CORPUS_ROOTS:
            root_parts = tuple(root.split("/"))
            if parts[: len(root_parts)] == root_parts:
                # Corpus pointer ids carry a JSONL fragment after the
                # dot (``file.jsonl#slug``); the suffix belongs to the
                # file, so check the fragment-stripped name, not the
                # pointer's last component.
                file_part = text.split("#", 1)[0]
                return Path(file_part).suffix.lower() in SUPPORTED_EXTENSIONS
        return False

    def list_entries(self, *, include_retired: bool = False) -> list[SourceDocumentEntry]:
        """Every stored entry, sorted by ``document_id``."""
        entries: list[SourceDocumentEntry] = []
        directory = self.documents_dir
        if not directory.is_dir():
            return entries
        for path in sorted(directory.glob("*.json")):
            match = _ENTRY_FILE_RE.match(path.name)
            if match is None:
                continue
            entry = SourceDocumentEntry.from_json(path.read_bytes())
            if not include_retired and not entry.is_registered:
                continue
            entries.append(entry)
        return sorted(entries, key=lambda entry: entry.document_id)

    # -- internals -----------------------------------------------------------

    def _build(
        self,
        document_id: str,
        *,
        sha256: str | None,
        file_present: bool,
        state: SourceDocumentState,
        revision: int,
        registered_at: datetime,
        registered_by: str,
        updated_at: datetime,
        updated_by: str,
        notes: str | None,
    ) -> SourceDocumentEntry:
        _validate_document_id(document_id)
        return SourceDocumentEntry(
            document_id=document_id,
            kind=classify_document_id(document_id),
            path=str(self.resolve_path(document_id)),
            sha256=sha256,
            file_present=file_present,
            state=state,
            revision=revision,
            registered_at=registered_at,
            registered_by=registered_by,
            updated_at=updated_at,
            updated_by=updated_by,
            notes=notes,
        )

    def _entry_path(self, document_id: str) -> Path:
        _validate_document_id(document_id)
        encoded = quote(document_id, safe="")
        return self.documents_dir / f"{_DOC_DIR_PREFIX}{encoded}{_DOC_DIR_SUFFIX}.json"


def _validate_document_id(document_id: str) -> None:
    """Reject ids the registry must not hold, before they touch the filesystem.

    Three cases, each a different failure mode: blank (no key at all), a path
    that tries to escape ``documents/`` (``..`` segments, absolute paths,
    drive letters), and a non-file kind — a ``*.jsonl#slug`` corpus pointer is
    a real citation but not a registered document (Ghi chú 4's chosen
    semantics: unregistered pointers surface in the blind-citation report).
    """
    text = (document_id or "").strip()
    if not text:
        raise RegistryPathError("document_id is blank")
    if text.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", text):
        raise RegistryPathError(f"{document_id!r} is an absolute path")
    if any(part == ".." for part in text.replace("\\", "/").split("/")):
        raise RegistryPathError(f"{document_id!r} escapes the repo root")
    kind = classify_document_id(text)
    if kind is not SourceDocumentKind.PDF_DOCUMENT:
        raise RegistryPathError(
            f"{document_id!r} is a {kind.value}, not a file-backed document; "
            "the registry holds documents only"
        )


def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` at ``path`` via a ``.tmp`` sibling + ``os.replace``.

    Same idiom as ``canonical_record_store._write_atomic``: a crash between
    the two steps leaves the destination absent rather than half-written. It is
    repeated rather than imported because that helper is private to its module;
    two six-line writers beat a cross-module private dependency.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
