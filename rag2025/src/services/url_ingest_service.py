"""URL → draft record — ticket T21.

The ticket's rule is one sentence: *the same URL with the same bytes is skipped*.
The rule is implemented the way the repo already implements idempotency —
compare SHA-256, write only when it differs — and the comparison state is the
store the drafts already live in, so there is no second index to keep in sync:

- ``record_id`` is ``url:<sha256(url)[:16]>`` — the URL *is* the identity, the
  way a content hash is the identity of an upload. Distinct URLs never collide;
  the same URL always lands on the same record.
- the fetched bytes' digest is stored on the record as ``source_sha256`` (a
  ``CanonicalRecord`` extra, preserved verbatim by the store). The next fetch
  compares against the latest version's value: equal → ``unchanged``, nothing
  written; different → a new version of the same record, back at ``draft``.

So the answer to "where is the URL→hash map?" is: in the record store, on the
record the URL names. A restart, a reindex, or a second worker reads it from
there without any migration.

Writes go through :class:`AdminLifecycleService` (create / edit) rather than the
raw store, so authorship is stamped from the authenticated actor, the text is
normalised to NFC, and the write reaches the audit ledger. Ticket T04's
``_write_upload_draft`` writes the raw store directly because it runs *inside*
the service that already emits the registry's audit event; this service has no
such companion write, so the lifecycle service is what keeps its drafts
auditable. Reads use the store directly — reading is not an auditable act.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Literal

from src.domain.canonical.record import (
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.infrastructure.source_document_registry import (
    SOURCE_DOCUMENTS_DIRNAME,
    UPLOAD_ID_PREFIX,
)
from src.services.admin_lifecycle import AdminLifecycleService, RecordNotFound
from src.services.web_fetch import (
    extract_html_text,
    fetch_html,
    validate_public_url,
)

__all__ = [
    "RECORD_ID_PREFIX",
    "UrlIngestEmpty",
    "UrlIngestError",
    "UrlIngestResult",
    "UrlIngestService",
    "record_id_for_url",
]

#: Prefix of a URL draft's ``record_id``. Distinct from ``upload:`` so an
#: operator reading the store can tell a pasted page from an uploaded file.
RECORD_ID_PREFIX = "url:"


class UrlIngestError(Exception):
    """Base for the service's refusals; the router maps these to 4xx."""


class UrlIngestEmpty(UrlIngestError):
    """The page fetched, but carries no readable text — no empty draft."""

    def __init__(self, url: str) -> None:
        super().__init__(f"{url!r} carries no readable text; nothing to draft")
        self.url = url


@dataclass(frozen=True)
class UrlIngestResult:
    """What one ingest did, and what it produced.

    ``status`` is the whole idempotency answer in one value:

    - ``created`` — no record existed; version 1 was written;
    - ``updated`` — the bytes differ from the stored digest; a new version;
    - ``unchanged`` — the bytes are identical; *nothing was written*.
    """

    status: Literal["created", "updated", "unchanged"]
    record_id: str
    source_url: str
    sha256: str
    version: int
    chars: int


def record_id_for_url(url: str) -> str:
    """The stable record id of a URL: ``url:<sha256(url)[:16]>``.

    Hashed rather than embedded so the id is path-safe whatever the URL
    contains — the store turns the id into a directory name.
    """
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return f"{RECORD_ID_PREFIX}{digest[:16]}"


def document_id_for_url(url: str) -> str:
    """The registry document id a fetched page is written under.

    Deterministic from the URL (the same hash ``record_id_for_url`` uses),
    so a re-fetch of the same page reuses the id instead of minting a second
    document.

    The id is a store-owned ``uploads/`` location (``UPLOAD_ID_PREFIX``) — the
    one shape the ``/content`` serving policy opens unconditionally
    (``SourceDocumentRegistry.is_servable``). A bare ``url-page-….html`` id
    under ``documents/`` would be registered but *unservable*, so the citation
    chip on a published URL draft would 404; ``uploads/`` ids resolve under
    ``<root>/source_documents/`` (``resolve_path``) instead. The ``.txt``
    suffix makes ``/content`` stream the raw page as ``text/plain`` — the
    fetched bytes are an untrusted third-party page, so serving them as
    ``text/html`` would be a stored-XSS vector, and ``.txt`` also keeps the id
    a kind the registry's ``classify_document_id`` accepts.
    """
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return f"{UPLOAD_ID_PREFIX}url-page-{digest[:16]}.txt"


def _stored_sha256(record: CanonicalRecord) -> str | None:
    """The content digest a stored version carries, if it has one."""
    extra = record.model_extra or {}
    value = extra.get("source_sha256")
    return value if isinstance(value, str) else None


class UrlIngestService:
    """Fetch a pasted URL and draft / refresh the record it names.

    ``fetch``, ``extract`` and ``validate`` are injectable so a test can run the
    whole path with no socket and no DNS; left alone they resolve to the shared
    core in :mod:`src.services.web_fetch` at call time.
    """

    def __init__(
        self,
        lifecycle: AdminLifecycleService,
        *,
        fetch: Callable[[str], bytes] | None = None,
        extract: Callable[[bytes], str] | None = None,
        validate: Callable[[str], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._lifecycle = lifecycle
        self._fetch = fetch
        self._extract = extract
        self._validate = validate
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def lifecycle(self) -> AdminLifecycleService:
        return self._lifecycle

    def ingest(self, url: str, *, actor_id: str, role: str | None = None) -> UrlIngestResult:
        """Fetch ``url`` and draft / refresh / skip its record.

        Raises :class:`src.services.web_fetch.UnsafeUrlError` for a URL this
        process will not fetch, the web-fetch errors for a page that could not
        be read, and :class:`UrlIngestEmpty` for a page with no text. In every
        one of those cases nothing was written.
        """
        validate = self._validate or validate_public_url
        safe_url = validate(url).strip()
        fetch = self._fetch or fetch_html
        extract = self._extract or extract_html_text

        body = fetch(safe_url)
        digest = hashlib.sha256(body).hexdigest()
        record_id = record_id_for_url(safe_url)

        latest = self._latest(record_id)
        if latest is not None and _stored_sha256(latest) == digest:
            return UrlIngestResult(
                status="unchanged",
                record_id=record_id,
                source_url=safe_url,
                sha256=digest,
                version=latest.version,
                chars=len(latest.text),
            )

        text = extract(body).strip()
        if not text:
            raise UrlIngestEmpty(safe_url)

        document_id = self._register_source_document(body, digest, safe_url, actor_id)

        if latest is None:
            created = self._lifecycle.create(
                self._record(record_id, safe_url, text, digest, actor_id, document_id),
                actor_id=actor_id,
                role=role,
            )
            return UrlIngestResult(
                status="created",
                record_id=record_id,
                source_url=safe_url,
                sha256=digest,
                version=created.version,
                chars=len(created.text),
            )

        changes: dict[str, object] = {
            "text": text,
            "source_url": safe_url,
            "source_sha256": digest,
        }
        if document_id is not None:
            changes["source_document_id"] = document_id
        updated = self._lifecycle.edit(
            record_id,
            actor_id=actor_id,
            role=role,
            expected_version=latest.version,
            reason=f"re-fetched {safe_url}",
            **changes,
        )
        return UrlIngestResult(
            status="updated",
            record_id=record_id,
            source_url=safe_url,
            sha256=digest,
            version=updated.version,
            chars=len(updated.text),
        )

    # -- internals -----------------------------------------------------------

    def _latest(self, record_id: str) -> CanonicalRecord | None:
        """The current version of ``record_id``, or ``None`` if it has none."""
        try:
            return self._lifecycle.get(record_id)
        except RecordNotFound:  # "no record yet" is not an error here
            return None

    def _register_source_document(
        self, body: bytes, digest: str, url: str, actor_id: str
    ) -> str | None:
        """Write the fetched page into the source-document registry and return
        the id the record should carry.

        The registry is optional (test-built lifecycle services have none);
        without it there is nowhere to put the bytes, so the record goes back
        to carrying only the URL — which is the pre-fix behaviour, and a
        deployment that wants to *publish* a URL draft must wire the registry.
        Registration is idempotent: a re-fetch of the same page rewrites the
        same file and updates the entry instead of raising, so a retry cannot
        mint a second document.
        """
        registry = self._lifecycle.registry
        if registry is None:
            return None
        document_id = document_id_for_url(url)
        # ``uploads/`` ids resolve under ``<root>/source_documents/``
        # (``resolve_path``), not ``documents/`` (the registry's entry-JSON
        # dir) — write the blob where the /content policy will actually open
        # it, exactly as the HOI_DAP source document does.
        target = (
            registry.root
            / SOURCE_DOCUMENTS_DIRNAME
            / document_id[len(UPLOAD_ID_PREFIX):]
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        now = self._clock()
        if registry.is_registered(document_id):
            registry.update(
                document_id,
                sha256=digest,
                file_present=True,
                actor_id=actor_id,
                notes=f"re-fetched {url}",
                now=now,
            )
        else:
            registry.create(
                document_id,
                sha256=digest,
                file_present=True,
                actor_id=actor_id,
                notes=f"fetched from {url}",
                now=now,
            )
        return document_id

    def _record(
        self,
        record_id: str,
        url: str,
        text: str,
        digest: str,
        actor_id: str,
        document_id: str | None,
    ) -> CanonicalRecord:
        """The draft a first fetch creates: one ``THONG_BAO`` version 1.

        ``source_document_id`` is the registry id the fetched page was written
        under (``None`` when no registry is wired, which leaves the record
        unpublishable by design); ``source_url`` is the provenance the page
        came from.
        """
        now = self._clock()
        return CanonicalRecord(
            record_id=record_id,
            record_type=RecordType.THONG_BAO,
            version=1,
            admission_year=now.year,
            effective_from=date(now.year, 1, 1),
            status=RecordStatus.DRAFT,
            authority_level=AuthorityLevel.HUSC_OFFICIAL,
            source_url=url,
            source_document_id=document_id,
            text=text,
            authored_by=actor_id,
            source_sha256=digest,
        )
