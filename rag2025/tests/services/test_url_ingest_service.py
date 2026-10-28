"""URL ingest service — ticket T21.

The three acceptance criteria of the ticket are three tests here:

1. a pasted notification URL drafts a ``THONG_BAO`` carrying ``source_url``;
2. the same URL with the same bytes is *skipped* — no new version, nothing
   written;
3. changed bytes append a new version, back at ``draft``.

No network: the fetch, the extraction and the URL guard are all injected, so
the test exercises the service's own logic rather than ``requests``.
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.audit import AuditAction  # noqa: E402
from src.infrastructure.audit_ledger import AuditLedger  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.services.admin_lifecycle import AdminLifecycleService  # noqa: E402
from src.services.url_ingest_service import (  # noqa: E402
    RECORD_ID_PREFIX,
    UrlIngestEmpty,
    UrlIngestService,
    document_id_for_url,
    record_id_for_url,
)

from src.services.web_fetch import UnsafeUrlError, validate_public_url  # noqa: E402

URL = "https://tuyensinh.husc.edu.vn/thongbao.php?id=63"
PAGE_V1 = b"<html><body><main>Thong bao tuyen sinh 2026: hoc phi 12000000</main></body></html>"
PAGE_V2 = b"<html><body><main>Thong bao tuyen sinh 2026: hoc phi 13500000</main></body></html>"


class _Fetcher:
    """A fetch double: hands out queued pages, counts calls."""

    def __init__(self, *pages: bytes):
        self._pages = list(pages)
        self.calls: list[str] = []

    def queue(self, *pages: bytes) -> "_Fetcher":
        self._pages = list(pages)
        return self

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        return self._pages.pop(0) if len(self._pages) > 1 else self._pages[0]


def _extract(body: bytes) -> str:
    """The real extractor is tested in ``test_web_fetch``; here it is a stub."""
    return body.decode("utf-8").split("<main>")[-1].split("</main>")[0]


@pytest.fixture
def lifecycle(tmp_path: Path) -> AdminLifecycleService:
    store = CanonicalRecordStore(tmp_path / "store")
    ledger = AuditLedger(tmp_path / "store" / "audit" / "ledger.jsonl")
    return AdminLifecycleService(store, ledger=ledger, clock=lambda: datetime(2026, 10, 7, tzinfo=timezone.utc))


@pytest.fixture
def fetcher() -> _Fetcher:
    return _Fetcher(PAGE_V1)


@pytest.fixture
def service(lifecycle: AdminLifecycleService, fetcher: _Fetcher) -> UrlIngestService:
    return UrlIngestService(
        lifecycle,
        fetch=fetcher,
        extract=_extract,
        validate=lambda url: url,  # the guard has its own tests
        clock=lambda: datetime(2026, 10, 7, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# criterion 1: URL → draft with source_url
# ---------------------------------------------------------------------------


def test_first_fetch_creates_a_draft_with_source_url(
    service: UrlIngestService, lifecycle: AdminLifecycleService
) -> None:
    result = service.ingest(URL, actor_id="cb.quan-tri", role="admin")

    assert result.status == "created"
    assert result.version == 1
    assert result.record_id == record_id_for_url(URL)
    assert result.sha256 == hashlib.sha256(PAGE_V1).hexdigest()

    stored = lifecycle.get(result.record_id)
    assert stored.status.value == "draft"
    assert stored.record_type.value == "THONG_BAO"
    assert stored.source_url == URL
    assert stored.authority_level.value == "husc_official"
    assert "hoc phi 12000000" in stored.text
    assert stored.authored_by == "cb.quan-tri"
    # the digest is the idempotency state, stored on the version it describes
    assert (stored.model_extra or {}).get("source_sha256") == result.sha256
    assert stored.source_document_id is None  # no registry wired: unpublishable


def test_first_fetch_is_audited(service: UrlIngestService, lifecycle: AdminLifecycleService) -> None:
    result = service.ingest(URL, actor_id="cb.quan-tri", role="admin")
    entries = lifecycle.ledger.entries()
    assert [entry.event.action for entry in entries] == [AuditAction.CREATE]
    assert entries[0].event.target == result.record_id
    assert entries[0].event.actor_id == "cb.quan-tri"


# ---------------------------------------------------------------------------
# criterion 2: same URL, same bytes → skipped, nothing written
# ---------------------------------------------------------------------------


def test_unchanged_page_is_skipped(
    service: UrlIngestService, lifecycle: AdminLifecycleService, fetcher: _Fetcher
) -> None:
    first = service.ingest(URL, actor_id="cb.quan-tri", role="admin")
    before = lifecycle.store.versions(first.record_id)
    entries_before = len(lifecycle.ledger.entries())

    second = service.ingest(URL, actor_id="cb.quan-tri", role="admin")

    assert second.status == "unchanged"
    assert second.version == 1
    assert second.record_id == first.record_id
    assert second.sha256 == first.sha256
    # nothing new on disk and nothing new in the ledger
    assert lifecycle.store.versions(first.record_id) == before
    assert len(lifecycle.ledger.entries()) == entries_before
    assert fetcher.calls == [URL, URL]  # it did re-fetch; it just did not write


def test_unchanged_page_reports_the_stored_text_length(
    service: UrlIngestService, lifecycle: AdminLifecycleService
) -> None:
    service.ingest(URL, actor_id="cb.quan-tri", role="admin")
    second = service.ingest(URL, actor_id="cb.quan-tri", role="admin")
    assert second.chars == len(lifecycle.get(second.record_id).text)


# ---------------------------------------------------------------------------
# criterion 3: changed bytes → a new version
# ---------------------------------------------------------------------------


def test_changed_page_appends_a_version(
    lifecycle: AdminLifecycleService, fetcher: _Fetcher
) -> None:
    service = UrlIngestService(
        lifecycle, fetch=fetcher, extract=_extract, validate=lambda url: url
    )
    fetcher.queue(PAGE_V1, PAGE_V2)

    first = service.ingest(URL, actor_id="cb.quan-tri", role="admin")
    second = service.ingest(URL, actor_id="cb.quan-tri", role="admin")

    assert (first.status, second.status) == ("created", "updated")
    assert second.record_id == first.record_id
    assert second.version == 2
    assert second.sha256 == hashlib.sha256(PAGE_V2).hexdigest()
    assert lifecycle.store.versions(first.record_id) == (1, 2)

    latest = lifecycle.get(first.record_id)
    assert latest.status.value == "draft"  # a new version needs review again
    assert latest.source_url == URL
    assert "13500000" in latest.text
    assert (latest.model_extra or {}).get("source_sha256") == second.sha256


def test_changed_page_is_audited_as_an_edit(
    lifecycle: AdminLifecycleService, fetcher: _Fetcher
) -> None:
    fetcher.queue(PAGE_V1, PAGE_V2)
    service = UrlIngestService(
        lifecycle, fetch=fetcher, extract=_extract, validate=lambda url: url
    )
    service.ingest(URL, actor_id="cb.quan-tri", role="admin")
    service.ingest(URL, actor_id="cb.quan-tri", role="admin")
    assert [entry.event.action for entry in lifecycle.ledger.entries()] == [
        AuditAction.CREATE,
        AuditAction.EDIT,
    ]


def test_distinct_urls_are_distinct_records(
    service: UrlIngestService, lifecycle: AdminLifecycleService, fetcher: _Fetcher
) -> None:
    other = URL.replace("id=63", "id=64")
    fetcher.queue(PAGE_V1, PAGE_V1)
    first = service.ingest(URL, actor_id="cb.quan-tri", role="admin")
    second = service.ingest(other, actor_id="cb.quan-tri", role="admin")
    assert first.record_id != second.record_id
    assert set(lifecycle.store.record_ids()) == {first.record_id, second.record_id}


# ---------------------------------------------------------------------------
# refusals: nothing is written when the page cannot become a record
# ---------------------------------------------------------------------------


def test_textless_page_is_refused_and_writes_nothing(
    lifecycle: AdminLifecycleService, fetcher: _Fetcher
) -> None:
    fetcher.queue(b"<html><body><nav>a</nav></body></html>")
    service = UrlIngestService(
        lifecycle, fetch=fetcher, extract=lambda _body: "", validate=lambda url: url
    )
    with pytest.raises(UrlIngestEmpty):
        service.ingest(URL, actor_id="cb.quan-tri", role="admin")
    assert lifecycle.store.record_ids() == ()
    assert lifecycle.ledger.entries() == ()


def test_unsafe_url_is_refused_before_fetching(lifecycle: AdminLifecycleService) -> None:
    """The default guard is on: a private host never reaches the fetch."""
    fetcher = _Fetcher(PAGE_V1)
    service = UrlIngestService(lifecycle, fetch=fetcher, extract=_extract)
    with pytest.raises(UnsafeUrlError):
        service.ingest("http://127.0.0.1/thongbao", actor_id="cb.quan-tri", role="admin")
    assert fetcher.calls == []
    assert lifecycle.store.record_ids() == ()


def test_validate_returns_the_url_the_record_is_keyed_by(lifecycle: AdminLifecycleService) -> None:
    """The guard's return value is what is fetched and what names the record."""
    fetcher = _Fetcher(PAGE_V1)
    service = UrlIngestService(
        lifecycle,
        fetch=fetcher,
        extract=_extract,
        validate=lambda _url: validate_public_url("http://93.184.216.34/thongbao.php?id=63"),
    )
    result = service.ingest("https://paste-me.example/x", actor_id="cb.quan-tri", role="admin")
    assert result.source_url == "http://93.184.216.34/thongbao.php?id=63"
    assert fetcher.calls == ["http://93.184.216.34/thongbao.php?id=63"]
    assert result.record_id == record_id_for_url("http://93.184.216.34/thongbao.php?id=63")


# ---------------------------------------------------------------------------
# record ids
# ---------------------------------------------------------------------------


def test_record_id_is_stable_and_namespaced() -> None:
    assert record_id_for_url(URL) == record_id_for_url(URL)
    assert record_id_for_url(URL).startswith(RECORD_ID_PREFIX)
    assert record_id_for_url(URL) != record_id_for_url(URL + "&x=1")
    assert "/" not in record_id_for_url(URL)  # path-safe, whatever the URL holds


def test_record_id_survives_a_restart(tmp_path: Path) -> None:
    """The id is derived from the URL, so a new process finds the same record."""
    store = CanonicalRecordStore(tmp_path / "store")
    ledger = AuditLedger(tmp_path / "store" / "audit" / "ledger.jsonl")
    first = UrlIngestService(
        AdminLifecycleService(store, ledger=ledger),
        fetch=_Fetcher(PAGE_V1),
        extract=_extract,
        validate=lambda url: url,
    )
    created = first.ingest(URL, actor_id="cb.quan-tri", role="admin")

    # a second service over the same store, as after a restart
    second = UrlIngestService(
        AdminLifecycleService(CanonicalRecordStore(tmp_path / "store"), ledger=ledger),
        fetch=_Fetcher(PAGE_V1),
        extract=_extract,
        validate=lambda url: url,
    )
    again = second.ingest(URL, actor_id="cb.quan-tri", role="admin")
    assert again.status == "unchanged"


def test_fetched_page_is_registerable_and_publishable(tmp_path: Path) -> None:
    """Criterion 4: a URL draft carries a ``source_document_id`` the
    registry vouches for, so it can pass publication validation.

    The lifecycle service is wired with the source-document registry
    (as production wiring does); the fetched page is written under a
    deterministic id, registered, and named on the record.
    """
    from src.infrastructure.source_document_registry import SourceDocumentRegistry
    from src.services.review_preview import validate_for_publication

    root = tmp_path / "store"
    store = CanonicalRecordStore(root)
    ledger = AuditLedger(root / "audit" / "ledger.jsonl")
    registry = SourceDocumentRegistry(root, root)
    lifecycle = AdminLifecycleService(
        store, ledger=ledger, registry=registry,
        clock=lambda: datetime(2026, 10, 7, tzinfo=timezone.utc),
    )
    service = UrlIngestService(
        lifecycle, fetch=_Fetcher(PAGE_V1), extract=_extract,
        validate=lambda url: url,
        clock=lambda: datetime(2026, 10, 7, tzinfo=timezone.utc),
    )
    result = service.ingest(URL, actor_id="cb.quan-tri", role="admin")

    stored = lifecycle.get(result.record_id)
    assert stored.source_document_id is not None
    assert stored.source_document_id == document_id_for_url(URL)
    assert registry.is_registered(stored.source_document_id)
    # ... and the /content serving policy actually serves it — a registered
    # id that is not servable would 404 the citation chip on the draft.
    assert registry.is_servable(stored.source_document_id)
    entry = registry.get(stored.source_document_id)
    assert entry is not None and entry.sha256 == result.sha256
    # the page bytes landed where /content opens them (the store-owned
    # ``source_documents/`` dir, not the registry's entry-JSON ``documents/``)
    assert registry.resolve_path(stored.source_document_id).read_bytes() == PAGE_V1
    assert validate_for_publication(stored) == ()


def test_refetch_updates_the_registered_document(tmp_path: Path) -> None:
    """A re-fetch with different bytes rewrites the same document id
    and updates its entry — a retry cannot mint a second document."""
    from src.infrastructure.source_document_registry import SourceDocumentRegistry

    root = tmp_path / "store"
    store = CanonicalRecordStore(root)
    ledger = AuditLedger(root / "audit" / "ledger.jsonl")
    registry = SourceDocumentRegistry(root, root)
    lifecycle = AdminLifecycleService(store, ledger=ledger, registry=registry)
    fetcher = _Fetcher()
    fetcher.queue(PAGE_V1, PAGE_V2)
    service = UrlIngestService(lifecycle, fetch=fetcher, extract=_extract,
                               validate=lambda url: url)

    first = service.ingest(URL, actor_id="cb.quan-tri", role="admin")
    second = service.ingest(URL, actor_id="cb.quan-tri", role="admin")

    assert second.status == "updated"
    latest = lifecycle.get(first.record_id)
    assert latest.source_document_id == document_id_for_url(URL)
    entry = registry.get(document_id_for_url(URL))
    assert entry is not None and entry.sha256 == second.sha256
    assert registry.resolve_path(document_id_for_url(URL)).read_bytes() == PAGE_V2
    # idempotent: same id across versions, not a second document
    assert first.sha256 == hashlib.sha256(PAGE_V1).hexdigest()
    assert second.sha256 == hashlib.sha256(PAGE_V2).hexdigest()
