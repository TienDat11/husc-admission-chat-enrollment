"""Admin URL → draft route — ticket T21, acceptance criteria 1–5.

The route is asserted at the HTTP seam over a real admin router, exactly as
ticket T04's upload is: the store on ``app.state`` is what a caller reads back
to see whether a draft was written. No socket is opened — the shared fetch core
is monkeypatched and the URLs name literal public addresses, so no DNS runs
either.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAGSettings  # noqa: E402
from src.app.lifecycle import configure_canonical_record_store  # noqa: E402
from src.domain.audit import AuditAction  # noqa: E402
from src.infrastructure.audit_ledger import AuditLedger  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.routers.admin import router as admin_router  # noqa: E402
from src.services import url_ingest_service as url_ingest_module  # noqa: E402

ACTORS: dict[str, tuple[str, str]] = {
    "admin": ("cb.quan-tri", "admin"),
    "viewer": ("cb.chi-xem", "viewer"),
}
TOKENS = {role: f"tok-{role}" for role in ACTORS}

#: A literal public address: no resolver is consulted, so the SSRF guard passes
#: deterministically and the test never depends on DNS.
URL = "http://93.184.216.34/thongbao.php?id=63"
PAGE_V1 = b"<html><body><main>Thong bao tuyen sinh 2026: hoc phi 12000000</main></body></html>"
PAGE_V2 = b"<html><body><main>Thong bao tuyen sinh 2026: hoc phi 13500000</main></body></html>"


def _headers(role: str) -> dict[str, str]:
    return {"X-Admin-Token": TOKENS[role]}


def _configure_identities(monkeypatch) -> None:
    monkeypatch.setenv(
        "ADMIN_IDENTITIES",
        json.dumps(
            [
                {"token": TOKENS[role], "actor_id": actor, "role": role}
                for role, (actor, _) in ACTORS.items()
            ]
        ),
    )
    monkeypatch.setenv("ADMIN_API_TOKEN", "shared-ops-token")


class _Pages:
    """The fetch double: queued pages, one per call, and a call log."""

    def __init__(self, *pages: bytes):
        self.pages = list(pages)
        self.calls: list[str] = []

    def __call__(self, url: str, **kwargs) -> bytes:
        self.calls.append(url)
        return self.pages.pop(0) if len(self.pages) > 1 else self.pages[0]


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "canonical_store"


@pytest.fixture
def app(store_root: Path, monkeypatch) -> FastAPI:
    _configure_identities(monkeypatch)
    application = FastAPI()
    application.include_router(admin_router)
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=store_root)
    )
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def disk(store_root: Path) -> CanonicalRecordStore:
    """An independent store over the same root: reads what the route wrote."""
    return CanonicalRecordStore(store_root)


@pytest.fixture
def ledger(store_root: Path) -> AuditLedger:
    return AuditLedger(store_root / "audit" / "ledger.jsonl")


@pytest.fixture
def pages(monkeypatch) -> _Pages:
    """Patch the fetch the service resolves at call time."""
    double = _Pages(PAGE_V1)
    monkeypatch.setattr(url_ingest_module, "fetch_html", double)
    return double


def _ingest(client: TestClient, url: str = URL, role: str = "admin"):
    return client.post("/admin/source-documents/from-url", json={"url": url}, headers=_headers(role))


# ---------------------------------------------------------------------------
# acceptance criterion 1: URL → 201 + a THONG_BAO draft carrying source_url
# ---------------------------------------------------------------------------


def test_url_creates_draft_with_source_url(
    client: TestClient, disk: CanonicalRecordStore, ledger: AuditLedger, pages: _Pages
) -> None:
    response = _ingest(client)
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["status"] == "created"
    assert payload["source_url"] == URL
    assert payload["version"] == 1
    assert payload["record_id"].startswith("url:")
    assert payload["chars"] > 0

    draft = disk.get_record(payload["record_id"]).latest
    assert draft.status.value == "draft"
    assert draft.record_type.value == "THONG_BAO"
    assert draft.source_url == URL
    assert "hoc phi 12000000" in draft.text
    assert draft.authored_by == ACTORS["admin"][0]
    assert draft.source_document_id is not None

    # the write reached the ledger, on the action the repo already uses
    assert any(
        entry.event.action is AuditAction.CREATE and entry.event.target == payload["record_id"]
        for entry in ledger.entries()
    )
    assert pages.calls == [URL]


# ---------------------------------------------------------------------------
# acceptance criterion 2: same URL, same content → skipped, no new draft
# ---------------------------------------------------------------------------


def test_unchanged_page_is_skipped(
    client: TestClient, disk: CanonicalRecordStore, ledger: AuditLedger, pages: _Pages
) -> None:
    first = _ingest(client)
    assert first.status_code == 201
    record_id = first.json()["record_id"]
    versions_before = disk.versions(record_id)
    entries_before = len(ledger.entries())

    second = _ingest(client)
    assert second.status_code == 200, second.text
    payload = second.json()
    assert payload["status"] == "unchanged"
    assert payload["record_id"] == record_id
    assert payload["version"] == 1
    assert payload["sha256"] == first.json()["sha256"]

    assert disk.versions(record_id) == versions_before  # nothing written
    assert len(ledger.entries()) == entries_before  # nothing audited


# ---------------------------------------------------------------------------
# acceptance criterion 3: changed content → a new draft version
# ---------------------------------------------------------------------------


def test_changed_page_appends_a_version(
    client: TestClient, disk: CanonicalRecordStore, pages: _Pages
) -> None:
    pages.pages = [PAGE_V1, PAGE_V2]
    first = _ingest(client)
    assert first.status_code == 201

    second = _ingest(client)
    assert second.status_code == 200, second.text
    payload = second.json()
    assert payload["status"] == "updated"
    assert payload["record_id"] == first.json()["record_id"]
    assert payload["version"] == 2

    assert disk.versions(payload["record_id"]) == (1, 2)
    latest = disk.get_record(payload["record_id"]).latest
    assert latest.status.value == "draft"
    assert "13500000" in latest.text
    assert (latest.model_extra or {}).get("source_sha256") == payload["sha256"]


# ---------------------------------------------------------------------------
# acceptance criterion 4: no real network — the SSRF guard and fetch errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/thongbao",
        "http://localhost/thongbao",
        "http://169.254.169.254/latest/meta-data/",
        "ftp://tuyensinh.husc.edu.vn/thongbao",
    ],
)
def test_unsafe_url_is_400_and_never_fetched(
    client: TestClient, disk: CanonicalRecordStore, pages: _Pages, url: str
) -> None:
    response = _ingest(client, url=url)
    assert response.status_code == 400, response.text
    assert pages.calls == []  # refused before any socket
    assert disk.record_ids() == ()


def test_upstream_failure_is_502(
    client: TestClient, pages: _Pages, monkeypatch
) -> None:
    from src.services.web_fetch import WebFetchError

    def boom(url: str, **kwargs):
        raise WebFetchError("HTTP 500 on the page")

    monkeypatch.setattr(url_ingest_module, "fetch_html", boom)
    response = _ingest(client)
    assert response.status_code == 502, response.text


def test_textless_page_is_422_and_writes_nothing(
    client: TestClient, disk: CanonicalRecordStore, pages: _Pages
) -> None:
    pages.pages = [b"<html><body><nav>Trang chu</nav></body></html>"]
    response = _ingest(client)
    assert response.status_code == 422, response.text
    assert disk.record_ids() == ()


def test_body_over_the_cap_is_400(client: TestClient, pages: _Pages, monkeypatch) -> None:
    """The cap itself is enforced in ``web_fetch`` (its own tests); here the
    route's mapping of that refusal is what matters."""
    from src.services.web_fetch import WebFetchTooLarge

    def too_large(url: str, **kwargs):
        raise WebFetchTooLarge(f"{url!r} exceeds the 5242880-byte cap")

    monkeypatch.setattr(url_ingest_module, "fetch_html", too_large)
    response = _ingest(client)
    assert response.status_code == 400, response.text
    assert "cap" in response.json()["detail"]


# ---------------------------------------------------------------------------
# acceptance criterion 5: the route carries the records.create grant
# ---------------------------------------------------------------------------


def test_missing_credential_is_422(client: TestClient, pages: _Pages) -> None:
    response = client.post("/admin/source-documents/from-url", json={"url": URL})
    assert response.status_code == 422, response.text
    assert pages.calls == []


def test_viewer_lacks_records_create(
    client: TestClient, disk: CanonicalRecordStore, pages: _Pages
) -> None:
    response = _ingest(client, role="viewer")
    assert response.status_code == 403, response.text
    assert pages.calls == []
    assert disk.record_ids() == ()


def test_empty_url_is_422(client: TestClient, pages: _Pages) -> None:
    response = client.post(
        "/admin/source-documents/from-url", json={"url": ""}, headers=_headers("admin")
    )
    assert response.status_code == 422, response.text
    assert pages.calls == []
