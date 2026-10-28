"""TDD: the admin CRUD surface — ticket 19.

Ticket 19's acceptance, asserted at the HTTP seam (the spec's "mọi test admin
chạy qua HTTP layer" rule) rather than against internal helpers:

- the list endpoint filters by ngành / năm / trạng thái, and returns each
  record's *latest* version;
- a role without a grant cannot call the action: the endpoint answers 403 and
  the store stays byte-identical (tested per (role, action) pair, not by
  asserting the UI hid a button);
- editing a published record appends a new version and leaves the published
  version's bytes on disk untouched — read back through a second store over the
  same root, so the assertion is about real bytes, not a returned object;
- archiving goes through the lifecycle (``supersede`` -> ``archived``) and never
  rewrites the published version it archives;
- a record that cannot legally be published (no provenance) is refused with a
  4xx that names the problem instead of a 500;
- ``GET /admin/identity`` reports the caller's own role and grants, so the UI
  can hide what the role does not hold.

Identity is the ticket-13 credential registry: nothing in a request names a
role, so these tests present a credential and assert what the server decided.
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
from rag2025.src.app.lifecycle import (  # noqa: E402
    configure_canonical_record_store,
)
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.routers.admin import router as admin_router  # noqa: E402

ACTORS: dict[str, tuple[str, str]] = {
    "admin": ("cb.quan-tri", "admin"),
    "editor": ("cb.bien-tap", "editor"),
    "reviewer": ("cb.phan-bien", "reviewer"),
    "publisher": ("cb.xuat-ban", "publisher"),
    "viewer": ("cb.chi-xem", "viewer"),
}
TOKENS = {role: f"tok-{role}" for role in ACTORS}
OPS_TOKEN = "shared-ops-token"

# One record per (ngành, năm) combination the filter tests need.
REC_A = "CHI_TIEU:nganh:7220104:2026:a"  # Hán Nôm, 2026
REC_B = "CHI_TIEU:nganh:7480201:2026:b"  # Công nghệ thông tin, 2026
REC_C = "HOC_PHI:nganh:7480201:2025:c"  # Công nghệ thông tin, 2025

def _headers(role: str) -> dict[str, str]:
    return {"X-Admin-Token": TOKENS[role]}

def _configure(monkeypatch) -> None:
    monkeypatch.setenv(
        "ADMIN_IDENTITIES",
        json.dumps(
            [
                {"token": TOKENS[role], "actor_id": actor, "role": role}
                for role, (actor, _) in ACTORS.items()
            ]
        ),
    )
    monkeypatch.setenv("ADMIN_API_TOKEN", OPS_TOKEN)

def _record_body(record_id: str, **overrides) -> dict:
    """A draft record as it crosses the HTTP boundary (JSON, not models)."""
    fields = {
        "record_id": record_id,
        "record_type": "CHI_TIEU",
        "version": 1,
        "admission_year": 2026,
        "effective_from": "2026-01-01",
        "effective_to": "2026-12-31",
        "status": "draft",
        "authority_level": "husc_official",
        "source_document_id": "de-an-tuyen-sinh-2026",
        "source_url": "https://tuyensinh.husc.edu.vn/de-an-2026.pdf",
        "source_page": 24,
        "text": "Chỉ tiêu tuyển sinh 2026.",
    }
    fields.update(overrides)
    return fields

@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "canonical_store"

@pytest.fixture
def app(store_root: Path, monkeypatch) -> FastAPI:
    _configure(monkeypatch)
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
    """A second, independent store over the same root — reads real bytes."""
    return CanonicalRecordStore(store_root)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _create(client: TestClient, record_id: str, **overrides):
    return client.post(
        "/admin/records",
        json={"record": _record_body(record_id, **overrides)},
        headers=_headers("editor"),
    )

def _transition(client: TestClient, record_id: str, action: str, **body):
    actor = {
        "submit": "editor",
        "approve": "reviewer",
        "reject": "reviewer",
        "publish": "publisher",
        "supersede": "publisher",
    }[action]
    return client.post(
        f"/admin/records/{record_id}/transitions",
        json={"action": action, **body},
        headers=_headers(actor),
    )

def _publish(client: TestClient, record_id: str, versions: int = 1) -> None:
    """Walk draft -> review -> approved -> published from version ``versions``."""
    assert _transition(client, record_id, "submit", expected_version=versions).status_code == 200
    assert _transition(client, record_id, "approve", expected_version=versions + 1).status_code == 200
    response = _transition(client, record_id, "publish", expected_version=versions + 2)
    assert response.status_code == 200, response.text

def _list(client: TestClient, *, role: str = "viewer", **params) -> list[dict]:
    response = client.get("/admin/records", params=params, headers=_headers(role))
    assert response.status_code == 200, response.text
    return response.json()["records"]

def _ids(records: list[dict]) -> list[str]:
    return [record["record_id"] for record in records]

# ---------------------------------------------------------------------------
# the filterable list
# ---------------------------------------------------------------------------

def test_list_is_mounted_on_the_router():
    """Both ticket-19 routes exist on the router ``src/main.py`` mounts."""
    registered = {
        (method, route.path)
        for route in admin_router.routes
        for method in (getattr(route, "methods", None) or ())
    }
    assert ("GET", "/admin/records") in registered
    assert ("GET", "/admin/identity") in registered

def test_list_filters_by_nganh_year_and_status(client):
    """The three filters the admissions office asked for, each on its own."""
    assert _create(client, REC_A, values={"ma_nganh": "7220104", "ten_nganh": "Hán Nôm"}).status_code == 200
    assert _create(
        client,
        REC_B,
        values={"ma_nganh": "7480201", "ten_nganh": "Công nghệ thông tin"},
    ).status_code == 200
    assert _create(
        client,
        REC_C,
        admission_year=2025,
        values={"ten_nganh": "Công nghệ thông tin"},
    ).status_code == 200

    # No filter: everything, newest admission year first, then record_id.
    assert _ids(_list(client)) == [REC_A, REC_B, REC_C]

    # ngành by name, with diacritics and without ("han nom" -> "Hán Nôm").
    assert _ids(_list(client, nganh="Hán Nôm")) == [REC_A]
    assert _ids(_list(client, nganh="han nom")) == [REC_A]
    # ngành by mã ngành, which matches the id and the payload alike.
    assert _ids(_list(client, nganh="7480201")) == [REC_B, REC_C]
    # ngành is a substring match over the folded name.
    assert _ids(_list(client, nganh="cong nghe")) == [REC_B, REC_C]

    assert _ids(_list(client, year=2025)) == [REC_C]
    assert _ids(_list(client, year=2026)) == [REC_A, REC_B]

    assert _ids(_list(client, status="draft")) == [REC_A, REC_B, REC_C]
    assert _list(client, status="published") == []

    # Filters combine with AND.
    assert _ids(_list(client, year=2026, nganh="cong nghe")) == [REC_B]
    assert _list(client, year=2025, nganh="han nom") == []

    # The filter reads the record's *latest* version: publishing REC_A moves it
    # out of `status=draft` without changing its ngành or year.
    _publish(client, REC_A)
    assert _ids(_list(client, status="published")) == [REC_A]
    assert _ids(_list(client, status="draft")) == [REC_B, REC_C]
    assert _ids(_list(client, nganh="han nom", status="published")) == [REC_A]

def test_list_returns_the_latest_version_after_an_edit(client):
    """A list row is the current version, not version 1 forever."""
    assert _create(client, REC_A, values={"ma_nganh": "7220104"}).status_code == 200
    edited = client.post(
        f"/admin/records/{REC_A}/edits",
        json={"expected_version": 1, "text": "Chỉ tiêu tuyển sinh 2026 (đã rà soát)."},
        headers=_headers("editor"),
    )
    assert edited.status_code == 200, edited.text

    [row] = _list(client)
    assert row["version"] == 2
    assert row["text"] == "Chỉ tiêu tuyển sinh 2026 (đã rà soát)."

def test_list_refuses_a_caller_without_records_read(client):
    """The ops credential holds no records.* action, so the list is closed to it."""
    assert _create(client, REC_A).status_code == 200

    forbidden = client.get("/admin/records", headers={"X-Admin-Token": OPS_TOKEN})
    assert forbidden.status_code == 403
    assert "records.read" in forbidden.json()["detail"]

    # A viewer may read (the positive half — a matrix that denies everything
    # would pass the negative test above).
    assert client.get("/admin/records", headers=_headers("viewer")).status_code == 200

    # No credential at all never reaches the handler.
    assert client.get("/admin/records").status_code == 422

# ---------------------------------------------------------------------------
# RBAC — the endpoint, not the UI
# ---------------------------------------------------------------------------

def test_low_privilege_roles_cannot_call_the_crud_endpoints(client, disk):
    """Every (role, action) pair ticket 19 must not leak, refused at the route."""
    assert _create(client, REC_A).status_code == 200

    attempts = [
        # A viewer may only read.
        ("viewer", "post", "/admin/records", {"record": _record_body(REC_B)}),
        ("viewer", "post", f"/admin/records/{REC_A}/edits", {"expected_version": 1, "text": "x"}),
        ("viewer", "post", f"/admin/records/{REC_A}/transitions", {"action": "submit"}),
        # An editor writes drafts and submits, but never approves or publishes.
        ("editor", "post", f"/admin/records/{REC_A}/transitions", {"action": "approve"}),
        ("editor", "post", f"/admin/records/{REC_A}/transitions", {"action": "publish"}),
        # A reviewer decides, but does not author or publish.
        ("reviewer", "post", "/admin/records", {"record": _record_body(REC_B)}),
        ("reviewer", "post", f"/admin/records/{REC_A}/transitions", {"action": "publish"}),
        # A publisher publishes, but does not edit admissions content.
        ("publisher", "post", f"/admin/records/{REC_A}/edits", {"expected_version": 1, "text": "x"}),
        ("publisher", "post", "/admin/records", {"record": _record_body(REC_B)}),
    ]
    for role, method, url, body in attempts:
        response = getattr(client, method)(url, json=body, headers=_headers(role))
        assert response.status_code == 403, f"{role} {method} {url} -> {response.status_code}"
        assert role in response.json()["detail"]

    # Nothing was written by any of the refused calls.
    assert disk.versions(REC_A) == (1,)
    assert disk.versions(REC_B) == ()
    assert disk.get_version(REC_A, 1).status.value == "draft"

def test_identity_reports_the_callers_own_grants(client):
    """The UI hides what the role does not hold — so it must be told the grants."""
    viewer = client.get("/admin/identity", headers=_headers("viewer"))
    assert viewer.status_code == 200
    assert viewer.json() == {
        "actor_id": "cb.chi-xem",
        "role": "viewer",
        "actions": ["records.read"],
    }

    editor = client.get("/admin/identity", headers=_headers("editor")).json()
    assert editor["actor_id"] == "cb.bien-tap"
    assert "records.create" in editor["actions"]
    assert "records.edit" in editor["actions"]
    assert "records.submit" in editor["actions"]
    assert "records.publish" not in editor["actions"]
    assert "records.approve" not in editor["actions"]

    ops = client.get("/admin/identity", headers={"X-Admin-Token": OPS_TOKEN}).json()
    assert ops["role"] == "ops"
    assert [action for action in ops["actions"] if action.startswith("records.")] == []

    assert client.get("/admin/identity", headers={"X-Admin-Token": "nope"}).status_code == 403
    assert client.get("/admin/identity").status_code == 422

# ---------------------------------------------------------------------------
# editing never overwrites a published version
# ---------------------------------------------------------------------------

def test_edit_appends_a_version_and_leaves_the_published_bytes_untouched(client, disk):
    """The ticket's immutability claim, checked against the bytes on disk."""
    assert _create(client, REC_A, values={"ma_nganh": "7220104", "ten_nganh": "Hán Nôm"}).status_code == 200
    _publish(client, REC_A)

    published_path = disk.version_path(REC_A, 4)
    published_bytes = published_path.read_bytes()
    published = disk.get_version(REC_A, 4)
    assert published.status.value == "published"
    assert published.authored_by == "cb.bien-tap"

    edited = client.post(
        f"/admin/records/{REC_A}/edits",
        json={
            "expected_version": 4,
            "text": "Chỉ tiêu tuyển sinh 2026: 32 (đã rà soát).",
            "values": {"ma_nganh": "7220104", "ten_nganh": "Hán Nôm", "chi_tieu": 32},
        },
        headers=_headers("editor"),
    )
    assert edited.status_code == 200, edited.text
    body = edited.json()
    assert body["record"]["version"] == 5
    assert body["record"]["status"] == "draft"
    assert body["event"]["action"] == "edit"
    assert body["event"]["actor_id"] == "cb.bien-tap"

    # The published version is byte-identical, still published, and still the
    # version the students' index would be built from.
    assert published_path.read_bytes() == published_bytes
    assert disk.get_version(REC_A, 4).status.value == "published"
    assert disk.get_version(REC_A, 4).text == published.text
    # The new draft carries the edited structured payload and no publish stamp.
    draft = disk.get_version(REC_A, 5)
    assert draft.status.value == "draft"
    assert draft.published_at is None
    assert draft.values["chi_tieu"] == 32

def test_a_stale_edit_is_refused_and_changes_nothing(client, disk):
    """Two editors cannot silently lose each other's work."""
    assert _create(client, REC_A).status_code == 200
    first = client.post(
        f"/admin/records/{REC_A}/edits",
        json={"expected_version": 1, "text": "bản 1"},
        headers=_headers("editor"),
    )
    assert first.status_code == 200
    stale = client.post(
        f"/admin/records/{REC_A}/edits",
        json={"expected_version": 1, "text": "bản cũ"},
        headers=_headers("editor"),
    )
    assert stale.status_code == 409
    assert disk.versions(REC_A) == (1, 2)
    assert disk.get_version(REC_A, 2).text == "bản 1"

# ---------------------------------------------------------------------------
# archive goes through the lifecycle
# ---------------------------------------------------------------------------

def test_supersede_archives_and_keeps_the_archived_version_intact(client, disk):
    """Soft delete is the lifecycle's `supersede`, never a hard delete."""
    replacement = "CHI_TIEU:nganh:7220104:2027:d"
    assert _create(client, REC_A).status_code == 200
    _publish(client, REC_A)

    archived_bytes = disk.version_path(REC_A, 4).read_bytes()

    assert _create(client, replacement, admission_year=2027, supersedes=REC_A).status_code == 200
    _publish(client, replacement)

    response = _transition(client, REC_A, "supersede", expected_version=4)
    assert response.status_code == 200, response.text
    assert response.json()["record"]["status"] == "archived"
    assert response.json()["event"]["action"] == "supersede"

    # Archived, not deleted: every version is still stored, and the version
    # that was live is byte-identical.
    assert disk.versions(REC_A) == (1, 2, 3, 4, 5)
    assert disk.version_path(REC_A, 4).read_bytes() == archived_bytes
    assert disk.is_superseded(REC_A) is True
    assert _ids(_list(client, status="archived")) == [REC_A]

def test_supersede_without_a_published_replacement_is_refused(client, disk):
    """Archiving a live fact with nothing to replace it is a revocation, not an edit."""
    assert _create(client, REC_A).status_code == 200
    _publish(client, REC_A)

    response = _transition(client, REC_A, "supersede", expected_version=4)
    assert response.status_code == 409
    assert "no published record names it" in response.json()["detail"]
    assert disk.versions(REC_A) == (1, 2, 3, 4)
    assert disk.get_version(REC_A, 4).status.value == "published"

# ---------------------------------------------------------------------------
# validation reaches the caller as a 4xx that names the problem
# ---------------------------------------------------------------------------

def test_an_invalid_record_is_refused_with_field_level_errors(client, disk):
    """Reversed effective window: refused at the boundary, nothing written."""
    response = client.post(
        "/admin/records",
        json={"record": _record_body(REC_A, effective_from="2026-12-31", effective_to="2026-06-01")},
        headers=_headers("editor"),
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    # A request-validation failure carries the field path, so the UI can render
    # the message at the offending input instead of as a form-level banner.
    assert isinstance(detail, list)
    assert detail[0]["loc"][:2] == ["body", "record"]
    assert "effective_from" in detail[0]["msg"]
    assert disk.record_ids() == ()

def test_publishing_without_provenance_is_a_422_not_a_500(client, disk):
    """The schema's "no fact without provenance" rule, at the HTTP seam."""
    assert _create(client, REC_A, source_document_id=None).status_code == 200
    assert _transition(client, REC_A, "submit", expected_version=1).status_code == 200
    assert _transition(client, REC_A, "approve", expected_version=2).status_code == 200

    response = _transition(client, REC_A, "publish", expected_version=3)
    assert response.status_code == 422, response.text
    assert "source_document_id" in response.json()["detail"]
    # Refused before the write: the record is still `approved`.
    assert disk.versions(REC_A) == (1, 2, 3)
    assert disk.get_version(REC_A, 3).status.value == "approved"
