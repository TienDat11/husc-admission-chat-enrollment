"""Audit ledger HTTP surface — ticket 14.

Two things are asserted here, at the seam that produces them:

1. The audit surface is **read-only**: three GET routes, no mutation route in
   the router's own table (the ticket's "không tồn tại endpoint xóa hoặc sửa
   event" is the absence of a route, not a guard on one), and every route goes
   through ticket 13's ``require_action``.
2. The writes ticket 12 already performs **land in the ledger**, through the
   real admin router, with before/after references that resolve through a
   *separate* store instance over the same root — so a passing assertion means
   the bytes are on disk, not that an in-memory object was handed back.

The app fixture wires the ledger the way the running app will: the store comes
from ``configure_canonical_record_store`` (the real startup wiring) and the
lifecycle service is then built over that same store *with* the ledger, which is
the one line Main adds at startup. The service-level section at the bottom
covers the publish seam that knows its release id — a release id the ticket-12
route cannot carry yet, because that route belongs to ticket 19.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAGSettings  # noqa: E402
from rag2025.src.app.lifecycle import configure_canonical_record_store  # noqa: E402
from src.domain.audit import LEGACY_RELEASE_ID, AuditAction, AuditEvent  # noqa: E402
from src.domain.canonical.lifecycle import LifecycleAction  # noqa: E402
from src.domain.canonical.record import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.infrastructure.audit_ledger import AuditLedger  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.routers.admin import router as admin_router  # noqa: E402
from src.routers.admin_audit import router as audit_router  # noqa: E402
from src.services.admin_lifecycle import AdminLifecycleService  # noqa: E402

ACTORS: dict[str, tuple[str, str]] = {
    "admin": ("cb.quan-tri", "admin"),
    "editor": ("cb.bien-tap", "editor"),
    "reviewer": ("cb.phan-bien", "reviewer"),
    "publisher": ("cb.xuat-ban", "publisher"),
    "viewer": ("cb.chi-xem", "viewer"),
}
TOKENS = {role: f"tok-{role}" for role in ACTORS}
OPS_TOKEN = "shared-ops-token"

RECORD_ID = "DIEM_CHUAN:NGANH:cntt:2026"
FIXED_NOW = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc)


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
    monkeypatch.setenv("ADMIN_API_TOKEN", OPS_TOKEN)


def _record(**overrides) -> dict:
    fields = {
        "record_id": RECORD_ID,
        "record_type": "DIEM_CHUAN",
        "version": 1,
        "admission_year": 2026,
        "effective_from": "2026-06-01",
        "effective_to": "2026-12-31",
        "status": "draft",
        "authority_level": "husc_official",
        "source_document_id": "de-an-tuyen-sinh-2026",
        "source_url": "https://tuyensinh.husc.edu.vn/de-an-2026.pdf",
        "source_page": 12,
        "text": "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 25,5 điểm.",
    }
    fields.update(overrides)
    return fields


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "canonical_store"


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "audit" / "ledger.jsonl"


@pytest.fixture
def ledger(ledger_path: Path) -> AuditLedger:
    return AuditLedger(ledger_path)


@pytest.fixture
def app(store_root: Path, ledger: AuditLedger, monkeypatch) -> FastAPI:
    """The app with the ticket-14 wiring: store, lifecycle service, ledger."""
    _configure_identities(monkeypatch)
    application = FastAPI()
    application.include_router(admin_router)
    application.include_router(audit_router)
    store = configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=store_root)
    )
    application.state.audit_ledger = ledger
    application.state.admin_lifecycle_service = AdminLifecycleService(
        store, ledger=ledger, clock=lambda: FIXED_NOW
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
# the surface is read-only
# ---------------------------------------------------------------------------


def test_the_audit_router_serves_only_read_routes():
    served = {
        (method, route.path)
        for route in audit_router.routes
        for method in (getattr(route, "methods", None) or ())
    }
    assert served == {
        ("GET", "/admin/audit/events"),
        ("GET", "/admin/audit/events/{event_id}"),
        ("GET", "/admin/audit/verify"),
    }
    # Spelled out, because "no endpoint deletes or edits an event" is exactly
    # the property a future edit could quietly add back.
    for method, path in served:
        assert method == "GET", f"{method} {path} is a mutating route"


def test_the_mounted_app_exposes_no_mutating_audit_route(app: FastAPI):
    """Checked on the app's own routing table, not just the router object."""
    openapi_paths = app.openapi()["paths"]
    for path, methods in openapi_paths.items():
        if path.startswith("/admin/audit"):
            assert set(methods) == {"get"}, f"{path} exposes {sorted(methods)}"


# ---------------------------------------------------------------------------
# authorization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method,url", [("get", "/admin/audit/events"), ("get", "/admin/audit/verify")])
def test_a_request_without_a_credential_is_refused(client, method, url):
    assert getattr(client, method)(url).status_code == 422


def test_an_unknown_credential_is_refused(client):
    response = client.get(
        "/admin/audit/events", headers={"X-Admin-Token": "not-a-registered-credential"}
    )
    assert response.status_code == 403


def test_the_shared_ops_token_cannot_read_the_ledger(client):
    """The machine credential holds no records.* action (ticket 13)."""
    response = client.get("/admin/audit/events", headers={"X-Admin-Token": OPS_TOKEN})
    assert response.status_code == 403


def test_a_viewer_can_read_the_ledger(client):
    """The spec's matrix gives viewer "read admin surface"; the ledger is it."""
    response = client.get("/admin/audit/events", headers=_headers("viewer"))
    assert response.status_code == 200
    assert response.json() == {"events": [], "count": 0}


def test_an_unwired_app_refuses_instead_of_reporting_an_empty_history(monkeypatch):
    """503, not an empty list: "nothing ever happened" is a different answer."""
    _configure_identities(monkeypatch)
    unwired = FastAPI()
    unwired.include_router(audit_router)
    with TestClient(unwired) as test_client:
        response = test_client.get("/admin/audit/events", headers=_headers("viewer"))
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------


def _append(ledger: AuditLedger, **overrides) -> AuditEvent:
    fields: dict = {
        "action": AuditAction.EDIT,
        "actor_id": "cb.bien-tap",
        "target": RECORD_ID,
        "timestamp": FIXED_NOW,
        "before_version": f"{RECORD_ID}@1",
        "after_version": f"{RECORD_ID}@2",
    }
    fields.update(overrides)
    event = AuditEvent(**fields)
    ledger.append(event)
    return event


def test_events_are_listed_newest_first_with_their_chain_position(client, ledger):
    _append(ledger, event_id="evt-1")
    _append(ledger, actor_id="cb.phan-bien", event_id="evt-2")

    body = client.get("/admin/audit/events", headers=_headers("viewer")).json()
    assert body["count"] == 2
    assert [item["event"]["event_id"] for item in body["events"]] == ["evt-2", "evt-1"]
    assert [item["seq"] for item in body["events"]] == [2, 1]
    assert body["events"][0]["prev_hash"] == body["events"][1]["entry_hash"]


def test_the_listing_filters_by_target_actor_and_action(client, ledger):
    _append(ledger, event_id="evt-1")
    _append(ledger, actor_id="cb.phan-bien", event_id="evt-2")
    _append(
        ledger,
        action=AuditAction.ROLLBACK,
        actor_id="cb.quan-tri",
        reason="hoàn nguyên vì công bố sai",
        event_id="evt-3",
    )

    def ids(**params):
        body = client.get(
            "/admin/audit/events", params=params, headers=_headers("viewer")
        ).json()
        return [item["event"]["event_id"] for item in body["events"]]

    assert ids() == ["evt-3", "evt-2", "evt-1"]
    assert ids(actor_id="cb.phan-bien") == ["evt-2"]
    assert ids(action="rollback") == ["evt-3"]
    assert ids(target=RECORD_ID) == ["evt-3", "evt-2", "evt-1"]
    assert ids(action="publish") == []


def test_an_unknown_action_filter_is_refused(client):
    response = client.get(
        "/admin/audit/events", params={"action": "delete_everything"}, headers=_headers("viewer")
    )
    assert response.status_code == 422


def test_one_event_is_fetchable_by_id_and_404s_when_unknown(client, ledger):
    _append(ledger, event_id="evt-1")
    _append(ledger, actor_id="cb.phan-bien", event_id="evt-2")

    found = client.get("/admin/audit/events/evt-2", headers=_headers("viewer"))
    assert found.status_code == 200
    assert found.json()["event"]["actor_id"] == "cb.phan-bien"
    assert found.json()["seq"] == 2

    missing = client.get("/admin/audit/events/evt-none", headers=_headers("viewer"))
    assert missing.status_code == 404


def test_verify_reports_an_intact_chain_then_the_tampered_index(client, ledger, ledger_path):
    _append(ledger, event_id="evt-1")
    _append(ledger, actor_id="cb.phan-bien", event_id="evt-2")
    _append(ledger, actor_id="cb.quan-tri", event_id="evt-3")

    intact = client.get("/admin/audit/verify", headers=_headers("viewer"))
    assert intact.status_code == 200
    assert intact.json()["ok"] is True
    assert intact.json()["checked"] == 3

    lines = ledger_path.read_bytes().splitlines()
    tampered = json.loads(lines[1])
    tampered["event"]["actor_id"] = "ke.gian-lan"
    lines[1] = json.dumps(tampered, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ledger_path.write_bytes(b"\n".join(lines) + b"\n")

    broken = client.get("/admin/audit/verify", headers=_headers("viewer"))
    assert broken.status_code == 200
    assert broken.json()["ok"] is False
    assert broken.json()["broken_index"] == 1
    assert broken.json()["checked"] == 1


# ---------------------------------------------------------------------------
# the writes ticket 12 already performs land in the ledger
# ---------------------------------------------------------------------------


def _create(client: TestClient):
    return client.post(
        "/admin/records", json={"record": _record()}, headers=_headers("editor")
    )


def _transition(client: TestClient, action: str, *, expected_version: int, role: str, reason=None):
    body: dict = {"action": action, "expected_version": expected_version}
    if reason is not None:
        body["reason"] = reason
    return client.post(
        f"/admin/records/{RECORD_ID}/transitions", json=body, headers=_headers(role)
    )


def _publish(client: TestClient) -> None:
    assert _create(client).status_code == 200
    assert _transition(client, "submit", expected_version=1, role="editor").status_code == 200
    assert _transition(client, "approve", expected_version=2, role="reviewer").status_code == 200
    assert _transition(client, "publish", expected_version=3, role="publisher").status_code == 200


def test_publishing_through_the_admin_router_records_a_publish_event(client, ledger, disk):
    _publish(client)

    published = ledger.find(action=AuditAction.PUBLISH)
    assert len(published) == 1
    event = published[0].event
    assert event.actor_id == "cb.xuat-ban"
    assert event.role == "publisher"
    assert event.target == RECORD_ID
    assert event.release_id, "a publish event must name its release"

    # Every write of the path is recorded, in order, with resolvable versions.
    assert [entry.event.action for entry in ledger.entries()] == [
        AuditAction.CREATE,
        AuditAction.SUBMIT,
        AuditAction.APPROVE,
        AuditAction.PUBLISH,
    ]
    assert event.before_version == f"{RECORD_ID}@3"
    assert event.after_version == f"{RECORD_ID}@4"
    assert disk.get_version(RECORD_ID, 3).status is RecordStatus.APPROVED
    assert disk.get_version(RECORD_ID, 4).status is RecordStatus.PUBLISHED


def test_a_publish_records_the_legacy_label_until_a_release_is_named(client, ledger):
    """The ticket-12 route carries no release id; the event says so with the
    label the pipeline already uses, never with a blank."""
    _publish(client)
    assert ledger.find(action=AuditAction.PUBLISH)[0].event.release_id == LEGACY_RELEASE_ID


def test_an_edit_that_moves_the_source_document_records_that_movement(client, ledger, disk):
    _create(client)
    edited = client.post(
        f"/admin/records/{RECORD_ID}/edits",
        json={
            "expected_version": 1,
            "source_document_id": "de-an-tuyen-sinh-2026-hieu-dinh",
            "reason": "đối chiếu bản hiệu đính của đề án",
        },
        headers=_headers("editor"),
    )
    assert edited.status_code == 200, edited.text

    changes = ledger.find(action=AuditAction.SOURCE_DOCUMENT_CHANGE)
    assert len(changes) == 1
    event = changes[0].event
    assert event.actor_id == "cb.bien-tap"
    assert event.before_version == f"{RECORD_ID}@1"
    assert event.after_version == f"{RECORD_ID}@2"
    assert event.reason == "đối chiếu bản hiệu đính của đề án"
    assert disk.resolve_provenance(event.after_version).source_document_id == (
        "de-an-tuyen-sinh-2026-hieu-dinh"
    )
    # One write, one instant: the edit and the provenance move are the same
    # change seen from two questions, so they must not be two timestamps apart.
    edit_event = ledger.find(action=AuditAction.EDIT)[0].event
    assert event.timestamp == edit_event.timestamp


def test_an_edit_that_leaves_provenance_alone_records_no_source_change(client, ledger):
    _create(client)
    edited = client.post(
        f"/admin/records/{RECORD_ID}/edits",
        json={"expected_version": 1, "text": "Điểm chuẩn ngành CNTT năm 2026 là 25,0 điểm."},
        headers=_headers("editor"),
    )
    assert edited.status_code == 200, edited.text
    assert ledger.find(action=AuditAction.SOURCE_DOCUMENT_CHANGE) == ()
    assert [entry.event.action for entry in ledger.entries()] == [
        AuditAction.CREATE,
        AuditAction.EDIT,
    ]


def test_a_reject_without_a_reason_is_refused_and_records_nothing(client, ledger):
    _create(client)
    assert _transition(client, "submit", expected_version=1, role="editor").status_code == 200
    before = ledger.path.read_bytes() if ledger.path.exists() else b""

    refused = _transition(client, "reject", expected_version=2, role="reviewer")
    assert refused.status_code == 422
    assert (ledger.path.read_bytes() if ledger.path.exists() else b"") == before
    assert ledger.find(action=AuditAction.REJECT) == ()


def test_a_reject_with_a_reason_records_the_justification(client, ledger):
    _create(client)
    assert _transition(client, "submit", expected_version=1, role="editor").status_code == 200
    rejected = _transition(
        client,
        "reject",
        expected_version=2,
        role="reviewer",
        reason="số liệu chưa đối chiếu được với đề án chính thức",
    )
    assert rejected.status_code == 200, rejected.text

    event = ledger.find(action=AuditAction.REJECT)[0].event
    assert event.reason == "số liệu chưa đối chiếu được với đề án chính thức"
    assert event.before_version == f"{RECORD_ID}@2"
    assert event.after_version == f"{RECORD_ID}@3"


# ---------------------------------------------------------------------------
# the publish seam that knows its release (tickets 16/19/22)
# ---------------------------------------------------------------------------


def test_a_publish_that_names_a_release_records_that_release_id(store_root, ledger):
    """The release id travels from the publish path into the event.

    Driven at the service, because the ticket-12 route cannot carry a release id
    — that parameter belongs to the router ticket 19 owns. The service is the
    same object the route calls.
    """
    store = CanonicalRecordStore(store_root)
    service = AdminLifecycleService(store, ledger=ledger, clock=lambda: FIXED_NOW)
    service.create(_canonical_record(), actor_id="cb.bien-tap", role="editor")
    service.transition(RECORD_ID, LifecycleAction.SUBMIT, actor_id="cb.bien-tap")
    service.transition(RECORD_ID, LifecycleAction.APPROVE, actor_id="cb.phan-bien")
    service.transition(
        RECORD_ID,
        LifecycleAction.PUBLISH,
        actor_id="cb.xuat-ban",
        release_id="R-2026-09-13",
    )

    event = ledger.find(action=AuditAction.PUBLISH)[0].event
    assert event.release_id == "R-2026-09-13"
    for reference in (event.before_version, event.after_version):
        record_id, _, version = reference.rpartition("@")
        assert store.get_version(record_id, int(version)).record_version_key == reference


def _canonical_record() -> CanonicalRecord:
    return CanonicalRecord(
        record_id=RECORD_ID,
        record_type=RecordType.DIEM_CHUAN,
        version=1,
        admission_year=2026,
        effective_from="2026-06-01",
        status=RecordStatus.DRAFT,
        authority_level=AuthorityLevel.HUSC_OFFICIAL,
        source_document_id="de-an-tuyen-sinh-2026",
        text="Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 25,5 điểm.",
    )


# ---------------------------------------------------------------------------
# The startup wiring itself — the `ledger=` line, which nothing above covers
# ---------------------------------------------------------------------------


def test_the_startup_wiring_alone_hands_its_ledger_to_the_lifecycle_service(
    tmp_path: Path, monkeypatch
) -> None:
    """`configure_canonical_record_store` must pass its ledger to the service.

    Every other test in this file builds the service itself, so none of them
    can see whether the real startup wiring connects the ledger. If that one
    line were dropped, publishes would stop being audited and the whole file
    would stay green — the audit trail would be empty in production and
    complete in the tests. This test uses the startup wiring alone, with no
    fixture overwriting either object.
    """
    _configure_identities(monkeypatch)
    application = FastAPI()
    application.include_router(admin_router)
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=tmp_path / "startup")
    )

    ledger = application.state.audit_ledger
    service = application.state.admin_lifecycle_service
    assert ledger.entries() == ()

    service.create(_canonical_record(), actor_id="cb.bien-tap", role="editor")

    entries = ledger.entries()
    assert len(entries) == 1, (
        "the startup wiring must hand its ledger to the lifecycle service; an "
        "empty ledger here means the `ledger=` argument was dropped"
    )
    assert entries[0].event.action is AuditAction.CREATE
    assert ledger.verify().ok


def test_the_startup_wiring_threads_a_release_id_into_the_publish_event(
    tmp_path: Path, monkeypatch
) -> None:
    """A publish through the startup-wired service records the release it names.

    Same seam as above, carried one step further: the release id has to survive
    the trip from the caller, through the service, into the event, or a publish
    cannot be rolled back by release later.
    """
    _configure_identities(monkeypatch)
    application = FastAPI()
    application.include_router(admin_router)
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=tmp_path / "startup")
    )

    ledger = application.state.audit_ledger
    service = application.state.admin_lifecycle_service
    service.create(_canonical_record(), actor_id="cb.bien-tap", role="editor")
    service.transition(RECORD_ID, LifecycleAction.SUBMIT, actor_id="cb.bien-tap")
    service.transition(RECORD_ID, LifecycleAction.APPROVE, actor_id="cb.phan-bien")
    service.transition(
        RECORD_ID,
        LifecycleAction.PUBLISH,
        actor_id="cb.xuat-ban",
        release_id="R-2026-09-13",
    )

    published = ledger.find(action=AuditAction.PUBLISH)
    assert len(published) == 1
    assert published[0].event.release_id == "R-2026-09-13"
    assert ledger.verify().ok


# ---------------------------------------------------------------------------
# the lazily built service — the branch a partially wired app lands on
# ---------------------------------------------------------------------------


def test_a_lazily_built_service_still_audits_the_write_it_serves(
    tmp_path: Path, monkeypatch
) -> None:
    """A store-wired app with no service yet must not write unaudited.

    The lazy fallback in ``_lifecycle_service`` builds the service on first
    request. If it forgets the ledger, every create/edit/publish it serves
    returns 200 with nothing in the audit trail — the exact silent divergence
    the service itself refuses to allow. Dropping ``app.state
    .admin_lifecycle_service`` is what puts a request on that branch.
    """
    _configure_identities(monkeypatch)
    application = FastAPI()
    application.include_router(admin_router)
    application.include_router(audit_router)
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=tmp_path / "lazy")
    )
    del application.state.admin_lifecycle_service
    ledger = application.state.audit_ledger

    with TestClient(application) as test_client:
        created = test_client.post(
            "/admin/records", json={"record": _record()}, headers=_headers("editor")
        )

    assert created.status_code == 200, created.text
    entries = ledger.entries()
    assert len(entries) == 1, (
        "a write served by the lazily built service must be audited; an empty "
        "ledger means the fallback built the service without its ledger"
    )
    assert entries[0].event.action is AuditAction.CREATE
    assert entries[0].event.target == RECORD_ID


def test_a_store_with_no_ledger_refuses_rather_than_writing_unaudited(
    tmp_path: Path, monkeypatch
) -> None:
    """No audit sink is absent configuration, not permission to skip auditing.

    The store and the ledger are wired together, so reaching this state means
    the app was assembled by hand. Refusing keeps the invariant the service
    enforces internally — a version is stored *and* recorded, or neither —
    rather than letting the HTTP layer be the one place that can bypass it.
    """
    _configure_identities(monkeypatch)
    application = FastAPI()
    application.include_router(admin_router)
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=tmp_path / "no-ledger")
    )
    del application.state.admin_lifecycle_service
    del application.state.audit_ledger

    with TestClient(application) as test_client:
        response = test_client.post(
            "/admin/records", json={"record": _record()}, headers=_headers("editor")
        )

    assert response.status_code == 503
    assert "audit ledger is not configured" in response.json()["detail"]
