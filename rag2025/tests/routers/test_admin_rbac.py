"""TDD: admin identity + role x action matrix — ticket 13.

Ticket 13 acceptance, asserted at the HTTP seam (the same seam the spec's
"mọi test admin chạy qua HTTP layer" rule names) — never against the internal
dependency, so a passing test means the *route* refused, not that some function
was called:

- every sensitive (role x action) pair has its own negative test: an editor
  cannot publish, a viewer cannot create/edit/submit/approve/publish, a reviewer
  cannot publish, a publisher cannot edit or approve, and an ops credential
  cannot touch the lifecycle at all;
- every role CAN do what the matrix grants it (the positive half — without it,
  a matrix that denies everything would pass the negative tests);
- an unknown credential is refused and an absent one never reaches a handler;
- a reviewer cannot approve a version they authored (four-eyes), while a
  different reviewer can;
- the emitted audit event carries the *caller's* ``actor_id`` and ``role``,
  proven by two different callers producing two different actor ids;
- an action the matrix does not list fails closed — both for a role the table
  forgot and for a lifecycle action with no mapped grant.

Identity is a credential -> (actor_id, role) registry in ``ADMIN_IDENTITIES``,
presented in ``X-Admin-Token``. Nothing in a request names a role, so the tests
below never send one: they present a credential and assert what the server
decided.
"""
from __future__ import annotations

import importlib.util
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
from src.domain import admin_identity as identity_policy  # noqa: E402
from src.domain.admin_identity import AdminAction, Role, role_allows  # noqa: E402
from src.domain.canonical.lifecycle import LifecycleAction  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.routers.admin import router as admin_router  # noqa: E402

# One actor per role, so a test can name the actor it means and an assertion on
# ``actor_id`` is unambiguous.
ACTORS: dict[str, tuple[str, str]] = {
    "admin": ("cb.quan-tri", "admin"),
    "editor": ("cb.bien-tap", "editor"),
    "reviewer": ("cb.phan-bien", "reviewer"),
    "publisher": ("cb.xuat-ban", "publisher"),
    "viewer": ("cb.chi-xem", "viewer"),
}
# A second reviewer: four-eyes must distinguish two people, not two roles.
SECOND_REVIEWER = ("cb.phan-bien-2", "reviewer")

TOKENS = {role: f"tok-{role}" for role in ACTORS}
OPS_TOKEN = "shared-ops-token"

MAIN_PATH = ROOT / "src" / "main.py"

# The origin the CORS preflight test drives. A configured production origin,
# not a localhost default, so the test reads as "a real admin UI can call this".
CORS_ORIGIN = "https://admin.husc.edu.vn"

RECORD_ID = "DIEM_CHUAN:NGANH:cntt:2026"

TEXT_VI = "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 25,5 điểm."


def _headers(role: str) -> dict[str, str]:
    return {"X-Admin-Token": TOKENS[role]}


def _configure(monkeypatch, *, four_eyes: bool | None = None) -> None:
    """Publish the registry the way a deployment does: environment."""
    entries = [
        {"token": TOKENS[role], "actor_id": actor_id, "role": role}
        for role, (actor_id, _) in ACTORS.items()
    ]
    entries.append(
        {"token": "tok-reviewer-2", "actor_id": SECOND_REVIEWER[0], "role": SECOND_REVIEWER[1]}
    )
    monkeypatch.setenv("ADMIN_IDENTITIES", json.dumps(entries))
    monkeypatch.setenv("ADMIN_API_TOKEN", OPS_TOKEN)
    if four_eyes is None:
        monkeypatch.delenv("ADMIN_FOUR_EYES", raising=False)
    else:
        monkeypatch.setenv("ADMIN_FOUR_EYES", "1" if four_eyes else "0")


def _record_body(**overrides) -> dict:
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
        "text": TEXT_VI,
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




@pytest.fixture(scope="module")
def real_app():
    """The app exactly as ``src/main.py`` builds it, CORS middleware included.

    Module scope: importing the real app pulls the service layer in, so the cost
    is paid once. ``ALLOWED_ORIGINS`` is pinned for the module so the preflight
    test drives a known origin — production pins it too (``main.py`` falls back
    to localhost defaults only when it is unset), and the header allow-list this
    test is about is fixed either way.
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("ALLOWED_ORIGINS", CORS_ORIGIN)
    try:
        spec = importlib.util.spec_from_file_location("husc_rbac_real_main", MAIN_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module.app
    finally:
        monkeypatch.undo()


def _create(client: TestClient, role: str = "editor", **overrides):
    return client.post(
        "/admin/records",
        json={"record": _record_body(**overrides)},
        headers=_headers(role),
    )


def _transition(client: TestClient, role: str, action: str, *, expected_version=None, reason=None):
    body: dict = {"action": action}
    if expected_version is not None:
        body["expected_version"] = expected_version
    if reason is not None:
        body["reason"] = reason
    return client.post(
        f"/admin/records/{RECORD_ID}/transitions", json=body, headers=_headers(role)
    )


def _advance_to_approved(client: TestClient) -> None:
    """draft -> review -> approved, each step by the role the matrix grants it."""
    assert _create(client, "editor").status_code == 200
    assert _transition(client, "editor", "submit", expected_version=1).status_code == 200
    assert _transition(client, "reviewer", "approve", expected_version=2).status_code == 200


# ---------------------------------------------------------------------------
# negative: one test per sensitive (role x action) pair
# ---------------------------------------------------------------------------


def test_editor_cannot_publish(client, disk):
    """The headline case: writing a draft does not carry the right to release it."""
    _advance_to_approved(client)
    before = disk.versions(RECORD_ID)

    refused = _transition(client, "editor", "publish", expected_version=3)

    assert refused.status_code == 403, refused.text
    assert "publish" in refused.json()["detail"]
    assert disk.versions(RECORD_ID) == before
    assert disk.get_record(RECORD_ID).latest.status == "approved"


def test_editor_cannot_approve_or_reject(client, disk):
    """An editor submitting its own work does not get to review it."""
    _create(client, "editor")
    assert _transition(client, "editor", "submit", expected_version=1).status_code == 200
    before = disk.versions(RECORD_ID)

    assert _transition(client, "editor", "approve", expected_version=2).status_code == 403
    assert (
        _transition(client, "editor", "reject", expected_version=2, reason="x").status_code
        == 403
    )
    assert disk.versions(RECORD_ID) == before


def test_viewer_cannot_create_edit_submit_or_publish(client, disk):
    """A viewer reads. Every write on the surface is refused, not just publish."""
    refused_create = _create(client, "viewer")
    assert refused_create.status_code == 403, refused_create.text

    # Seed a record as an editor so the later refusals are about the viewer's
    # role and not about the record being missing.
    _create(client, "editor")
    before = disk.versions(RECORD_ID)

    refused_edit = client.post(
        f"/admin/records/{RECORD_ID}/edits",
        json={"expected_version": 1, "text": "viewer edit"},
        headers=_headers("viewer"),
    )
    assert refused_edit.status_code == 403, refused_edit.text
    assert _transition(client, "viewer", "submit", expected_version=1).status_code == 403
    assert _transition(client, "viewer", "approve", expected_version=1).status_code == 403
    assert _transition(client, "viewer", "publish", expected_version=1).status_code == 403

    assert disk.versions(RECORD_ID) == before


def test_reviewer_cannot_publish(client, disk):
    """The spec's reviewer row ends with KHÔNG publish."""
    _advance_to_approved(client)
    before = disk.versions(RECORD_ID)

    refused = _transition(client, "reviewer", "publish", expected_version=3)

    assert refused.status_code == 403, refused.text
    assert disk.versions(RECORD_ID) == before


def test_publisher_cannot_edit_or_approve(client, disk):
    """Releasing approved content does not include changing or approving it."""
    _create(client, "editor")
    before = disk.versions(RECORD_ID)

    refused_edit = client.post(
        f"/admin/records/{RECORD_ID}/edits",
        json={"expected_version": 1, "text": "publisher edit"},
        headers=_headers("publisher"),
    )
    assert refused_edit.status_code == 403, refused_edit.text
    assert _transition(client, "publisher", "submit", expected_version=1).status_code == 403
    assert _transition(client, "publisher", "approve", expected_version=1).status_code == 403

    assert disk.versions(RECORD_ID) == before


def test_the_shared_ops_credential_cannot_publish(client, disk):
    """Decision 7: the shared token is a machine credential, not a bypass.

    Ticket 12 gated publish with this token; ticket 13 narrows it to role
    ``ops``, which the matrix grants nothing on the lifecycle. Publishing with
    it must fail *at the same point* as any other unpermitted role — before a
    version is written.
    """
    _advance_to_approved(client)
    before = disk.versions(RECORD_ID)

    refused = client.post(
        f"/admin/records/{RECORD_ID}/transitions",
        json={"action": "publish", "expected_version": 3},
        headers={"X-Admin-Token": OPS_TOKEN},
    )

    assert refused.status_code == 403, refused.text
    assert disk.versions(RECORD_ID) == before
    assert disk.get_record(RECORD_ID).latest.status == "approved"


def test_graph_update_uses_the_same_identity_model(real_app, monkeypatch):
    """The second, divergent auth path is gone.

    ``POST /v2/graph/update`` lives outside the admin router and used to carry
    its own token check (503 when the token was unconfigured, header optional) —
    fixing only the router would have been fixing half the exposure. It now
    resolves identity through the same registry, so the four outcomes below are
    the *same* four the admin routes give: no credential 422, unknown credential
    403, a role without the grant 403, and the machine credential past auth and
    into the handler (503 here, because this app never ran its lifespan, so no
    pipeline is loaded).

    That last one is the point of the machine credential: role ``ops`` holds
    ``graph.update`` and no records.* action, so this route keeps working while
    the same credential is refused on publish.
    """
    _configure(monkeypatch)
    client = TestClient(real_app)
    body = {"chunks": [{"id": "c1", "text": "điểm chuẩn 2026", "faq_type": "diem_chuan"}]}

    assert client.post("/v2/graph/update", json=body).status_code == 422
    assert (
        client.post("/v2/graph/update", json=body, headers={"X-Admin-Token": "nope"}).status_code
        == 403
    )
    for role in ("viewer", "editor", "reviewer", "publisher"):
        refused = client.post("/v2/graph/update", json=body, headers=_headers(role))
        assert refused.status_code == 403, f"{role} -> {refused.status_code}"

    machine = client.post("/v2/graph/update", json=body, headers={"X-Admin-Token": OPS_TOKEN})
    assert machine.status_code == 503, machine.text
    assert "Pipeline not initialized" in machine.json()["detail"]

    # The strongest form of "one model, not two": with the shared token *unset*,
    # an admin credential still authenticates. The old graph-update check was
    # driven by ADMIN_API_TOKEN alone, so it would have refused here with
    # "Admin token is not configured" regardless of who was asking.
    monkeypatch.delenv("ADMIN_API_TOKEN")
    admin = client.post("/v2/graph/update", json=body, headers=_headers("admin"))
    assert admin.status_code == 503, admin.text
    assert "Pipeline not initialized" in admin.json()["detail"]


def test_an_unregistered_credential_is_refused_everywhere(client, disk):
    """An unknown credential is not a lower role — it is no identity at all."""
    _create(client, "editor")
    before = disk.versions(RECORD_ID)
    unknown = {"X-Admin-Token": "not-a-registered-credential"}

    assert client.get(f"/admin/records/{RECORD_ID}", headers=unknown).status_code == 403
    assert (
        client.post(
            "/admin/records",
            json={"record": _record_body(record_id="DIEM_CHUAN:NGANH:khac:2026")},
            headers=unknown,
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"/admin/records/{RECORD_ID}/transitions",
            json={"action": "publish", "expected_version": 1},
            headers=unknown,
        ).status_code
        == 403
    )
    assert disk.versions(RECORD_ID) == before


def test_a_missing_credential_never_reaches_a_handler(client, disk):
    """No credential -> no request. The header is required, not optional."""
    _create(client, "editor")
    before = disk.versions(RECORD_ID)

    assert client.get(f"/admin/records/{RECORD_ID}").status_code == 422
    assert (
        client.post(
            f"/admin/records/{RECORD_ID}/transitions", json={"action": "publish"}
        ).status_code
        == 422
    )
    assert disk.versions(RECORD_ID) == before


# ---------------------------------------------------------------------------
# four-eyes: an approval by the author is not an approval
# ---------------------------------------------------------------------------


def test_reviewer_cannot_approve_their_own_version(client, disk):
    """A reviewer who authored the version under review must find another one.

    Run as an admin author: the point is the authorship rule itself, so the role
    must be one that *does* hold the approve grant. If four-eyes were merely
    "editors cannot approve", this would pass for the wrong reason.
    """
    assert _create(client, "admin").status_code == 200
    assert _transition(client, "admin", "submit", expected_version=1).status_code == 200
    before = disk.versions(RECORD_ID)

    refused = _transition(client, "admin", "approve", expected_version=2)

    assert refused.status_code == 403, refused.text
    assert "four-eyes" in refused.json()["detail"]
    # The refusal is a no-op: still in review, nothing written.
    assert disk.versions(RECORD_ID) == before
    assert disk.get_record(RECORD_ID).latest.status == "review"


def test_a_different_reviewer_can_approve_the_same_version(client):
    """The other half of four-eyes: a second pair of eyes is accepted."""
    assert _create(client, "editor").status_code == 200
    assert _transition(client, "editor", "submit", expected_version=1).status_code == 200

    approved = client.post(
        f"/admin/records/{RECORD_ID}/transitions",
        json={"action": "approve", "expected_version": 2},
        headers={"X-Admin-Token": "tok-reviewer-2"},
    )

    assert approved.status_code == 200, approved.text
    assert approved.json()["record"]["status"] == "approved"
    # The event names the approver, not the author.
    assert approved.json()["event"]["actor_id"] == SECOND_REVIEWER[0]


def test_authorship_follows_the_edited_version(client, disk):
    """Four-eyes tracks the version under review, not the record's first author.

    A version's author is whoever wrote *that* version. If the rule were read
    once from the record's original author, an approver could edit the content
    and then approve it — the exact bypass the rule exists to close.
    """
    assert _create(client, "editor").status_code == 200
    assert _transition(client, "editor", "submit", expected_version=1).status_code == 200
    # The editor's version is approved by someone else: legal.
    assert _transition(client, "reviewer", "approve", expected_version=2).status_code == 200

    # An admin edits the approved version, becoming the author of version 4.
    edited = client.post(
        f"/admin/records/{RECORD_ID}/edits",
        json={"expected_version": 3, "text": "bản hiệu chỉnh của quản trị"},
        headers=_headers("admin"),
    )
    assert edited.status_code == 200, edited.text
    assert edited.json()["record"]["authored_by"] == ACTORS["admin"][0]

    assert _transition(client, "editor", "submit", expected_version=4).status_code == 200
    before = disk.versions(RECORD_ID)

    # The admin may not approve the version it just wrote…
    refused = _transition(client, "admin", "approve", expected_version=5)
    assert refused.status_code == 403, refused.text
    assert disk.versions(RECORD_ID) == before

    # …while a reviewer, who did not write it, can.
    assert _transition(client, "reviewer", "approve", expected_version=5).status_code == 200


def test_four_eyes_can_be_turned_off_deliberately(monkeypatch, store_root: Path):
    """The control is configurable, and only an explicit off-value disables it.

    Configured at startup, like any other deployment decision — not per request,
    which is what keeps it from being something a caller can turn off.
    """
    _configure(monkeypatch, four_eyes=False)
    application = FastAPI()
    application.include_router(admin_router)
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=store_root)
    )

    with TestClient(application) as client:
        assert _create(client, "admin").status_code == 200
        assert _transition(client, "admin", "submit", expected_version=1).status_code == 200
        approved = _transition(client, "admin", "approve", expected_version=2)

    assert approved.status_code == 200, approved.text
    assert approved.json()["record"]["status"] == "approved"


# ---------------------------------------------------------------------------
# positive: each role can do what the matrix grants it
# ---------------------------------------------------------------------------


def test_viewer_can_read_the_admin_surface(client):
    """Read-only is a real grant, not a synonym for denied."""
    assert _create(client, "editor").status_code == 200

    fetched = client.get(f"/admin/records/{RECORD_ID}", headers=_headers("viewer"))
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["record_id"] == RECORD_ID

    history = client.get(
        f"/admin/records/{RECORD_ID}/versions", headers=_headers("viewer")
    )
    assert history.status_code == 200, history.text
    assert [v["version"] for v in history.json()["versions"]] == [1]


def test_editor_can_create_edit_and_submit(client):
    """CRUD draft + submit review, end to end."""
    assert _create(client, "editor").status_code == 200

    edited = client.post(
        f"/admin/records/{RECORD_ID}/edits",
        json={"expected_version": 1, "text": "bản nháp đã sửa"},
        headers=_headers("editor"),
    )
    assert edited.status_code == 200, edited.text
    assert edited.json()["record"]["status"] == "draft"

    submitted = _transition(client, "editor", "submit", expected_version=2)
    assert submitted.status_code == 200, submitted.text
    assert submitted.json()["record"]["status"] == "review"


def test_reviewer_can_approve_and_reject(client):
    """Both review outcomes, with reject carrying its required reason."""
    _create(client, "editor")
    assert _transition(client, "editor", "submit", expected_version=1).status_code == 200

    rejected = _transition(client, "reviewer", "reject", expected_version=2, reason="thiếu nguồn")
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["record"]["status"] == "draft"

    assert _transition(client, "editor", "submit", expected_version=3).status_code == 200
    approved = _transition(client, "reviewer", "approve", expected_version=4)
    assert approved.status_code == 200, approved.text
    assert approved.json()["record"]["status"] == "approved"


def test_publisher_can_publish_an_approved_record(client, disk):
    """The release step, and the only role that reaches it."""
    _advance_to_approved(client)

    published = _transition(client, "publisher", "publish", expected_version=3)

    assert published.status_code == 200, published.text
    assert published.json()["record"]["status"] == "published"
    assert disk.get_record(RECORD_ID).latest.status == "published"


def test_admin_can_do_every_lifecycle_action(client):
    """``admin`` is the superset row — including the approve its own author needs.

    Every step here is performed by a *different* admin identity from the
    author's, because four-eyes is about people, not roles: the version is
    created and submitted by the editor, then approved and published by admin.
    """
    _create(client, "editor")
    assert _transition(client, "editor", "submit", expected_version=1).status_code == 200
    assert _transition(client, "admin", "approve", expected_version=2).status_code == 200
    assert _transition(client, "admin", "publish", expected_version=3).status_code == 200

    edited = client.post(
        f"/admin/records/{RECORD_ID}/edits",
        json={"expected_version": 4, "text": "admin edit"},
        headers=_headers("admin"),
    )
    assert edited.status_code == 200, edited.text
    assert client.get(f"/admin/records/{RECORD_ID}", headers=_headers("admin")).status_code == 200


# ---------------------------------------------------------------------------
# identity enters the audit event
# ---------------------------------------------------------------------------


def test_the_event_names_the_real_caller_not_a_constant(client):
    """Two callers, two events, two different actor ids.

    A constant ``actor_id`` would satisfy "the field is populated"; it would not
    satisfy "every action knows who did it".
    """
    created = _create(client, "editor")
    assert created.status_code == 200
    create_event = created.json()["event"]
    assert create_event["actor_id"] == ACTORS["editor"][0]
    assert create_event["role"] == "editor"

    assert _transition(client, "editor", "submit", expected_version=1).status_code == 200
    approved = _transition(client, "reviewer", "approve", expected_version=2)
    assert approved.status_code == 200
    approve_event = approved.json()["event"]
    assert approve_event["actor_id"] == ACTORS["reviewer"][0]
    assert approve_event["role"] == "reviewer"

    assert approve_event["actor_id"] != create_event["actor_id"]
    # The event still describes the record, not the caller.
    assert approve_event["entity_id"] == RECORD_ID
    assert approve_event["after_status"] == "approved"


def test_an_edit_event_names_its_editor(client):
    """Edits are audited too, and carry the editing actor's identity."""
    _create(client, "editor")

    edited = client.post(
        f"/admin/records/{RECORD_ID}/edits",
        json={"expected_version": 1, "text": "bản sửa"},
        headers=_headers("admin"),
    )

    assert edited.status_code == 200, edited.text
    event = edited.json()["event"]
    assert event["action"] == "edit"
    assert event["actor_id"] == ACTORS["admin"][0]
    assert event["role"] == "admin"


def test_the_stored_version_records_its_author(client):
    """Four-eyes needs authorship on the version, so it must be readable there.

    The version carries ``authored_by`` stamped from the authenticated caller —
    the edit is a new claim of authorship, and the body cannot set the field.
    """
    claimed = _record_body(authored_by="someone.else")
    created = client.post(
        "/admin/records", json={"record": claimed}, headers=_headers("editor")
    )
    assert created.status_code == 200, created.text
    assert created.json()["record"]["authored_by"] == ACTORS["editor"][0]

    edited = client.post(
        f"/admin/records/{RECORD_ID}/edits",
        json={"expected_version": 1, "text": "bản sửa"},
        headers=_headers("admin"),
    )
    assert edited.status_code == 200, edited.text
    assert edited.json()["record"]["authored_by"] == ACTORS["admin"][0]


# ---------------------------------------------------------------------------
# fail closed: an action the matrix does not grant is denied
# ---------------------------------------------------------------------------


def test_a_role_absent_from_the_matrix_grants_nothing(monkeypatch, store_root: Path):
    """The matrix is the policy: removing a grant revokes it, with no code change.

    Publishing with a publisher credential succeeds only because the table lists
    ``records.publish`` for ``publisher``. Drop that one entry and the same
    request must be refused — which is what proves the routes consult the table
    rather than a hardcoded per-endpoint role check.
    """
    _configure(monkeypatch)
    stripped = {
        role: frozenset(action for action in granted if action is not AdminAction.RECORDS_PUBLISH)
        for role, granted in identity_policy.ROLE_MATRIX.items()
    }
    monkeypatch.setattr(identity_policy, "ROLE_MATRIX", stripped)

    application = FastAPI()
    application.include_router(admin_router)
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=store_root)
    )

    with TestClient(application) as client:
        _advance_to_approved(client)
        refused = _transition(client, "publisher", "publish", expected_version=3)
        # ...and the record is still not published.
        assert client.get(f"/admin/records/{RECORD_ID}", headers=_headers("viewer")).json()[
            "status"
        ] == "approved"

    assert refused.status_code == 403, refused.text


def test_an_unmapped_lifecycle_action_denies_for_every_role(monkeypatch, store_root: Path):
    """An action nobody has been granted is not one the transition route may run.

    ``approve`` is granted to reviewer and admin today; a lifecycle action added
    by a later ticket would arrive with no entry in the transition->grant map.
    The route must refuse that rather than fall back to a default, so the
    fallback is asserted to be "deny" for every role.
    """
    _configure(monkeypatch)
    monkeypatch.setattr(
        identity_policy,
        "_TRANSITION_ACTIONS",
        {
            action: grant
            for action, grant in identity_policy._TRANSITION_ACTIONS.items()
            if action is not LifecycleAction.APPROVE
        },
    )

    application = FastAPI()
    application.include_router(admin_router)
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=store_root)
    )

    with TestClient(application) as client:
        _create(client, "editor")
        assert _transition(client, "editor", "submit", expected_version=1).status_code == 200

        responses = {
            role: _transition(client, role, "approve", expected_version=2)
            for role in ("admin", "reviewer", "editor")
        }

    assert {role: response.status_code for role, response in responses.items()} == {
        "admin": 403,
        "reviewer": 403,
        "editor": 403,
    }


def test_the_credential_header_survives_cors_preflight(real_app):
    """The browser must be able to send the credential header at all.

    ``TestClient`` does not enforce CORS, so every other test here would pass
    while a real browser silently dropped the header and locked the admin UI out
    of its own backend. The app's CORS middleware allows a fixed header list, so
    this asserts the credential's header is *on* that list by driving the
    preflight a browser would send — the request path alone does not prove it.

    No credential is configured here on purpose: a preflight that reached the
    route would be refused for a missing header (422), not answered by CORS.
    """
    client = TestClient(real_app)

    preflight = client.options(
        "/admin/records",
        headers={
            "Origin": CORS_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-admin-token,content-type",
        },
    )

    assert preflight.status_code in (200, 204), preflight.text
    allowed = {
        header.strip().lower()
        for header in preflight.headers.get("access-control-allow-headers", "").split(",")
    }
    assert "x-admin-token" in allowed, preflight.headers
    assert "content-type" in allowed, preflight.headers


def test_the_matrix_denies_unknown_roles_and_actions():
    """The table's own contract, asserted directly: both axes fail closed.

    A route-level test cannot cover "a role that does not exist yet", so the
    vocabulary is checked where it lives.
    """
    assert role_allows("not-a-role", AdminAction.RECORDS_READ) is False
    assert role_allows(Role.ADMIN, "records.something-new") is False
    assert role_allows(Role.OPS, AdminAction.RECORDS_PUBLISH) is False
    assert role_allows(Role.VIEWER, AdminAction.RECORDS_READ) is True
