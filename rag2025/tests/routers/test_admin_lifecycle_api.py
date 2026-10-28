"""TDD: /admin/records lifecycle API — ticket 12.

Ticket 12 acceptance, asserted at the HTTP seam (per the spec's "mọi test admin
chạy qua HTTP layer"):

- the admin router is mounted on the REAL app built by ``src/main.py``, so the
  endpoints are reachable in the running app rather than only when a test
  mounts them by hand;
- a legal transition runs end-to-end (create -> edit -> submit -> approve ->
  publish) and the published version is readable back from disk;
- an illegal transition answers 4xx and leaves stored data untouched;
- a reject without a reason is refused;
- a stale ``expected_version`` is detected instead of silently overwriting;
- a published version is immutable — editing appends a new version;
- every transition emits an event;
- supersede is unreachable on a non-published record, cannot name itself, and
  needs a published replacement.

Ticket 13 replaced the shared-token interlock these endpoints used to carry
with real actor identity: every request now presents a credential from the
``ADMIN_IDENTITIES`` registry, and each transition is performed by an actor
whose role actually holds that grant (``_transition`` picks one per action).
The four-eyes rule stays ON here, so ``approve`` is always a different actor
from the version's author — which is also why the authoring actor below is an
editor rather than the admin used for ad-hoc writes.

Two app fixtures, deliberately: ``real_app`` is the app ``src/main.py`` builds
(the mount proof — a 404 there is a failure), while ``app`` is a minimal
hand-mounted app used for the behavioral cases, so their store root is a
throwaway directory. The behavioral tests read the store back through a
*separate* store instance over the same root, so a passing assertion means the
bytes are on disk — not that an in-memory object was returned to the caller.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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

MAIN_PATH = ROOT / "src" / "main.py"

# Ticket 13: identity is a credential -> (actor_id, role) registry, configured
# per deployment. One actor per role so a test can name the actor it means.
ACTORS: dict[str, tuple[str, str]] = {
    "admin": ("cb.quan-tri", "admin"),
    "editor": ("cb.bien-tap", "editor"),
    "reviewer": ("cb.phan-bien", "reviewer"),
    "publisher": ("cb.xuat-ban", "publisher"),
    "viewer": ("cb.chi-xem", "viewer"),
}
TOKENS = {role: f"tok-{role}" for role in ACTORS}
OPS_TOKEN = "shared-ops-token"


def _headers(role: str) -> dict[str, str]:
    return {"X-Admin-Token": TOKENS[role]}


# The authoring actor. Not the admin: an editor writes drafts, and four-eyes
# (on by default) requires the approver to be someone else.
HEADERS = _headers("editor")


def _configure_identities(monkeypatch) -> None:
    """Publish the actor registry the way a deployment does: environment."""
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


RECORD_ID = "DIEM_CHUAN:NGANH:cntt:2026"
REPLACEMENT_ID = "DIEM_CHUAN:NGANH:cntt:2027"

TEXT_VI = "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 25,5 điểm."


def _record_body(**overrides) -> dict:
    """A minimal draft record as it crosses the HTTP boundary (JSON, not models)."""
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
    _configure_identities(monkeypatch)
    application = FastAPI()
    application.include_router(admin_router)
    # The real startup wiring, over a throwaway root — the same call
    # app startup makes. A test that hand-built the store would not prove it.
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
    """The app exactly as ``src/main.py`` builds it, routers included.

    Loaded once per module: importing the real app pulls the service layer in
    (~10s). Module scope keeps that cost paid once instead of per test.
    """
    spec = importlib.util.spec_from_file_location("husc_real_main", MAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


# Every admin route the running app must serve: the ticket-12 lifecycle
# endpoints plus the two ops endpoints that were previously unreachable too.
ADMIN_ROUTES = (
    ("POST", "/admin/warmup"),
    ("POST", "/admin/reload-table"),
    ("POST", "/admin/records"),
    ("GET", "/admin/records/{record_id}"),
    ("GET", "/admin/records/{record_id}/versions"),
    ("POST", "/admin/records/{record_id}/edits"),
    ("POST", "/admin/records/{record_id}/transitions"),
)


def test_real_app_mounts_every_admin_route(real_app):
    """The router is mounted by the app, not by the test."""
    registered = {
        (method, route.path)
        for route in real_app.routes
        for method in (getattr(route, "methods", None) or ())
    }
    missing = [pair for pair in ADMIN_ROUTES if pair not in registered]
    assert not missing, f"not mounted on the real app: {missing}"


def test_real_app_lifecycle_endpoints_are_reachable_not_404(real_app, monkeypatch):
    """Reachable on the real app: the auth layer answers, so it is not a 404.

    Deliberately *not* used as a context manager: entering ``TestClient`` runs
    the app's lifespan, which loads embedding/reranker models. Mount proof needs
    the router table, not a warm model, so this drives the real app's routing
    and auth without paying for warmup.
    """
    _configure_identities(monkeypatch)
    client = TestClient(real_app)
    unknown = {"X-Admin-Token": "not-a-registered-credential"}

    probes = (
        ("post", "/admin/records", {"record": _record_body()}),
        ("get", f"/admin/records/{RECORD_ID}", None),
        ("get", f"/admin/records/{RECORD_ID}/versions", None),
        ("post", f"/admin/records/{RECORD_ID}/edits", {"expected_version": 1, "text": "x"}),
        ("post", f"/admin/records/{RECORD_ID}/transitions", {"action": "submit"}),
    )
    for method, url, body in probes:
        response = (
            client.post(url, json=body, headers=unknown)
            if method == "post"
            else client.get(url, headers=unknown)
        )
        assert response.status_code != 404, f"{method.upper()} {url} is not mounted"
        # The auth layer refused it — the endpoint is reachable and guarded.
        assert response.status_code in (403, 422), (
            f"{method.upper()} {url} -> {response.status_code}"
        )


def _create(client: TestClient, **overrides):
    """Create as the editor: authoring a draft is an editor's grant."""
    return client.post(
        "/admin/records", json={"record": _record_body(**overrides)}, headers=HEADERS
    )


# Which role each transition is performed by. Deliberately the actor the matrix
# grants it to, so these lifecycle tests exercise real permissions instead of an
# all-powerful caller, and so four-eyes (ON here) never sees a self-approval.
_TRANSITION_ACTOR = {
    "submit": "editor",
    "approve": "reviewer",
    "reject": "reviewer",
    "publish": "publisher",
    "supersede": "publisher",
    # Ticket 36: the retreat edges, performed by the role the matrix grants
    # them to (withdraw: publisher/admin; revoke_supersession: admin only).
    "withdraw": "publisher",
    "revoke_supersession": "admin",
}


def _transition(client: TestClient, action: str, *, expected_version=None, reason=None, headers=None):
    body: dict = {"action": action}
    if expected_version is not None:
        body["expected_version"] = expected_version
    if reason is not None:
        body["reason"] = reason
    if headers is None:
        headers = _headers(_TRANSITION_ACTOR[action])
    return client.post(
        f"/admin/records/{RECORD_ID}/transitions", json=body, headers=headers
    )


def _advance_to_published(client: TestClient) -> None:
    """Walk the legal path: draft -> review -> approved -> published."""
    assert _transition(client, "submit", expected_version=1).status_code == 200
    assert _transition(client, "approve", expected_version=2).status_code == 200
    response = _transition(client, "publish", expected_version=3)
    assert response.status_code == 200, response.text


def test_an_unwired_app_refuses_instead_of_writing_nowhere(monkeypatch):
    """503 is reserved for genuinely absent configuration.

    This is the other half of the end-to-end proof: the round-trip test passes
    because the store is really wired, not because a missing store is tolerated.
    Without startup wiring the endpoint must refuse, not silently accept a write
    it cannot persist.
    """
    _configure_identities(monkeypatch)
    unwired = FastAPI()
    unwired.include_router(admin_router)

    with TestClient(unwired) as test_client:
        response = test_client.post(
            "/admin/records",
            json={"record": _record_body()},
            headers=HEADERS,
        )

    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------


def test_create_edit_and_publish_round_trip_through_disk(client, disk, store_root):
    """The wiring is done when this passes: no 503, real files, real readback."""
    created = _create(client)
    assert created.status_code == 200, created.text
    assert created.json()["record"]["status"] == "draft"

    edited = client.post(
        f"/admin/records/{RECORD_ID}/edits",
        json={"expected_version": 1, "text": "Điểm chuẩn ngành CNTT năm 2026 là 25,5 điểm (đã rà soát)."},
        headers=HEADERS,
    )
    assert edited.status_code == 200, edited.text
    assert edited.json()["record"]["version"] == 2

    assert _transition(client, "submit", expected_version=2).status_code == 200
    assert _transition(client, "approve", expected_version=3).status_code == 200
    published = _transition(client, "publish", expected_version=4)
    assert published.status_code == 200, published.text
    assert published.json()["record"]["status"] == "published"

    # Read back through the HTTP layer…
    fetched = client.get(f"/admin/records/{RECORD_ID}", headers=HEADERS)
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "published"

    # …and independently from disk, with a fresh store instance.
    stored = disk.get_record(RECORD_ID)
    assert stored.latest.status == "published"
    assert stored.latest.version == 5
    assert stored.latest.text.startswith("Điểm chuẩn ngành CNTT năm 2026")
    assert disk.version_path(RECORD_ID, 5).is_file()

    # History is intact and append-only: five versions, none rewritten.
    assert disk.versions(RECORD_ID) == (1, 2, 3, 4, 5)


def test_published_version_is_immutable_and_edit_appends_a_draft(client, disk):
    _create(client)
    _advance_to_published(client)
    published_bytes = disk.version_path(RECORD_ID, 4).read_bytes()

    edited = client.post(
        f"/admin/records/{RECORD_ID}/edits",
        json={"expected_version": 4, "text": "Nội dung hiệu chỉnh."},
        headers=HEADERS,
    )
    assert edited.status_code == 200, edited.text
    body = edited.json()["record"]

    # The edit is a NEW version, back in draft — the reviewed text is not
    # silently republished.
    assert body["version"] == 5
    assert body["status"] == "draft"
    assert body["published_at"] is None

    # The published bytes on disk are untouched, byte for byte.
    assert disk.version_path(RECORD_ID, 4).read_bytes() == published_bytes
    assert disk.get_version(RECORD_ID, 4).status == "published"


# ---------------------------------------------------------------------------
# refusals leave data untouched
# ---------------------------------------------------------------------------


def test_illegal_transition_is_4xx_and_writes_nothing(client, disk):
    """The headline prohibition: no shortcut from draft to published."""
    _create(client)
    before = disk.versions(RECORD_ID)

    response = _transition(client, "publish", expected_version=1)

    assert response.status_code == 409, response.text
    assert disk.versions(RECORD_ID) == before
    assert disk.get_record(RECORD_ID).latest.status == "draft"


@pytest.mark.parametrize(
    "action,status_code",
    [("publish", 409), ("approve", 409), ("supersede", 409)],
)
def test_transitions_unreachable_from_draft_are_refused(client, disk, action, status_code):
    _create(client)
    before = disk.versions(RECORD_ID)

    response = _transition(client, action, expected_version=1)

    assert response.status_code == status_code, response.text
    assert disk.versions(RECORD_ID) == before


def test_reject_requires_a_reason(client, disk):
    _create(client)
    _transition(client, "submit", expected_version=1)

    refused = _transition(client, "reject", expected_version=2)
    assert refused.status_code == 422, refused.text
    # Still in review — a refused reject changes nothing.
    assert disk.get_record(RECORD_ID).latest.status == "review"

    accepted = _transition(client, "reject", expected_version=2, reason="thiếu nguồn chính thức")
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["record"]["status"] == "draft"


def test_stale_expected_version_is_detected_not_silently_overwritten(client, disk):
    """Two editors on the same record: the loser is told, not silently merged."""
    _create(client)
    first = client.post(
        f"/admin/records/{RECORD_ID}/edits",
        json={"expected_version": 1, "text": "Bản sửa của người thứ nhất."},
        headers=HEADERS,
    )
    assert first.status_code == 200

    # Second editor still believes version 1 is current.
    conflict = client.post(
        f"/admin/records/{RECORD_ID}/edits",
        json={"expected_version": 1, "text": "Bản sửa của người thứ hai."},
        headers=HEADERS,
    )

    assert conflict.status_code == 409, conflict.text
    assert "version conflict" in conflict.json()["detail"]
    # The first editor's work survived intact.
    assert disk.versions(RECORD_ID) == (1, 2)
    assert "người thứ nhất" in disk.get_record(RECORD_ID).latest.text


# ---------------------------------------------------------------------------
# the shared ops credential cannot touch the admissions lifecycle
# ---------------------------------------------------------------------------

def test_every_lifecycle_endpoint_refuses_the_shared_ops_token(client, disk):
    """Spec decision 7, as machine-visible behaviour: the shared token is ops-only.

    Ticket 12 gated these endpoints with that token behind a staged interlock and
    left a marker for ticket 13 to remove. 13 removed the mechanism: the token is
    now a credential with role ``ops``, and the role x action matrix grants that
    role no admissions action at all. This test makes re-granting it a
    deliberate, failing change rather than an oversight — the endpoint list is
    the same one the interlock test used.
    """
    _create(client)
    before = disk.versions(RECORD_ID)
    ops = {"X-Admin-Token": OPS_TOKEN}

    calls = [
        ("post", "/admin/records", {"record": _record_body(record_id=REPLACEMENT_ID)}),
        ("get", f"/admin/records/{RECORD_ID}", None),
        ("get", f"/admin/records/{RECORD_ID}/versions", None),
        ("post", f"/admin/records/{RECORD_ID}/edits", {"expected_version": 1, "text": "x"}),
        ("post", f"/admin/records/{RECORD_ID}/transitions", {"action": "publish"}),
    ]
    for method, url, body in calls:
        response = (
            client.post(url, json=body, headers=ops)
            if method == "post"
            else client.get(url, headers=ops)
        )
        assert response.status_code == 403, f"{method.upper()} {url} -> {response.status_code}"

    # Nothing was written by any of the refused calls.
    assert disk.versions(RECORD_ID) == before


def test_the_shared_ops_token_still_serves_the_ops_surface(client):
    """Unification is not removal: the machine credential keeps its own surface.

    ``/admin/warmup`` is infrastructure, not admissions content, so role ``ops``
    holds it. Without this, "the shared token cannot publish" would be provable
    by breaking the token everywhere rather than by narrowing it.
    """
    import app.lifecycle as lifecycle_mod

    with patch.object(lifecycle_mod, "AppServices") as services:
        embedding = MagicMock()
        services.return_value.get_embedding.return_value = embedding
        services.return_value.get_llm_client.return_value = MagicMock()

        response = client.post("/admin/warmup", headers={"X-Admin-Token": OPS_TOKEN})

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "warmed"


def test_publish_is_refused_without_a_valid_credential(client, disk):
    """No credential -> no publish.

    This guard survives ticket 13 unchanged: 13 replaces *how* a caller proves
    identity, not the requirement that publish is refused without one.
    """
    _create(client)
    _transition(client, "submit", expected_version=1)
    _transition(client, "approve", expected_version=2)
    before = disk.versions(RECORD_ID)

    wrong = _transition(client, "publish", expected_version=3, headers={"X-Admin-Token": "not-the-token"})
    assert wrong.status_code == 403, wrong.text

    missing = _transition(client, "publish", expected_version=3, headers={})
    assert missing.status_code == 422, missing.text

    assert disk.versions(RECORD_ID) == before
    assert disk.get_record(RECORD_ID).latest.status == "approved"


# ---------------------------------------------------------------------------
# audit events
# ---------------------------------------------------------------------------


def test_every_transition_emits_an_event(client):
    _create(client)

    submitted = _transition(client, "submit", expected_version=1)
    assert submitted.status_code == 200
    event = submitted.json()["event"]
    assert event["action"] == "submit"
    assert event["before_status"] == "draft"
    assert event["after_status"] == "review"
    assert event["before_version"] == 1
    assert event["after_version"] == 2
    assert event["entity"] == "canonical_record"
    assert event["entity_id"] == RECORD_ID
    assert event["actor_id"]  # identity travels with the event

    # reject is only legal from review, and carries its justification.
    rejected = _transition(client, "reject", expected_version=2, reason="cần bổ sung nguồn")
    assert rejected.status_code == 200, rejected.text
    reject_event = rejected.json()["event"]
    assert reject_event["action"] == "reject"
    assert reject_event["reason"] == "cần bổ sung nguồn"
    assert reject_event["after_status"] == "draft"

    assert _transition(client, "submit", expected_version=3).status_code == 200
    approved = _transition(client, "approve", expected_version=4)
    assert approved.json()["event"]["after_status"] == "approved"

    published = _transition(client, "publish", expected_version=5)
    assert published.json()["event"]["after_status"] == "published"

    # Every event lands in the app's stream (ticket 14 persists this).
    service = client.app.state.admin_lifecycle_service
    assert len(service.emitted_events()) == 6  # create + five transitions


def test_transitions_are_rejected_before_any_event_is_emitted(client):
    _create(client)
    service = client.app.state.admin_lifecycle_service
    before = len(service.emitted_events())

    assert _transition(client, "publish", expected_version=1).status_code == 409

    assert len(service.emitted_events()) == before


# ---------------------------------------------------------------------------
# supersede: a relation, not a status
# ---------------------------------------------------------------------------


def test_supersede_is_refused_on_a_record_that_is_not_published(client, disk):
    _create(client)
    _transition(client, "submit", expected_version=1)
    before = disk.versions(RECORD_ID)

    response = _transition(client, "supersede", expected_version=2)

    assert response.status_code == 409, response.text
    assert disk.versions(RECORD_ID) == before


def test_supersede_is_refused_without_a_published_replacement(client, disk):
    """Archiving with nothing in its place is a revocation, not a supersession."""
    _create(client)
    _advance_to_published(client)
    before = disk.versions(RECORD_ID)

    response = _transition(client, "supersede", expected_version=4)

    assert response.status_code == 409, response.text
    assert "no published record names it" in response.json()["detail"]
    assert disk.versions(RECORD_ID) == before
    assert disk.get_record(RECORD_ID).latest.status == "published"


def test_supersede_archives_when_a_published_record_replaces_it(client, disk):
    _create(client)
    _advance_to_published(client)

    # A replacement that is itself published and names the old record.
    replacement = _create(
        client,
        record_id=REPLACEMENT_ID,
        admission_year=2027,
        effective_from="2027-06-01",
        effective_to="2027-12-31",
        supersedes=RECORD_ID,
    )
    assert replacement.status_code == 200, replacement.text
    for version, action in ((1, "submit"), (2, "approve"), (3, "publish")):
        response = client.post(
            f"/admin/records/{REPLACEMENT_ID}/transitions",
            json={"action": action, "expected_version": version},
            headers=_headers(_TRANSITION_ACTOR[action]),
        )
        assert response.status_code == 200, response.text

    superseded = _transition(client, "supersede", expected_version=4)
    assert superseded.status_code == 200, superseded.text
    assert superseded.json()["record"]["status"] == "archived"

    # The relation is what makes it superseded; the status is what makes it
    # archived. They are orthogonal, per the spec.
    assert disk.is_superseded(RECORD_ID) is True
    assert disk.get_record(RECORD_ID).latest.status == "archived"
    # The replacement is untouched and still live.
    assert disk.get_record(REPLACEMENT_ID).latest.status == "published"


def test_a_record_naming_itself_does_not_supersede_itself(client, disk):
    """Self-naming is structurally impossible, so the record cannot self-archive.

    ``supersede`` takes no target from the caller — the replacement is resolved
    through the store's reverse index, which ignores a version naming its own
    ``record_id``. A self-reference therefore yields "no replacement", and the
    record stays published.
    """
    _create(client, supersedes=RECORD_ID)
    _advance_to_published(client)

    assert disk.is_superseded(RECORD_ID) is False

    response = _transition(client, "supersede", expected_version=4)

    assert response.status_code == 409, response.text
    assert disk.get_record(RECORD_ID).latest.status == "published"
    assert disk.versions(RECORD_ID) == (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Idempotency-Key on the transition route (ticket 27)
#
# The state machine already makes a second publish impossible — ``_TRANSITIONS``
# has no self-loop, so a replay is refused 409. That is not the same as a replay
# returning the first call's answer: an operator whose connection dropped
# mid-publish gets a 409 on retry and cannot tell "my publish landed" from
# "someone else published it". These tests cover the stronger guarantee.
# ---------------------------------------------------------------------------


def test_a_replayed_publish_returns_the_first_response(
    client: TestClient, disk: CanonicalRecordStore
):
    """Same key, same body → the original response, and no second transition."""
    _create(client)
    assert _transition(client, "submit", expected_version=1).status_code == 200
    assert _transition(client, "approve", expected_version=2).status_code == 200

    headers = {**_headers(_TRANSITION_ACTOR["publish"]), "Idempotency-Key": "pub-1"}
    first = _transition(client, "publish", expected_version=3, headers=headers)
    assert first.status_code == 200, first.text

    replay = _transition(client, "publish", expected_version=3, headers=headers)

    assert replay.status_code == 200, replay.text
    assert replay.headers["Idempotent-Replay"] == "true"
    assert replay.json() == first.json()
    # No new version was appended by the replay: publish stamps v4, and a second
    # execution would have appended v5.
    assert disk.versions(RECORD_ID) == (1, 2, 3, 4)


def test_a_replay_of_a_publish_that_would_now_be_illegal_still_returns_the_first_answer(
    client: TestClient, disk: CanonicalRecordStore
):
    """The point of the key: the retry succeeds even though the state moved on.

    Without the key this exact call is a 409 — the record is already published,
    so publishing again is illegal. With the key the operator learns what
    actually happened, which is the whole reason they sent one.
    """
    _create(client)
    assert _transition(client, "submit", expected_version=1).status_code == 200
    assert _transition(client, "approve", expected_version=2).status_code == 200
    headers = {**_headers(_TRANSITION_ACTOR["publish"]), "Idempotency-Key": "pub-2"}

    first = _transition(client, "publish", expected_version=3, headers=headers)
    assert first.status_code == 200

    # A *different* key against the same moved-on state is refused, proving the
    # state machine still guards the route — the key is what absorbs the retry,
    # not a weakening of the transition rules.
    unkeyed = _transition(client, "publish", expected_version=3)
    assert unkeyed.status_code == 409

    replay = _transition(client, "publish", expected_version=3, headers=headers)
    assert replay.status_code == 200
    assert replay.headers["Idempotent-Replay"] == "true"


def test_a_replayed_transition_emits_no_second_event(
    client: TestClient, store_root: Path
):
    """One PUBLISH event in the ledger, not two."""
    from src.domain.audit import AuditAction
    from src.infrastructure.audit_ledger import AuditLedger

    _create(client)
    assert _transition(client, "submit", expected_version=1).status_code == 200
    assert _transition(client, "approve", expected_version=2).status_code == 200
    headers = {**_headers(_TRANSITION_ACTOR["publish"]), "Idempotency-Key": "pub-3"}

    _transition(client, "publish", expected_version=3, headers=headers)
    ledger = AuditLedger(store_root / "audit" / "ledger.jsonl")
    after_first = len(ledger.find(action=AuditAction.PUBLISH))

    _transition(client, "publish", expected_version=3, headers=headers)

    assert len(ledger.find(action=AuditAction.PUBLISH)) == after_first


def test_a_failed_transition_frees_its_key(client: TestClient, disk):
    """An illegal transition must not consume the key.

    The operator fixes ``expected_version`` and retries under the same key; a
    key burned by the 409 would make the retry impossible.
    """
    _create(client)
    headers = {**_headers(_TRANSITION_ACTOR["submit"]), "Idempotency-Key": "sub-1"}

    stale = _transition(client, "submit", expected_version=99, headers=headers)
    assert stale.status_code == 409

    retry = _transition(client, "submit", expected_version=1, headers=headers)

    assert retry.status_code == 200, retry.text
    assert retry.json()["record"]["status"] == "review"


def test_the_same_key_with_a_different_action_is_refused(client: TestClient):
    """A key is scoped to one intention: reuse for another action is a 422.

    One actor, two actions — ``admin`` holds both ``submit`` and ``approve``.
    Keys are per actor, so a second *actor* reusing the string "shared-1" is a
    different claim by design (covered in ``test_admin_idempotency.py``); this
    test is the same actor reusing its own key for a different intention.
    """
    _create(client)
    admin_headers = _headers("admin")
    headers = {**admin_headers, "Idempotency-Key": "shared-1"}

    assert _transition(client, "submit", expected_version=1, headers=headers).status_code == 200

    reused = _transition(client, "approve", expected_version=2, headers=headers)

    assert reused.status_code == 422
    assert "different request" in reused.json()["detail"]


# ---------------------------------------------------------------------------
# The key names one record, not just one action (ticket 27)
# ---------------------------------------------------------------------------


def test_a_key_claimed_on_one_record_does_not_replay_onto_another(
    client: TestClient, disk: CanonicalRecordStore
):
    """The record is part of the request, so it is part of the key's identity.

    ``record_id`` travels in the URL path, not the body — and the guard
    fingerprints the body. Left out, the same actor reusing one key on a second
    record matched the stored fingerprint and was served the *first* record's
    response: the caller saw a success for record B while B was never touched.
    The same key for a different record is a different request and must be
    refused, not answered with someone else's result.
    """
    other_id = "DIEM_CHUAN:NGANH:ktpm:2026"
    _create(client)
    _create(client, record_id=other_id)

    admin_headers = _headers("admin")
    headers = {**admin_headers, "Idempotency-Key": "one-intention"}

    first = _transition(client, "submit", expected_version=1, headers=headers)
    assert first.status_code == 200, first.text

    # Same actor, same key, same body — different record. Not a replay.
    second = client.post(
        f"/admin/records/{other_id}/transitions",
        json={"action": "submit", "expected_version": 1},
        headers=headers,
    )

    assert second.status_code == 422, second.text
    assert "different request" in second.json()["detail"]

    # The refusal is real: record B is untouched, and record A's success stands.
    assert disk.get_record(other_id).latest.status == "draft"
    assert disk.versions(other_id) == (1,)
    assert disk.get_record(RECORD_ID).latest.status == "review"
