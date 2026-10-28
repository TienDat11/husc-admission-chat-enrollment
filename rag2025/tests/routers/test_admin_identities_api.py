"""Ticket 37 acceptance: the identity store + ``/admin/identities`` surface.

Every criterion in
``.scratch/husc-rag-admin-upgrade/issues/37-role-management-permission-audit.md``
maps to at least one test here, and each test is written so a plausible bug in
the code it covers FAILS:

- persistence is proven through a *fresh* ``IdentityStore`` instance over the
  same root (a router-local cache cannot fake it);
- the ledger is read through the raw JSONL bytes and a *second* ``AuditLedger``
  instance (an in-memory event list cannot fake it);
- env-vs-store precedence is proven by changing the environment mid-test
  (a restart with a stale registry) and re-authenticating;
- the reason guard is proven at both layers (router 422 + audit-domain
  ``ValueError``), including "the store file bytes did not change".

Fixture pattern copied from ``tests/routers/test_admin_rbac.py`` (ticket 13):
a bare ``FastAPI`` with the composed ``admin_router`` included, wired through
the real startup helper over a ``tmp_path`` root, tokens ``tok-<role>``.
"""
from __future__ import annotations

import json
import os
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
from rag2025.src.app.lifecycle import (  # noqa: E402
    configure_canonical_record_store,
)
from src.domain import audit as audit_domain  # noqa: E402
from src.domain.admin_identity import (  # noqa: E402
    AdminAction,
    Role,
    action_for_transition,
    granted_actions,
    role_allows,
)
from src.domain.canonical.lifecycle import LifecycleAction  # noqa: E402
from src.infrastructure import identity_store as store_module  # noqa: E402
from src.infrastructure.audit_ledger import AuditLedger  # noqa: E402
from src.infrastructure.identity_store import (  # noqa: E402
    IDENTITIES_DIRNAME,
    IDENTITIES_FILENAME,
    IdentityStore,
)
from src.routers import admin_identities as identities_module  # noqa: E402
from src.routers.admin import router as admin_router  # noqa: E402

# Same registry the ticket-13 suite publishes: one actor per role plus a
# second reviewer. The ops machine key stays env-only (never seeded).
ACTORS: dict[str, tuple[str, str]] = {
    "admin": ("cb.quan-tri", "admin"),
    "editor": ("cb.bien-tap", "editor"),
    "reviewer": ("cb.phan-bien", "reviewer"),
    "publisher": ("cb.xuat-ban", "publisher"),
    "viewer": ("cb.chi-xem", "viewer"),
}
SECOND_REVIEWER = ("cb.phan-bien-2", "reviewer")

TOKENS = {role: f"tok-{role}" for role in ACTORS}
OPS_TOKEN = "shared-ops-token"

RECORD_ID = "DIEM_CHUAN:NGANH:cntt:2026"

# A credential this suite registers through the API. Distinct enough that
# "this string never appears in a response body" cannot be satisfied by
# accident, and its first two chars ("to") still match the mask shape.
NEW_TOKEN = "tok-moi-7f3a9c"

LEDGER_PATH = ("audit", "ledger.jsonl")


def _headers(role: str) -> dict[str, str]:
    return {"X-Admin-Token": TOKENS[role]}


def _configure(monkeypatch, *, four_eyes: bool | None = None) -> None:
    """Publish the env registry exactly as a deployment does."""
    entries = [
        {"token": TOKENS[role], "actor_id": actor_id, "role": role}
        for role, (actor_id, _) in ACTORS.items()
    ]
    entries.append(
        {
            "token": "tok-reviewer-2",
            "actor_id": SECOND_REVIEWER[0],
            "role": SECOND_REVIEWER[1],
        }
    )
    monkeypatch.setenv("ADMIN_IDENTITIES", json.dumps(entries))
    monkeypatch.setenv("ADMIN_API_TOKEN", OPS_TOKEN)
    if four_eyes is None:
        monkeypatch.delenv("ADMIN_FOUR_EYES", raising=False)
    else:
        monkeypatch.setenv("ADMIN_FOUR_EYES", "1" if four_eyes else "0")


def _build_app(store_root: Path, monkeypatch, *, four_eyes=None) -> FastAPI:
    _configure(monkeypatch, four_eyes=four_eyes)
    application = FastAPI()
    application.include_router(admin_router)
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=store_root)
    )
    return application


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "canonical_store"


@pytest.fixture
def app(store_root: Path, monkeypatch) -> FastAPI:
    return _build_app(store_root, monkeypatch)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def _permission_events(store_root: Path) -> list[dict]:
    """The permission_change events as a second ledger instance reads them."""
    return [
        entry.event.to_dict()
        for entry in _fresh_ledger(store_root).find(action="permission_change")
    ]


def _client_for(store_root: Path, monkeypatch, *, four_eyes=None) -> TestClient:
    return TestClient(_build_app(store_root, monkeypatch, four_eyes=four_eyes))


def _identities_file(store_root: Path) -> Path:
    return store_root / IDENTITIES_DIRNAME / IDENTITIES_FILENAME


def _fresh_store(store_root: Path) -> IdentityStore:
    """A second, independent store over the same root — reads real bytes."""
    return IdentityStore(store_root)


def _fresh_ledger(store_root: Path) -> AuditLedger:
    """A second, independent ledger over the same file the app writes."""
    return AuditLedger(store_root.joinpath(*LEDGER_PATH))


def _ledger_text(store_root: Path) -> str:
    path = store_root.joinpath(*LEDGER_PATH)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _ledger_line_count(store_root: Path) -> int:
    return len([line for line in _ledger_text(store_root).splitlines() if line.strip()])


def _create(
    client: TestClient,
    *,
    actor_id: str = "cb.moi",
    credential: str = NEW_TOKEN,
    role: str = "editor",
    subject_id: str | None = None,
    reason: str = "  cap quyen theo de nghi so 47  ",
    headers: dict[str, str] | None = None,
):
    body: dict = {
        "actor_id": actor_id,
        "role": role,
        "credential": credential,
        "reason": reason,
    }
    if subject_id is not None:
        body["subject_id"] = subject_id
    return client.post(
        "/admin/identities", json=body, headers=headers or _headers("admin")
    )


def _patch(
    client: TestClient,
    subject_id: str,
    *,
    role: str | None = None,
    disabled: bool | None = None,
    reason: str = "dieu chinh theo phan cong moi",
    extra: dict | None = None,
    headers: dict[str, str] | None = None,
):
    body: dict = {"reason": reason}
    if role is not None:
        body["role"] = role
    if disabled is not None:
        body["disabled"] = disabled
    if extra:
        body.update(extra)
    return client.patch(
        f"/admin/identities/{subject_id}",
        json=body,
        headers=headers or _headers("admin"),
    )


def _prime(client: TestClient) -> None:
    """Force one authenticated read so the env bootstrap has run."""
    assert client.get("/admin/identity", headers=_headers("admin")).status_code == 200


# ---------------------------------------------------------------------------
# criterion 1 — the store is really consulted (persistence without a restart)
# ---------------------------------------------------------------------------


def test_created_identity_persists_in_a_fresh_store_instance(client, store_root):
    """A POST writes the store, not a router-local cache.

    The read-back is a brand-new ``IdentityStore`` over the same root: it
    shares no state with the app, so the entry can only be there if it is on
    disk. The env string never changed, either.
    """
    env_before = os.environ["ADMIN_IDENTITIES"]

    created = _create(client)
    assert created.status_code == 200, created.text

    entry = _fresh_store(store_root).get("cb.moi")
    assert entry is not None, "created identity is not on disk"
    assert entry.actor_id == "cb.moi"
    assert entry.role.value == "editor"
    assert entry.disabled is False
    assert os.environ["ADMIN_IDENTITIES"] == env_before


def test_created_identity_authenticates_without_restart(client):
    """The point of the store: the next request with the new credential works.

    Env-only resolution answers 403 here — the credential exists only in the
    store, which the resolution order must consult.
    """
    assert _create(client).status_code == 200

    me = client.get("/admin/identity", headers={"X-Admin-Token": NEW_TOKEN})
    assert me.status_code == 200, me.text
    body = me.json()
    assert body["actor_id"] == "cb.moi"
    assert body["role"] == "editor"
    assert body["actions"] == sorted(
        action.value for action in granted_actions(Role.EDITOR)
    )


# ---------------------------------------------------------------------------
# criterion 2 — env bootstrap, then the store wins over a changed env
# ---------------------------------------------------------------------------


def test_env_bootstrap_seeds_the_store_on_first_request(client, store_root):
    """Empty store + configured env → the humans resolve AND the file appears.

    The shared ops credential is deliberately not seeded (machine key, not a
    person), and no mutation happened: the six env humans are exactly what
    lands.
    """
    assert not _identities_file(store_root).exists()

    _prime(client)

    path = _identities_file(store_root)
    assert path.exists(), "bootstrap did not write the store file"
    doc = json.loads(path.read_text(encoding="utf-8"))
    subjects = {entry["actor_id"] for entry in doc["entries"]}
    assert subjects == {a for a, _ in ACTORS.values()} | {SECOND_REVIEWER[0]}
    assert OPS_TOKEN not in path.read_text(encoding="utf-8")


def test_store_wins_after_environment_changes(client, store_root, monkeypatch):
    """A stale env must not silently rewrite what the API granted.

    This is the test the "drop the seed guard" mutation breaks: an
    unconditional seed would rewrite the store from the degraded env on the
    next request, deleting the API-created subject and resurrecting the
    env-claimed admin grant.
    """
    _prime(client)
    assert _create(client).status_code == 200

    # Simulate a restart with an older environment: cb.moi does not exist
    # there, and it claims cb.bien-tap is an admin.
    degraded = [
        {"token": TOKENS["admin"], "actor_id": ACTORS["admin"][0], "role": "admin"},
        {"token": TOKENS["editor"], "actor_id": ACTORS["editor"][0], "role": "admin"},
    ]
    monkeypatch.setenv("ADMIN_IDENTITIES", json.dumps(degraded))

    # The API-created identity still resolves...
    me = client.get("/admin/identity", headers={"X-Admin-Token": NEW_TOKEN})
    assert me.status_code == 200, me.text
    assert me.json()["role"] == "editor"

    # ...and the env's inflated claim loses to the stored role.
    editor_me = client.get("/admin/identity", headers=_headers("editor"))
    assert editor_me.status_code == 200
    assert editor_me.json()["role"] == "editor", "env silently overwrote the store"

    # The store file itself was not rewritten by the changed env.
    doc = json.loads(_identities_file(store_root).read_text(encoding="utf-8"))
    assert {e["actor_id"] for e in doc["entries"]} == (
        {a for a, _ in ACTORS.values()} | {SECOND_REVIEWER[0], "cb.moi"}
    )


# ---------------------------------------------------------------------------
# criterion 3 — exactly one PERMISSION_CHANGE per mutation, and it lands
# ---------------------------------------------------------------------------


def test_each_mutation_appends_exactly_one_permission_change_event(client, store_root):
    """create → 1 event, role change → 1, disable → 1.

    Deleting the emit in any single branch fails that step's count. The
    before/after role states are asserted per event, and the chain still
    verifies at the end.
    """
    _prime(client)
    assert _permission_events(store_root) == []

    assert _create(client).status_code == 200
    events = _permission_events(store_root)
    assert len(events) == 1, "create must append exactly one event"
    assert events[0]["before_version"] == "cb.moi@none"
    assert events[0]["after_version"] == "cb.moi@editor"

    assert _patch(client, "cb.moi", role="reviewer").status_code == 200
    events = _permission_events(store_root)
    assert len(events) == 2, "role change must append exactly one event"
    assert events[1]["before_version"] == "cb.moi@editor"
    assert events[1]["after_version"] == "cb.moi@reviewer"

    assert _patch(client, "cb.moi", disabled=True).status_code == 200
    events = _permission_events(store_root)
    assert len(events) == 3, "disable must append exactly one event"
    assert events[2]["before_version"] == "cb.moi@reviewer"
    assert events[2]["after_version"] == "cb.moi@disabled"

    verification = _fresh_ledger(store_root).verify()
    assert verification.ok is True
    assert verification.checked == len(_fresh_ledger(store_root).entries())


def test_event_actor_is_the_resolved_identity_not_a_body_field(client, store_root):
    """Neither the create body nor a spoofed PATCH field names the actor.

    ``actor_id`` in the create body is the *target* (cb.moi); the event's
    actor must be the caller the credential proved. A PATCH body carrying an
    extra ``actor_id`` must be ignored entirely.
    """
    _prime(client)
    assert _create(client).status_code == 200

    event = _permission_events(store_root)[-1]
    assert event["actor_id"] == ACTORS["admin"][0]
    assert event["role"] == "admin"
    assert event["target"] == "cb.moi"
    assert event["target_kind"] == "subject"
    # Blank-wrapped reason is stored stripped.
    assert event["reason"] == "cap quyen theo de nghi so 47"

    spoofed = _patch(
        client, "cb.moi", role="reviewer", extra={"actor_id": "ke-gia-mao"}
    )
    assert spoofed.status_code == 200, spoofed.text
    event = _permission_events(store_root)[-1]
    assert event["actor_id"] == ACTORS["admin"][0], "body forged the event actor"
    assert event["target"] == "cb.moi"


def test_permission_change_events_land_in_the_ledger_file(client, store_root):
    """Producer criterion at file level: the JSONL on disk carries them.

    Read as raw bytes — nothing in this test goes through the app's objects.
    """
    _prime(client)
    assert _create(client).status_code == 200
    assert _patch(client, "cb.moi", role="admin").status_code == 200

    raw = _ledger_text(store_root)
    assert raw.count('"action":"permission_change"') == 2, raw
    assert '"target_kind":"subject"' in raw
    assert '"actor_id":"cb.quan-tri"' in raw


def test_permission_change_has_a_real_producer_outside_the_audit_domain():
    """B14-1's zero-producer gap: the enum must have an actual caller.

    Before ticket 37, ``PERMISSION_CHANGE`` appeared only in
    ``domain/audit.py``. This walks ``src/`` and requires the mutation router
    to call the emitter seam — a call site, not a mention.
    """
    callers = {
        path.name
        for path in (ROOT / "src").rglob("*.py")
        if ".record_permission_change(" in path.read_text(encoding="utf-8")
    }
    assert "admin_identities.py" in callers, (
        "no router calls record_permission_change — the emitter is gone"
    )


# ---------------------------------------------------------------------------
# criterion 4 — the reason guard at BOTH layers, with nothing written
# ---------------------------------------------------------------------------


def test_router_rejects_blank_reason_before_any_write(client, store_root):
    """Layer 1 (router): blank reason → 422, store bytes and ledger untouched.

    Removing the router's ``_require_reason`` check would let the write reach
    ``store.add`` (file changes) before the ledger's own guard raises — this
    assertion fails on either symptom.
    """
    _prime(client)
    before = _identities_file(store_root).read_bytes()
    ledger_before = _ledger_line_count(store_root)

    refused = _create(client, reason="   ")
    assert refused.status_code == 422, refused.text

    assert _identities_file(store_root).read_bytes() == before, "blank reason wrote"
    assert _ledger_line_count(store_root) == ledger_before, "blank reason emitted"


def test_router_rejects_blank_reason_on_patch_before_any_write(client, store_root):
    """Layer 1 on PATCH: 422 before ``store.update``, file byte-identical."""
    _prime(client)
    assert _create(client).status_code == 200
    before = _identities_file(store_root).read_bytes()
    ledger_before = _ledger_line_count(store_root)

    refused = _patch(client, "cb.moi", role="reviewer", reason="  ")
    assert refused.status_code == 422, refused.text

    assert _identities_file(store_root).read_bytes() == before
    assert _ledger_line_count(store_root) == ledger_before


def test_audit_domain_refuses_permission_change_without_reason():
    """Layer 2 (domain): PERMISSION_CHANGE is in REASON_REQUIRED itself.

    A route that forgot to pass a reason cannot build a valid event even if
    the router check were removed — the enum membership is the backstop.
    """
    assert audit_domain.AuditAction.PERMISSION_CHANGE in audit_domain.REASON_REQUIRED
    with pytest.raises(ValueError):
        audit_domain.AuditEvent(
            action=audit_domain.AuditAction.PERMISSION_CHANGE,
            actor_id="cb.quan-tri",
            target="cb.moi",
            timestamp=datetime.now(timezone.utc),
            target_kind=audit_domain.TargetKind.SUBJECT,
            reason="   ",
        )


# ---------------------------------------------------------------------------
# criterion 5 — RBAC: only roles.manage gets in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["publisher", "reviewer", "editor", "viewer"])
def test_roles_without_manage_are_refused_on_all_three_routes(client, role):
    """publisher/reviewer/editor/viewer → 403 on list, create, and patch."""
    listed = client.get("/admin/identities", headers=_headers(role))
    assert listed.status_code == 403, listed.text

    created = _create(client, headers=_headers(role))
    assert created.status_code == 403, created.text

    patched = _patch(
        client, ACTORS["editor"][0], role="viewer", headers=_headers(role)
    )
    assert patched.status_code == 403, patched.text


def test_ops_credential_cannot_manage_roles(client):
    """The shared machine key has no roles.manage → 403, not a superuser."""
    response = client.get("/admin/identities", headers={"X-Admin-Token": OPS_TOKEN})
    assert response.status_code == 403, response.text


def test_missing_credential_header_is_422(client):
    """No header at all is a malformed request, refused before any handler."""
    response = client.get("/admin/identities")
    assert response.status_code == 422, response.text


def test_unknown_credential_is_403(client):
    """A credential nobody issued is refused like ticket 13, not 401/422."""
    response = client.get(
        "/admin/identities", headers={"X-Admin-Token": "tok-khong-ton-tai"}
    )
    assert response.status_code == 403, response.text


def test_roles_manage_is_admin_only_in_the_matrix():
    """The grant itself: admin yes, every other role no.

    Removing ``ROLES_MANAGE`` from the admin row makes the admin assertion
    fail; adding it to any other row makes that row's assertion fail.
    """
    assert role_allows(Role.ADMIN, AdminAction.ROLES_MANAGE) is True
    for role in (Role.VIEWER, Role.EDITOR, Role.REVIEWER, Role.PUBLISHER, Role.OPS):
        assert role_allows(role, AdminAction.ROLES_MANAGE) is False, role


def test_admin_identity_endpoint_reports_roles_manage(client):
    """The FE gate reads the grant from ``GET /admin/identity``.

    If the matrix loses ROLES_MANAGE, the literal disappears from the admin's
    own action list and the FE can no longer show the screen.
    """
    response = client.get("/admin/identity", headers=_headers("admin"))
    assert response.status_code == 200, response.text
    assert "roles.manage" in response.json()["actions"]


# ---------------------------------------------------------------------------
# criterion 6 — transitions cannot be pointed at roles.manage
# ---------------------------------------------------------------------------

_TRANSITION_VALUES = tuple(a.value for a in LifecycleAction)


def test_action_for_transition_never_resolves_roles_manage():
    """``roles.manage`` is not a lifecycle action and maps to nothing.

    The deny-by-``None`` rule means a transition endpoint can never be the
    back door for a permission grant.
    """
    assert "roles.manage" not in _TRANSITION_VALUES
    assert action_for_transition("roles.manage") is None
    for action in _TRANSITION_VALUES:
        assert action_for_transition(action) is not AdminAction.ROLES_MANAGE


def test_transition_route_rejects_roles_manage_with_422(client):
    """The wire behaviour: POST transitions with action "roles.manage" → 422.

    Validation rejects the enum value before any record lookup, so no record
    needs to exist and no data can change.
    """
    response = client.post(
        f"/admin/records/{RECORD_ID}/transitions",
        json={"action": "roles.manage"},
        headers=_headers("admin"),
    )
    assert response.status_code == 422, response.text


# ---------------------------------------------------------------------------
# criterion 7 — four-eyes: self-change blocked, others not, flag read ONCE
# ---------------------------------------------------------------------------


def test_four_eyes_on_blocks_self_change_without_writing_or_emitting(
    store_root, monkeypatch
):
    """With the control ON, an admin PATCHing their own subject → 403.

    The store file and the ledger must not move: the block happens before any
    write. A comparison on the *body* rather than the resolved identity would
    let the same call through.
    """
    client = _client_for(store_root, monkeypatch, four_eyes=True)
    with client:
        _prime(client)
        before = _identities_file(store_root).read_bytes()
        ledger_before = _ledger_line_count(store_root)

        refused = _patch(client, ACTORS["admin"][0], role="reviewer")
        assert refused.status_code == 403, refused.text
        assert _identities_file(store_root).read_bytes() == before
        assert _ledger_line_count(store_root) == ledger_before

        # Someone else is NOT blocked — the control is about self-service,
        # not about freezing the admin out of their own job.
        allowed = _patch(client, ACTORS["editor"][0], role="viewer")
        assert allowed.status_code == 200, allowed.text
        events = _permission_events(store_root)
        assert len(events) == 1
        assert events[0]["review_id"] == "four_eyes=on"


def test_four_eyes_off_allows_self_change_and_records_the_flag(
    store_root, monkeypatch
):
    """With the control OFF, the same self-PATCH succeeds; the event says off.

    The recorded flag must reflect the switch at mutation time, so an auditor
    can tell a supervised change from an unsupervised one.
    """
    client = _client_for(store_root, monkeypatch, four_eyes=False)
    with client:
        _prime(client)
        allowed = _patch(client, ACTORS["admin"][0], role="reviewer")
        assert allowed.status_code == 200, allowed.text
        events = _permission_events(store_root)
        assert len(events) == 1
        assert events[0]["review_id"] == "four_eyes=off"
        assert events[0]["target"] == ACTORS["admin"][0]


def test_mutation_flow_reads_the_four_eyes_flag_exactly_once(client, monkeypatch):
    """One read per mutation — the check and the event share one value.

    A flow that re-reads the switch mid-request can block on one reading and
    record the opposite, which is worse than either setting.
    """
    reads: list[bool] = []

    def counting() -> bool:
        reads.append(False)
        return False

    monkeypatch.setattr(identities_module, "four_eyes_enabled", counting)
    _prime(client)

    assert _create(client).status_code == 200
    assert len(reads) == 1, "create must read the flag exactly once"

    assert _patch(client, "cb.moi", role="reviewer").status_code == 200
    assert len(reads) == 2, "patch must read the flag exactly once"


def test_flag_flip_between_reads_cannot_split_check_and_event(
    client, store_root, monkeypatch
):
    """The env must not be re-consulted after the flag was pinned.

    The wrapper returns OFF on the first read and flips the env to ON behind
    the flow's back. Correct code records ``four_eyes=off`` (the value the
    check saw) and reads once; a re-read-mid-flow records ``on`` and fails.
    """
    real = identities_module.four_eyes_enabled
    state = {"calls": 0}

    def flipping() -> bool:
        state["calls"] += 1
        value = real()
        if state["calls"] == 1:
            monkeypatch.setenv("ADMIN_FOUR_EYES", "1")
        return value

    monkeypatch.setenv("ADMIN_FOUR_EYES", "0")
    monkeypatch.setattr(identities_module, "four_eyes_enabled", flipping)
    _prime(client)

    assert _create(client).status_code == 200
    assert state["calls"] == 1, "flow re-read the flag"
    event = _permission_events(store_root)[-1]
    assert event["review_id"] == "four_eyes=off", (
        "event recorded a different flag value than the check saw"
    )


# ---------------------------------------------------------------------------
# criterion 8 — no response body ever carries a plaintext credential
# ---------------------------------------------------------------------------


def test_mask_credential_shape():
    """2 revealed chars, at least 4 stars; ≤4-char secrets fully masked."""
    assert identities_module.mask_credential("abcd") == "****"
    assert identities_module.mask_credential("abcde") == "ab****"
    assert identities_module.mask_credential("tok-admin") == "to" + "*" * 7


def test_plaintext_credentials_absent_from_list_create_and_patch(client):
    """Every response body masks; a leaked screenshot leaks nothing.

    Removing the mask in ``_view`` puts the real token back in all three
    bodies and fails every assertion here.
    """
    _prime(client)

    created = _create(client)
    assert created.status_code == 200
    assert NEW_TOKEN not in created.text
    assert created.json()["credential"] == "to" + "*" * 12

    patched = _patch(client, "cb.moi", role="reviewer")
    assert patched.status_code == 200
    assert NEW_TOKEN not in patched.text

    listed = client.get("/admin/identities", headers=_headers("admin"))
    assert listed.status_code == 200
    assert NEW_TOKEN not in listed.text
    for token in TOKENS.values():
        assert token not in listed.text, "a seeded token leaked"
    assert "tok-reviewer-2" not in listed.text
    assert OPS_TOKEN not in listed.text
    # Every listed credential is a mask: 2 chars + only stars.
    assert len(listed.json()) == 7
    for row in listed.json():
        assert row["credential"][:2] and set(row["credential"][2:]) <= {"*"}


# ---------------------------------------------------------------------------
# criterion 9 — disable takes effect immediately
# ---------------------------------------------------------------------------


def test_disabled_identity_denies_on_the_very_next_request(client, store_root):
    """PATCH disabled=true → the credential is dead now, in this process.

    A per-process identity cache would keep answering 200 here; a store that
    only marked-but-ignored ``disabled`` likewise. The fresh store instance
    proves the flag is on disk, not just in the response.
    """
    assert _create(client).status_code == 200
    assert client.get(
        "/admin/identity", headers={"X-Admin-Token": NEW_TOKEN}
    ).status_code == 200

    disabled = _patch(client, "cb.moi", disabled=True)
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["disabled"] is True

    refused = client.get("/admin/identity", headers={"X-Admin-Token": NEW_TOKEN})
    assert refused.status_code == 403, refused.text
    assert refused.json()["detail"] == "admin identity is disabled"

    # The env never knew this token; a store-blind resolver would miss the
    # disable, and the on-disk entry proves the flag persisted.
    assert _fresh_store(store_root).find(NEW_TOKEN).disabled is True

    # And the revocation is auditable: after_role None → @disabled.
    event = _permission_events(store_root)[-1]
    assert event["after_version"] == "cb.moi@disabled"


def test_disable_survives_a_new_process(store_root, monkeypatch):
    """A second app over the same root refuses the disabled credential too.

    This is the "revocation needs a restart to take effect" failure mode:
    in-memory caches die with the process, the store does not.
    """
    first = _client_for(store_root, monkeypatch)
    with first:
        assert _create(first).status_code == 200
        assert _patch(first, "cb.moi", disabled=True).status_code == 200

    second = _client_for(store_root, monkeypatch)
    with second:
        refused = second.get(
            "/admin/identity", headers={"X-Admin-Token": NEW_TOKEN}
        )
        assert refused.status_code == 403, refused.text
        admin_ok = second.get("/admin/identity", headers=_headers("admin"))
        assert admin_ok.status_code == 200


# ---------------------------------------------------------------------------
# criterion 10 — GET /admin/identity reflects the store before/after a PATCH
# ---------------------------------------------------------------------------


def test_identity_actions_change_after_a_role_patch(client):
    """The caller's own action list follows the stored role, live.

    Env-only resolution would freeze the list at whatever env said at boot;
    a stale cache would freeze it at the first request. Both fail here.
    """
    assert _create(client).status_code == 200

    before = client.get("/admin/identity", headers={"X-Admin-Token": NEW_TOKEN})
    assert before.status_code == 200
    actions_before = before.json()["actions"]
    assert "records.edit" in actions_before
    assert "records.approve" not in actions_before

    assert _patch(client, "cb.moi", role="reviewer").status_code == 200

    after = client.get("/admin/identity", headers={"X-Admin-Token": NEW_TOKEN})
    assert after.status_code == 200
    actions_after = after.json()["actions"]
    assert "records.approve" in actions_after
    assert "records.edit" not in actions_after
    assert actions_before != actions_after


# ---------------------------------------------------------------------------
# supporting error mappings (the 409/404 seam)
# ---------------------------------------------------------------------------


def test_conflicts_and_unknown_subject_map_to_409_and_404(client):
    """Duplicate subject/actor/credential → 409; PATCH unknown → 404.

    The 409s are what keep one credential from opening two doors; if the
    credential conflict check were dropped, the last POST would succeed and
    ``find`` would become ambiguous.
    """
    _prime(client)
    assert _create(client).status_code == 200

    dup_subject = _create(
        client, actor_id="cb-khac", subject_id="cb.moi", credential="tok-khac-9"
    )
    assert dup_subject.status_code == 409, dup_subject.text

    dup_credential = _create(client, actor_id="cb-khac", credential=NEW_TOKEN)
    assert dup_credential.status_code == 409, dup_credential.text

    missing = _patch(client, "cb-khong-co", role="viewer")
    assert missing.status_code == 404, missing.text


# ---------------------------------------------------------------------------
# timing-safe lookup (hmac.compare_digest, no early return)
# ---------------------------------------------------------------------------


def test_find_compares_every_entry_without_early_return(client, store_root, monkeypatch):
    """``find`` must do the same work for every credential.

    With the matching credential first, an early ``return`` would compare one
    entry and stop; ``==`` instead of ``compare_digest`` would skip the
    constant-time helper entirely. Both fail the count assertions.
    """
    _prime(client)
    # Order on disk: the six seeded humans, then cb.moi (created last).
    assert _create(client).status_code == 200
    store = _fresh_store(store_root)
    entries = store.entries()
    assert len(entries) == 7
    first_credential = entries[0].credential

    calls: list[tuple[str, str]] = []
    real = store_module.hmac.compare_digest

    def recording(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(store_module.hmac, "compare_digest", recording)

    calls.clear()
    found = store.find(first_credential)
    assert found is not None and found.credential == first_credential
    assert len(calls) == len(entries), "find short-circuited on the first match"

    calls.clear()
    assert store.find("tok-hoan-toan-khong-co") is None
    assert len(calls) == len(entries), "unknown credential skipped comparisons"
