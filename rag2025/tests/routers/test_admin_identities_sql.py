"""U10 - the admin identity surface on the PostgreSQL identity tables.

Skipped without DATABASE_URL (see ``tests.pg_fixtures``).
"""
from __future__ import annotations

import threading
import uuid

from src.domain.admin_identity import Role
from src.infrastructure.identity_store import IdentityEntry, hash_password
from src.infrastructure.sql_identity_repo import LastAdminError, SqlIdentityRepository
from tests.pg_fixtures import ADMIN_ACTOR, ADMIN_TOKEN, fresh_db, sql_admin_client  # noqa: F401

ADMIN_HEADERS = {"X-Admin-Token": ADMIN_TOKEN}
PASSWORD = "correct-horse-1"


def _create(client, actor_id: str, role: str = "editor", **extra):
    body = {"actor_id": actor_id, "role": role, "reason": "U10 test", **extra}
    return client.post("/admin/identities", json=body, headers=ADMIN_HEADERS)


def _patch(client, subject_id: str, **body):
    return client.patch(
        f"/admin/identities/{subject_id}",
        json={"reason": "U10 test", **body},
        headers=ADMIN_HEADERS,
    )


def _subject_of(client, actor_id: str) -> str:
    rows = client.get("/admin/identities", headers=ADMIN_HEADERS).json()
    return next(row["subject_id"] for row in rows if row["actor_id"] == actor_id)


def _cookie_for(client, actor_id: str, password: str) -> dict[str, str]:
    resp = client.post("/admin/login", json={"actor_id": actor_id, "password": password})
    assert resp.status_code == 200, resp.text
    return {"Cookie": f"admin_session={resp.cookies['admin_session']}"}


def test_issued_header_token_authenticates_and_no_secret_is_ever_listed(fresh_db, tmp_path, monkeypatch):  # noqa: F811
    with sql_admin_client(fresh_db, tmp_path, monkeypatch) as client:
        created = _create(client, "cb.quan-tri-2", role="admin", issue_header_token=True)
        assert created.status_code == 200, created.text
        body = created.json()
        token = body["header_token"]
        assert isinstance(token, str) and len(token) >= 40
        for forbidden in ("password", "password_hash", "header_token_hash"):
            assert forbidden not in body

        listing = client.get("/admin/identities", headers=ADMIN_HEADERS)
        assert listing.status_code == 200
        assert token not in listing.text
        assert "password_hash" not in listing.text
        assert "header_token_hash" not in listing.text

        assert client.get("/admin/identities", headers={"X-Admin-Token": token}).status_code == 200


def test_duplicate_actor_id_is_409_identity_exists(fresh_db, tmp_path, monkeypatch):  # noqa: F811
    with sql_admin_client(fresh_db, tmp_path, monkeypatch) as client:
        assert _create(client, "cb.bien-tap-2", password=PASSWORD).status_code == 200
        dup = _create(client, "cb.bien-tap-2", password="another-pass-2")
        assert dup.status_code == 409
        assert "IDENTITY_EXISTS" in dup.text


def test_admin_cannot_change_own_role_or_disable_self_without_four_eyes(fresh_db, tmp_path, monkeypatch):  # noqa: F811
    with sql_admin_client(fresh_db, tmp_path, monkeypatch) as client:
        own = _subject_of(client, ADMIN_ACTOR)
        role_change = _patch(client, own, role="editor")
        assert role_change.status_code == 403
        assert "SELF_ROLE_CHANGE_FORBIDDEN" in role_change.text
        disable = _patch(client, own, disabled=True)
        assert disable.status_code == 403
        assert "SELF_ROLE_CHANGE_FORBIDDEN" in disable.text


def test_disabling_an_identity_refuses_its_live_session_with_session_expired(fresh_db, tmp_path, monkeypatch):  # noqa: F811
    with sql_admin_client(fresh_db, tmp_path, monkeypatch) as client:
        assert _create(client, "cb.bien-tap-2", password=PASSWORD).status_code == 200
        cookie = _cookie_for(client, "cb.bien-tap-2", PASSWORD)
        subject = _subject_of(client, "cb.bien-tap-2")
        assert _patch(client, subject, disabled=True).status_code == 200

        resp = client.get("/admin/identities", headers=cookie)
        assert resp.status_code == 401
        assert "SESSION_EXPIRED" in resp.text


def test_changing_a_role_revokes_the_live_session(fresh_db, tmp_path, monkeypatch):  # noqa: F811
    with sql_admin_client(fresh_db, tmp_path, monkeypatch) as client:
        assert _create(client, "cb.bien-tap-2", role="admin", password=PASSWORD).status_code == 200
        cookie = _cookie_for(client, "cb.bien-tap-2", PASSWORD)
        subject = _subject_of(client, "cb.bien-tap-2")
        assert _patch(client, subject, role="viewer").status_code == 200

        resp = client.get("/admin/identities", headers=cookie)
        assert resp.status_code == 401
        assert "SESSION_EXPIRED" in resp.text


def test_disabled_header_identity_is_403_account_disabled(fresh_db, tmp_path, monkeypatch):  # noqa: F811
    with sql_admin_client(fresh_db, tmp_path, monkeypatch) as client:
        created = _create(client, "cb.quan-tri-3", role="admin", issue_header_token=True)
        assert created.status_code == 200, created.text
        token = created.json()["header_token"]
        assert _patch(client, created.json()["subject_id"], disabled=True).status_code == 200

        resp = client.get("/admin/identities", headers={"X-Admin-Token": token})
        assert resp.status_code == 403
        assert "ACCOUNT_DISABLED" in resp.text


def test_concurrent_demotions_of_the_two_remaining_admins_leave_one(fresh_db, tmp_path, monkeypatch):  # noqa: F811
    with sql_admin_client(fresh_db, tmp_path, monkeypatch) as client:
        first_subject = _subject_of(client, ADMIN_ACTOR)

    repo = SqlIdentityRepository(fresh_db)
    second = repo.add(
        IdentityEntry(
            subject_id=str(uuid.uuid4()),
            actor_id="cb.quan-tri-2",
            role=Role.ADMIN,
            password_hash=hash_password(PASSWORD),
        ),
        created_by="test",
    )

    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def demote(subject_id: str) -> None:
        barrier.wait()
        try:
            repo.update(subject_id, role=Role.EDITOR, disabled=None, password=None)
            outcomes.append("ok")
        except LastAdminError:
            outcomes.append("last-admin")

    threads = [threading.Thread(target=demote, args=(s,)) for s in (first_subject, second.subject_id)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sorted(outcomes) == ["last-admin", "ok"]
    admins = [entry for entry in repo.entries() if entry.role is Role.ADMIN and not entry.disabled]
    assert len(admins) == 1
