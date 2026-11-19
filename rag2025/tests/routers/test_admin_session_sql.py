"""U10 - password login, lockout and logout on the PostgreSQL identity tables.

Skipped without DATABASE_URL (see ``tests.pg_fixtures``).
"""
from __future__ import annotations

from tests.pg_fixtures import ADMIN_TOKEN, fresh_db, sql_admin_client  # noqa: F401

ADMIN_HEADERS = {"X-Admin-Token": ADMIN_TOKEN}
PASSWORD = "correct-horse-1"


def _create_editor(client, actor_id: str) -> None:
    resp = client.post(
        "/admin/identities",
        json={"actor_id": actor_id, "role": "editor", "password": PASSWORD, "reason": "U10 test"},
        headers=ADMIN_HEADERS,
    )
    assert resp.status_code == 200, resp.text


def _login(client, actor_id: str, password: str):
    return client.post("/admin/login", json={"actor_id": actor_id, "password": password})


def test_five_wrong_passwords_lock_the_right_one_out_and_the_lock_survives_restart(fresh_db, tmp_path, monkeypatch):  # noqa: F811
    with sql_admin_client(fresh_db, tmp_path, monkeypatch) as client:
        _create_editor(client, "cb.bien-tap-2")
        for _ in range(5):
            assert _login(client, "cb.bien-tap-2", "wrong-password").status_code == 401

        locked = _login(client, "cb.bien-tap-2", PASSWORD)
        assert locked.status_code == 423
        assert "ACCOUNT_LOCKED" in locked.text
        assert int(locked.headers["Retry-After"]) > 0

    with sql_admin_client(fresh_db, tmp_path, monkeypatch) as restarted:
        still = _login(restarted, "cb.bien-tap-2", PASSWORD)
        assert still.status_code == 423
        assert "Retry-After" in still.headers


def test_four_wrong_passwords_do_not_lock_the_account(fresh_db, tmp_path, monkeypatch):  # noqa: F811
    with sql_admin_client(fresh_db, tmp_path, monkeypatch) as client:
        _create_editor(client, "cb.bien-tap-2")
        for _ in range(4):
            assert _login(client, "cb.bien-tap-2", "wrong-password").status_code == 401

        ok = _login(client, "cb.bien-tap-2", PASSWORD)
        assert ok.status_code == 200
        assert "admin_session" in ok.cookies


def test_unknown_actor_failures_lock_in_memory_with_the_same_refusal_shape(fresh_db, tmp_path, monkeypatch):  # noqa: F811
    with sql_admin_client(fresh_db, tmp_path, monkeypatch) as client:
        for _ in range(5):
            refused = _login(client, "nobody-here", "wrong-password")
            assert refused.status_code == 401
            assert "invalid credentials" in refused.text

        locked = _login(client, "nobody-here", "wrong-password")
        assert locked.status_code == 423
        assert "Retry-After" in locked.headers


def test_disabled_account_cannot_log_in_even_with_the_right_password(fresh_db, tmp_path, monkeypatch):  # noqa: F811
    with sql_admin_client(fresh_db, tmp_path, monkeypatch) as client:
        _create_editor(client, "cb.bien-tap-2")
        rows = client.get("/admin/identities", headers=ADMIN_HEADERS).json()
        subject = next(row["subject_id"] for row in rows if row["actor_id"] == "cb.bien-tap-2")
        patched = client.patch(
            f"/admin/identities/{subject}",
            json={"disabled": True, "reason": "U10 test"},
            headers=ADMIN_HEADERS,
        )
        assert patched.status_code == 200, patched.text

        refused = _login(client, "cb.bien-tap-2", PASSWORD)
        assert refused.status_code == 403
        assert "ACCOUNT_DISABLED" in refused.text


def test_logout_revokes_the_session_it_was_issued_for(fresh_db, tmp_path, monkeypatch):  # noqa: F811
    with sql_admin_client(fresh_db, tmp_path, monkeypatch) as client:
        _create_editor(client, "cb.bien-tap-2")
        login = _login(client, "cb.bien-tap-2", PASSWORD)
        assert login.status_code == 200
        cookie = {"Cookie": f"admin_session={login.cookies['admin_session']}"}

        assert client.post("/admin/logout", headers=cookie).status_code == 200

        resp = client.get("/admin/identities", headers=cookie)
        assert resp.status_code == 401
        assert "SESSION_EXPIRED" in resp.text
