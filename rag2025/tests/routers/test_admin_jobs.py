"""Acceptance for the ingestion job surface (plan §PL-5, §PL-7.1, unit U2).

Three layers:
1. The grant table matches §PL-7.1 exactly (no test-only leniency).
2. HTTP refusal per role: a denied role never reaches a handler.
3. State transitions against PostgreSQL, when ``DATABASE_URL`` is set.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.admin_identity import AdminAction, Role, role_allows  # noqa: E402
from src.infrastructure.db import create_db_engine  # noqa: E402
from src.routers.admin_jobs import router as jobs_router  # noqa: E402

OPS_TOKEN = "shared-ops-token"
TOKENS = {
    "viewer": "tok-viewer",
    "editor": "tok-editor",
    "reviewer": "tok-reviewer",
    "publisher": "tok-publisher",
    "reviewer_publisher": "tok-reviewer-publisher",
    "admin": "tok-admin",
}

# §PL-7.1 lines 607-608, transcribed. A change here is a change to the plan.
EXPECTED_JOBS_READ = {
    Role.EDITOR,
    Role.REVIEWER,
    Role.PUBLISHER,
    Role.REVIEWER_PUBLISHER,
    Role.ADMIN,
    Role.OPS,
}
EXPECTED_JOBS_MANAGE = {Role.ADMIN, Role.OPS}


@pytest.mark.parametrize("role", list(Role))
def test_jobs_read_is_granted_exactly_as_plan_7_1(role):
    assert role_allows(role, AdminAction.JOBS_READ) is (role in EXPECTED_JOBS_READ)


@pytest.mark.parametrize("role", list(Role))
def test_jobs_manage_is_granted_exactly_as_plan_7_1(role):
    assert role_allows(role, AdminAction.JOBS_MANAGE) is (role in EXPECTED_JOBS_MANAGE)


def _configure(monkeypatch) -> None:
    entries = [
        {"token": TOKENS[role], "actor_id": f"cb.{role}", "role": role} for role in TOKENS
    ]
    monkeypatch.setenv("ADMIN_IDENTITIES", json.dumps(entries))
    monkeypatch.setenv("ADMIN_API_TOKEN", OPS_TOKEN)
    monkeypatch.delenv("ADMIN_FOUR_EYES", raising=False)


@pytest.fixture
def app(monkeypatch) -> FastAPI:
    _configure(monkeypatch)
    application = FastAPI()
    application.include_router(jobs_router)
    return application


@pytest.fixture
def client(app: FastAPI):
    with TestClient(app) as test_client:
        yield test_client


def _headers(token: str) -> dict[str, str]:
    return {"X-Admin-Token": token}


def test_viewer_cannot_list_jobs(client):
    response = client.get("/admin/jobs", headers=_headers(TOKENS["viewer"]))
    assert response.status_code == 403


def test_editor_may_read_jobs_but_not_manage_them(client):
    job = str(uuid.uuid4())
    # Authorised, so the request reaches the handler; no job store is wired here.
    assert client.get("/admin/jobs", headers=_headers(TOKENS["editor"])).status_code == 503
    assert client.post(f"/admin/jobs/{job}/retry", headers=_headers(TOKENS["editor"])).status_code == 403
    assert client.post(f"/admin/jobs/{job}/cancel", headers=_headers(TOKENS["editor"])).status_code == 403


def test_publisher_and_reviewer_cannot_manage_jobs(client):
    job = str(uuid.uuid4())
    for role in ("publisher", "reviewer"):
        assert client.post(f"/admin/jobs/{job}/cancel", headers=_headers(TOKENS[role])).status_code == 403


def test_admin_and_ops_may_manage_jobs(client):
    job = str(uuid.uuid4())
    # Authorised but unwired: 503 proves the grant passed and the store is absent.
    assert client.post(f"/admin/jobs/{job}/retry", headers=_headers(TOKENS["admin"])).status_code == 503
    assert client.post(f"/admin/jobs/{job}/cancel", headers=_headers(OPS_TOKEN)).status_code == 503


def test_workers_endpoint_requires_jobs_read(client):
    assert client.get("/admin/ops/workers", headers=_headers(TOKENS["viewer"])).status_code == 403
    assert client.get("/admin/ops/workers", headers=_headers(OPS_TOKEN)).status_code == 503


def test_unknown_credential_is_refused(client):
    assert client.get("/admin/jobs", headers=_headers("not-registered")).status_code in (401, 403)


# --- PostgreSQL-backed transitions -------------------------------------------

DATABASE_URL = os.environ.get("DATABASE_URL", "")
needs_pg = pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set (needs migrated PostgreSQL)")


@pytest.fixture
def pg_client(app: FastAPI):
    # The app's engine lives on the TestClient's event loop.
    app.state.db_engine = create_db_engine(DATABASE_URL)
    with TestClient(app) as test_client:
        yield test_client
    app.state.db_engine = None


def _run(coro):
    return asyncio.run(coro)


async def _with_private_engine(work):
    # A private NullPool engine per helper call, so no pooled connection crosses
    # from one event loop to another.
    engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
    try:
        return await work(engine)
    finally:
        await engine.dispose()


async def _insert_job(state: str, lease_owner: str | None = None) -> str:
    job_id = str(uuid.uuid4())

    async def work(engine):
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    """
INSERT INTO ingestion_jobs (job_id, kind, payload, state, idempotency_key, created_by,
                            lease_owner, lease_expires_at, heartbeat_at)
VALUES (CAST(:job AS uuid), 'parse', '{}'::jsonb, :state, :key, 'cb.test',
        :owner, CASE WHEN CAST(:owner AS text) IS NULL THEN NULL ELSE now() + interval '60 seconds' END,
        CASE WHEN CAST(:owner AS text) IS NULL THEN NULL ELSE now() END)
"""
                ),
                {"job": job_id, "state": state, "key": f"test-{job_id}", "owner": lease_owner},
            )

    await _with_private_engine(work)
    return job_id


async def _delete_job(job_id: str) -> None:
    async def work(engine):
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM ingestion_jobs WHERE job_id = CAST(:job AS uuid)"), {"job": job_id}
            )

    await _with_private_engine(work)


@needs_pg
def test_retry_moves_failed_terminal_to_queued(pg_client):
    job = _run(_insert_job("failed_terminal"))
    try:
        response = pg_client.post(f"/admin/jobs/{job}/retry", headers=_headers(TOKENS["admin"]))
        assert response.status_code == 200, response.text
        assert response.json()["state"] == "queued"
        assert response.json()["actor_id"] == "cb.admin"
    finally:
        _run(_delete_job(job))


@needs_pg
def test_retry_refuses_a_job_that_is_not_terminal(pg_client):
    job = _run(_insert_job("queued"))
    try:
        response = pg_client.post(f"/admin/jobs/{job}/retry", headers=_headers(TOKENS["admin"]))
        assert response.status_code == 409
    finally:
        _run(_delete_job(job))


@needs_pg
def test_cancel_idle_job_is_cancelled_outright(pg_client):
    job = _run(_insert_job("queued"))
    try:
        response = pg_client.post(f"/admin/jobs/{job}/cancel", headers=_headers(TOKENS["admin"]))
        assert response.status_code == 200, response.text
        assert response.json()["state"] == "cancelled"
    finally:
        _run(_delete_job(job))


@needs_pg
def test_cancel_leased_job_waits_for_worker_ack(pg_client):
    job = _run(_insert_job("claimed", lease_owner="worker-1"))
    try:
        response = pg_client.post(f"/admin/jobs/{job}/cancel", headers=_headers(TOKENS["admin"]))
        assert response.status_code == 200, response.text
        assert response.json()["state"] == "cancel_requested"
    finally:
        _run(_delete_job(job))


@needs_pg
def test_cancel_refuses_a_completed_job(pg_client):
    job = _run(_insert_job("completed"))
    try:
        response = pg_client.post(f"/admin/jobs/{job}/cancel", headers=_headers(TOKENS["admin"]))
        assert response.status_code == 409
    finally:
        _run(_delete_job(job))


@needs_pg
def test_get_unknown_job_is_404(pg_client):
    response = pg_client.get(f"/admin/jobs/{uuid.uuid4()}", headers=_headers(TOKENS["editor"]))
    assert response.status_code == 404
