"""T02: the index-jobs screen reports the real runner state.

AC(i)  : a runner whose ``.started`` is True makes the rebuild step runnable.
AC(ii) : no ``NO_RUNNER_REASON`` text anywhere; ``rebuild.reason`` is None.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.routers.admin import router as admin_router  # noqa: E402
from src.services.release_status import IndexJobsService, RetryStep  # noqa: E402


class _FakeJob:
    """A live job whose retry list carries the rebuild step under test."""

    state = MagicMock(value="failed")
    saved = True
    serving = False
    staged_dir = None
    error = RuntimeError("embedder hết quota")


# ── unit level: IndexJobsService directly ───────────────────────────────────


def test_rebuild_step_is_runnable_when_runner_is_live(tmp_path: Path) -> None:
    """AC(i): runner.started=True → rebuild.runnable is True."""
    runner = MagicMock()
    runner.started = True

    service = IndexJobsService(
        corpus_dir=tmp_path / "corpus",
        job=_FakeJob(),
        runner=runner,
    )
    steps = {s.step: s for s in service.status().jobs[0].retry}
    rebuild = steps["rebuild"]
    assert isinstance(rebuild, RetryStep)
    assert rebuild.runnable is True


def test_rebuild_reason_is_none_when_runner_is_live(tmp_path: Path) -> None:
    """AC(ii): no 'chưa có runner' / NO_RUNNER_REASON text; reason is None."""
    runner = MagicMock()
    runner.started = True

    service = IndexJobsService(
        corpus_dir=tmp_path / "corpus",
        job=_FakeJob(),
        runner=runner,
    )
    rebuild = {s.step: s for s in service.status().jobs[0].retry}["rebuild"]
    assert rebuild.reason is None
    serialised = str(rebuild)
    assert "chưa có runner" not in serialised
    assert "B15-1" not in serialised
    assert "NO_RUNNER" not in serialised


def test_rebuild_step_is_not_runnable_when_runner_is_absent(tmp_path: Path) -> None:
    """No runner → rebuild.runnable is False (no regression)."""
    service = IndexJobsService(
        corpus_dir=tmp_path / "corpus",
        job=_FakeJob(),
    )
    rebuild = {s.step: s for s in service.status().jobs[0].retry}["rebuild"]
    assert rebuild.runnable is False


def test_rebuild_step_is_not_runnable_when_runner_not_started(tmp_path: Path) -> None:
    """Runner present but .started is False → rebuild.runnable is False."""
    runner = MagicMock()
    runner.started = False

    service = IndexJobsService(
        corpus_dir=tmp_path / "corpus",
        job=_FakeJob(),
        runner=runner,
    )
    rebuild = {s.step: s for s in service.status().jobs[0].retry}["rebuild"]
    assert rebuild.runnable is False


# ── HTTP level: GET /admin/index-jobs with a real app ────────────────────────


@pytest.fixture()
def app_with_runner(tmp_path: Path, monkeypatch):
    """A FastAPI app with the admin router, a live job, and a started runner."""
    from src.services import release_status

    corpus_dir = tmp_path / "canonical_corpus"
    monkeypatch.setattr(release_status, "DEFAULT_CORPUS_DIR", corpus_dir)

    app = FastAPI()
    app.include_router(admin_router)

    runner = MagicMock()
    runner.started = True
    app.state.reindex_runner = runner
    app.state.reindex_job = _FakeJob()
    app.state.reindex_outbox = None

    return app


def test_http_index_jobs_rebuild_runnable_when_runner_live(
    app_with_runner, monkeypatch
) -> None:
    """AC(i) over HTTP: runner started → rebuild step runnable=True.

    Also asserts the entire JSON response contains no obsolete text (AC(ii)).
    """
    # Bypass auth: monkeypatch the token dependency
    from src.routers.admin_auth import current_identity  # noqa: F401

    client = TestClient(app_with_runner)

    # The admin routes require a valid credential; use the ACTORS pattern.
    import os
    from config.settings import RAGSettings

    actors = json.dumps(
        [{"token": "tok-admin", "actor_id": "a-admin", "role": "admin"}]
    )
    monkeypatch.setenv("ADMIN_IDENTITIES", actors)

    # Force the app to re-read the identities (they are read at startup)
    # Build a fresh client so the dependency picks up the env var.
    client = TestClient(app_with_runner)
    resp = client.get(
        "/admin/index-jobs",
        headers={"X-Admin-Token": "tok-admin"},
    )
    assert resp.status_code == 200, resp.text

    body = resp.json()
    jobs = body["jobs"]
    assert jobs, f"expected at least one job row: {body}"

    rebuild = next((s for s in jobs[0]["retry"] if s["step"] == "rebuild"), None)
    assert rebuild is not None, f"no rebuild step in retry: {jobs[0]['retry']}"
    assert rebuild["runnable"] is True

    # AC(ii): the entire response must not contain the obsolete reason text.
    full = json.dumps(body)
    assert "chưa có runner" not in full
    assert "B15-1" not in full
    assert "NO_RUNNER" not in full
    assert rebuild["reason"] is None
