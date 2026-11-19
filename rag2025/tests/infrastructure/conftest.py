"""Shared live-PostgreSQL fixtures for the U2 infrastructure tests.

Tests that need PostgreSQL skip with a clear reason when ``DATABASE_URL`` is
unset. When it is set, ``migrated_database_url`` brings the schema to head with
``alembic upgrade head`` (idempotent) before the test runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

RAG2025_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        pytest.skip("DATABASE_URL not set; live PostgreSQL test skipped")
    return url


def run_alembic(url: str, *args: str) -> None:
    env = {**os.environ, "DATABASE_URL": url}
    env["PYTHONPATH"] = os.pathsep.join([str(RAG2025_ROOT), str(RAG2025_ROOT / "src")])
    proc = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=RAG2025_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


@pytest.fixture(scope="session")
def migrated_database_url(database_url: str) -> str:
    run_alembic(database_url, "upgrade", "head")
    return database_url


@pytest.fixture
def alembic_run(database_url: str):
    """Callable that runs ``alembic <args>`` against DATABASE_URL."""
    return lambda *args: run_alembic(database_url, *args)
