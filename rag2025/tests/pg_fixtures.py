"""Shared PostgreSQL fixtures for the A3 store tests.

``fresh_db`` yields the psycopg-form URL of a brand-new database migrated to
head. It skips without DATABASE_URL, so pure tests stay runnable without a
database. Import it with ``from tests.pg_fixtures import fresh_db``.
"""
from __future__ import annotations

import os
import uuid

import psycopg
import pytest
from sqlalchemy.engine import make_url

import scripts.migrate_file_stores as migrate
from src.infrastructure.sql_record_store import _conninfo
from tests.infrastructure.conftest import run_alembic


@pytest.fixture
def fresh_db():
    """Yields a psycopg-form sync URL of a brand-new migrated database."""
    base = os.getenv("DATABASE_URL", "").strip()
    if not base:
        pytest.skip("DATABASE_URL not set; PostgreSQL apply tests skipped")
    admin_url = _conninfo(make_url(migrate.sync_database_url(base)).set(database="postgres").render_as_string(hide_password=False))
    try:
        admin = psycopg.connect(admin_url, autocommit=True)
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL not reachable at DATABASE_URL: {exc}")
    name = "a3apply_" + uuid.uuid4().hex[:12]
    with admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    target = make_url(base).set(database=name)
    run_alembic(target.render_as_string(hide_password=False), "upgrade", "head")
    try:
        yield migrate.sync_database_url(target.render_as_string(hide_password=False))
    finally:
        with psycopg.connect(admin_url, autocommit=True) as c:
            c.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
