"""PostgreSQL engine, session factory, and the schema-revision write freeze.

``assert_schema_head`` compares ``alembic_version`` with the head revision. On
mismatch it engages ``WRITE_FREEZE`` (reason ``schema_revision``): admin write
routes answer 503 ``WRITES_FROZEN`` and ``/health/ready`` answers 503, while
public chat keeps working.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

RAG2025_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = RAG2025_ROOT / "alembic.ini"
SCHEMA_REVISION_REASON = "schema_revision"


class WriteFreeze:
    """Process-wide write freeze. Read by the admin write guard and /health/ready."""

    def __init__(self) -> None:
        self.active = False
        self.reason: Optional[str] = None
        self.detail: dict[str, Any] = {}

    def engage(self, reason: str, detail: Optional[dict[str, Any]] = None) -> None:
        self.active = True
        self.reason = reason
        self.detail = dict(detail or {})

    def release(self) -> None:
        self.active = False
        self.reason = None
        self.detail = {}


WRITE_FREEZE = WriteFreeze()


def database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    # Normalize Neon connection strings for asyncpg:
    #   postgresql:// -> postgresql+asyncpg://
    #   sslmode=require -> ssl=require  (asyncpg uses 'ssl', not 'sslmode')
    #   channel_binding=require -> dropped  (not supported by asyncpg)
    url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    url = url.replace("sslmode=require", "ssl=require")
    url = url.replace("&channel_binding=require", "").replace("?channel_binding=require", "?")
    # Clean up a stray '?' if all query params were removed
    if url.endswith("?"):
        url = url[:-1]
    return url


def head_revision() -> str:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(RAG2025_ROOT / "alembic"))
    return ScriptDirectory.from_config(cfg).get_current_head()


def create_db_engine(url: Optional[str] = None) -> AsyncEngine:
    # NullPool: one connection per use, so the engine is safe across the event
    # loops that TestClient and uvicorn create. Pooling can be added later.
    return create_async_engine(url or database_url(), poolclass=NullPool)


@asynccontextmanager
async def db_engine_lifespan(app):
    """Share one NullPool engine on app.state.db_engine. No DATABASE_URL -> None, no crash."""
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        app.state.db_engine = None
        yield
        return
    engine = create_db_engine(url)
    app.state.db_engine = engine
    try:
        yield
    finally:
        await engine.dispose()
        app.state.db_engine = None


async def active_release_id(engine: AsyncEngine, *, strict: bool = False) -> Optional[str]:
    """release_id of the active release, or None when none is active.

    Default (``strict=False``): an unreachable DB also yields None. ``strict=True``
    re-raises, so a readiness probe can tell "no active release" from "DB down".
    """
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT release_id FROM releases WHERE status = 'active'"))
            return result.scalar_one_or_none()
    except Exception:
        if strict:
            raise
        return None


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)


async def current_revision(engine: AsyncEngine) -> Optional[str]:
    """Return alembic_version.version_num, or None when the table is missing."""
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version_num FROM alembic_version"))
            return result.scalar_one_or_none()
    except Exception:
        return None


async def assert_schema_head(engine: AsyncEngine, expected_head: Optional[str] = None) -> bool:
    """True when the DB is at head. Otherwise engage WRITE_FREEZE and return False."""
    expected = expected_head or head_revision()
    current = await current_revision(engine)
    if current == expected:
        if WRITE_FREEZE.reason == SCHEMA_REVISION_REASON:
            WRITE_FREEZE.release()
        return True
    WRITE_FREEZE.engage(SCHEMA_REVISION_REASON, {"expected": expected, "current": current})
    return False
