"""active_release_id: default swallows DB errors, strict=True surfaces them.

Health readiness needs strict=True so an unreachable DB is a 503, not a 200
reporting ``served_release_id: null``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure.db import active_release_id  # noqa: E402

# Nothing listens on port 1; the connect fails fast without a real database.
UNREACHABLE_URL = "postgresql+asyncpg://nobody@127.0.0.1:1/none"


def _run_with_engine(coro_factory):
    async def runner():
        engine = create_async_engine(UNREACHABLE_URL, poolclass=NullPool)
        try:
            return await coro_factory(engine)
        finally:
            await engine.dispose()

    return asyncio.run(runner())


def test_default_returns_none_when_db_unreachable():
    assert _run_with_engine(lambda engine: active_release_id(engine)) is None


def test_strict_raises_when_db_unreachable():
    with pytest.raises(Exception):
        _run_with_engine(lambda engine: active_release_id(engine, strict=True))
