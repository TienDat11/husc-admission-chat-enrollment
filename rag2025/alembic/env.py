"""Alembic environment for the rag2025 PostgreSQL schema.

The URL comes from DATABASE_URL (asyncpg driver, e.g.
postgresql+asyncpg://rag@localhost:5439/husc). Migrations are hand-written DDL,
so there is no ORM metadata to autogenerate from (target_metadata is None).
"""

from __future__ import annotations

import asyncio
import os

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine


def _database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL must be set to run alembic migrations")
    # Normalize Neon connection strings for asyncpg:
    #   postgresql:// -> postgresql+asyncpg://
    #   sslmode=require -> ssl=require
    #   channel_binding=require -> dropped  (not supported by asyncpg)
    url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    url = url.replace("sslmode=require", "ssl=require")
    url = url.replace("&channel_binding=require", "").replace("?channel_binding=require", "?")
    if url.endswith("?"):
        url = url[:-1]
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=None)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async_migrations() -> None:
    engine = create_async_engine(_database_url())
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_do_run_migrations)
            await connection.commit()
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(_run_async_migrations())
