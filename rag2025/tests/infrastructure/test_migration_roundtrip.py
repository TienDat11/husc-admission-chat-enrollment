"""Alembic baseline (0001) must round-trip: downgrade base -> upgrade head, twice.

Needs a live PostgreSQL (DATABASE_URL). This drops and recreates the 23 baseline
tables in the URL's database, so point it only at the dev database.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

async def _schema_facts(url: str) -> dict:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            tables = (
                await conn.execute(
                    text(
                        "select count(*) from information_schema.tables "
                        "where table_schema='public' and table_name<>'alembic_version'"
                    )
                )
            ).scalar_one()
            indexes = {
                row[0]
                for row in await conn.execute(
                    text("select indexname from pg_indexes where schemaname='public'")
                )
            }
            return {"tables": tables, "indexes": indexes}
    finally:
        await engine.dispose()


def test_baseline_roundtrip_and_required_indexes(database_url: str, alembic_run) -> None:
    alembic_run("downgrade", "base")
    alembic_run("upgrade", "head")
    alembic_run("downgrade", "base")
    alembic_run("upgrade", "head")

    facts = asyncio.run(_schema_facts(database_url))
    assert facts["tables"] >= 23, facts
    assert "uq_current_coordinate" in facts["indexes"], facts["indexes"]
    assert "uq_one_active_release" in facts["indexes"], facts["indexes"]
