"""Seed the browser e2e stack. Idempotent: run before every backend start.

Creates one admin identity per role (password from E2E_PASSWORD), a probe
identity for the lockout test, and one active release so the overview and
ops screens have data. Needs STORE_BACKEND=postgres and DATABASE_URL.
"""

from __future__ import annotations

import os
import uuid

import psycopg

from scripts.migrate_file_stores import sync_database_url
from src.domain.admin_identity import Role
from src.infrastructure.identity_store import IdentityEntry, hash_password
from src.infrastructure.sql_identity_repo import SqlIdentityRepository
from src.infrastructure.sql_record_store import _conninfo

E2E_PASSWORD = os.environ.get("E2E_PASSWORD", "e2e-password-123")
LOCKOUT_PROBE = "lockout-probe"
RELEASE_ID = "e2e-release"

ROLE_ACTORS: dict[Role, str] = {
    Role.VIEWER: "e2e-viewer",
    Role.EDITOR: "e2e-editor",
    Role.REVIEWER: "e2e-reviewer",
    Role.PUBLISHER: "e2e-publisher",
    Role.REVIEWER_PUBLISHER: "e2e-reviewer-publisher",
    Role.ADMIN: "e2e-admin",
    Role.OPS: "e2e-ops",
}


def _subject(actor_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"e2e:{actor_id}"))


def _entries() -> list[IdentityEntry]:
    password_hash = hash_password(E2E_PASSWORD)
    actors = [(actor, role) for role, actor in ROLE_ACTORS.items()]
    actors.append((LOCKOUT_PROBE, Role.VIEWER))
    return [
        IdentityEntry(
            subject_id=_subject(actor),
            actor_id=actor,
            role=role,
            password_hash=password_hash,
        )
        for actor, role in actors
    ]


def _seed_release(conninfo: str) -> None:
    with psycopg.connect(conninfo, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO releases (release_id, corpus_stamp, lancedb_table, manifest, status, "
            "created_by, activated_by, activated_at) "
            "SELECT %s, '{}'::jsonb, 'e2e', '{}'::jsonb, 'active', 'e2e-seed', 'e2e-seed', now() "
            "WHERE NOT EXISTS (SELECT 1 FROM releases WHERE status = 'active') "
            "ON CONFLICT (release_id) DO NOTHING",
            (RELEASE_ID,),
        )


def main() -> None:
    url = sync_database_url(os.environ["DATABASE_URL"])
    SqlIdentityRepository(url).seed(_entries())
    _seed_release(_conninfo(url))
    print(f"e2e seed ok: {len(ROLE_ACTORS) + 1} identities, release {RELEASE_ID}")


if __name__ == "__main__":
    main()
