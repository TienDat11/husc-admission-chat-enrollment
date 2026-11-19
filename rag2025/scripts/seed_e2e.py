"""Seed the browser e2e stack. Idempotent: run before every backend start.

Creates one admin identity per role (password from E2E_PASSWORD), a probe
identity for the lockout test, one active release so the overview and
ops screens have data, plus deterministic unanswered questions (one overdue)
and chat feedback rows so the Frame 7 e2e spec has stable targets.
Needs STORE_BACKEND=postgres and DATABASE_URL.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

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


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _seed_unanswered(conninfo: str) -> None:
    """Insert deterministic unanswered questions, one of them overdue (>48h)."""
    now = _now()
    overdue_at = now - timedelta(hours=72)
    rows = [
        {
            "trace_id": "e2e-unanswered-open",
            "question": "Học phí ngành Hán Nôm năm 2026 bao nhiêu?",
            "normalized_hash": "e2e-han-nom-hoc-phi-2026",
            "occurrence_count": 1,
            "asked_at": now.isoformat(),
            "route": "/v2/ask",
            "status_code": "NO_INFO",
            "resolved_at": None,
            "resolved_by_record_id": None,
            "out_of_scope": False,
            "sla_due_at": (now + timedelta(hours=48)).isoformat(),
        },
        {
            "trace_id": "e2e-unanswered-overdue",
            "question": "Ngày thi tuyển sinh ngành Công nghệ thông tin 2026",
            "normalized_hash": "e2e-cntt-ngay-thi-2026",
            "occurrence_count": 1,
            "asked_at": overdue_at.isoformat(),
            "route": "/v2/ask",
            "status_code": "OUT_OF_DOMAIN",
            "resolved_at": None,
            "resolved_by_record_id": None,
            "out_of_scope": False,
            "sla_due_at": (overdue_at + timedelta(hours=48)).isoformat(),
        },
    ]
    with psycopg.connect(conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    """
                    INSERT INTO unanswered_questions
                    (trace_id, question, normalized_hash, occurrence_count, asked_at,
                     route, status_code, resolved_at, resolved_by_record_id,
                     out_of_scope, sla_due_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (trace_id) DO NOTHING
                    """,
                    (
                        row["trace_id"], row["question"], row["normalized_hash"],
                        row["occurrence_count"], row["asked_at"], row["route"],
                        row["status_code"], row["resolved_at"], row["resolved_by_record_id"],
                        row["out_of_scope"], row["sla_due_at"],
                    ),
                )


def _seed_feedback(conninfo: str) -> None:
    """Insert a couple of chat_feedback rows (one with a redacted phone number)."""
    now = _now()
    rows = [
        {
            "feedback_id": str(uuid.uuid4()),
            "trace_id": "e2e-feedback-ok",
            "helpful": True,
            "reason": "Giúp ích rất nhiều, cảm ơn.",
            "answered_at": now.isoformat(),
            "linked_item": False,
            "release_id": None,
            "created_at": now.isoformat(),
        },
        {
            "feedback_id": str(uuid.uuid4()),
            "trace_id": "e2e-feedback-phone",
            "helpful": False,
            "reason": "Sai rồi, liên hệ 0905123456 để xử lý.",
            "answered_at": now.isoformat(),
            "linked_item": False,
            "release_id": None,
            "created_at": now.isoformat(),
        },
    ]
    with psycopg.connect(conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    """
                    INSERT INTO chat_feedback
                    (feedback_id, trace_id, helpful, reason, answered_at,
                     linked_item, release_id, created_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (feedback_id) DO NOTHING
                    """,
                    (
                        row["feedback_id"], row["trace_id"], row["helpful"],
                        row["reason"], row["answered_at"], row["linked_item"],
                        row["release_id"], row["created_at"],
                    ),
                )


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
    _seed_unanswered(_conninfo(url))
    _seed_feedback(_conninfo(url))
    print(
        f"e2e seed ok: {len(ROLE_ACTORS) + 1} identities, release {RELEASE_ID}, "
        "unanswered questions + feedback"
    )


if __name__ == "__main__":
    main()
