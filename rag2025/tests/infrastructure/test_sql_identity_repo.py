"""U10 - SqlIdentityRepository invariants on the PostgreSQL identity tables.

Skipped without DATABASE_URL (see ``tests.pg_fixtures``).
"""
from __future__ import annotations

import threading
import uuid

from src.domain.admin_identity import Role
from src.infrastructure.identity_store import IdentityEntry, hash_password
from src.infrastructure.sql_audit_ledger import SqlAuditLedger
from src.infrastructure.sql_identity_repo import LastAdminError, PermissionAudit, SqlIdentityRepository
from tests.pg_fixtures import ADMIN_ACTOR, ADMIN_TOKEN, fresh_db, sql_admin_client  # noqa: F401

PASSWORD = "correct-horse-1"


def test_concurrent_demotions_of_the_two_remaining_admins_leave_one(fresh_db, tmp_path, monkeypatch):  # noqa: F811
    with sql_admin_client(fresh_db, tmp_path, monkeypatch) as client:
        rows = client.get("/admin/identities", headers={"X-Admin-Token": ADMIN_TOKEN}).json()
        first_subject = next(row["subject_id"] for row in rows if row["actor_id"] == ADMIN_ACTOR)

    repo = SqlIdentityRepository(fresh_db)
    second = repo.add(
        IdentityEntry(
            subject_id=str(uuid.uuid4()),
            actor_id="cb.quan-tri-2",
            role=Role.ADMIN,
            password_hash=hash_password(PASSWORD),
        ),
        created_by="test",
        audit=PermissionAudit(actor_id=ADMIN_ACTOR, actor_role="admin", reason="test setup"),
    )

    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def demote(subject_id: str) -> None:
        barrier.wait()
        try:
            repo.update(
                subject_id,
                role=Role.EDITOR,
                disabled=None,
                password=None,
                audit=PermissionAudit(actor_id=ADMIN_ACTOR, actor_role="admin", reason="concurrency test"),
            )
            outcomes.append("ok")
        except LastAdminError:
            outcomes.append("last-admin")

    threads = [threading.Thread(target=demote, args=(s,)) for s in (first_subject, second.subject_id)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sorted(outcomes) == ["last-admin", "ok"]
    admins = [entry for entry in repo.entries() if entry.role is Role.ADMIN and not entry.disabled]
    assert len(admins) == 1


def test_an_identity_write_and_its_audit_row_commit_together(fresh_db):
    repo = SqlIdentityRepository(fresh_db)
    audit = PermissionAudit(actor_id=ADMIN_ACTOR, actor_role="admin", reason="atomicity check")
    before = len(SqlAuditLedger(fresh_db).find(action="permission_change"))
    subject = str(uuid.uuid4())
    repo.add(
        IdentityEntry(
            subject_id=subject,
            actor_id="cb.atomic",
            role=Role.EDITOR,
            password_hash=hash_password(PASSWORD),
        ),
        created_by="test",
        audit=audit,
    )
    repo.update(subject, disabled=True, audit=audit)
    events = SqlAuditLedger(fresh_db).find(action="permission_change")
    assert len(events) - before == 2
