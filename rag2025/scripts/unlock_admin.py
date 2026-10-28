#!/usr/bin/env python3
"""Clear a login-locked admin actor — T08 rollback path.

The in-process lockout (5 wrong passwords → 15-minute lock) lives in
``src/routers/admin_session.py`` and is cleared when the API process restarts.
This script is the "I cannot wait 15 minutes" path:

1. It records a permission-change audit event (the unlock decision) into the
   canonical ledger so the action is traceable.
2. It prints the command to restart the API (the lock itself disappears with
   the process).

Usage:
    python -m scripts.unlock_admin --actor-id <actor_id> --data-root <path>
    python -m scripts.unlock_admin --actor-id <actor_id>  # reads CANONICAL_STORE_ROOT

If the lockout was caused by a *forgotten* password (not a brute-force), also
rotate the credential via the API (PATCH /admin/identities/<subject_id>) so
any in-flight session cookie becomes stale on the next read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from src.domain.audit import AuditAction, AuditEvent, TargetKind
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.domain.audit import AuditAction, AuditEvent, TargetKind


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record an audit event for a manual admin lockout unlock"
    )
    parser.add_argument("--actor-id", required=True, help="The actor_id to unlock")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Canonical store root (default: CANONICAL_STORE_ROOT env var)",
    )
    parser.add_argument(
        "--reason",
        default="manual unlock via CLI (T08)",
        help="Reason recorded in the audit event",
    )
    args = parser.parse_args()

    data_root = Path(
        args.data_root or os.environ.get("CANONICAL_STORE_ROOT", "")
    )
    if not data_root or not data_root.exists():
        print(
            "error: --data-root not given and CANONICAL_STORE_ROOT is unset "
            "or the path does not exist.",
            file=sys.stderr,
        )
        return 1

    # Build a valid AuditEvent. PERMISSION_CHANGE requires a non-blank reason.
    event = AuditEvent(
        action=AuditAction.PERMISSION_CHANGE,
        actor_id="cli-admin",
        role="admin",
        target=args.actor_id,
        target_kind=TargetKind.SUBJECT,
        timestamp=datetime.now(timezone.utc),
        reason=f"lockout_reset: {args.reason}",
    )

    # Append to the ledger in the same format the API writes.
    audit_file = data_root / "audit" / "ledger.jsonl"
    audit_file.parent.mkdir(parents=True, exist_ok=True)

    last_seq = 0
    if audit_file.exists():
        lines = [
            line for line in audit_file.read_text().splitlines() if line.strip()
        ]
        if lines:
            last_seq = json.loads(lines[-1]).get("seq", 0)

    entry = {"seq": last_seq + 1, "event": event.to_dict()}
    with open(audit_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"audit event written to {audit_file}")
    print(
        "\nThe in-process lockout is cleared by restarting the API:\n"
        "  uvicorn src.main:app --reload  (dev)\n"
        "  or restart the Docker container (production)\n"
        "Restart now, or wait for the 15-minute window to expire."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
