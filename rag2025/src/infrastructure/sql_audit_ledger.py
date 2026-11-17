"""Sync PostgreSQL audit ledger over ``audit_events`` (same surface as AuditLedger).

Interoperates with ``sql_audit_log.append_audit_event`` (async writer) and with
the migrator's legacy rows: rows migrated from the file ledger carry
``request_id == LEGACY_REQUEST_ID`` and are checked by linkage only.
"""

from __future__ import annotations

# Marker the migrator writes on every row it loads from the file ledger.
# The file-ledger hash format differs from the SQL chain and migrated event
# fields were partly lost, so verify() checks only linkage for these rows.
LEGACY_REQUEST_ID = "legacy:file-ledger"
