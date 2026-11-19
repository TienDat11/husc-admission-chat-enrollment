"""Migrate legacy file stores into PostgreSQL rows (U3, Task 2a slice).

Loaders are pure functions: store root in, row dicts out. Only the real run
(apply_to_database) connects to PostgreSQL; nothing here writes to the legacy store
except --scrub-identities after a committed real run.

Deviation from the brief: a published record whose source_document_id is not in
source_documents aborts the real run (MigrationError) instead of nulling the FK, because
the DDL CHECK (status <> 'published' OR source_document_id IS NOT NULL) forbids NULL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.infrastructure.canonical_record_store import (
    _RECORD_DIR_PREFIX,
    _RECORD_DIR_SUFFIX,
    RECORDS_DIRNAME,
    RETRACTIONS_DIRNAME,
)
from src.infrastructure.source_document_registry import SOURCE_DOCUMENTS_DIRNAME, UPLOAD_ID_PREFIX
from src.infrastructure.sql_audit_ledger import LEGACY_REQUEST_ID

UNANCHORED_PREFIX = "unanchored:"
UNANCHORED_BLOCKER = "UNANCHORED_DRAFT"

# Top-level legacy keys that map to a canonical_records column or to
# structured_payload['values']. Every other top-level key is kept verbatim
# under structured_payload['legacy_extra'].
MAPPED_KEYS = frozenset(
    {
        "record_id",
        "version",
        "record_type",
        "admission_year",
        "admission_round",
        "status",
        "authority_level",
        "effective_from",
        "effective_to",
        "text",
        "values",
        "major_codes",
        "admission_methods",
        "source_document_id",
        "source_url",
        "source_page",
        "supersedes",
        "authored_by",
        "published_at",
    }
)

_MALFORMED = (KeyError, ValueError, TypeError, AttributeError)


class MigrationError(Exception):
    """A legacy file could not be read; the whole run must abort."""


@dataclass
class RecordsLoad:
    rows: list[dict] = field(default_factory=list)
    id_map: dict[str, str] = field(default_factory=dict)
    quarantined_rows: int = 0
    source_rows: int = 0


@dataclass(frozen=True)
class _Version:
    legacy: str
    version: int
    path: Path
    payload: dict


def _parse_legacy_id(record_id: str) -> tuple[str, str, int, str | None] | None:
    """(scope, subject, year, method token) from ``TYPE:scope:subject:year[:method]:tail``."""
    parts = record_id.split(":")
    if len(parts) not in (5, 6):
        return None
    scope, subject = parts[1], parts[2]
    if not scope or not subject:
        return None
    try:
        year = int(parts[3])
    except ValueError:
        return None
    method = parts[4] if len(parts) == 6 and parts[4] else None
    return scope, subject, year, method


def derive_coordinate_key(legacy_payload: dict) -> str | None:
    """PL-6.6 coordinate key for one legacy record version, or None if not derivable."""
    record_id = legacy_payload.get("record_id")
    record_type = legacy_payload.get("record_type")
    year = legacy_payload.get("admission_year")
    if not record_id or not record_type or year is None:
        return None
    parsed = _parse_legacy_id(record_id)
    if parsed is None:
        return None
    scope, subject, id_year, id_method = parsed
    if id_year != year:
        return None
    round_ = legacy_payload.get("admission_round") or "all"
    methods = list(legacy_payload.get("admission_methods") or [])
    if not methods and id_method:
        methods = [id_method]
    if methods:
        methods_hash = hashlib.sha256("|".join(sorted(methods)).encode("utf-8")).hexdigest()[:12]
    else:
        methods_hash = "all"
    return f"{record_type}:{scope}:{subject}:{year}:{round_}:{methods_hash}"


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise MigrationError(f"unreadable legacy file: {path} ({exc})") from exc


def _checked_key(version: _Version) -> str | None:
    """derive_coordinate_key, with malformed payload values raised as MigrationError."""
    try:
        return derive_coordinate_key(version.payload)
    except _MALFORMED as exc:
        raise MigrationError(f"malformed legacy record: {version.path} ({exc})") from exc


def _record_versions(records_dir: Path) -> list[_Version]:
    """Every record version under ``records_dir``, validated, sorted by (legacy_id, version)."""
    found: dict[tuple[str, int], _Version] = {}
    if not records_dir.is_dir():
        return []
    for directory in sorted(records_dir.iterdir()):
        if not (
            directory.is_dir()
            and directory.name.startswith(_RECORD_DIR_PREFIX)
            and directory.name.endswith(_RECORD_DIR_SUFFIX)
        ):
            continue
        for path in sorted(directory.glob("v*.json")):
            payload = _read_json(path)
            if not isinstance(payload, dict):
                raise MigrationError(f"malformed legacy record: {path} (not a JSON object)")
            legacy = payload.get("record_id")
            version = payload.get("version")
            if not isinstance(legacy, str) or not legacy:
                raise MigrationError(f"malformed legacy record: {path} (record_id)")
            if type(version) is not int or version < 1:
                raise MigrationError(f"malformed legacy record: {path} (version)")
            published_at = payload.get("published_at")
            if published_at is not None and not isinstance(published_at, str):
                raise MigrationError(f"malformed legacy record: {path} (published_at)")
            source_document_id = payload.get("source_document_id")
            if source_document_id is not None and not isinstance(source_document_id, str):
                raise MigrationError(f"malformed legacy record: {path} (source_document_id)")
            identity = (legacy, version)
            if identity in found:
                raise MigrationError(
                    f"duplicate legacy version {legacy}@{version}: "
                    f"{found[identity].path} and {path}"
                )
            found[identity] = _Version(legacy, version, path, payload)
    return [found[identity] for identity in sorted(found)]


def _add_blocker(structured: dict, name: str) -> None:
    """Append a blocker to any existing list rather than overwriting it."""
    blockers = list(structured.get("blockers") or [])
    if name not in blockers:
        blockers.append(name)
    structured["blockers"] = blockers


def load_records(root: str | Path) -> RecordsLoad:
    """Read every legacy record version under ``root/records`` into canonical_records rows.

    The decision is per legacy record_id: if any of its versions has a problem,
    every version of it is quarantined under ``unanchored:<legacy_id>``.
    """
    versions = _record_versions(Path(root) / RECORDS_DIRNAME)
    by_legacy: dict[str, list[_Version]] = {}
    for version in versions:
        by_legacy.setdefault(version.legacy, []).append(version)

    # The key of a legacy record comes from its lowest version, so every
    # version of that record lands on one coordinate.
    legacy_key: dict[str, str | None] = {}
    problem: dict[str, str] = {}
    for legacy, group in by_legacy.items():
        legacy_key[legacy] = _checked_key(group[0])
        for version in group:
            if _checked_key(version) is None:
                problem[legacy] = "coordinate"
            elif version.payload.get("status") == "published" and not (
                version.payload.get("source_document_id") or ""
            ).strip():
                problem[legacy] = "no_source"

    # Coordinate duplicates between different legacy records: one winner keeps it.
    groups: dict[str, list[str]] = {}
    for legacy in sorted(by_legacy):
        key = legacy_key[legacy]
        if legacy in problem or key is None:
            continue
        groups.setdefault(key, []).append(legacy)
    for key, legacies in groups.items():
        if len(legacies) < 2:
            continue
        published = [
            (legacy, max(
                (v.payload.get("published_at") or "")
                for v in by_legacy[legacy]
                if v.payload.get("status") == "published"
            ))
            for legacy in legacies
            if any(v.payload.get("status") == "published" for v in by_legacy[legacy])
        ]
        if published:
            # max() keeps the first maximal item, so ties go to the sorted-first legacy id.
            winner = max(published, key=lambda item: item[1])[0]
        else:
            winner = legacies[0]
        for legacy in legacies:
            if legacy != winner:
                problem[legacy] = "duplicate"

    out = RecordsLoad(source_rows=len(versions))
    for version in versions:
        legacy = version.legacy
        payload = version.payload
        key = legacy_key[legacy]
        parsed = _parse_legacy_id(legacy)
        scope, subject = (parsed[0], parsed[1]) if parsed else ("", "")
        structured: dict = {
            "legacy_record_id": legacy,
            "values": payload.get("values") or {},
            "legacy_extra": {k: v for k, v in payload.items() if k not in MAPPED_KEYS},
        }
        try:
            if legacy in problem or key is None:
                record_id = f"{UNANCHORED_PREFIX}{legacy}"
                status = "draft"
                _add_blocker(structured, UNANCHORED_BLOCKER)
                methods_hash = "all"
                column_type = payload.get("record_type")
                column_year = payload.get("admission_year")
                column_round = payload.get("admission_round") or "all"
                out.quarantined_rows += 1
            else:
                # Key columns come from the emitted key, so every version of this
                # legacy id agrees with the coordinate it sits on.
                record_id = key
                status = payload.get("status")
                head, _, methods_hash = key.rpartition(":")
                record_type, scope, subject, year, round_ = head.split(":", 4)
                column_type, column_year, column_round = record_type, int(year), round_
            out.id_map[legacy] = record_id
            out.rows.append(
                {
                    "record_id": record_id,
                    "version": version.version,
                    "coordinate_key": record_id,
                    "record_type": column_type,
                    "scope": scope,
                    "subject": subject,
                    "admission_year": column_year,
                    "admission_round": column_round,
                    "methods_hash": methods_hash,
                    "status": status,
                    "authority_level": payload.get("authority_level"),
                    "effective_from": payload.get("effective_from"),
                    "effective_to": payload.get("effective_to"),
                    "text": payload.get("text"),
                    "structured_payload": structured,
                    "major_codes": list(payload.get("major_codes") or []),
                    "admission_methods": list(payload.get("admission_methods") or []),
                    "source_document_id": payload.get("source_document_id"),
                    "source_url": payload.get("source_url"),
                    "source_page": payload.get("source_page"),
                    "supersedes": payload.get("supersedes"),
                    "authored_by": payload.get("authored_by"),
                    "published_at": payload.get("published_at"),
                }
            )
        except _MALFORMED as exc:
            raise MigrationError(f"malformed legacy record: {version.path} ({exc})") from exc
    for row in out.rows:
        if row["supersedes"]:
            row["supersedes"] = _remap_reference(row["supersedes"], out.id_map)
    return out


def _remap_reference(reference: str, id_map: dict[str, str]) -> str:
    """Map the record part of ``"legacy@N"`` through ``id_map``; unmapped names pass through."""
    name, version = split_version_key(reference)
    mapped = id_map.get(name, name)
    return mapped if version is None else f"{mapped}@{version}"


def split_version_key(reference: str) -> tuple[str, int | None]:
    """``"a@3"`` -> ``("a", 3)``; a bare id -> ``(id, None)``.

    Unlike ``canonical_record_store._named_record_id``, the version is kept.
    """
    head, sep, tail = reference.rpartition("@")
    if sep and head and tail.isdigit():
        return head, int(tail)
    return reference, None


def load_retractions(root: str | Path) -> list[dict]:
    """Read ``supersession_retractions`` markers into supersession_retractions rows."""
    base = Path(root) / RETRACTIONS_DIRNAME
    rows = []
    if not base.is_dir():
        return rows
    for path in sorted(base.glob("*/*.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise MigrationError(f"malformed retraction file: {path} (not a JSON object)")
        target = payload.get("target_record_id")
        claim = payload.get("superseding_version_key")
        if not isinstance(target, str) or not target or not isinstance(claim, str) or not claim:
            raise MigrationError(
                f"malformed retraction file: {path} (target_record_id / superseding_version_key)"
            )
        try:
            superseding_id, superseding_version = split_version_key(claim)
            rows.append(
                {
                    "retraction_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{target}|{claim}")),
                    "target_record_id": target,
                    "superseded_version": None,
                    "superseding_record_id": superseding_id,
                    "superseding_version": superseding_version,
                    "revoked_by": payload.get("retracted_by"),
                    "revoked_at": payload.get("at"),
                    "reason": payload.get("reason"),
                }
            )
        except _MALFORMED as exc:
            raise MigrationError(f"malformed retraction file: {path} ({exc})") from exc
    return rows




# ---- Task 2b-1 (U3): releases, outbox, conflicts, idempotency, unanswered, feedback ----
# Pure loaders: root -> rows shaped for the target table columns in
# alembic/versions/0001_baseline.py. Nothing is written. Every row is kept
# (len(rows) == source_rows); NOT NULL gaps are None plus an unfillable entry.


@dataclass
class RowsLoad:
    rows: list[dict]
    source_rows: int
    redacted_rows: int = 0
    unfillable: list[dict] = field(default_factory=list)
    id_map: dict = field(default_factory=dict)


def _json_object(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MigrationError(f"{path}: unreadable JSON ({exc})") from exc
    if not isinstance(obj, dict):
        raise MigrationError(f"{path}: expected a JSON object at top level")
    return obj


def _split_lines(text: str) -> list[str]:
    # Newline-only split: str.splitlines() also breaks on U+2028, U+0085, \x0b,
    # \x0c and \x1c-\x1e, which the real writers emit raw inside JSON strings.
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def _jsonl_objects(path: Path) -> list[tuple[int, dict]]:
    if not path.is_file():
        return []
    try:
        lines = _split_lines(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise MigrationError(f"{path}: unreadable ({exc})") from exc
    out: list[tuple[int, dict]] = []
    for lineno, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except ValueError as exc:
            raise MigrationError(f"{path}: line {lineno}: invalid JSON ({exc})") from exc
        if not isinstance(obj, dict):
            raise MigrationError(f"{path}: line {lineno}: expected a JSON object")
        out.append((lineno, obj))
    return out


def _reject_non_text(path: Path, lineno: int, obj: dict, names: tuple[str, ...]) -> None:
    for name in names:
        value = obj.get(name)
        if value is not None and not isinstance(value, str):
            raise MigrationError(f"{path}: line {lineno}: '{name}' must be a string")


def _not_null_gaps(legacy_id, row: dict, columns: list[str]) -> list[dict]:
    return [{"legacy_id": legacy_id, "column": c} for c in columns if row.get(c) is None]


def _redact(text: str) -> str:
    # Lazy import: records/retractions loaders must stay importable without src on sys.path.
    # guardrail.py imports the top-level 'services' package, so src must be on sys.path here.
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.append(src)
    from src.services.log_redaction import redact_pii

    return redact_pii(text)


def _redact_text_fields(row: dict, names: tuple[str, ...]) -> bool:
    changed = False
    for name in names:
        value = row.get(name)
        if isinstance(value, str):
            clean = _redact(value)
            if clean != value:
                changed = True
            row[name] = clean
    return changed


def load_releases(root: str | Path) -> RowsLoad:
    base = Path(root) / "releases"
    if not base.is_dir():
        return RowsLoad(rows=[], source_rows=0)
    rows: list[dict] = []
    unfill: list[dict] = []
    manifests = sorted(p for p in base.glob("*/release_manifest.json") if p.is_file())
    for path in manifests:
        m = _json_object(path)
        legacy_id = path.parent.name
        row = {
            "release_id": m.get("release_id"),
            "parent_release_id": m.get("parent_release_id"),
            "corpus_stamp": None,
            "lancedb_table": m.get("lancedb_table"),
            "lancedb_version": m.get("lancedb_version"),
            "graphml_uri": m.get("graph_path"),
            "manifest": m,
            "golden_report": None,
            "status": None,
            "created_by": None,
            "activated_by": None,
            "created_at": m.get("created_at"),
            "activated_at": None,
        }
        unfill += _not_null_gaps(
            legacy_id, row, ["release_id", "corpus_stamp", "lancedb_table", "status", "created_by"]
        )
        rows.append(row)
    return RowsLoad(rows=rows, source_rows=len(manifests), unfillable=unfill)


def load_outbox(root: str | Path) -> RowsLoad:
    entries = _jsonl_objects(Path(root) / "reindex_outbox.jsonl")
    rows: list[dict] = []
    unfill: list[dict] = []
    id_map: dict = {}
    for _, e in entries:
        legacy_entry = e.get("entry_id")
        event_id = None
        if isinstance(legacy_entry, str):
            # Legacy ids look like "<release>|<record@version>", not uuids.
            event_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "husc-outbox:" + legacy_entry))
            id_map[legacy_entry] = event_id
        row = {
            "event_id": event_id,
            "record_id": e.get("record_id"),
            "record_version": e.get("version"),
            # Legacy action ("publish") is not a reindex_outbox event_type; Task 3 maps it.
            "event_type": None,
            "payload": e,
            "release_id": e.get("release_id"),
            "available_at": e.get("appended_at"),
            "processed_at": None,
        }
        unfill += _not_null_gaps(
            legacy_entry, row, ["event_id", "record_id", "record_version", "event_type"]
        )
        rows.append(row)
    return RowsLoad(rows=rows, source_rows=len(entries), unfillable=unfill, id_map=id_map)


def load_conflicts(root: str | Path) -> RowsLoad:
    path = Path(root) / "conflicts" / "queue.json"
    if not path.is_file():
        return RowsLoad(rows=[], source_rows=0)
    doc = _json_object(path)
    items = doc.get("items")
    if not isinstance(items, list):
        raise MigrationError(f"{path}: 'items' must be a list")
    rows: list[dict] = []
    unfill: list[dict] = []
    redacted = 0
    for item in items:
        if not isinstance(item, dict):
            raise MigrationError(f"{path}: conflict item is not a JSON object")
        resolution = item.get("resolution")
        if resolution is not None and not isinstance(resolution, dict):
            raise MigrationError(f"{path}: conflict resolution must be a JSON object")
        clean_res = dict(resolution) if resolution is not None else None
        reason = (clean_res or {}).get("reason")
        if reason is not None and not isinstance(reason, str):
            raise MigrationError(f"{path}: conflict resolution 'reason' must be a string")
        override_reason = None
        row_redacted = False
        if isinstance(reason, str):
            override_reason = _redact(reason)
            row_redacted = override_reason != reason
            clean_res["reason"] = override_reason
        if row_redacted:
            redacted += 1
        row = {
            "conflict_id": item.get("conflict_id"),
            "coordinate_key": None,
            "record_ids": None,
            "old_record_id": None,
            "new_record_id": None,
            "conflict_type": None,
            "field_diffs": {"legacy_item": {k: v for k, v in item.items() if k != "resolution"}},
            "status": item.get("status"),
            "resolution": (
                json.dumps(clean_res, ensure_ascii=False, sort_keys=True)
                if clean_res is not None
                else None
            ),
            "override_reason": override_reason,
            "resolved_by": (clean_res or {}).get("resolved_by"),
            "resolved_at": (clean_res or {}).get("resolved_at"),
        }
        unfill += _not_null_gaps(
            item.get("conflict_id"),
            row,
            ["conflict_id", "coordinate_key", "record_ids", "conflict_type", "status"],
        )
        rows.append(row)
    return RowsLoad(rows=rows, source_rows=len(items), redacted_rows=redacted, unfillable=unfill)


def load_idempotency(root: str | Path) -> RowsLoad:
    base = Path(root) / "idempotency"
    if not base.is_dir():
        return RowsLoad(rows=[], source_rows=0)
    rows: list[dict] = []
    unfill: list[dict] = []
    files = sorted(p for p in base.glob("*.json") if p.is_file())
    for path in files:
        rec = _json_object(path)
        legacy_id = path.stem
        row = {
            # The on-disk file is named by the key digest; the digest is the key.
            "key": legacy_id,
            "scope": rec.get("operation"),
            "request_hash": rec.get("request_fingerprint"),
            "status_code": rec.get("status_code"),
            "response_body": rec.get("response"),
            "created_at": rec.get("created_at"),
            "expires_at": None,
        }
        unfill += _not_null_gaps(
            legacy_id,
            row,
            ["key", "scope", "request_hash", "status_code", "response_body", "expires_at"],
        )
        rows.append(row)
    return RowsLoad(rows=rows, source_rows=len(files), unfillable=unfill)


def load_unanswered(root: str | Path) -> RowsLoad:
    path = Path(root) / "unanswered" / "queue.jsonl"
    entries = _jsonl_objects(path)
    rows: list[dict] = []
    unfill: list[dict] = []
    redacted = 0
    for lineno, q in entries:
        _reject_non_text(path, lineno, q, ("question",))
        row = {
            "trace_id": q.get("trace_id"),
            "question": q.get("question"),
            "normalized_hash": None,
            "asked_at": q.get("asked_at"),
            "route": q.get("route"),
            "status_code": q.get("status_code"),
            "resolved_at": q.get("resolved_at"),
            "resolved_by_record_id": q.get("resolved_by_record_id"),
            "out_of_scope": bool(q.get("out_of_scope", False)),
            "sla_due_at": None,
        }
        if _redact_text_fields(row, ("question",)):
            redacted += 1
        unfill += _not_null_gaps(
            q.get("trace_id"),
            row,
            ["trace_id", "question", "normalized_hash", "asked_at", "route", "status_code", "sla_due_at"],
        )
        rows.append(row)
    return RowsLoad(rows=rows, source_rows=len(entries), redacted_rows=redacted, unfillable=unfill)


def load_feedback(root: str | Path) -> RowsLoad:
    path = Path(root) / "unanswered" / "feedback.jsonl"
    entries = _jsonl_objects(path)
    rows: list[dict] = []
    unfill: list[dict] = []
    redacted = 0
    for lineno, f in entries:
        _reject_non_text(path, lineno, f, ("reason",))
        row = {
            # Legacy feedback rows carry no feedback id; Task 3 assigns one.
            "feedback_id": None,
            "trace_id": f.get("trace_id"),
            "helpful": f.get("helpful"),
            "reason": f.get("reason"),
            "answered_at": f.get("answered_at"),
            "linked_item": bool(f.get("linked_item", False)),
            "release_id": None,
        }
        if _redact_text_fields(row, ("reason",)):
            redacted += 1
        unfill += _not_null_gaps(
            f.get("trace_id"), row, ["feedback_id", "trace_id", "helpful", "answered_at"]
        )
        rows.append(row)
    return RowsLoad(rows=rows, source_rows=len(entries), redacted_rows=redacted, unfillable=unfill)


# -- Task 2b-2 (U3): source documents, identities, audit ledger ---------------

_HEX_CHARS = frozenset("0123456789abcdefABCDEF")
_ZERO_DIGEST = bytes(32)
_KIND_BY_EXT = {".pdf": "pdf", ".xlsx": "xlsx", ".docx": "docx"}
_DOC_STATUS = {"registered": "uploaded", "retired": "archived"}
_IDENTITY_SCHEMA = "1.0"
_IDENTITY_NAMESPACE = "husc-identity:"


def _digest_bytes(value: object, where: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HEX_CHARS:
        raise MigrationError(f"{where} must be 64 hex characters")
    return bytes.fromhex(value)


def _parse_iso(where: str, value: object) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise MigrationError(f"{where} is not an ISO-8601 timestamp ({exc})") from exc


def _legacy_version(value: object) -> tuple[int | None, dict | None]:
    # Digit-only versions become ints; anything else is kept verbatim as legacy.
    if value is None:
        return None, None
    text = str(value)
    if text.isascii() and text.isdigit():
        return int(text), None
    return None, {"legacy_version": text}


def load_source_documents(root: str | Path) -> RowsLoad:
    docs_dir = Path(root) / "documents"
    paths = sorted(p for p in docs_dir.glob("doc-*_d.json") if p.is_file()) if docs_dir.is_dir() else []
    rows: list[dict] = []
    unfill: list[dict] = []
    for path in paths:
        e = _json_object(path)
        doc_id = e.get("document_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise MigrationError(f"{path}: document_id must be a non-empty string")
        location = doc_id.partition("#")[0]
        original_name = location.rsplit("/", 1)[-1]
        kind = _KIND_BY_EXT.get(Path(original_name).suffix.lower())
        sha = e.get("sha256")
        row = {
            "document_id": doc_id,
            "sha256": None if sha is None else _digest_bytes(sha, f"{path}: sha256"),
            "kind": kind,
            "storage_uri": e.get("path"),
            "original_name": original_name,
            "page_count": None,
            "admission_year": None,
            "status": _DOC_STATUS.get(e.get("state")),
            "uploaded_by": e.get("registered_by"),
            "created_at": _parse_iso(f"{path}: registered_at", e.get("registered_at")),
            "updated_at": _parse_iso(f"{path}: updated_at", e.get("updated_at")),
        }
        unfill += _not_null_gaps(
            doc_id,
            row,
            ["document_id", "sha256", "kind", "storage_uri", "original_name", "status",
             "uploaded_by", "created_at", "updated_at"],
        )
        rows.append(row)
    rows.sort(key=lambda r: r["document_id"])
    return RowsLoad(rows=rows, source_rows=len(paths), unfillable=unfill)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_document_blobs(root: str | Path) -> RowsLoad:
    """source_documents/ bytes, each matched to the documents/ entry that registers it.

    A blob is a row only when a documents/ entry registers it as ``uploads/<relpath>`` with
    the same sha256. Any other blob is an orphan: it counts in source_rows but not in rows,
    so the store's mismatch is the orphan count until the operator reconciles it.
    """
    base = Path(root) / SOURCE_DOCUMENTS_DIRNAME
    blobs = sorted(p for p in base.rglob("*") if p.is_file()) if base.is_dir() else []
    docs_dir = Path(root) / "documents"
    registered: dict[str, object] = {}
    for path in sorted(docs_dir.glob("doc-*_d.json")) if docs_dir.is_dir() else []:
        entry = _json_object(path)
        doc_id = entry.get("document_id")
        if isinstance(doc_id, str) and doc_id.startswith(UPLOAD_ID_PREFIX):
            registered[doc_id] = entry.get("sha256")
    rows: list[dict] = []
    for blob in blobs:
        doc_id = UPLOAD_ID_PREFIX + blob.relative_to(base).as_posix()
        digest = _file_sha256(blob)
        if registered.get(doc_id) == digest:
            rows.append({"document_id": doc_id, "sha256": digest})
    rows.sort(key=lambda r: r["document_id"])
    return RowsLoad(rows=rows, source_rows=len(blobs))


def load_identities(root: str | Path) -> RowsLoad:
    path = Path(root) / "identity" / "identities.json"
    if not path.is_file():
        return RowsLoad(rows=[], source_rows=0)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise MigrationError(f"{path}: unreadable ({exc})") from exc
    if not text.strip():
        return RowsLoad(rows=[], source_rows=0)
    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise MigrationError(f"{path}: unreadable JSON ({exc})") from exc
    if not isinstance(doc, dict):
        raise MigrationError(f"{path}: expected a JSON object at top level")
    if doc.get("schema_version") != _IDENTITY_SCHEMA:
        raise MigrationError(
            f"{path}: unsupported schema_version {doc.get('schema_version')!r}"
        )
    entries = doc.get("entries")
    if not isinstance(entries, list):
        raise MigrationError(f"{path}: 'entries' must be a list")
    rows: list[dict] = []
    unfill: list[dict] = []
    id_map: dict = {}
    for raw in entries:
        if not isinstance(raw, dict):
            raise MigrationError(f"{path}: every entry must be a JSON object")
        legacy_subject = raw.get("subject_id")
        if not isinstance(legacy_subject, str) or not legacy_subject:
            raise MigrationError(f"{path}: entry subject_id must be a non-empty string")
        try:
            subject = str(uuid.UUID(legacy_subject))
        except ValueError:
            subject = str(uuid.uuid5(uuid.NAMESPACE_URL, _IDENTITY_NAMESPACE + legacy_subject))
        id_map[legacy_subject] = subject
        credential = raw.get("credential")
        if not isinstance(credential, str) or not credential:
            raise MigrationError(f"{path}: entry credential must be a non-empty string")
        actor = raw.get("actor_id")
        row = {
            "subject_id": subject,
            "actor_id": actor,
            "role": raw.get("role"),
            "password_hash": raw.get("password_hash"),
            # Only the digest is stored; the plaintext credential never leaves this loop.
            "header_token_hash": hashlib.sha256(credential.encode("utf-8")).digest(),
            "disabled": bool(raw.get("disabled", False)),
            "created_by": None,
        }
        unfill += _not_null_gaps(actor, row, ["subject_id", "actor_id", "role", "created_by"])
        rows.append(row)
    rows.sort(key=lambda r: r["actor_id"] or "")
    return RowsLoad(rows=rows, source_rows=len(entries), unfillable=unfill, id_map=id_map)


def load_audit(root: str | Path) -> RowsLoad:
    path = Path(root) / "audit" / "ledger.jsonl"
    if not path.is_file():
        return RowsLoad(rows=[], source_rows=0)
    try:
        lines = _split_lines(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise MigrationError(f"{path}: unreadable ({exc})") from exc
    rows: list[dict] = []
    unfill: list[dict] = []
    prev_entry: bytes | None = None
    prev_seq: int | None = None
    for lineno, raw in enumerate(lines, start=1):
        where = f"{path}: line {lineno}:"
        if not raw.strip():
            raise MigrationError(f"{where} blank line in audit ledger")
        try:
            entry = json.loads(raw)
        except ValueError as exc:
            raise MigrationError(f"{where} invalid JSON ({exc})") from exc
        if not isinstance(entry, dict):
            raise MigrationError(f"{where} expected a JSON object")
        event = entry.get("event")
        if not isinstance(event, dict):
            raise MigrationError(f"{where} 'event' must be a JSON object")
        seq = entry.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool):
            raise MigrationError(f"{where} 'seq' must be an integer")
        # Hashes are recorded exactly as written; they are checked, never recomputed.
        prev_digest = _digest_bytes(entry.get("prev_hash"), f"{where} prev_hash")
        entry_digest = _digest_bytes(entry.get("entry_hash"), f"{where} entry_hash")
        if prev_entry is None:
            if prev_digest != _ZERO_DIGEST:
                raise MigrationError(f"{where} first entry must have the genesis prev_hash")
        elif prev_digest != prev_entry:
            raise MigrationError(f"{where} prev_hash does not match previous entry_hash")
        if prev_seq is not None and seq <= prev_seq:
            raise MigrationError(f"{where} seq must be strictly increasing")
        raw_event_id = event.get("event_id")
        if not isinstance(raw_event_id, str):
            raise MigrationError(f"{where} event_id must be a uuid hex string")
        try:
            event_id = str(uuid.UUID(hex=raw_event_id))
        except ValueError as exc:
            raise MigrationError(f"{where} event_id must be a uuid hex ({exc})") from exc
        before_version, before = _legacy_version(event.get("before_version"))
        after_version, after = _legacy_version(event.get("after_version"))
        row = {
            "event_id": event_id,
            "seq": seq,
            "aggregate_type": event.get("target_kind"),
            "aggregate_id": event.get("target"),
            "actor_id": event.get("actor_id"),
            "role": event.get("role"),
            "action": event.get("action"),
            "before": before,
            "after": after,
            "reason": event.get("reason"),
            "before_version": before_version,
            "after_version": after_version,
            "release_id": event.get("release_id"),
            "request_id": LEGACY_REQUEST_ID,
            "created_at": _parse_iso(f"{where} timestamp", event.get("timestamp")),
            "prev_hash": None if prev_digest == _ZERO_DIGEST else prev_digest,
            "entry_hash": entry_digest,
        }
        unfill += _not_null_gaps(
            event_id,
            row,
            ["aggregate_type", "aggregate_id", "actor_id", "role", "action", "created_at"],
        )
        rows.append(row)
        prev_entry = entry_digest
        prev_seq = seq
    return RowsLoad(rows=rows, source_rows=len(lines), unfillable=unfill)


# ---- Task 2b-3 (U3): dry-run report and CLI ----
# Read-only. The report carries counts, hashes and id maps; it never carries row contents.

_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "canonical_store"


def _serialize(value: object) -> str:
    if isinstance(value, bytes):
        return "hex:" + value.hex()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"not canonically serializable: {type(value).__name__}")


def _canonical_json(obj: object) -> str:
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_serialize
    )


def _canonical_hash(rows: object) -> str:
    return hashlib.sha256(_canonical_json(rows).encode("utf-8")).hexdigest()


def _loaders() -> list[tuple[str, Callable]]:
    # Resolved by global name at call time so tests can monkeypatch a loader.
    return [
        ("records", load_records),
        ("retractions", load_retractions),
        ("releases", load_releases),
        ("outbox", load_outbox),
        ("conflicts", load_conflicts),
        ("idempotency", load_idempotency),
        ("unanswered", load_unanswered),
        ("feedback", load_feedback),
        ("source_documents", load_source_documents),
        ("source_document_blobs", load_source_document_blobs),
        ("identities", load_identities),
        ("audit", load_audit),
    ]


def _parts(load: object) -> tuple[list, int]:
    if isinstance(load, list):
        return load, len(load)
    return load.rows, load.source_rows


def build_report(root: str | Path) -> dict:
    stores: list[dict] = []
    id_map: dict[str, dict] = {}
    unfillable: dict[str, int] = {}
    redacted_rows = 0
    quarantined_rows = 0
    normalised = _normalised_report([])
    for name, loader in _loaders():
        first = loader(root)
        rows, source_rows = _parts(first)
        if name == "records":
            normalised = _normalised_report(_superseded_keys(_supersede_extra_open_and_published(rows)[0]))
        digest = _canonical_hash(rows)
        second_rows, _ = _parts(loader(root))
        drift = 0 if _canonical_hash(second_rows) == digest else 1
        stores.append({
            "name": name,
            "source_rows": source_rows,
            "target_rows": len(rows),
            "mismatch": abs(source_rows - len(rows)) + drift,
            "hash": digest,
        })
        redacted_rows += getattr(first, "redacted_rows", 0)
        quarantined_rows += getattr(first, "quarantined_rows", 0)
        mapping = getattr(first, "id_map", None)
        # Identity subject ids default to actor_id (may be a person's email): never report them.
        if mapping and name != "identities":
            id_map[name] = dict(mapping)
        gaps = len(getattr(first, "unfillable", []))
        if gaps:
            unfillable[name] = gaps
    report = {
        "stores": stores,
        "redacted_rows": redacted_rows,
        "quarantined_rows": quarantined_rows,
        "id_map": id_map,
        "unfillable": unfillable,
        "normalised": normalised,
        "dry_run": True,
    }
    report["report_hash"] = hashlib.sha256(_canonical_json(report).encode("utf-8")).hexdigest()
    return report


# ---- Task 3a (U3): real run -- one transaction, idempotent, PostgreSQL ----
# Every store is written inside one transaction. Re-running inserts nothing new:
# each INSERT ... ON CONFLICT (natural key) DO NOTHING, and no existing row is updated.

LEGACY_ACTOR = "legacy-migration"
LEGACY_REASON = "legacy migration: no reason recorded"
LEGACY_REASON_PREFIX = "legacy migration: "
_ROLE_FALLBACK = "viewer"  # an identity with no role gets the least privilege
_OUTBOX_EVENT_TYPES = {
    "upsert": "upsert",
    "publish": "upsert",
    "tombstone": "tombstone",
    "supersede": "supersede",
    "revoke_supersession": "revoke_supersession",
}
_SCRUBBED_CREDENTIAL = re.compile(r"^scrubbed:[0-9a-f]{64}$")
_AUDIT_LOCK_TIMEOUT = "10s"  # bounds the wait for the audit chain advisory lock
# report store -> (table, key column, key cast) for the post-commit PostgreSQL re-count.
_TARGET_TABLES = {
    "source_documents": ("source_documents", "document_id", "text"),
    "source_document_blobs": ("source_documents", "document_id", "text"),
    "identities": ("admin_identities", "subject_id", "uuid"),
    "releases": ("releases", "release_id", "text"),
    "outbox": ("reindex_outbox", "event_id", "uuid"),
    "conflicts": ("conflict_entries", "conflict_id", "text"),
    "idempotency": ("idempotency_keys", "key", "text"),
    "unanswered": ("unanswered_questions", "trace_id", "text"),
    "feedback": ("chat_feedback", "feedback_id", "uuid"),
    "retractions": ("supersession_retractions", "retraction_id", "uuid"),
    "audit": ("audit_events", "event_id", "uuid"),
}


def sync_database_url(url: str) -> str:
    """DATABASE_URL (asyncpg form) -> the psycopg form used by the real run."""
    return url.strip().replace("+asyncpg", "+psycopg", 1)


def _conninfo(url: str) -> str:
    # libpq takes postgresql://, not the SQLAlchemy dialect form.
    return url.replace("+psycopg", "", 1)


class _Tally:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.defaulted: dict[str, dict[str, int]] = {}
        self.inserted: dict[str, int] = {}
        self.keys: dict[str, set] = {}
        self.skipped_idempotency = 0
        self.dangling = 0
        self.normalised_keys: list[str] = []

    def default(self, store: str, column: str) -> None:
        cols = self.defaulted.setdefault(store, {})
        cols[column] = cols.get(column, 0) + 1

    def seen(self, store: str, key: object) -> None:
        self.keys.setdefault(store, set()).add(key)


def _utc(value: object, where: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError as exc:
            raise MigrationError(f"{where} is not an ISO-8601 timestamp") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _ts(t: _Tally, store: str, column: str, value: object, *siblings: object) -> datetime:
    """NOT NULL timestamp: the value, else the nearest sibling timestamp on the row, else the run's now."""
    found = _utc(value, f"{store}.{column}")
    if found is not None:
        return found
    t.default(store, column)
    for sibling in siblings:
        found = _utc(sibling, f"{store}.{column}")
        if found is not None:
            return found
    return t.now


def _day(value: object, where: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise MigrationError(f"{where} is not a date") from exc


def _text(t: _Tally, store: str, column: str, value: object, fallback: str) -> object:
    if value is None:
        t.default(store, column)
        return fallback
    return value


def _write(cur, t: _Tally, store: str, table: str, row: dict, target: str, key: object,
           overriding: str = "") -> None:
    columns = ", ".join(row)
    holes = ", ".join(f"%({name})s" for name in row)
    sql = f"INSERT INTO {table} ({columns}) {overriding}VALUES ({holes}) ON CONFLICT {target} DO NOTHING"
    try:
        cur.execute(sql, row)
    except psycopg.Error as exc:
        # message_primary echoes offending row values; keep only the SQLSTATE, constraint and table.
        diag = exc.diag
        raise MigrationError(
            f"{store}: {key}: rejected by PostgreSQL (sqlstate {diag.sqlstate or '-'}, "
            f"constraint {diag.constraint_name or '-'}, table {table})"
        ) from exc
    if cur.rowcount == 1:
        t.inserted[store] = t.inserted.get(store, 0) + 1


def _write_source_documents(cur, t: _Tally, rows: list[dict]) -> None:
    store = "source_documents"
    for r in rows:
        doc_id = r["document_id"]
        location = doc_id.partition("#")[0]
        kind = r["kind"]
        if kind is None:
            t.default(store, "kind")
            kind = "url" if location.startswith(("http://", "https://")) else "pdf"
        sha = r["sha256"]
        if sha is None:
            t.default(store, "sha256")
            sha = hashlib.sha256(doc_id.encode("utf-8")).digest()
        row = {
            "document_id": doc_id,
            "sha256": sha,
            "kind": kind,
            "storage_uri": _text(t, store, "storage_uri", r["storage_uri"], location),
            "original_name": r["original_name"],
            "status": _text(t, store, "status", r["status"], "uploaded"),
            "uploaded_by": _text(t, store, "uploaded_by", r["uploaded_by"], LEGACY_ACTOR),
            "created_at": _ts(t, store, "created_at", r["created_at"], r["updated_at"]),
            "updated_at": _ts(t, store, "updated_at", r["updated_at"], r["created_at"]),
        }
        t.seen(store, doc_id)
        _write(cur, t, store, "source_documents", row, "(document_id)", doc_id)


def _write_identities(cur, t: _Tally, rows: list[dict]) -> None:
    store = "identities"
    for r in rows:
        subject = r["subject_id"]
        actor = r["actor_id"]
        if actor is None:
            t.default(store, "actor_id")
            actor = f"legacy:{subject}"
        role = r["role"]
        if role is None:
            t.default(store, "role")
            role = _ROLE_FALLBACK
        created = _ts(t, store, "created_at", None)
        row = {
            "subject_id": subject,
            "actor_id": actor,
            "role": role,
            "password_hash": r["password_hash"],
            "header_token_hash": r["header_token_hash"],
            "disabled": r["disabled"],
            "created_by": _text(t, store, "created_by", r["created_by"], LEGACY_ACTOR),
            "created_at": created,
            "updated_at": _ts(t, store, "updated_at", None, created),
        }
        t.seen(store, subject)
        _write(cur, t, store, "admin_identities", row, "(subject_id)", actor)


def _parents_first(rows: list[dict], parent_of: dict) -> list[dict]:
    loaded = {r["release_id"] for r in rows}
    done: set[str] = set()
    order: list[dict] = []
    pending = list(rows)
    while pending:
        ready = [
            r for r in pending
            if parent_of[r["release_id"]] not in loaded or parent_of[r["release_id"]] in done
        ]
        if not ready:
            raise MigrationError("releases: parent_release_id forms a cycle")
        order.extend(ready)
        done.update(r["release_id"] for r in ready)
        pending = [r for r in pending if r["release_id"] not in done]
    return order


def _write_releases(cur, t: _Tally, rows: list[dict], known: set[str]) -> None:
    store = "releases"
    for r in rows:
        if r["release_id"] is None:
            raise MigrationError(f"{store}: a release manifest has no release_id")
    parent_of: dict[str, str | None] = {}
    for r in rows:
        parent = r["parent_release_id"]
        if parent is not None and parent not in known:
            t.default(store, "parent_release_id")
            parent = None
        parent_of[r["release_id"]] = parent
    for r in _parents_first(rows, parent_of):
        rid = r["release_id"]
        stamp = r["corpus_stamp"]
        if stamp is None:
            t.default(store, "corpus_stamp")
            stamp = {}
        row = {
            "release_id": rid,
            "parent_release_id": parent_of[rid],
            "corpus_stamp": Jsonb(stamp),
            "lancedb_table": _text(t, store, "lancedb_table", r["lancedb_table"], LEGACY_ACTOR),
            "lancedb_version": r["lancedb_version"],
            "graphml_uri": r["graphml_uri"],
            "manifest": Jsonb(r["manifest"]),
            "status": _text(t, store, "status", r["status"], "ready"),
            "created_by": _text(t, store, "created_by", r["created_by"], LEGACY_ACTOR),
            "created_at": _ts(t, store, "created_at", r["created_at"]),
        }
        t.seen(store, rid)
        _write(cur, t, store, "releases", row, "(release_id)", rid)


_OPEN_STATUSES = ("draft", "review", "approved")


_NORMALISED_KEY_CAP = 200


def _supersede_extra_open_and_published(rows: list[dict]) -> tuple[list[dict], int]:
    """Archive every version the SQL model cannot hold as a current one.

    The file store could hold several open versions, several published ones, and open
    versions below a published one (a chain that write_record would never produce). The
    SQL model allows one open version (uq_one_open_draft) and one published version per
    coordinate (uq_current_coordinate). Kept as they are: the newest published version,
    and the newest open version that sits above it (or the newest open one if there is
    no published version). Everything else becomes ``archived`` with a ``_superseded``
    flag and the original status in ``_legacy_status``. Returns new row dicts (inputs are
    not mutated) and how many rows were changed.
    """
    def group(row: dict) -> str | None:
        status = row["status"] or "draft"
        if status in _OPEN_STATUSES:
            return "open"
        if status == "published":
            return "published"
        return None

    newest_pub: dict[str, int] = {}
    for index, row in enumerate(rows):
        if group(row) == "published":
            rid = row["record_id"]
            if rid not in newest_pub or row["version"] > rows[newest_pub[rid]]["version"]:
                newest_pub[rid] = index
    newest_open: dict[str, int] = {}
    for index, row in enumerate(rows):
        if group(row) != "open":
            continue
        rid = row["record_id"]
        if rid in newest_pub and row["version"] < rows[newest_pub[rid]]["version"]:
            continue  # below the newest published: never a current open version
        if rid not in newest_open or row["version"] > rows[newest_open[rid]]["version"]:
            newest_open[rid] = index
    out: list[dict] = []
    changed = 0
    for index, row in enumerate(rows):
        kind = group(row)
        rid = row["record_id"]
        keep = (
            kind is None
            or (kind == "published" and newest_pub[rid] == index)
            or (kind == "open" and newest_open.get(rid) == index)
        )
        if keep:
            out.append(row)
        else:
            out.append(dict(row, status="archived", _superseded=True, _legacy_status=row["status"]))
            changed += 1
    return out, changed


def _superseded_keys(rows: list[dict]) -> list[str]:
    """``record_id@version`` for each row rewritten by the normalisation, in input order."""
    return [f"{r['record_id']}@{r['version']}" for r in rows if r.get("_superseded")]


def _normalised_report(keys: list[str]) -> dict:
    return {"records": len(keys), "keys": keys[:_NORMALISED_KEY_CAP]}


def _write_records(cur, t: _Tally, rows: list[dict], known_docs: set[str]) -> None:
    store = "records"
    rows, _ = _supersede_extra_open_and_published(rows)
    t.normalised_keys.extend(_superseded_keys(rows))
    for r in rows:
        key = f"{r['record_id']}@{r['version']}"
        structured = dict(r["structured_payload"])
        legacy_status = r.get("_legacy_status", r["status"])
        if r.get("_superseded"):
            structured["legacy_status"] = legacy_status or "draft"
        source = r["source_document_id"]
        if source is not None and source not in known_docs:
            structured["legacy_source_document_id"] = source
            source = None
            t.dangling += 1
            if not r.get("_superseded") and legacy_status == "published":
                raise MigrationError(
                    f"{store}: {key}: published record cites source document {structured['legacy_source_document_id']!r}, "
                    "which is not in source_documents"
                )
        if legacy_status is None:
            t.default(store, "status")
        status = r["status"] or "draft"
        year = r["admission_year"]
        if year is None:
            t.default(store, "admission_year")
            year = t.now.year
        archive = None
        if r.get("_superseded"):
            archive = "superseded"
        elif status == "archived":
            t.default(store, "archive_reason")
            archive = "archived"
        effective_from = _day(r["effective_from"], f"{store}.effective_from")
        if effective_from is None:
            t.default(store, "effective_from")
            effective_from = date(year, 1, 1)
        published = _utc(r["published_at"], f"{store}.published_at")
        created = _ts(t, store, "created_at", None, published)
        row = {
            "record_id": r["record_id"],
            "version": r["version"],
            "coordinate_key": r["coordinate_key"],
            "record_type": _text(t, store, "record_type", r["record_type"], LEGACY_ACTOR),
            "scope": r["scope"],
            "subject": r["subject"],
            "admission_year": year,
            "admission_round": r["admission_round"],
            "methods_hash": r["methods_hash"],
            "status": status,
            "archive_reason": archive,
            "authority_level": _text(t, store, "authority_level", r["authority_level"], LEGACY_ACTOR),
            "effective_from": effective_from,
            "effective_to": _day(r["effective_to"], f"{store}.effective_to"),
            "text": _text(t, store, "text", r["text"], ""),
            "structured_payload": Jsonb(structured),
            "major_codes": Jsonb(r["major_codes"]),
            "admission_methods": Jsonb(r["admission_methods"]),
            "source_document_id": source,
            "source_url": r["source_url"],
            "source_page": r["source_page"],
            "supersedes": r["supersedes"],
            "authored_by": _text(t, store, "authored_by", r["authored_by"], LEGACY_ACTOR),
            "published_at": published,
            "created_at": created,
            "updated_at": _ts(t, store, "updated_at", None, created),
        }
        t.seen(store, (r["record_id"], r["version"]))
        _write(cur, t, store, "canonical_records", row, "(record_id, version)", key)


def _write_outbox(cur, t: _Tally, rows: list[dict], known: set[str]) -> None:
    store = "outbox"
    for r in rows:
        event_id = r["event_id"]
        if event_id is None:
            raise MigrationError(f"{store}: an outbox entry has no entry_id")
        payload = r["payload"]
        event_type = r["event_type"]
        if event_type is None:
            t.default(store, "event_type")
            event_type = _OUTBOX_EVENT_TYPES.get(payload.get("action"), "upsert")
        release = r["release_id"]
        if release is not None and release not in known:
            t.default(store, "release_id")
            release = None
        version = r["record_version"]
        if version is None:
            t.default(store, "record_version")
            version = 1
        row = {
            "event_id": event_id,
            "record_id": _text(t, store, "record_id", r["record_id"], LEGACY_ACTOR),
            "record_version": version,
            "event_type": event_type,
            "payload": Jsonb(payload),
            "release_id": release,
            "available_at": _ts(t, store, "available_at", r["available_at"]),
        }
        t.seen(store, event_id)
        _write(cur, t, store, "reindex_outbox", row, "(event_id)", event_id)


def _write_conflicts(cur, t: _Tally, rows: list[dict]) -> None:
    store = "conflicts"
    for r in rows:
        cid = r["conflict_id"]
        if cid is None:
            raise MigrationError(f"{store}: an item has no conflict_id")
        legacy = r["field_diffs"]["legacy_item"]
        record_ids = legacy.get("record_ids")
        if record_ids is None:
            t.default(store, "record_ids")
            record_ids = []
        row = {
            "conflict_id": cid,
            "coordinate_key": _text(t, store, "coordinate_key", legacy.get("coordinate_key"), LEGACY_ACTOR),
            "record_ids": Jsonb(record_ids),
            "old_record_id": legacy.get("old_record_id"),
            "new_record_id": legacy.get("new_record_id"),
            "conflict_type": _text(t, store, "conflict_type", legacy.get("conflict_type"), LEGACY_ACTOR),
            "field_diffs": Jsonb(r["field_diffs"]),
            "status": _text(t, store, "status", r["status"], "open"),
            "resolution": r["resolution"],
            "override_reason": r["override_reason"],
            "resolved_by": r["resolved_by"],
            "created_at": _ts(t, store, "created_at", legacy.get("created_at"), r["resolved_at"]),
            "resolved_at": _utc(r["resolved_at"], f"{store}.resolved_at"),
        }
        t.seen(store, cid)
        _write(cur, t, store, "conflict_entries", row, "(conflict_id)", cid)


def _write_idempotency(cur, t: _Tally, rows: list[dict]) -> None:
    store = "idempotency"
    for r in rows:
        if r["status_code"] is None:
            # An in_progress claim has no response: it is ephemeral and never migrated.
            t.skipped_idempotency += 1
            continue
        created = _ts(t, store, "created_at", r["created_at"])
        response = r["response_body"]
        if response is None:
            t.default(store, "response_body")
            response = {}
        row = {
            "key": r["key"],
            "scope": _text(t, store, "scope", r["scope"], LEGACY_ACTOR),
            "request_hash": _text(t, store, "request_hash", r["request_hash"], LEGACY_ACTOR),
            "status_code": r["status_code"],
            "response_body": Jsonb(response),
            "created_at": created,
            "expires_at": _ts(t, store, "expires_at", None, created),
        }
        t.seen(store, r["key"])
        _write(cur, t, store, "idempotency_keys", row, "(key)", r["key"])


def _write_unanswered(cur, t: _Tally, rows: list[dict]) -> None:
    store = "unanswered"
    for r in rows:
        trace = r["trace_id"]
        question = r["question"]
        if question is None:
            t.default(store, "question")
            question = ""
        normalized = r["normalized_hash"]
        if normalized is None:
            t.default(store, "normalized_hash")
            normalized = hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()
        asked = _ts(t, store, "asked_at", r["asked_at"], r["resolved_at"])
        status_code = r["status_code"]
        if status_code is None:
            t.default(store, "status_code")
            status_code = LEGACY_ACTOR
        row = {
            "trace_id": trace,
            "question": question,
            "normalized_hash": normalized,
            "asked_at": asked,
            "route": _text(t, store, "route", r["route"], LEGACY_ACTOR),
            "status_code": str(status_code),
            "resolved_at": _utc(r["resolved_at"], f"{store}.resolved_at"),
            "resolved_by_record_id": r["resolved_by_record_id"],
            "out_of_scope": r["out_of_scope"],
            "sla_due_at": _ts(t, store, "sla_due_at", None, asked),
        }
        t.seen(store, trace)
        _write(cur, t, store, "unanswered_questions", row, "(trace_id)", trace)


def _write_feedback(cur, t: _Tally, rows: list[dict]) -> None:
    store = "feedback"
    for r in rows:
        if r["helpful"] is None:
            raise MigrationError(f"{store}: trace {r['trace_id']!r} has no 'helpful' value")
        # The legacy row has no id: derive a stable one so a re-run finds the same row.
        identity = _canonical_json([r["trace_id"], r["helpful"], r["reason"], r["answered_at"]])
        feedback_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "husc-feedback:" + identity))
        row = {
            "feedback_id": feedback_id,
            "trace_id": r["trace_id"],
            "helpful": r["helpful"],
            "reason": r["reason"],
            "answered_at": _ts(t, store, "answered_at", r["answered_at"]),
            "linked_item": r["linked_item"],
            "release_id": r["release_id"],
        }
        t.seen(store, feedback_id)
        _write(cur, t, store, "chat_feedback", row, "(feedback_id)", feedback_id)


def _write_retractions(cur, t: _Tally, rows: list[dict], id_map: dict) -> None:
    store = "retractions"
    for r in rows:
        target = r["target_record_id"]
        if r["superseded_version"] is None or r["superseding_version"] is None:
            raise MigrationError(
                f"{store}: {target}: superseded or superseding version is not known"
            )
        reason = r["reason"]
        if reason is None:
            t.default(store, "reason")
            reason = LEGACY_REASON
        elif len(reason) < 10:
            t.default(store, "reason")
            reason = LEGACY_REASON_PREFIX + reason
        row = {
            "retraction_id": r["retraction_id"],
            "target_record_id": id_map.get(target, target),
            "superseded_version": r["superseded_version"],
            "superseding_record_id": id_map.get(r["superseding_record_id"], r["superseding_record_id"]),
            "superseding_version": r["superseding_version"],
            "revoked_by": _text(t, store, "revoked_by", r["revoked_by"], LEGACY_ACTOR),
            "revoked_at": _ts(t, store, "revoked_at", r["revoked_at"]),
            "reason": reason,
        }
        t.seen(store, r["retraction_id"])
        _write(cur, t, store, "supersession_retractions", row, "(retraction_id)", target)


def _write_audit(cur, t: _Tally, rows: list[dict]) -> None:
    store = "audit"
    cur.execute(f"SET LOCAL lock_timeout = '{_AUDIT_LOCK_TIMEOUT}'")
    try:
        cur.execute("SELECT pg_advisory_xact_lock(hashtext('audit_chain'))")
    except psycopg.errors.LockNotAvailable as exc:
        raise MigrationError(
            f"{store}: could not take the audit chain lock within {_AUDIT_LOCK_TIMEOUT}; another writer holds it"
        ) from exc
    cur.execute("SELECT event_id::text FROM audit_events")
    present = {r[0] for r in cur.fetchall()}
    # Migrating genesis-chain rows after live app rows would fork the audit chain: refuse first.
    foreign = present - {r["event_id"] for r in rows}
    if foreign:
        raise MigrationError(
            f"{store}: audit_events already has rows not in the ledger ({len(foreign)} row(s)); "
            "migrating ledger rows after them would fork the audit chain"
        )
    for r in rows:
        event_id = r["event_id"]
        t.seen(store, event_id)
        if event_id in present:
            continue
        row = {
            "event_id": event_id,
            "seq": r["seq"],
            "aggregate_type": _text(t, store, "aggregate_type", r["aggregate_type"], LEGACY_ACTOR),
            "aggregate_id": _text(t, store, "aggregate_id", r["aggregate_id"], LEGACY_ACTOR),
            "actor_id": _text(t, store, "actor_id", r["actor_id"], LEGACY_ACTOR),
            "role": _text(t, store, "role", r["role"], LEGACY_ACTOR),
            "action": _text(t, store, "action", r["action"], LEGACY_ACTOR),
            "before": None if r["before"] is None else Jsonb(r["before"]),
            "after": None if r["after"] is None else Jsonb(r["after"]),
            "reason": r["reason"],
            "before_version": r["before_version"],
            "after_version": r["after_version"],
            "release_id": r["release_id"],
            "request_id": r["request_id"],
            "created_at": _ts(t, store, "created_at", r["created_at"]),
            "prev_hash": r["prev_hash"],
            "entry_hash": r["entry_hash"],
        }
        _write(cur, t, store, "audit_events", row, "(event_id)", event_id, overriding="OVERRIDING SYSTEM VALUE ")
    cur.execute("SELECT setval(pg_get_serial_sequence('audit_events', 'seq'), (SELECT max(seq) FROM audit_events))")


def _count_targets(conn, t: _Tally) -> dict[str, int]:
    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        for store, (table, column, cast) in _TARGET_TABLES.items():
            keys = sorted(t.keys.get(store, ()))
            counts[store] = 0
            if keys:
                cur.execute(f"SELECT count(*) FROM {table} WHERE {column} = ANY(%s::{cast}[])", (keys,))
                counts[store] = cur.fetchone()[0]
        pairs = t.keys.get("records", set())
        counts["records"] = 0
        if pairs:
            cur.execute(
                "SELECT record_id, version FROM canonical_records WHERE record_id = ANY(%s::text[])",
                (sorted({p[0] for p in pairs}),),
            )
            counts["records"] = sum(1 for row in cur.fetchall() if (row[0], row[1]) in pairs)
    return counts


def apply_to_database(root: str | Path, conn, *, now: datetime) -> dict:
    """Write every legacy store into PostgreSQL in one transaction; return the real-run report.

    Any failure rolls the whole transaction back and raises (MigrationError or psycopg.Error).
    """
    run_at = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    base = build_report(root)
    loads = {name: loader(root) for name, loader in _loaders()}
    rows = {name: _parts(load)[0] for name, load in loads.items()}
    record_ids = getattr(loads["records"], "id_map", {})
    tally = _Tally(run_at)
    for row in rows["source_document_blobs"]:
        tally.seen("source_document_blobs", row["document_id"])
    with conn.transaction():
        with conn.cursor() as cur:
            _write_source_documents(cur, tally, rows["source_documents"])
            cur.execute("SELECT document_id FROM source_documents")
            known_docs = {r[0] for r in cur.fetchall()}
            _write_identities(cur, tally, rows["identities"])
            cur.execute("SELECT release_id FROM releases")
            known_releases = {r[0] for r in cur.fetchall()} | {r["release_id"] for r in rows["releases"]}
            _write_releases(cur, tally, rows["releases"], known_releases)
            _write_records(cur, tally, rows["records"], known_docs)
            _write_outbox(cur, tally, rows["outbox"], known_releases)
            _write_conflicts(cur, tally, rows["conflicts"])
            _write_idempotency(cur, tally, rows["idempotency"])
            _write_unanswered(cur, tally, rows["unanswered"])
            _write_feedback(cur, tally, rows["feedback"])
            _write_retractions(cur, tally, rows["retractions"], record_ids)
            _write_audit(cur, tally, rows["audit"])
    targets = _count_targets(conn, tally)
    # Rows skipped on purpose (in_progress claims) are reported under "skipped", not as mismatch.
    skipped_by_store = {"idempotency": tally.skipped_idempotency}
    stores = []
    for entry in base["stores"]:
        name = entry["name"]
        stores.append({
            "name": name,
            "source_rows": entry["source_rows"],
            "target_rows": targets[name],
            "mismatch": abs(entry["source_rows"] - skipped_by_store.get(name, 0) - targets[name]),
            "hash": entry["hash"],
            "inserted": tally.inserted.get(name, 0),
        })
    report = {k: v for k, v in base.items() if k != "report_hash"}
    report.update({
        "stores": stores,
        "dry_run": False,
        "defaulted": tally.defaulted,
        "skipped": {"idempotency": tally.skipped_idempotency},
        "dangling_source_document_id": tally.dangling,
        "normalised": _normalised_report(tally.normalised_keys),
        "run_at": run_at.isoformat(),
    })
    return report


def scrub_identities_file(root: str | Path) -> int:
    """Replace each plaintext credential with ``scrubbed:<sha256 hex>``; atomic (tmp + os.replace)."""
    path = Path(root) / "identity" / "identities.json"
    if not path.is_file():
        return 0
    doc = _json_object(path)
    scrubbed = 0
    for entry in doc.get("entries") or []:
        credential = entry.get("credential")
        if isinstance(credential, str) and credential and not _SCRUBBED_CREDENTIAL.match(credential):
            entry["credential"] = "scrubbed:" + hashlib.sha256(credential.encode("utf-8")).hexdigest()
            scrubbed += 1
    tmp = Path(str(path) + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, indent=2, ensure_ascii=False))
            fh.flush()
            os.fsync(fh.fileno())
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return scrubbed


def run_real_run(root: str | Path, url: str, *, now: datetime, scrub_identities: bool = False) -> dict:
    with psycopg.connect(_conninfo(sync_database_url(url))) as conn:
        report = apply_to_database(root, conn, now=now)
    if scrub_identities:
        identities = next(s for s in report["stores"] if s["name"] == "identities")
        if identities["target_rows"] != identities["source_rows"]:
            raise MigrationError("identities: PostgreSQL does not hold every identity; file not scrubbed")
        try:
            scrub_identities_file(root)
        except OSError as exc:
            raise MigrationError(
                "data committed; identities scrub failed - re-run is safe "
                f"({exc.strerror or type(exc).__name__})"
            ) from exc
    return report


def _failure_text(exc: BaseException) -> str:
    message = str(exc)
    filename = getattr(exc, "filename", None)
    if filename and str(filename) not in message:
        message = f"{filename}: {message}"
    return message


def _pg_failure_text(exc: psycopg.Error) -> str:
    # str(psycopg.Error) carries server message text with row values; print only the code.
    diag = exc.diag
    return (
        f"PostgreSQL error {type(exc).__name__} (sqlstate {diag.sqlstate or '-'}, "
        f"constraint {diag.constraint_name or '-'})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy file stores (U3).")
    parser.add_argument("--dry-run", action="store_true", help="read-only report; never connects to a database")
    parser.add_argument("--report", required=True, metavar="PATH", help="where to write the JSON report")
    parser.add_argument("--root", default=str(_DEFAULT_ROOT), metavar="DIR", help="legacy canonical_store root")
    parser.add_argument("--confirm-frozen", action="store_true", help="required for a real run: the legacy stores are frozen")
    parser.add_argument(
        "--database-url", default=None, metavar="URL",
        help="PostgreSQL URL; prefer the DATABASE_URL env var (argv is visible to other users)",
    )
    parser.add_argument(
        "--scrub-identities", action="store_true",
        help="real run only: after PostgreSQL verification, replace plaintext credentials in identity/identities.json",
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        if args.scrub_identities:
            print("--scrub-identities needs a real run, not --dry-run", file=sys.stderr)
            return 2
        try:
            report = build_report(args.root)
        except (MigrationError, OSError, TypeError) as exc:
            print(_failure_text(exc), file=sys.stderr)
            return 2
    else:
        if not args.confirm_frozen:
            print(
                "real run not implemented without --confirm-frozen: freeze the legacy stores first",
                file=sys.stderr,
            )
            return 2
        url = args.database_url or os.environ.get("DATABASE_URL")
        if not url:
            print("real run needs --database-url or DATABASE_URL", file=sys.stderr)
            return 2
        try:
            report = run_real_run(
                args.root, url, now=datetime.now(timezone.utc), scrub_identities=args.scrub_identities
            )
        except (MigrationError, OSError, TypeError, psycopg.Error) as exc:
            print(_pg_failure_text(exc) if isinstance(exc, psycopg.Error) else _failure_text(exc), file=sys.stderr)
            return 2
    target = Path(args.report)
    tmp = Path(str(target) + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        print(_failure_text(exc), file=sys.stderr)
        return 2
    mismatch_total = sum(s["mismatch"] for s in report["stores"])
    print(f"wrote {args.report}: {len(report['stores'])} stores, mismatch total {mismatch_total}")
    if not args.dry_run:
        unexplained = [s["name"] for s in report["stores"] if s["mismatch"] > 0]
        if unexplained:
            print(
                "post-commit mismatch in store(s) " + ", ".join(unexplained)
                + ": PostgreSQL does not hold every source row (see the report)",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
