"""Migrate legacy file stores into PostgreSQL rows (U3, Task 2a slice).

Loaders are pure functions: store root in, row dicts out. Nothing here
connects to a database or writes to the legacy store.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from src.infrastructure.canonical_record_store import (
    _RECORD_DIR_PREFIX,
    _RECORD_DIR_SUFFIX,
    RECORDS_DIRNAME,
    RETRACTIONS_DIRNAME,
)

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
    return out


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


def _jsonl_objects(path: Path) -> list[tuple[int, dict]]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
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
