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


class MigrationError(Exception):
    """A legacy file could not be read; the whole run must abort."""


@dataclass
class RecordsLoad:
    rows: list[dict] = field(default_factory=list)
    id_map: dict[str, str] = field(default_factory=dict)
    quarantined_rows: int = 0
    source_rows: int = 0


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


def _record_files(records_dir: Path) -> list[tuple[str, int, Path, dict]]:
    entries = []
    if not records_dir.is_dir():
        return entries
    for directory in sorted(records_dir.iterdir()):
        if not (
            directory.is_dir()
            and directory.name.startswith(_RECORD_DIR_PREFIX)
            and directory.name.endswith(_RECORD_DIR_SUFFIX)
        ):
            continue
        for path in sorted(directory.glob("v*.json")):
            payload = _read_json(path)
            if not isinstance(payload, dict) or not payload.get("record_id"):
                raise MigrationError(f"malformed legacy record: {path}")
            entries.append((payload["record_id"], int(payload.get("version", 1)), path, payload))
    entries.sort(key=lambda e: (e[0], e[1]))
    return entries


def _anchor_problem(payload: dict, key: str | None) -> str | None:
    if key is None:
        return "coordinate"
    if payload.get("status") == "published" and not (payload.get("source_document_id") or "").strip():
        return "no_source"
    return None


def load_records(root: str | Path) -> RecordsLoad:
    """Read every legacy record version under ``root/records`` into canonical_records rows."""
    entries = _record_files(Path(root) / RECORDS_DIRNAME)
    keys = [derive_coordinate_key(payload) for _, _, _, payload in entries]
    problems = [_anchor_problem(p, k) for (_, _, _, p), k in zip(entries, keys)]

    # Coordinate duplicates between different legacy records: one winner keeps it.
    groups: dict[str, list[int]] = {}
    for index, (key, problem) in enumerate(zip(keys, problems)):
        if key is not None and problem is None:
            groups.setdefault(key, []).append(index)
    for key, indexes in groups.items():
        legacy_ids = {entries[i][0] for i in indexes}
        if len(legacy_ids) < 2:
            continue
        published = [i for i in indexes if entries[i][3].get("status") == "published"]
        if published:
            winner_index = max(
                published,
                key=lambda i: entries[i][3].get("published_at") or "",
            )
            winner = entries[winner_index][0]
        else:
            winner = entries[indexes[0]][0]
        for i in indexes:
            if entries[i][0] != winner:
                problems[i] = "duplicate"

    out = RecordsLoad(source_rows=len(entries))
    for (legacy, version, _, payload), key, problem in zip(entries, keys, problems):
        parsed = _parse_legacy_id(legacy)
        scope, subject = (parsed[0], parsed[1]) if parsed else ("", "")
        structured = dict(payload.get("values") or {})
        structured["legacy_record_id"] = legacy
        if problem is None:
            record_id = key
            status = payload.get("status")
            methods_hash = key.rsplit(":", 1)[1]
        else:
            record_id = f"{UNANCHORED_PREFIX}{legacy}"
            status = "draft"
            structured["blockers"] = [UNANCHORED_BLOCKER]
            methods_hash = "all"
            out.quarantined_rows += 1
        out.id_map[legacy] = record_id
        out.rows.append(
            {
                "record_id": record_id,
                "version": version,
                "coordinate_key": record_id,
                "record_type": payload.get("record_type"),
                "scope": scope,
                "subject": subject,
                "admission_year": payload.get("admission_year"),
                "admission_round": payload.get("admission_round") or "all",
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
            raise MigrationError(f"malformed retraction file: {path}")
        target = payload["target_record_id"]
        claim = payload["superseding_version_key"]
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
    return rows
