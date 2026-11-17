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
                out.quarantined_rows += 1
            else:
                record_id = key
                status = payload.get("status")
                methods_hash = key.rsplit(":", 1)[1]
            out.id_map[legacy] = record_id
            out.rows.append(
                {
                    "record_id": record_id,
                    "version": version.version,
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
