"""Serve probe — ticket T16: release-bound probe log feeding K3.

K3 = minutes from ``upload_received_at`` to ``successfully_served_at``. The
two timestamps live in different processes (publish vs query), so the probe
log joins them: the publish hook appends ONE ``published`` entry per release,
and the query path appends ONE ``served`` follow-up line the first time a
query returns a chunk stamped with that release.

Path: ``<data_dir>/kpi/probe_log.jsonl`` (``data_dir`` is the canonical-store
root, the same root the unanswered queue lives under). Append-only: readers
tolerate BOTH a follow-up ``served`` line and an updated ``published`` line
carrying ``served_at`` — latest non-null ``served_at`` wins, first ``published``
line wins for the rest.

Entry schema (``published`` line)::

    {"event": "published", "release_id": "<hook release id>",
     "chunk_release_id": "<release_id stamped in the served chunks, or null>",
     "record_id": "<probe record served by the release>",
     "probe_question": "<first 200 chars of the probe record's text>",
     "upload_received_at": "<ISO, or null when genuinely unavailable>",
     "upload_received_at_known": false,
     "published_at": "<manifest clock, ISO>",
     "served_at": null}

Follow-up (``served`` line)::

    {"event": "served", "release_id": "<same>", "served_at": "<ISO>"}

Why two release ids: the publish hook mints its own ``release_id`` AFTER the
re-index job already stamped the chunks, so the served chunks may carry an
older id. A query stamps an open probe entry when a returned chunk matches
the entry's ``release_id`` OR ``chunk_release_id`` AND belongs to the
entry's ``record_id`` (spec 02 §9: a chunk of THAT record) — whichever
release convention the table uses, the stamp fires. ``upload_received_at`` comes from the
source-document registry's ``registered_at`` for the probe record's source
document; when that link is missing the entry records ``null`` with
``upload_received_at_known=false`` — never an invented timestamp.

ponytail: probe_question is a text snippet, not a real question; upgrade to a
per-record_type question template when K3 needs asking rather than quoting.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROBE_DIRNAME = "kpi"
PROBE_FILENAME = "probe_log.jsonl"

EVENT_PUBLISHED = "published"
EVENT_SERVED = "served"


def probe_log_path(data_dir: str | Path) -> Path:
    """``<data_dir>/kpi/probe_log.jsonl`` — the one probe-log location."""
    return Path(data_dir) / PROBE_DIRNAME / PROBE_FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_line(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_probe_entry(
    *,
    data_dir: str | Path,
    release_id: str,
    chunk_release_id: str | None = None,
    record_id: str | None = None,
    probe_question: str | None = None,
    upload_received_at: str | None = None,
    upload_received_at_known: bool = False,
    published_at: str | None = None,
) -> dict[str, Any]:
    """Append ONE ``published`` entry for a release; return the row written."""
    row: dict[str, Any] = {
        "event": EVENT_PUBLISHED,
        "release_id": release_id,
        "chunk_release_id": chunk_release_id,
        "record_id": record_id,
        "probe_question": probe_question,
        "upload_received_at": upload_received_at,
        "upload_received_at_known": bool(upload_received_at_known),
        "published_at": published_at or _now_iso(),
        "served_at": None,
    }
    _append_line(probe_log_path(data_dir), row)
    return row


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def read_probe_entries(path: str | Path) -> dict[str, dict[str, Any]]:
    """Merge the log into ``release_id -> entry``, tolerating both served
    conventions: a follow-up ``served`` line or an updated ``published``
    line carrying ``served_at``. First ``published`` line wins for the static
    fields; latest non-null ``served_at`` wins."""
    merged: dict[str, dict[str, Any]] = {}
    for row in _read_rows(Path(path)):
        rid = row.get("release_id")
        if not rid:
            continue
        rid = str(rid)
        entry = merged.setdefault(
            rid,
            {
                "release_id": rid,
                "chunk_release_id": None,
                "record_id": None,
                "probe_question": None,
                "upload_received_at": None,
                "upload_received_at_known": False,
                "published_at": None,
                "served_at": None,
            },
        )
        if row.get("event") == EVENT_PUBLISHED or (
            "published_at" in row and entry["published_at"] is None
        ):
            for key in (
                "chunk_release_id",
                "record_id",
                "probe_question",
                "upload_received_at",
                "upload_received_at_known",
                "published_at",
            ):
                if entry[key] is None and row.get(key) is not None:
                    entry[key] = row[key]
        if row.get("served_at"):
            entry["served_at"] = row["served_at"]
    return merged


def stamp_served(
    *,
    data_dir: str | Path,
    release_id: str,
    served_at: str | None = None,
) -> bool:
    """Append a ``served`` follow-up for ``release_id``. Returns True only
    when a line was written: unknown releases and already-served entries
    write nothing (a query over a pre-probe release must not spam the log)."""
    entries = read_probe_entries(probe_log_path(data_dir))
    entry = entries.get(str(release_id))
    if entry is None or entry.get("served_at"):
        return False
    _append_line(
        probe_log_path(data_dir),
        {
            "event": EVENT_SERVED,
            "release_id": str(release_id),
            "served_at": served_at or _now_iso(),
        },
    )
    return True


def _chunk_identity(chunk: Any) -> tuple[list[str], Optional[str]]:
    """(release ids, record id) one returned chunk belongs to.

    Accepts a chunk dict (``metadata.release_id`` / top-level
    ``release_id``; ``metadata.record_id`` / top-level ``doc_id``)
    OR a ``RetrievedDocument``-shaped object (``.metadata`` dict
    attr, or ``.to_dict()``) — the v2 path hands over
    ``rag_result.documents`` objects, the v3/legacy paths hand
    dicts. The record id is the probe's identity: spec 02 §9
    closes a probe only when a chunk OF THE PROBE'S RECORD is
    served, so another record of the same release must not count.
    """
    if hasattr(chunk, "to_dict") and not isinstance(chunk, dict):
        try:
            chunk = chunk.to_dict()
        except Exception:
            return [], None
    if isinstance(chunk, dict):
        meta = chunk.get("metadata")
        releases = []
        if isinstance(meta, dict) and meta.get("release_id"):
            releases.append(str(meta["release_id"]))
        if chunk.get("release_id"):
            releases.append(str(chunk["release_id"]))
        record_id = None
        if isinstance(meta, dict) and meta.get("record_id"):
            record_id = str(meta["record_id"])
        elif chunk.get("doc_id"):
            record_id = str(chunk["doc_id"])
        return releases, record_id
    releases = []
    meta = getattr(chunk, "metadata", None)
    if isinstance(meta, dict) and meta.get("release_id"):
        releases.append(str(meta["release_id"]))
    if getattr(chunk, "release_id", None):
        releases.append(str(chunk.release_id))
    record_id = None
    if isinstance(meta, dict) and meta.get("record_id"):
        record_id = str(meta["record_id"])
    elif getattr(chunk, "doc_id", None):
        record_id = str(chunk.doc_id)
    return releases, record_id


def stamp_served_from_chunks(
    data_dir: str | Path, chunks: list[Any]
) -> int:
    """Stamp every open probe entry whose RECORD a returned chunk
    belongs to. A chunk closes a probe when its release
    (``metadata.release_id`` or top-level ``release_id``) equals
    the entry's ``release_id`` OR ``chunk_release_id``, AND its
    record id equals the entry's ``record_id`` — spec 02 §9: the
    probe is served when a chunk of THAT record comes back, so
    another record from the same release does not count. An entry
    whose ``record_id`` is None (unresolvable at publish time)
    degrades to release-only matching. Fail-soft by contract —
    callers swallow exceptions — but never writes when nothing
    matches."""
    wanted_releases: set[str] = set()
    records_by_release: dict[str, set[str]] = {}
    for chunk in chunks or []:
        releases, record_id = _chunk_identity(chunk)
        for rid in releases:
            wanted_releases.add(rid)
            if record_id:
                records_by_release.setdefault(rid, set()).add(record_id)
    if not wanted_releases:
        return 0
    path = probe_log_path(data_dir)
    entries = read_probe_entries(path)
    to_stamp = []
    for eid, entry in entries.items():
        if entry.get("served_at"):
            continue
        candidates = {eid, entry.get("chunk_release_id") or ""} & wanted_releases
        if not candidates:
            continue
        record_id = entry.get("record_id")
        if record_id is None:
            to_stamp.append(eid)
            continue
        if any(record_id in records_by_release.get(rid, ()) for rid in candidates):
            to_stamp.append(eid)
    for eid in sorted(to_stamp):
        _append_line(path, {"event": EVENT_SERVED, "release_id": eid, "served_at": _now_iso()})
    return len(to_stamp)


def k3_durations_minutes(
    entries: dict[str, dict[str, Any]],
) -> list[tuple[str, float]]:
    """``(release_id, minutes)`` for entries with BOTH ``upload_received_at``
    and ``served_at`` known. Entries missing either are skipped, never
    zero-filled."""
    out: list[tuple[str, float]] = []
    for rid, entry in entries.items():
        up = entry.get("upload_received_at")
        sv = entry.get("served_at")
        if not up or not sv or sv is True:
            continue
        try:
            minutes = (
                datetime.fromisoformat(str(sv)) - datetime.fromisoformat(str(up))
            ).total_seconds() / 60
        except ValueError:
            continue
        out.append((rid, minutes))
    return sorted(out)
