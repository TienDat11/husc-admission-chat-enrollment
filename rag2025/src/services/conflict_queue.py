"""The conflict queue and the human override — ticket 38.

When two official sources state the same fact differently, the system must
**stop** and hand the decision to an admissions officer. This module is the
place a conflict lives: it is detected from data, persisted as a queue item,
and closed only by an explicit, audited override.

Two rules are load-bearing here and are enforced structurally, not by
convention:

1. **Detection never selects.** :meth:`ConflictQueueService.detect` reads the
   seed report (``cross_source_conflicts``) and compares published record
   versions with the same coordinates, and it writes **nothing** to the record
   store — not a majority value, not the higher-authority one. A conflict with
   a 2-to-1 statement split queues exactly like a 1-to-1 split; the count of
   agreeing sources is not a decision. The byte-identity of
   ``records/<id>/*.json`` across a full detect → queue run is asserted by the
   tests, so a future "auto-pick the majority" branch fails loudly.
2. **An override carries a reason and an actor.** ``resolve`` builds an
   :class:`~src.domain.audit.AuditEvent` with
   :attr:`~src.domain.audit.AuditAction.CONFLICT_OVERRIDE` — the first real
   producer of that vocabulary term — and appends it to the hash-chained
   ledger. An empty reason cannot produce a valid event at all: the router
   checks the body first, and :data:`~src.domain.audit.REASON_REQUIRED`
   refuses at the model second, two independent enforcement points on the
   ticket-12 pattern.

A resolution is one-shot: an item that is ``resolved`` answers a second
attempt with :class:`ConflictAlreadyResolvedError` (409 at the edge), so two
POSTs for one conflict never append two events (note 5 — the lock is the
item's state, not a time window).

``after_version`` is the bridge to ``TargetKind.RECORD`` (note 2): the winning
statement names a source document, the service finds the stored version whose
provenance and field value match that statement, and the event records that
version. No matching stored version means the decision has nothing to point
at, so the resolve is refused (404) rather than recorded against a phantom.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.domain.audit import AuditAction, AuditEvent, TargetKind
from src.domain.canonical.record import CanonicalRecord, RecordStatus
from src.infrastructure.audit_ledger import AuditLedger
from src.infrastructure.canonical_record_store import CanonicalRecordStore

__all__ = [
    "ConflictAlreadyResolvedError",
    "ConflictError",
    "ConflictItem",
    "ConflictItemNotFoundError",
    "ConflictQueueService",
    "ConflictWinnerNotInStatementsError",
    "ConflictWinnerRecordNotFoundError",
    "DetectionResult",
    "ORIGIN_PUBLISHED_RECORDS",
    "ORIGIN_SEED_REPORT",
    "STATUS_OPEN",
    "STATUS_RESOLVED",
]

# The two states an item moves through. ``resolved`` is terminal for the item:
# re-opening is a new detection with a new decision, not an edit of the old one.
STATUS_OPEN = "open"
STATUS_RESOLVED = "resolved"

# Where a queued conflict came from. Both are real detections (note 1): the
# seed report proves the pipeline, live published-record comparison keeps the
# queue alive for conflicts the seed never saw.
ORIGIN_SEED_REPORT = "seed_report"
ORIGIN_PUBLISHED_RECORDS = "published_records"

QUEUE_DIRNAME = "conflicts"
QUEUE_FILENAME = "queue.json"
QUEUE_SCHEMA_VERSION = "1.0"

# The fact fields the live detector compares, mirroring the seed script's
# conflict vocabulary (scripts/seed_canonical_records.py). A field outside this
# set is free text where two sources may differ without contradicting each
# other; these six are the countable facts students are quoted.
CONFLICT_FIELDS: tuple[str, ...] = (
    "chi_tieu",
    "to_hop_codes",
    "hoc_phi_tin_chi_vnd",
    "diem_trung_tuyen_num",
    "thoi_gian_dao_tao_nam",
    "ten_nganh",
)


class ConflictError(Exception):
    """Base class for conflict-queue failures."""


class ConflictItemNotFoundError(ConflictError):
    """No queued item has that ``conflict_id``."""


class ConflictAlreadyResolvedError(ConflictError):
    """The item exists but its decision was already recorded (note 5)."""


class ConflictWinnerNotInStatementsError(ConflictError):
    """The chosen winner names a source that states nothing in this conflict."""


class ConflictWinnerRecordNotFoundError(ConflictError):
    """No stored record version carries the winning statement to point at."""


def _canonical(value: object) -> str:
    """Deterministic comparison form of one statement value.

    Same convention the seed script hashes with: compact JSON, sorted keys,
    no ``\\uXXXX`` escaping, so ``"Hán Nôm"`` and ``"Hán - Nôm"`` never compare
    equal and two lists compare by content, not by identity.
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _conflict_id(
    record_type: str, ma_nganh: str, admission_year: int, field_name: str
) -> str:
    """Stable identity of a conflict: a hash of its four coordinates.

    The coordinates are serialized through canonical JSON with sorted keys, so
    the id depends on *what* is in conflict and never on the order statements
    or sources were inserted in, nor on the process that computed it. Two runs
    over the same data — or over the same data with the statements dict built
    in a different order — yield the same id; an id derived from ``hash()`` of
    an insertion-ordered tuple would not.
    """
    payload = json.dumps(
        {
            "record_type": str(record_type),
            "ma_nganh": str(ma_nganh),
            "admission_year": int(admission_year),
            "field": str(field_name),
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class ConflictItem:
    """One queued contradiction between official sources.

    ``statements`` maps ``source_document_id`` to the value that source states;
    the map is the evidence an officer reads before deciding. ``resolution`` is
    present only once a human has decided, and carries the winner, the record
    version the audit event pointed at, and who decided.
    """

    conflict_id: str
    record_type: str
    ma_nganh: str
    admission_year: int
    field: str
    statements: dict[str, Any] = field(default_factory=dict)
    origin: str = ORIGIN_SEED_REPORT
    status: str = STATUS_OPEN
    resolution: dict[str, Any] | None = None

    def coordinates(self) -> tuple[str, str, int, str]:
        return (self.record_type, self.ma_nganh, self.admission_year, self.field)


@dataclass(frozen=True)
class DetectionResult:
    """What one detection pass found, with its dedup accounting.

    ``deduped`` and ``dedup_reason`` exist so the API can state them (criterion
    9): when the seed report and the live comparison describe the same
    coordinates, the item is queued once and the response says so, rather than
    silently dropping or double-counting it.
    """

    items: tuple[ConflictItem, ...]
    deduped: int = 0
    dedup_reason: str | None = None


def _ma_nganh_of(record: CanonicalRecord) -> str | None:
    """The ``ma_nganh`` a stored version is about.

    Seeded records carry it in their ``values`` dict; a hand-authored record id
    of the canonical shape ``<TYPE>:<kind>:<ma_nganh>:<year>:<slug>`` names it
    in segment 2, so the fallback keeps the live detector working on records
    whose values dict omits the key.
    """
    values = getattr(record, "values", None)
    if isinstance(values, Mapping):
        ma = values.get("ma_nganh")
        if ma is not None:
            return str(ma)
    parts = record.record_id.split(":")
    if len(parts) >= 3 and parts[2]:
        return parts[2]
    return None


class ConflictQueueService:
    """Detect, persist, and resolve source conflicts.

    The queue is **data**, not a derived view: every detection pass upserts into
    the persisted file, so a decision made before a restart survives it, and a
    conflict that stops being reported by the seed still shows as ``resolved``
    rather than re-opening itself. All reads/mutations of the file happen under
    one process lock, because two officers can resolve from two requests.
    """

    def __init__(
        self,
        store: CanonicalRecordStore,
        ledger: AuditLedger | None,
        seed_report_path: str | Path | None = None,
        queue_path: str | Path | None = None,
    ) -> None:
        self._store = store
        self._ledger = ledger
        # Default seed report: the file the canonical seeding writes **next to
        # the store it seeded** (``seed_canonical_records.py`` REPORT_FILENAME
        # under ``RAGSettings.CANONICAL_STORE_ROOT``). That store-root copy is
        # the app's source of truth; the older ``data/canonical_records`` copy
        # is kept as a fallback so a deployment whose seeding predates the
        # store-root convention still answers. Resolving against ``store.root``
        # — not a hardcoded sibling directory — is what keeps this module from
        # reading a parallel store the app never serves.
        if seed_report_path is None:
            in_store = Path(store.root) / "seed_report.json"
            seed_report_path = in_store if in_store.is_file() else (
                Path(__file__).resolve().parents[2]
                / "data"
                / "canonical_records"
                / "seed_report.json"
            )
        self._seed_report_path = Path(seed_report_path)
        if queue_path is None:
            queue_path = Path(store.root) / QUEUE_DIRNAME / QUEUE_FILENAME
        self._queue_path = Path(queue_path)
        self._lock = threading.Lock()

    # -- properties ----------------------------------------------------------

    @property
    def queue_path(self) -> Path:
        """Where the queue persists. Beside the records, never inside them."""
        return self._queue_path

    @property
    def seed_report_path(self) -> Path:
        return self._seed_report_path

    # -- detection ------------------------------------------------------------

    def detect(self) -> DetectionResult:
        """Every current conflict, from both sources, deduplicated by id.

        Pure with respect to the record store: it reads the seed report and the
        published versions and writes nothing. That is the load-bearing
        invariant — the queue is a to-do list for a human, never an
        auto-applied decision.
        """
        by_id: dict[str, ConflictItem] = {}
        for item in self._detect_from_seed_report():
            by_id[item.conflict_id] = item
        deduped = 0
        dedup_reasons: list[str] = []
        for item in self._detect_from_published():
            existing = by_id.get(item.conflict_id)
            if existing is None:
                by_id[item.conflict_id] = item
                continue
            # Same coordinates reported twice: the seed row wins the statement
            # map (it is the audited provenance record) and the collision is
            # counted, never hidden.
            deduped += 1
            dedup_reasons.append(
                f"{existing.record_type}/{existing.ma_nganh}/{existing.admission_year}"
                f"/{existing.field} reported by both seed_report and published records"
            )
        items = tuple(
            sorted(by_id.values(), key=lambda item: item.coordinates() + (item.conflict_id,))
        )
        reason = "; ".join(dedup_reasons) if dedup_reasons else None
        return DetectionResult(items=items, deduped=deduped, dedup_reason=reason)

    def _detect_from_seed_report(self) -> list[ConflictItem]:
        """``cross_source_conflicts`` rows, each as an open item."""
        if not self._seed_report_path.is_file():
            return []
        data = json.loads(self._seed_report_path.read_text(encoding="utf-8"))
        rows = data.get("cross_source_conflicts") or []
        items: list[ConflictItem] = []
        for row in rows:
            try:
                coordinates = (
                    str(row["record_type"]),
                    str(row["ma_nganh"]),
                    int(row["admission_year"]),
                    str(row["field"]),
                )
                statements = dict(row.get("statements") or {})
            except (KeyError, TypeError, ValueError):
                # A malformed row is a seed-data bug; queueing a half-read
                # conflict would send an officer to decide on missing
                # evidence, so skip it rather than invent one.
                continue
            if len({_canonical(v) for v in statements.values()}) < 2:
                continue
            record_type, ma_nganh, admission_year, field_name = coordinates
            items.append(
                ConflictItem(
                    conflict_id=_conflict_id(*coordinates),
                    record_type=record_type,
                    ma_nganh=ma_nganh,
                    admission_year=admission_year,
                    field=field_name,
                    statements=statements,
                    origin=ORIGIN_SEED_REPORT,
                )
            )
        return items

    def _detect_from_published(self) -> list[ConflictItem]:
        """Published versions with the same coordinates stating different values.

        Grouped by ``(record_type, ma_nganh, admission_year, field)`` over the
        six countable fact fields; a group whose published statements disagree
        is a conflict. Authority and effective window are deliberately **not**
        tie-breakers here — picking between them is the officer's job, and a
        detector that ranked sources would be the auto-selection this ticket
        exists to remove.
        """
        grouped: dict[tuple[str, str, int, str], dict[str, str]] = {}
        for record_id in self._store.record_ids():
            for version in self._store.versions(record_id):
                record = self._store.get_version(record_id, version)
                if record.status is not RecordStatus.PUBLISHED:
                    continue
                ma_nganh = _ma_nganh_of(record)
                if ma_nganh is None:
                    continue
                values = getattr(record, "values", None)
                if not isinstance(values, Mapping):
                    continue
                source = (record.source_document_id or "").strip()
                if not source:
                    continue
                record_type = str(record.record_type.value)
                for field_name in CONFLICT_FIELDS:
                    if field_name not in values:
                        continue
                    key = (record_type, ma_nganh, int(record.admission_year), field_name)
                    # Latest version of one source wins its own statement;
                    # a source correcting itself is not a source disagreeing
                    # with another. Versions ascend, so last write is newest.
                    grouped.setdefault(key, {})[source] = _canonical(values[field_name])
        items: list[ConflictItem] = []
        for (record_type, ma_nganh, admission_year, field_name), by_source in grouped.items():
            if len(set(by_source.values())) < 2:
                continue
            statements = {source: json.loads(canonical) for source, canonical in by_source.items()}
            items.append(
                ConflictItem(
                    conflict_id=_conflict_id(record_type, ma_nganh, admission_year, field_name),
                    record_type=record_type,
                    ma_nganh=ma_nganh,
                    admission_year=admission_year,
                    field=field_name,
                    statements=statements,
                    origin=ORIGIN_PUBLISHED_RECORDS,
                )
            )
        return items

    # -- queue ----------------------------------------------------------------

    def refresh(self) -> tuple[list[ConflictItem], DetectionResult]:
        """Detect, upsert into the persisted queue, save, and return both.

        An existing item keeps its ``status`` and ``resolution`` — a decided
        conflict is not re-opened by being detected again, and an undetected
        one is not dropped from the file while it still has a decision to
        show. New items enter as ``open``.
        """
        with self._lock:
            detection = self.detect()
            existing = self._load_locked()
            merged: dict[str, ConflictItem] = {item.conflict_id: item for item in existing}
            for item in detection.items:
                prior = merged.get(item.conflict_id)
                if prior is None:
                    merged[item.conflict_id] = item
                else:
                    merged[item.conflict_id] = replace(prior, statements=item.statements)
            ordered = sorted(merged.values(), key=lambda i: i.coordinates() + (i.conflict_id,))
            self._save_locked(ordered)
            return list(ordered), detection

    def items(self) -> list[ConflictItem]:
        """The persisted queue as it stands, without detecting."""
        with self._lock:
            return self._load_locked()

    def _load_locked(self) -> list[ConflictItem]:
        if not self._queue_path.is_file():
            return []
        data = json.loads(self._queue_path.read_text(encoding="utf-8"))
        items: list[ConflictItem] = []
        for row in data.get("items") or []:
            items.append(
                ConflictItem(
                    conflict_id=str(row["conflict_id"]),
                    record_type=str(row["record_type"]),
                    ma_nganh=str(row["ma_nganh"]),
                    admission_year=int(row["admission_year"]),
                    field=str(row["field"]),
                    statements=dict(row.get("statements") or {}),
                    origin=str(row.get("origin") or ORIGIN_SEED_REPORT),
                    status=str(row.get("status") or STATUS_OPEN),
                    resolution=dict(row["resolution"]) if row.get("resolution") else None,
                )
            )
        return items

    def _save_locked(self, items: list[ConflictItem]) -> None:
        payload = {
            "schema_version": QUEUE_SCHEMA_VERSION,
            "items": [asdict(item) for item in items],
        }
        data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1) + "\n").encode(
            "utf-8"
        )
        self._queue_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._queue_path.with_name(self._queue_path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, self._queue_path)

    # -- resolution -------------------------------------------------------------

    def resolve(
        self,
        conflict_id: str,
        winner_source_document_id: str,
        reason: str | None,
        actor_id: str,
        role: str | None,
    ) -> ConflictItem:
        """Record a human decision: the winner's statement, with a reason.

        Appends one ``conflict_override`` event and flips the item to
        ``resolved`` — and nothing else. No record version is written: the
        override is the accountable statement of which source wins; applying
        it to the data is the lifecycle's job through its own audited
        transitions. The event is built **before** the queue is touched, so a
        refusal (no reason, no winner record) leaves the item ``open`` rather
        than resolved-but-unaudited.
        """
        if self._ledger is None:
            raise ConflictError("conflict resolution requires the audit ledger")
        with self._lock:
            items = self._load_locked()
            item = next((i for i in items if i.conflict_id == conflict_id), None)
            if item is None:
                raise ConflictItemNotFoundError(f"no conflict item {conflict_id!r} in the queue")
            if item.status == STATUS_RESOLVED:
                raise ConflictAlreadyResolvedError(
                    f"conflict {conflict_id!r} was already resolved by "
                    f"{(item.resolution or {}).get('resolved_by')!r}"
                )
            if winner_source_document_id not in item.statements:
                raise ConflictWinnerNotInStatementsError(
                    f"{winner_source_document_id!r} states nothing in conflict {conflict_id!r}"
                )
            winner = self._find_winner_version(item, winner_source_document_id)
            if winner is None:
                raise ConflictWinnerRecordNotFoundError(
                    f"no stored record version carries the statement of "
                    f"{winner_source_document_id!r} for this conflict"
                )
            # Second enforcement point: REASON_REQUIRED in the audit model.
            # The router refuses a blank reason first; even if that check were
            # removed, this constructor raises and nothing is appended.
            event = AuditEvent(
                action=AuditAction.CONFLICT_OVERRIDE,
                actor_id=actor_id,
                role=role,
                target=winner.record_id,
                target_kind=TargetKind.RECORD,
                timestamp=datetime.now(timezone.utc),
                after_version=winner.record_version_key,
                reason=(reason or "").strip(),
                review_id=conflict_id,
            )
            self._ledger.append(event)
            resolved = replace(
                item,
                status=STATUS_RESOLVED,
                resolution={
                    "winner_source_document_id": winner_source_document_id,
                    "winner_statement": item.statements[winner_source_document_id],
                    "winner_record_version": winner.record_version_key,
                    "reason": (reason or "").strip(),
                    "resolved_by": actor_id,
                    "resolved_by_role": role,
                    "resolved_at": event.timestamp.isoformat(),
                    "event_id": event.event_id,
                },
            )
            merged = [resolved if i.conflict_id == conflict_id else i for i in items]
            self._save_locked(merged)
            return resolved

    def _find_winner_version(
        self, item: ConflictItem, winner_source_document_id: str
    ) -> CanonicalRecord | None:
        """The stored version the winning statement lives in.

        Matches provenance (the source document) **and** content (the field
        value equals the statement) **and** coordinates, across every stored
        version regardless of status — the decision is about which source is
        right, and the record holding that source's statement may still be a
        draft awaiting its own lifecycle. When several versions qualify, the
        highest ``(version, record_id)`` wins, the same preference rule the
        store applies to supersession candidates.
        """
        wanted_statement = _canonical(item.statements[winner_source_document_id])
        best: CanonicalRecord | None = None
        for record_id in self._store.record_ids():
            for version in self._store.versions(record_id):
                record = self._store.get_version(record_id, version)
                if str(record.record_type.value) != item.record_type:
                    continue
                if int(record.admission_year) != item.admission_year:
                    continue
                if _ma_nganh_of(record) != item.ma_nganh:
                    continue
                if (record.source_document_id or "").strip() != winner_source_document_id:
                    continue
                values = getattr(record, "values", None)
                if not isinstance(values, Mapping) or item.field not in values:
                    continue
                if _canonical(values[item.field]) != wanted_statement:
                    continue
                if best is None or (record.version, record.record_id) > (
                    best.version,
                    best.record_id,
                ):
                    best = record
        return best
