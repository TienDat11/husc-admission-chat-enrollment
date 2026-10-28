"""Conflict-queue service — ticket 38.

The load-bearing claims this file measures, at the service layer (the HTTP
seam is ``tests/routers/test_admin_conflicts_api.py``):

- detection reads data and writes nothing, ever — not even when a 2-to-1
  majority exists and a machine *could* pick a winner (criterion 2);
- conflict ids are deterministic across runs and across interpreters, so the
  queue dedups by identity rather than by insertion order (criterion 7);
- a resolve with a blank reason cannot produce an event even when the router's
  own check is bypassed, because the audit model refuses it (criterion 3,
  second enforcement point);
- a recorded decision survives a restart — the queue is data, not a view
  (criterion 4 persistence);
- running the detector over the real corpus leaves every stored version
  byte-identical (criterion 8).
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.audit import AuditAction  # noqa: E402
from src.domain.canonical.record import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.infrastructure.audit_ledger import AuditLedger  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.services.conflict_queue import (  # noqa: E402
    STATUS_OPEN,
    STATUS_RESOLVED,
    ConflictAlreadyResolvedError,
    ConflictItemNotFoundError,
    ConflictQueueService,
    ConflictWinnerNotInStatementsError,
    ConflictWinnerRecordNotFoundError,
)

REAL_SEED_REPORT = ROOT / "data" / "canonical_records" / "seed_report.json"
REAL_STORE_ROOT = ROOT / "data" / "canonical_store"

# The pair criterion 1 names: three record types report it, each stating
# exactly two distinct values across three sources.
PAIR_COORDS = ("NGANH", "7220104", 2026, "ten_nganh")
WINNER_DOC = "new_data/THONG_TIN_TUYEN_SINH_202611.pdf"


def _published(
    record_id: str,
    record_type: str,
    ma_nganh: str,
    year: int,
    field_name: str,
    value: str,
    source_doc: str,
) -> CanonicalRecord:
    """One published version carrying ``value`` for ``field_name``."""
    return CanonicalRecord(
        record_id=record_id,
        record_type=RecordType(record_type),
        version=1,
        admission_year=year,
        effective_from=date(2026, 1, 1),
        status=RecordStatus.PUBLISHED,
        authority_level=AuthorityLevel.HUSC_OFFICIAL,
        source_document_id=source_doc,
        text=f"{field_name}={value}",
        values={"ma_nganh": ma_nganh, field_name: value},
    )


def _winner_store(tmp_path: Path) -> CanonicalRecordStore:
    """A store holding the winning statement for the seed pair, so resolve can
    point its audit event at a real version."""
    store = CanonicalRecordStore(tmp_path / "store")
    store.write_record(
        _published(
            "NGANH:kind:7220104:2026:tt202611",
            "NGANH",
            "7220104",
            2026,
            "ten_nganh",
            "Hán - Nôm",
            WINNER_DOC,
        )
    )
    return store


def _snapshot(root: Path) -> dict[str, bytes]:
    """Every file under ``<root>/records`` as path -> bytes."""
    records = root / "records"
    if not records.is_dir():
        return {}
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(records.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def store(tmp_path: Path) -> CanonicalRecordStore:
    return CanonicalRecordStore(tmp_path / "store")


@pytest.fixture
def service(store: CanonicalRecordStore, tmp_path: Path) -> ConflictQueueService:
    """Real seed report, throwaway store and queue — detection sees both."""
    return ConflictQueueService(
        store,
        AuditLedger(tmp_path / "audit" / "ledger.jsonl"),
        seed_report_path=REAL_SEED_REPORT,
        queue_path=tmp_path / "queue" / "queue.json",
    )


# -- criterion 1 + 9: the seed report's conflicts enter the queue -------------


def test_seed_pair_7220104_ten_nganh_2026_is_queued_with_both_values(
    service: ConflictQueueService,
) -> None:
    """The criterion-1 pair: exactly two distinct values, statements keyed by
    source document. A detector that filtered on one field name loses this item
    and the test goes RED."""
    result = service.detect()
    items = [
        item
        for item in result.items
        if (item.ma_nganh, item.admission_year, item.field) == ("7220104", 2026, "ten_nganh")
    ]
    assert items, "the 7220104/ten_nganh/2026 conflict is not queued"
    for item in items:
        assert set(item.statements.values()) == {"Hán Nôm", "Hán - Nôm"}
        assert len(item.statements) == 3
        assert item.status == STATUS_OPEN


def test_detect_twice_yields_same_ids_no_dupes(service: ConflictQueueService) -> None:
    """Criterion 7 in-process: two runs over the same seed report return the
    same id set, the same count, and no duplicate ids."""
    first = service.detect()
    second = service.detect()
    ids_a = [item.conflict_id for item in first.items]
    ids_b = [item.conflict_id for item in second.items]
    assert len(ids_a) == len(set(ids_a)), "duplicate conflict ids within one run"
    assert sorted(ids_a) == sorted(ids_b)
    assert len(first.items) == len(second.items)


# -- criterion 7 across processes: hash() would not survive -------------------

_PROBE = (
    "import json, tempfile;"
    "from src.services.conflict_queue import ConflictQueueService;"
    "from src.infrastructure.canonical_record_store import CanonicalRecordStore;"
    f"svc = ConflictQueueService(CanonicalRecordStore(tempfile.mkdtemp()), None,"
    f" seed_report_path=r'{REAL_SEED_REPORT}');"
    "print(json.dumps(sorted(i.conflict_id for i in svc.detect().items)))"
)


def _ids_in_fresh_interpreter() -> list[str]:
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_conflict_ids_survive_a_fresh_interpreter() -> None:
    """Criterion 7's mutation case: an id built from ``hash()`` of an
    insertion-ordered structure shifts between processes (PYTHONHASHSEED is
    randomised), so two interpreters would disagree. sha256 over the four
    coordinates cannot."""
    assert _ids_in_fresh_interpreter() == _ids_in_fresh_interpreter()


# -- criterion 2: detection never selects a winner ----------------------------


def test_detection_never_writes_to_the_store_even_with_a_majority(
    store: CanonicalRecordStore, tmp_path: Path
) -> None:
    """Three published sources state the same fact, two agreeing (a machine
    *could* take the majority). Detect + refresh must leave every record byte
    identical and the item open. Mutation: an auto-majority branch writes the
    winner value -> the byte-identity assert goes RED."""
    statements = {"doc-a": "Hán Nôm", "doc-b": "Hán Nôm", "doc-c": "Hán - Nôm"}
    for doc, value in statements.items():
        store.write_record(
            _published(
                f"NGANH:kind:7220999:2026:{doc}",
                "NGANH",
                "7220999",
                2026,
                "ten_nganh",
                value,
                doc,
            )
        )
    before = _snapshot(store.root)
    service = ConflictQueueService(
        store,
        None,
        seed_report_path=tmp_path / "no-seed.json",
        queue_path=tmp_path / "queue.json",
    )
    result = service.detect()
    item = next(i for i in result.items if i.ma_nganh == "7220999")
    assert item.status == STATUS_OPEN
    assert set(item.statements) == set(statements)

    service.refresh()
    assert _snapshot(store.root) == before
    for record_id in store.record_ids():
        assert store.versions(record_id) == (1,), "detection appended a version"


# -- criterion 3: the blank-reason refusal exists at the audit layer too ------


def test_resolve_with_blank_reason_raises_and_changes_nothing(tmp_path: Path) -> None:
    """Bypassing the router, the service still refuses: the winner record
    resolves fine, and then ``AuditEvent`` requires a reason for
    ``conflict_override``. The item stays open and the ledger stays empty.
    Mutation: dropping CONFLICT_OVERRIDE from REASON_REQUIRED appends an
    event -> RED."""
    ledger = AuditLedger(tmp_path / "audit" / "ledger.jsonl")
    service = ConflictQueueService(
        _winner_store(tmp_path),
        ledger,
        seed_report_path=REAL_SEED_REPORT,
        queue_path=tmp_path / "queue.json",
    )
    service.refresh()
    item = next(i for i in service.items() if i.coordinates() == PAIR_COORDS)
    with pytest.raises(ValueError):
        service.resolve(
            item.conflict_id,
            winner_source_document_id=WINNER_DOC,
            reason="     ",
            actor_id="cb.quan-tri",
            role="admin",
        )
    assert ledger.find(action=AuditAction.CONFLICT_OVERRIDE) == ()
    still = next(i for i in service.items() if i.conflict_id == item.conflict_id)
    assert still.status == STATUS_OPEN


def test_resolve_rejects_unknown_item_and_foreign_winner(
    store: CanonicalRecordStore, tmp_path: Path
) -> None:
    """The three refusals a resolve can hit, each before anything is recorded:
    no such item (404 in HTTP terms), winner states nothing here (422), and —
    on an empty store — no version carries the winning statement (404)."""
    service = ConflictQueueService(
        store,
        AuditLedger(tmp_path / "audit" / "ledger.jsonl"),
        seed_report_path=REAL_SEED_REPORT,
        queue_path=tmp_path / "queue.json",
    )
    service.refresh()
    item = next(i for i in service.items() if i.coordinates() == PAIR_COORDS)
    with pytest.raises(ConflictItemNotFoundError):
        service.resolve(
            "0" * 16,
            winner_source_document_id="x",
            reason="r",
            actor_id="a",
            role="admin",
        )
    with pytest.raises(ConflictWinnerNotInStatementsError):
        service.resolve(
            item.conflict_id,
            winner_source_document_id="doc-that-states-nothing",
            reason="quyết định",
            actor_id="cb.quan-tri",
            role="admin",
        )
    # The winner is in the statements, but no stored version carries it.
    with pytest.raises(ConflictWinnerRecordNotFoundError):
        service.resolve(
            item.conflict_id,
            winner_source_document_id=WINNER_DOC,
            reason="quyết định",
            actor_id="cb.quan-tri",
            role="admin",
        )


# -- criterion 4: a real decision, persisted ----------------------------------


def test_resolve_records_one_event_and_survives_a_restart(tmp_path: Path) -> None:
    """Criterion 4 + note 5: resolve appends exactly one ``conflict_override``
    event pointing at the winner's version, flips the item to ``resolved``,
    refuses a second resolve, and a fresh service over the same files still
    shows the decision."""
    ledger = AuditLedger(tmp_path / "audit" / "ledger.jsonl")
    queue_path = tmp_path / "queue.json"
    winner_store = _winner_store(tmp_path)
    service = ConflictQueueService(
        winner_store,
        ledger,
        seed_report_path=REAL_SEED_REPORT,
        queue_path=queue_path,
    )
    service.refresh()
    item = next(i for i in service.items() if i.coordinates() == PAIR_COORDS)
    resolved = service.resolve(
        item.conflict_id,
        winner_source_document_id=WINNER_DOC,
        reason="  căn cứ quyết định 345/QĐ  ",
        actor_id="cb.quan-tri",
        role="admin",
    )
    assert resolved.status == STATUS_RESOLVED
    assert resolved.resolution["reason"] == "căn cứ quyết định 345/QĐ"

    events = ledger.find(action=AuditAction.CONFLICT_OVERRIDE)
    assert len(events) == 1
    event = events[0].event
    assert event.actor_id == "cb.quan-tri"
    assert event.role == "admin"
    assert event.target == "NGANH:kind:7220104:2026:tt202611"
    assert event.after_version == "NGANH:kind:7220104:2026:tt202611@1"
    assert event.reason == "căn cứ quyết định 345/QĐ"
    assert ledger.verify().ok

    with pytest.raises(ConflictAlreadyResolvedError):
        service.resolve(
            item.conflict_id,
            winner_source_document_id=WINNER_DOC,
            reason="quyết định lại",
            actor_id="cb.quan-tri",
            role="admin",
        )
    assert len(ledger.find(action=AuditAction.CONFLICT_OVERRIDE)) == 1

    # A restart is a new service object over the same files.
    reopened = ConflictQueueService(
        winner_store,
        ledger,
        seed_report_path=REAL_SEED_REPORT,
        queue_path=queue_path,
    )
    after_restart = next(i for i in reopened.items() if i.conflict_id == item.conflict_id)
    assert after_restart.status == STATUS_RESOLVED
    assert after_restart.resolution["resolved_by"] == "cb.quan-tri"


# -- criterion 8: the real corpus is untouched --------------------------------


def test_real_corpus_versions_unchanged_after_detect_and_queue(tmp_path: Path) -> None:
    """Detector + queue over the *real* store: every record's version list and
    every record byte stay exactly as they were. The queue file is redirected
    to tmp so the repo's data directory is not a test artifact."""
    assert REAL_STORE_ROOT.is_dir(), "real corpus missing — cannot measure criterion 8"
    store = CanonicalRecordStore(REAL_STORE_ROOT)
    record_ids = store.record_ids()
    assert record_ids, "real corpus has no records to protect"
    versions_before = {rid: store.versions(rid) for rid in record_ids}
    bytes_before = _snapshot(store.root)

    service = ConflictQueueService(
        store,
        None,
        seed_report_path=REAL_SEED_REPORT,
        queue_path=tmp_path / "queue.json",
    )
    result = service.detect()
    service.refresh()

    assert {rid: store.versions(rid) for rid in record_ids} == versions_before
    assert _snapshot(store.root) == bytes_before
    # The seed conflicts are still all there, none lost to the live pass.
    assert len([i for i in result.items if i.origin == "seed_report"]) == 74


# -- default seed report: store root first, repo copy as fallback --------------


def test_default_seed_report_prefers_the_file_inside_the_store_root(
    tmp_path: Path,
) -> None:
    """A store root carrying its own ``seed_report.json`` — the layout the
    seeder writes under ``CANONICAL_STORE_ROOT`` — must win over the repo
    copy. With a one-row report in the root, the service detects *that* row
    and nothing of the repo's 74; the assertion names the resolved path, so
    precedence is explicit, not inferred from a count."""
    root = tmp_path / "store"
    root.mkdir()
    report = {
        "cross_source_conflicts": [
            {
                "record_type": "NGANH",
                "ma_nganh": "9999999",
                "admission_year": 2026,
                "field": "to_hop_xet_tuyen",
                "statements": {"doc-a.pdf": "A:B", "doc-b.pdf": "C:D"},
            }
        ]
    }
    (root / "seed_report.json").write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8"
    )
    service = ConflictQueueService(CanonicalRecordStore(root), None)
    assert service.seed_report_path == root / "seed_report.json"
    items = service.detect().items
    assert [item.ma_nganh for item in items] == ["9999999"]


def test_default_seed_report_falls_back_to_the_repo_copy(tmp_path: Path) -> None:
    """An empty store root (a deployment seeded before the store-root
    convention, or a fresh test root) resolves to the repo's
    ``data/canonical_records`` copy — the path is named, not implied."""
    root = tmp_path / "store"
    root.mkdir()
    service = ConflictQueueService(CanonicalRecordStore(root), None)
    assert service.seed_report_path == REAL_SEED_REPORT
