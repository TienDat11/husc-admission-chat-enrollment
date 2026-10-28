"""The two emitters ticket 14's audit found missing — B14-1 closure.

The audit of 2026-09-13 said ``PERMISSION_CHANGE`` and ``CONFLICT_OVERRIDE``
were in the enum and in ``REASON_REQUIRED`` but had **no emitter anywhere in
``src/``**: the ledger could record them and refused to record them without a
reason, yet no code path ever tried. "ai đổi quyền của tôi?" was unanswerable
from the ledger. This file pins the two seams that close that gap, each driven
end-to-end through a real ledger file on disk:

- ``AuditLedger.record_permission_change`` — the seam ticket 37's identity
  routes call. Asserted here at the seam because the mutation endpoint is
  ticket 37's deliverable; the seam *is* the emitter, and the encoding
  (``subject_id@role`` in the version fields) is the seam's contract.
- ``ConflictQueueService.resolve`` — ticket 38's real production path, which
  appends the ``conflict_override`` event before flipping the queue item.
  Asserted end-to-end (detect → queue → resolve → ledger) because ticket 38's
  own API test is still in flight.

The other four kinds the ticket names (publish, rollback, source-document
change, reject) already had emitters, and their firing is asserted where they
are driven: ``tests/routers/test_admin_audit.py`` (HTTP publish/reject/edit,
reading the real ledger) and ``tests/routers/test_admin_rollback.py`` (HTTP
rollback appending a new event over the old ones).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.audit import AuditAction, TargetKind  # noqa: E402
from src.domain.canonical.record import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.infrastructure.audit_ledger import AuditLedger  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.services.conflict_queue import (  # noqa: E402
    ConflictQueueService,
    STATUS_OPEN,
    STATUS_RESOLVED,
)

SUBJECT_ID = "cb.bien-tap"
ACTOR = "cb.quan-tri"


# ---------------------------------------------------------------------------
# PERMISSION_CHANGE — the record_permission_change seam
# ---------------------------------------------------------------------------


def _grant(ledger: AuditLedger, **overrides):
    fields = {
        "actor_id": ACTOR,
        "actor_role": "admin",
        "subject_id": SUBJECT_ID,
        "before_role": None,
        "after_role": "editor",
        "reason": "nâng quyền biên tập theo quyết định 123",
    }
    fields.update(overrides)
    return ledger.record_permission_change(**fields)


def test_a_grant_appends_a_subject_event_naming_grantor_before_after(tmp_path: Path):
    """The event answers "who changed whose role, from what, to what, why?"."""
    ledger = AuditLedger(tmp_path / "audit" / "ledger.jsonl")

    entry = _grant(ledger)

    event = entry.event
    assert event.action is AuditAction.PERMISSION_CHANGE
    assert event.actor_id == ACTOR
    assert event.role == "admin"
    assert event.target == SUBJECT_ID
    assert event.target_kind is TargetKind.SUBJECT
    # Role state travels in the version fields, encoded subject_id@role;
    # a first grant has no prior role, so "none" — never a bare None.
    assert event.before_version == f"{SUBJECT_ID}@none"
    assert event.after_version == f"{SUBJECT_ID}@editor"
    assert event.reason == "nâng quyền biên tập theo quyết định 123"

    # Read back through a second instance over the same file: the event that
    # landed is the event the chain carries, and the chain is intact.
    reopened = AuditLedger(ledger.path)
    found = reopened.find(action=AuditAction.PERMISSION_CHANGE)
    assert len(found) == 1
    assert found[0].event.to_dict() == event.to_dict()
    assert reopened.verify().ok


def test_a_revocation_encodes_the_disabled_state(tmp_path: Path):
    """after_role=None means "no role any more", recorded as ``@disabled``."""
    ledger = AuditLedger(tmp_path / "audit" / "ledger.jsonl")
    _grant(ledger)

    revoked = _grant(
        ledger,
        before_role="editor",
        after_role=None,
        reason="nghỉ việc, thu hồi quyền theo quyết định 200",
    )

    assert revoked.event.before_version == f"{SUBJECT_ID}@editor"
    assert revoked.event.after_version == f"{SUBJECT_ID}@disabled"
    # Rollback-style history: the grant is still there, the revocation is a
    # newer event, and the chain verifies.
    history = ledger.find(action=AuditAction.PERMISSION_CHANGE)
    assert [e.event.after_version for e in history] == [
        f"{SUBJECT_ID}@editor",
        f"{SUBJECT_ID}@disabled",
    ]
    assert ledger.verify().ok


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_a_permission_change_without_a_reason_is_refused_and_nothing_is_written(
    tmp_path: Path, reason
):
    """The mandatory-reason rule bites at the seam, before the file exists."""
    path = tmp_path / "audit" / "ledger.jsonl"
    ledger = AuditLedger(path)

    with pytest.raises(ValueError):
        _grant(ledger, reason=reason)

    assert not path.exists()


# ---------------------------------------------------------------------------
# CONFLICT_OVERRIDE — ConflictQueueService.resolve
# ---------------------------------------------------------------------------

DE_AN = "de-an-tuyen-sinh-2026"
THONG_BAO = "thong-bao-tuyen-sinh-2026"


def _published(store: CanonicalRecordStore, record_id: str, source: str, diem: float) -> None:
    store.write_record(
        CanonicalRecord(
            record_id=record_id,
            record_type=RecordType.DIEM_CHUAN,
            version=1,
            admission_year=2026,
            effective_from="2026-06-01",
            status=RecordStatus.PUBLISHED,
            authority_level=AuthorityLevel.HUSC_OFFICIAL,
            source_document_id=source,
            text=f"Điểm chuẩn ngành CNTT năm 2026 là {diem} điểm.",
            values={"ma_nganh": "cntt", "diem_trung_tuyen_num": diem},
        )
    )


def _conflict_service(tmp_path: Path) -> tuple[ConflictQueueService, AuditLedger]:
    store = CanonicalRecordStore(tmp_path / "canonical_store")
    # Two published sources state different admission scores for the same
    # coordinates — the live detector's definition of a conflict.
    _published(store, "DIEM_CHUAN:NGANH:cntt:2026:de-an", DE_AN, 25.5)
    _published(store, "DIEM_CHUAN:NGANH:cntt:2026:thong-bao", THONG_BAO, 24.0)
    ledger = AuditLedger(tmp_path / "audit" / "ledger.jsonl")
    service = ConflictQueueService(
        store,
        ledger,
        seed_report_path=tmp_path / "no-seed-report.json",
        queue_path=tmp_path / "conflicts" / "queue.json",
    )
    return service, ledger


def test_resolving_a_conflict_appends_the_override_event(tmp_path: Path):
    """The decision an officer makes through the queue reaches the ledger."""
    service, ledger = _conflict_service(tmp_path)
    items, _ = service.refresh()
    assert len(items) == 1
    assert items[0].status == STATUS_OPEN

    resolved = service.resolve(
        items[0].conflict_id,
        winner_source_document_id=DE_AN,
        reason="đề án là nguồn gốc, thông báo lấy số liệu bản nháp",
        actor_id=ACTOR,
        role="reviewer",
    )

    assert resolved.status == STATUS_RESOLVED
    events = ledger.find(action=AuditAction.CONFLICT_OVERRIDE)
    assert len(events) == 1
    event = events[0].event
    assert event.actor_id == ACTOR
    assert event.role == "reviewer"
    # The event points at the winning record version, and names the conflict
    # it decided, so "why does this version win?" is answerable from the
    # ledger alone.
    assert event.after_version == "DIEM_CHUAN:NGANH:cntt:2026:de-an@1"
    assert event.review_id == items[0].conflict_id
    assert event.reason == "đề án là nguồn gốc, thông báo lấy số liệu bản nháp"
    assert ledger.verify().ok


def test_an_override_without_a_reason_refuses_and_leaves_the_item_open(tmp_path: Path):
    """A blank reason cannot resolve: the queue stays honest about the decision."""
    service, ledger = _conflict_service(tmp_path)
    items, _ = service.refresh()

    for reason in (None, "", "  "):
        with pytest.raises(ValueError):
            service.resolve(
                items[0].conflict_id,
                winner_source_document_id=DE_AN,
                reason=reason,
                actor_id=ACTOR,
                role="reviewer",
            )

    assert ledger.find(action=AuditAction.CONFLICT_OVERRIDE) == ()
    assert service.items()[0].status == STATUS_OPEN
