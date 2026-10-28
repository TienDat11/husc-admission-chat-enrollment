"""T15: claim a draft — service-level contract of ``AdminLifecycleService.claim``.

The HTTP seam is asserted in ``tests/routers/test_admin_claim_draft.py``; this
file asserts the same rule where it is enforced, in the service, so a change
to the route's mapping cannot hide a change to the rule itself:

- a claim appends one version: the caller's ``authored_by``, the status still
  ``draft``, every other field unchanged, the prior version intact;
- it is legal only while the record is a draft with ``authored_by = None`` —
  the two refusals name their own case and write nothing;
- the claim is recorded in the durable ledger as a ``claim`` event, carrying
  the same before/after version keys the other lifecycle writes do.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.audit import AuditAction  # noqa: E402
from src.domain.canonical.lifecycle import LifecycleAction  # noqa: E402
from src.domain.canonical.record import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.infrastructure.audit_ledger import AuditLedger  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.services.admin_lifecycle import (  # noqa: E402
    AdminLifecycleService,
    ClaimRefusedError,
    FourEyesViolation,
)

RECORD_ID = "DIEM_CHUAN:NGANH:cntt:2026"
NOW = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc)


def _seed() -> CanonicalRecord:
    """A seed-shaped draft: version 1, ``draft``, ``authored_by = null``."""
    return CanonicalRecord(
        record_id=RECORD_ID,
        record_type=RecordType.DIEM_CHUAN,
        version=1,
        admission_year=2026,
        effective_from="2026-06-01",
        status=RecordStatus.DRAFT,
        authority_level=AuthorityLevel.HUSC_OFFICIAL,
        source_document_id="de-an-tuyen-sinh-2026",
        text="Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 25,5 điểm.",
    )


def _seeded_service(store_root: Path) -> AdminLifecycleService:
    """A service over a store that holds exactly one seed draft — written
    straight to the store, the way the seed script writes it, so the draft is
    truly unclaimed rather than created through the service (which would have
    stamped an author already)."""
    store = CanonicalRecordStore(store_root)
    store.write_record(_seed())
    return AdminLifecycleService(store, clock=lambda: NOW)


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "canonical_store"


def test_a_claim_appends_one_version_authored_by_the_caller(store_root: Path):
    service = _seeded_service(store_root)
    store = service.store

    claimed = service.claim(RECORD_ID, actor_id="cb.bien-tap", role="editor")

    assert claimed.version == 2
    assert claimed.status is RecordStatus.DRAFT
    assert claimed.authored_by == "cb.bien-tap"
    # Everything else is the seed's: the claim is an authorship move, not a
    # content move.
    assert claimed.text == store.get_version(RECORD_ID, 1).text
    assert claimed.published_at is None
    # The seed version stays exactly where it was — still unclaimed, still v1.
    seed = store.get_version(RECORD_ID, 1)
    assert seed.authored_by is None
    assert seed.version == 1
    assert store.versions(RECORD_ID) == (1, 2)


def test_a_claim_leaves_the_draft_ready_for_a_four_eyes_approve(store_root: Path):
    """The point of the operation: after a claim + submit, a *different*
    person can approve — the exact pair T15 acceptance (a) and (b) pin down."""
    service = _seeded_service(store_root)

    claimed = service.claim(RECORD_ID, actor_id="cb.bien-tap", role="editor")
    service.transition(
        RECORD_ID,
        LifecycleAction.SUBMIT,
        actor_id="cb.bien-tap",
        expected_version=claimed.version,
    )

    with pytest.raises(FourEyesViolation):
        service.transition(
            RECORD_ID,
            LifecycleAction.APPROVE,
            actor_id="cb.bien-tap",
            expected_version=3,
        )
    approved = service.transition(
        RECORD_ID,
        LifecycleAction.APPROVE,
        actor_id="cb.phan-bien",
        expected_version=3,
    )
    assert approved.status is RecordStatus.APPROVED


def test_a_reclaim_of_a_draft_claimed_by_someone_else_is_refused(store_root: Path):
    service = _seeded_service(store_root)
    service.claim(RECORD_ID, actor_id="cb.bien-tap", role="editor")

    with pytest.raises(ClaimRefusedError) as raised:
        service.claim(RECORD_ID, actor_id="cb.quan-tri", role="admin")

    assert raised.value.record_id == RECORD_ID
    assert "cb.bien-tap" in str(raised.value)
    assert service.store.versions(RECORD_ID) == (1, 2)


def test_a_reclaim_of_one_s_own_draft_is_refused(store_root: Path):
    """Claiming your own draft again is a version that changes nothing: it is
    refused, documented as a refusal rather than a silent no-op version."""
    service = _seeded_service(store_root)
    service.claim(RECORD_ID, actor_id="cb.bien-tap", role="editor")

    with pytest.raises(ClaimRefusedError) as raised:
        service.claim(RECORD_ID, actor_id="cb.bien-tap", role="editor")

    assert "yours" in str(raised.value)
    assert service.store.versions(RECORD_ID) == (1, 2)


def test_a_claim_on_a_decided_status_is_refused_and_writes_nothing(
    store_root: Path,
):
    """review / approved / published: a claim would demote a decision back to
    draft, so it refuses — naming the status it refused on, leaving the
    record's version count untouched."""
    service = _seeded_service(store_root)
    service.claim(RECORD_ID, actor_id="cb.bien-tap", role="editor")
    submitted = service.transition(
        RECORD_ID,
        LifecycleAction.SUBMIT,
        actor_id="cb.bien-tap",
        expected_version=2,
    )

    with pytest.raises(ClaimRefusedError) as raised:
        service.claim(RECORD_ID, actor_id="cb.quan-tri", role="admin", expected_version=submitted.version)

    assert "review" in str(raised.value)
    assert service.store.versions(RECORD_ID) == (1, 2, 3)


def test_a_claim_is_recorded_in_the_ledger_as_a_claim_event(
    store_root: Path,
    tmp_path: Path,
):
    """The durable half: a claim is one of the lifecycle writes the ledger
    records — ``claim`` with the same before/after version keys the other
    writes carry, so "who claimed the seed draft, and when" is a query, not a
    diff-reading exercise."""
    store = CanonicalRecordStore(store_root)
    ledger = AuditLedger(tmp_path / "audit" / "ledger.jsonl")

    store.write_record(_seed())
    service = AdminLifecycleService(store, ledger=ledger, clock=lambda: NOW)

    claimed = service.claim(
        RECORD_ID,
        actor_id="cb.bien-tap",
        role="editor",
        reason="nhận nháp seed",
    )

    entries = ledger.find(action=AuditAction.CLAIM)
    assert len(entries) == 1
    event = entries[0].event
    assert event.action is AuditAction.CLAIM
    assert event.actor_id == "cb.bien-tap"
    assert event.role == "editor"
    assert event.target == RECORD_ID
    assert event.before_version == f"{RECORD_ID}@1"
    assert event.after_version == claimed.record_version_key == f"{RECORD_ID}@2"
    assert event.reason == "nhận nháp seed"
    # The chain is intact with the new action kind in it.
    assert ledger.verify().ok is True
