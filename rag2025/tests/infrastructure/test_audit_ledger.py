"""Append-only hash-chained audit ledger — ticket 14.

The acceptance this file exists to prove:

- a chain that cannot detect its own break is decoration, so a stored entry is
  mutated **on disk** and ``verify()`` must name the exact broken index;
- rollback is a new event and the prior log is byte-identical afterwards — the
  old bytes are asserted, not the old count;
- there is no way to change or remove a stored event, because no such method
  exists rather than because a guard refuses one.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.audit import AuditAction, AuditEvent, TargetKind  # noqa: E402
from src.domain.canonical.record import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.infrastructure.audit_ledger import (  # noqa: E402
    GENESIS_HASH,
    AuditEntry,
    AuditLedger,
    AuditLedgerCorruptError,
    ChainVerification,
)
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402

RECORD_ID = "DIEM_CHUAN:NGANH:cntt:2026"
NOW = datetime(2026, 9, 13, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "audit" / "ledger.jsonl"


@pytest.fixture
def ledger(ledger_path: Path) -> AuditLedger:
    return AuditLedger(ledger_path)


def _edit_event(actor: str, *, seq_hint: int = 1, event_id: str | None = None) -> AuditEvent:
    """A valid event; ``seq_hint`` only varies the version references."""
    return AuditEvent(
        action=AuditAction.EDIT,
        actor_id=actor,
        target=RECORD_ID,
        timestamp=NOW,
        before_version=f"{RECORD_ID}@{seq_hint}",
        after_version=f"{RECORD_ID}@{seq_hint + 1}",
        event_id=event_id or f"evt-{actor}",
    )


def _store_with_versions(tmp_path: Path, count: int) -> CanonicalRecordStore:
    """A real store holding ``count`` versions, so references can be resolved."""
    store = CanonicalRecordStore(tmp_path / "canonical_store")
    for version in range(1, count + 1):
        store.write_record(
            CanonicalRecord(
                record_id=RECORD_ID,
                record_type=RecordType.DIEM_CHUAN,
                version=version,
                admission_year=2026,
                effective_from="2026-06-01",
                status=RecordStatus.DRAFT,
                authority_level=AuthorityLevel.HUSC_OFFICIAL,
                source_document_id="de-an-tuyen-sinh-2026",
                text=f"Điểm chuẩn ngành CNTT năm 2026, bản {version}.",
            )
        )
    return store


def _lines(path: Path) -> list[bytes]:
    return path.read_bytes().splitlines()


def _rewrite(path: Path, lines: list[bytes]) -> None:
    """Write ``lines`` back as the file — a tamper, done by the test only."""
    path.write_bytes(b"\n".join(lines) + b"\n")


# ---------------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------------


def test_entries_link_to_their_predecessor(ledger, ledger_path):
    first = ledger.append(_edit_event("cb.bien-tap", seq_hint=1))
    second = ledger.append(_edit_event("cb.bien-tap", seq_hint=2))
    third = ledger.append(_edit_event("cb.phan-bien", seq_hint=3))

    assert (first.seq, second.seq, third.seq) == (1, 2, 3)
    assert first.prev_hash == GENESIS_HASH
    assert second.prev_hash == first.entry_hash
    assert third.prev_hash == second.entry_hash
    assert len(_lines(ledger_path)) == 3

    verification = ledger.verify()
    assert verification.ok is True
    assert verification.checked == 3
    assert verification.broken_index is None
    assert verification.last_hash == third.entry_hash


def test_the_chain_continues_across_a_reopen(ledger, ledger_path):
    ledger.append(_edit_event("cb.bien-tap", seq_hint=1))
    reopened = AuditLedger(ledger_path)
    appended = reopened.append(_edit_event("cb.bien-tap", seq_hint=2))

    assert appended.seq == 2
    assert appended.prev_hash == ledger.entries()[0].entry_hash
    assert reopened.verify().ok is True


def test_a_blank_ledger_verifies_as_empty(ledger):
    verification = ledger.verify()
    assert verification.ok is True
    assert verification.checked == 0
    assert verification.last_hash is None


def test_every_required_kind_is_recordable_and_verifies(ledger):
    """The five kinds the ticket names all survive the chain intact.

    Publish and source-document-change are emitted by the lifecycle service
    (see ``tests/routers/test_admin_audit.py``); rollback, permission change and
    conflict override reach the ledger through this same append, which is the
    seam the endpoints that perform them call.
    """
    kinds = (
        AuditAction.PUBLISH,
        AuditAction.ROLLBACK,
        AuditAction.PERMISSION_CHANGE,
        AuditAction.SOURCE_DOCUMENT_CHANGE,
        AuditAction.CONFLICT_OVERRIDE,
    )
    for index, kind in enumerate(kinds):
        is_version_event = kind is not AuditAction.PERMISSION_CHANGE
        ledger.append(
            AuditEvent(
                action=kind,
                actor_id="cb.quan-tri",
                role="admin",
                target="cb.bien-tap" if kind is AuditAction.PERMISSION_CHANGE else RECORD_ID,
                target_kind=(
                    TargetKind.SUBJECT if kind is AuditAction.PERMISSION_CHANGE else TargetKind.RECORD
                ),
                timestamp=NOW,
                before_version=f"{RECORD_ID}@{index + 1}" if is_version_event else None,
                after_version=f"{RECORD_ID}@{index + 2}" if is_version_event else None,
                reason="lý do của quyết định" if kind is not AuditAction.PUBLISH else None,
                release_id="R-2026-09-13" if kind is AuditAction.PUBLISH else None,
                event_id=f"evt-{index}",
            )
        )

    assert ledger.verify().ok is True
    for kind in kinds:
        assert len(ledger.find(action=kind)) == 1


# ---------------------------------------------------------------------------
# tamper detection
# ---------------------------------------------------------------------------


def test_verify_names_the_exact_index_of_an_edited_entry(ledger, ledger_path):
    """The acceptance case: mutate a stored entry on disk, find it by index."""
    ledger.append(_edit_event("cb.bien-tap", seq_hint=1, event_id="evt-1"))
    ledger.append(_edit_event("cb.bien-tap", seq_hint=2, event_id="evt-2"))
    ledger.append(_edit_event("cb.phan-bien", seq_hint=3, event_id="evt-3"))

    lines = _lines(ledger_path)
    tampered = json.loads(lines[1])
    tampered["event"]["actor_id"] = "ke.gian-lan"
    lines[1] = json.dumps(tampered, ensure_ascii=False, sort_keys=True).encode("utf-8")
    _rewrite(ledger_path, lines)

    verification = ledger.verify()
    assert verification.ok is False
    assert verification.broken_index == 1
    assert verification.checked == 1
    assert "entry_hash" in verification.reason


def test_verify_catches_a_tamper_whose_hash_was_recomputed_too(ledger, ledger_path):
    """An attacker who knows the scheme must still break the link to the next
    entry — editing an entry and its own hash cannot hide the edit."""
    ledger.append(_edit_event("cb.bien-tap", seq_hint=1, event_id="evt-1"))
    ledger.append(_edit_event("cb.bien-tap", seq_hint=2, event_id="evt-2"))
    ledger.append(_edit_event("cb.phan-bien", seq_hint=3, event_id="evt-3"))

    lines = _lines(ledger_path)
    tampered = json.loads(lines[1])
    tampered["event"]["actor_id"] = "ke.gian-lan"
    forged = AuditEvent.from_dict(tampered["event"])
    entry = ledger.entries()[1]
    tampered["entry_hash"] = AuditEntry(
        seq=entry.seq, prev_hash=entry.prev_hash, entry_hash="", event=forged
    ).computed_hash()
    lines[1] = json.dumps(tampered, ensure_ascii=False, sort_keys=True).encode("utf-8")
    _rewrite(ledger_path, lines)

    verification = ledger.verify()
    assert verification.ok is False
    assert verification.broken_index == 2
    assert "prev_hash" in verification.reason


def test_verify_catches_a_deleted_entry_in_the_middle(ledger, ledger_path):
    ledger.append(_edit_event("cb.bien-tap", seq_hint=1, event_id="evt-1"))
    ledger.append(_edit_event("cb.bien-tap", seq_hint=2, event_id="evt-2"))
    ledger.append(_edit_event("cb.phan-bien", seq_hint=3, event_id="evt-3"))

    lines = _lines(ledger_path)
    _rewrite(ledger_path, [lines[0], lines[2]])

    verification = ledger.verify()
    assert verification.ok is False
    assert verification.broken_index == 1


def test_verify_catches_a_deleted_reason_on_a_reject(ledger, ledger_path):
    """Stripping a justification is tampering, and the model refuses it on
    read before the hash even matters."""
    ledger.append(
        AuditEvent(
            action=AuditAction.REJECT,
            actor_id="cb.phan-bien",
            target=RECORD_ID,
            timestamp=NOW,
            reason="số liệu chưa đối chiếu được với đề án",
            event_id="evt-reject",
        )
    )

    lines = _lines(ledger_path)
    tampered = json.loads(lines[0])
    tampered["event"]["reason"] = None
    lines[0] = json.dumps(tampered, ensure_ascii=False, sort_keys=True).encode("utf-8")
    _rewrite(ledger_path, lines)

    verification = ledger.verify()
    assert verification.ok is False
    assert verification.broken_index == 0


def test_a_truncated_tail_is_the_one_thing_the_chain_cannot_see(ledger, ledger_path):
    """Documented boundary: the head hash is the anchor for that, which is why
    ``verify()`` returns it."""
    ledger.append(_edit_event("cb.bien-tap", seq_hint=1))
    ledger.append(_edit_event("cb.bien-tap", seq_hint=2))
    lines = _lines(ledger_path)
    _rewrite(ledger_path, lines[:1])

    verification = ledger.verify()
    assert verification.ok is True
    assert verification.checked == 1
    assert verification.last_hash == ledger.entries()[0].entry_hash


# ---------------------------------------------------------------------------
# append-only: no update, no delete
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["update", "delete", "remove", "rewrite", "truncate", "pop", "clear", "purge", "replace"],
)
def test_the_ledger_has_no_mutating_method(ledger, name):
    """The ticket's "no endpoint deletes or edits an event" is structural: the
    operation does not exist to be guarded."""
    assert not hasattr(ledger, name)


def test_appending_never_rewrites_what_is_already_on_disk(ledger, ledger_path):
    ledger.append(_edit_event("cb.bien-tap", seq_hint=1))
    first_bytes = ledger_path.read_bytes()
    ledger.append(_edit_event("cb.bien-tap", seq_hint=2))

    assert ledger_path.read_bytes().startswith(first_bytes)


def test_a_rollback_appends_a_new_event_and_leaves_the_prior_log_byte_identical(
    tmp_path, ledger, ledger_path
):
    """Rollback is a new event; the old log survives byte for byte.

    The rollback *write* belongs to ticket 22; what is asserted here is the
    ledger half of it — the record is appended, the earlier bytes are untouched,
    and both version references resolve through the store.
    """
    store = _store_with_versions(tmp_path, 2)
    published = ledger.append(
        AuditEvent(
            action=AuditAction.PUBLISH,
            actor_id="cb.xuat-ban",
            target=RECORD_ID,
            timestamp=NOW,
            before_version=f"{RECORD_ID}@1",
            after_version=f"{RECORD_ID}@2",
            release_id="R-2026-09-13",
            event_id="evt-publish",
        )
    )
    prior_bytes = ledger_path.read_bytes()

    # The rollback appends version 3 carrying version 1's content — an append,
    # never a rewrite, which is the store's own rule.
    reverted = store.get_version(RECORD_ID, 1).model_copy(update={"version": 3})
    store.write_record(reverted)
    rollback = ledger.append(
        AuditEvent(
            action=AuditAction.ROLLBACK,
            actor_id="cb.quan-tri",
            role="admin",
            target=RECORD_ID,
            timestamp=NOW,
            before_version=f"{RECORD_ID}@2",
            after_version=f"{RECORD_ID}@3",
            reason="hoàn nguyên vì điểm chuẩn công bố sai",
            event_id="evt-rollback",
        )
    )

    after_bytes = ledger_path.read_bytes()
    assert after_bytes.startswith(prior_bytes), "the earlier log must be untouched"
    assert len(after_bytes) > len(prior_bytes)
    assert rollback.seq == published.seq + 1
    assert rollback.prev_hash == published.entry_hash
    assert ledger.verify().ok is True

    # Both sides of the rollback resolve through the store.
    for reference in (rollback.event.before_version, rollback.event.after_version):
        record_id, _, version = reference.rpartition("@")
        assert store.get_version(record_id, int(version)).record_version_key == reference


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------


def test_find_filters_by_target_actor_and_action(ledger):
    ledger.append(_edit_event("cb.bien-tap", seq_hint=1))
    ledger.append(_edit_event("cb.phan-bien", seq_hint=2))
    ledger.append(
        AuditEvent(
            action=AuditAction.PERMISSION_CHANGE,
            actor_id="cb.quan-tri",
            target="cb.bien-tap",
            target_kind=TargetKind.SUBJECT,
            timestamp=NOW,
            reason="nâng quyền biên tập theo quyết định 123",
            event_id="evt-perm",
        )
    )

    assert len(ledger.find(actor_id="cb.phan-bien")) == 1
    assert len(ledger.find(action=AuditAction.EDIT)) == 2
    assert len(ledger.find(target="cb.bien-tap")) == 1
    assert len(ledger.find(target=RECORD_ID, actor_id="cb.bien-tap")) == 1
    assert ledger.find(action="not-an-action") == ()


def test_get_returns_the_entry_by_event_id(ledger):
    ledger.append(_edit_event("cb.bien-tap", seq_hint=1, event_id="evt-1"))
    ledger.append(_edit_event("cb.phan-bien", seq_hint=2, event_id="evt-2"))

    assert ledger.get("evt-2").event.actor_id == "cb.phan-bien"
    assert ledger.get("evt-1").seq == 1
    assert ledger.get("evt-nonexistent") is None


def test_a_corrupt_line_makes_the_strict_read_raise_rather_than_skip(ledger, ledger_path):
    ledger.append(_edit_event("cb.bien-tap", seq_hint=1))
    with ledger_path.open("ab") as handle:
        handle.write(b"{not json at all}\n")

    with pytest.raises(AuditLedgerCorruptError):
        ledger.entries()
    # ...and the diagnostic read reports it instead of raising.
    verification: ChainVerification = ledger.verify()
    assert verification.ok is False
    assert verification.broken_index == 1


def test_appending_to_a_corrupt_tail_refuses_rather_than_extends(ledger, ledger_path):
    ledger.append(_edit_event("cb.bien-tap", seq_hint=1))
    with ledger_path.open("ab") as handle:
        handle.write(b"{not json at all}\n")

    with pytest.raises(AuditLedgerCorruptError):
        ledger.append(_edit_event("cb.bien-tap", seq_hint=2))
