"""TDD: rag2025/src/services/reindex_outbox.py — ticket 15.

Acceptance (``issues/15-incremental-reindex-outbox.md``):
- the outbox identifies the affected canonical id set for a published version;
- every entry carries the corpus stamp, taken from ticket 06's ``version_stamp()``
  rather than from a constant invented here — that stamp is the trigger a
  record-level id set cannot provide;
- publishing the same version twice is one entry (the drain is repeatable);
- a malformed line is an error, never a silently skipped change.

Everything is asserted through the outbox's public surface and the bytes it
wrote to disk.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.services.chunk_identity import version_stamp  # noqa: E402
from src.services.reindex_outbox import (  # noqa: E402
    OutboxAppendError,
    OutboxEntry,
    OutboxError,
    OutboxFormatError,
    ReindexOutbox,
)

RELEASE_ID = "R-2026-09-01"
RECORD_ID = "DIEM_CHUAN:NGANH:cntt:2026"
OLD_ID = "DIEM_CHUAN:NGANH:cntt:2025"
SUPERSEDING_ID = "DIEM_CHUAN:NGANH:cntt:2026:revision"

def _record(record_id: str = RECORD_ID, version: int = 1, **overrides) -> CanonicalRecord:
    fields = {
        "record_id": record_id,
        "record_type": RecordType.DIEM_CHUAN,
        "version": version,
        "admission_year": 2026,
        "effective_from": date(2026, 6, 1),
        "effective_to": date(2026, 12, 31),
        "status": RecordStatus.PUBLISHED,
        "authority_level": AuthorityLevel.HUSC_OFFICIAL,
        "source_document_id": "de-an-tuyen-sinh-2026",
        "source_url": "https://tuyensinh.husc.edu.vn/de-an-2026.pdf",
        "source_page": 12,
        "published_at": datetime(2026, 9, 1, tzinfo=timezone.utc),
        "text": "Điểm chuẩn ngành Công nghệ thông tin (mã ngành 7480201) là 25,5 điểm.",
    }
    fields.update(overrides)
    return CanonicalRecord(**fields)

@pytest.fixture
def store(tmp_path: Path) -> CanonicalRecordStore:
    return CanonicalRecordStore(tmp_path / "canonical")

@pytest.fixture
def outbox(tmp_path: Path) -> ReindexOutbox:
    return ReindexOutbox(tmp_path / "outbox" / "reindex_outbox.jsonl")

def test_entry_carries_the_stamp_ticket_06_defines(outbox: ReindexOutbox) -> None:
    """The stamp is ticket 06's, not a constant this module invented."""
    entry = outbox.entry(_record(), release_id=RELEASE_ID)

    assert entry.stamp == version_stamp()
    assert set(entry.stamp) == {
        "chunker_version",
        "unit_rules_version",
        "chunk_manifest_version",
    }

def test_entry_is_refused_when_it_carries_no_stamp() -> None:
    """A stamp-less entry cannot answer "is this corpus still current?"."""
    with pytest.raises(OutboxFormatError):
        OutboxEntry(
            entry_id="x",
            record_id=RECORD_ID,
            version=1,
            record_version_key=f"{RECORD_ID}@1",
            affected_record_ids=(RECORD_ID,),
            stamp={},
            appended_at="2026-09-01T00:00:00+00:00",
        )

def test_entry_always_lists_its_own_record_as_affected() -> None:
    """An entry that does not name the record it published re-indexes nothing."""
    with pytest.raises(OutboxFormatError):
        OutboxEntry(
            entry_id="x",
            record_id=RECORD_ID,
            version=1,
            record_version_key=f"{RECORD_ID}@1",
            affected_record_ids=(OLD_ID,),
            stamp=version_stamp(),
            appended_at="2026-09-01T00:00:00+00:00",
        )

def test_affected_ids_name_the_record_and_what_it_supersedes(
    outbox: ReindexOutbox,
) -> None:
    """A publication changes the served answer for its own id and its predecessor."""
    record = _record(record_id=SUPERSEDING_ID, supersedes=OLD_ID)

    entry = outbox.record_publication(record, release_id=RELEASE_ID)

    assert entry.affected_record_ids == tuple(sorted({SUPERSEDING_ID, OLD_ID}))

def test_affected_ids_read_a_version_key_spelling(outbox: ReindexOutbox) -> None:
    """``supersedes`` may name a bare id or a ``record_id@version`` key."""
    record = _record(record_id=SUPERSEDING_ID, supersedes=f"{OLD_ID}@3")

    entry = outbox.record_publication(record, release_id=RELEASE_ID)

    assert entry.affected_record_ids == tuple(sorted({SUPERSEDING_ID, OLD_ID}))

def test_affected_ids_include_a_record_that_supersedes_this_one(
    outbox: ReindexOutbox, store: CanonicalRecordStore
) -> None:
    """The live/superseded decision for this id changed too — it is affected."""
    store.write_record(_record(record_id=OLD_ID))
    store.write_record(_record(record_id=SUPERSEDING_ID, supersedes=OLD_ID))
    superseding = store.get_record(SUPERSEDING_ID).latest

    entry = outbox.record_publication(superseding, release_id=RELEASE_ID, store=store)

    assert OLD_ID in entry.affected_record_ids
    assert SUPERSEDING_ID in entry.affected_record_ids

def test_publishing_the_same_version_twice_appends_one_entry(
    outbox: ReindexOutbox,
) -> None:
    """Idempotent by construction — not by a caller remembering to check."""
    record = _record()

    first = outbox.record_publication(record, release_id=RELEASE_ID)
    second = outbox.record_publication(record, release_id=RELEASE_ID)

    assert first.entry_id == second.entry_id
    assert len(outbox.entries()) == 1
    assert len(outbox.path.read_text(encoding="utf-8").splitlines()) == 1

def test_a_new_version_of_the_same_record_is_a_new_entry(
    outbox: ReindexOutbox,
) -> None:
    outbox.record_publication(_record(version=1), release_id=RELEASE_ID)
    outbox.record_publication(_record(version=2), release_id=RELEASE_ID)

    assert len(outbox.entries()) == 2
    assert outbox.affected_record_ids() == (RECORD_ID,)

def test_entries_round_trip_through_the_file(outbox: ReindexOutbox) -> None:
    record = _record(record_id=SUPERSEDING_ID, supersedes=OLD_ID)
    written = outbox.record_publication(record, release_id=RELEASE_ID)

    reopened = ReindexOutbox(outbox.path)
    [read_back] = reopened.entries()

    assert read_back == written
    assert read_back.affected_record_ids == written.affected_record_ids
    assert read_back.stamp == written.stamp

def test_drain_returns_pending_entries_once_and_advances_the_cursor(
    outbox: ReindexOutbox,
) -> None:
    outbox.record_publication(_record(record_id="A"), release_id=RELEASE_ID)
    outbox.record_publication(_record(record_id="B"), release_id=RELEASE_ID)

    first = outbox.drain()
    second = outbox.drain()

    assert [e.record_id for e in first.entries] == ["A", "B"]
    assert first.affected_record_ids == ("A", "B")
    assert second.entries == ()
    assert second.affected_record_ids == ()

def test_pending_ids_cover_only_undrained_entries(outbox: ReindexOutbox) -> None:
    outbox.record_publication(_record(record_id="A"), release_id=RELEASE_ID)
    outbox.drain()
    outbox.record_publication(_record(record_id="B"), release_id=RELEASE_ID)

    assert outbox.affected_record_ids() == ("A", "B")
    assert outbox.pending_affected_record_ids() == ("B",)

def test_cursor_survives_reopening_the_outbox(outbox: ReindexOutbox) -> None:
    outbox.record_publication(_record(record_id="A"), release_id=RELEASE_ID)
    outbox.drain()

    assert ReindexOutbox(outbox.path).pending_affected_record_ids() == ()

def test_every_entry_states_the_stamp_of_the_run_that_wrote_it(
    outbox: ReindexOutbox,
) -> None:
    """Not just the first: a corpus can span publications across a code change."""
    outbox.record_publication(_record(record_id="A"), release_id=RELEASE_ID)
    outbox.record_publication(_record(record_id="B"), release_id=RELEASE_ID)

    assert [e.stamp for e in outbox.entries()] == [version_stamp(), version_stamp()]

def test_a_malformed_line_is_an_error_not_a_skipped_change(
    outbox: ReindexOutbox,
) -> None:
    outbox.record_publication(_record(), release_id=RELEASE_ID)
    with outbox.path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")

    with pytest.raises(OutboxFormatError):
        outbox.entries()

def test_an_entry_missing_its_stamp_on_disk_is_an_error(
    outbox: ReindexOutbox,
) -> None:
    payload = outbox.entry(_record(), release_id=RELEASE_ID).to_dict()
    payload.pop("stamp")
    outbox.path.parent.mkdir(parents=True, exist_ok=True)
    outbox.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(OutboxFormatError):
        outbox.entries()

def test_an_unwritable_outbox_names_the_ids_it_failed_to_queue(
    tmp_path: Path,
) -> None:
    """The publish has already committed; the failure is the missing trigger.

    A bare ``OSError`` would read as "the publish failed". It did not — the
    version is stored. What is missing is the entry that would re-index it, so
    the error has to name the ids, or the divergence is only discovered as a
    stale answer months later.
    """
    blocked = tmp_path / "outbox"
    blocked.mkdir(parents=True)
    (blocked / "reindex_outbox.jsonl").mkdir()  # a directory where the file goes
    outbox = ReindexOutbox(blocked / "reindex_outbox.jsonl")
    record = _record(record_id=SUPERSEDING_ID, supersedes=OLD_ID)

    with pytest.raises(OutboxAppendError) as excinfo:
        outbox.record_publication(record, release_id=RELEASE_ID)

    message = str(excinfo.value)
    assert SUPERSEDING_ID in message and OLD_ID in message
    assert "already committed" in message
    assert outbox.pending() == ()

def test_retrying_an_append_that_failed_is_idempotent(tmp_path: Path) -> None:
    """So the caller's recovery is "call it again", not "reconcile by hand"."""
    outbox = ReindexOutbox(tmp_path / "outbox" / "reindex_outbox.jsonl")
    record = _record()
    outbox.record_publication(record, release_id=RELEASE_ID)

    outbox.record_publication(record, release_id=RELEASE_ID)

    assert len(outbox.entries()) == 1

def test_a_cursor_that_cannot_be_written_loses_no_work(tmp_path: Path) -> None:
    """The entries stay pending, so the next drain returns them — safe, not silent."""
    outbox = ReindexOutbox(tmp_path / "outbox" / "reindex_outbox.jsonl")
    outbox.record_publication(_record(record_id="A"), release_id=RELEASE_ID)
    outbox.cursor_path.mkdir(parents=True)

    with pytest.raises(OutboxError, match="cursor is behind"):
        outbox.drain()

    assert outbox.cursor() == 0
    assert outbox.pending_affected_record_ids() == ("A",)
