"""TDD: rag2025/src/services/published_materializer.py — ticket 05.

Acceptance (``issues/05-published-only-materializer.md``):
- draft / review / approved records are excluded, each by its own test — a
  combined "not published" test would pass even if one status leaked;
- published records are admitted, with their provenance;
- archived records are filtered by the question's temporal policy, and
  superseded records are filtered — ``superseded`` being a relation derived from
  the store, never a status value;
- ``supersession_index()`` is built once per materialize and the per-record
  ``is_superseded()`` is never called (that call rescans every stored version,
  making materialization O(n^2));
- two runs over one release produce a byte-identical manifest, and
  re-materializing does not duplicate a record;
- the release id is recorded in the output manifest;
- an empty store does not produce a manifest that reads as a successful
  materialization of real data.

The last acceptance item runs on ticket 28's seed store — real records extracted
from the official PDFs and the existing corpus. The seed store is not a fixture,
so that test skips (loudly, never silently passing) when it is absent.

Everything is asserted through the materializer's public surface and the bytes
it wrote.
"""
from __future__ import annotations

import json
import shutil
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
from src.domain.canonical.lifecycle import LifecycleAction  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.services.admin_lifecycle import AdminLifecycleService  # noqa: E402
from src.services.published_materializer import (  # noqa: E402
    MaterializationState,
    PublishedMaterializer,
    TemporalPolicy,
)

from config.settings import RAGSettings  # noqa: E402

RELEASE_ID = "R-2026-08-01"
RECORD_ID = "DIEM_CHUAN:NGANH:cntt:2026"
OLD_ID = "DIEM_CHUAN:NGANH:cntt:2025"
NEW_ID = "DIEM_CHUAN:NGANH:cntt:2026:revision"
DOC_ID = "de-an-tuyen-sinh-2026"
URL = "https://tuyensinh.husc.edu.vn/de-an-2026.pdf"
TEXT = "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 25,5 điểm."
EDITED_TEXT = "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 26,0 điểm (đính chính)."

# Ticket 28's real seed store — the one the APP reads, i.e. the configured
# ``CANONICAL_STORE_ROOT``. Hardcoding a parallel path here made this test skip
# silently forever once the seeder's default was aligned with settings.
SEED_STORE = Path(RAGSettings().CANONICAL_STORE_ROOT)

def _record(**overrides) -> CanonicalRecord:
    """A minimal valid published record; each test overrides what it probes."""
    fields = {
        "record_id": RECORD_ID,
        "record_type": RecordType.DIEM_CHUAN,
        "version": 1,
        "admission_year": 2026,
        "effective_from": date(2026, 6, 1),
        "effective_to": date(2026, 12, 31),
        "status": RecordStatus.PUBLISHED,
        "authority_level": AuthorityLevel.HUSC_OFFICIAL,
        "source_document_id": DOC_ID,
        "source_url": URL,
        "source_page": 12,
        "published_at": datetime(2026, 6, 1, tzinfo=timezone.utc),
        "text": TEXT,
    }
    fields.update(overrides)
    return CanonicalRecord(**fields)

@pytest.fixture
def store(tmp_path: Path) -> CanonicalRecordStore:
    return CanonicalRecordStore(tmp_path / "canonical")

@pytest.fixture
def materializer(store: CanonicalRecordStore) -> PublishedMaterializer:
    return PublishedMaterializer(store)

def _admitted_ids(result) -> set[str]:
    return {record.record_id for record in result.admitted}

def _exclusion(result, record_id: str) -> str | None:
    for item in result.excluded:
        if item.record_id == record_id:
            return item.reason
    return None

def _write(store: CanonicalRecordStore, **overrides) -> CanonicalRecord:
    return store.write_record(_record(**overrides))

def _counting(store: CanonicalRecordStore, *names: str) -> dict[str, int]:
    """Count how often the materializer calls each named store method."""
    counts = {name: 0 for name in names}
    for name in names:
        original = getattr(store, name)

        def wrapper(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        setattr(store, name, wrapper)
    return counts

# ---------------------------------------------------------------------------
# status gates — one test per status, never one combined "not published" test
# ---------------------------------------------------------------------------

def test_a_draft_record_is_excluded(store, materializer, tmp_path):
    AdminLifecycleService(store).create(
        _record(status=RecordStatus.DRAFT), actor_id="author"
    )

    result = materializer.materialize(RELEASE_ID, tmp_path / "out")

    assert _admitted_ids(result) == set()
    assert _exclusion(result, RECORD_ID) == "draft"

def test_a_record_under_review_is_excluded(store, materializer, tmp_path):
    service = AdminLifecycleService(store)
    service.create(_record(status=RecordStatus.DRAFT), actor_id="author")
    service.transition(RECORD_ID, LifecycleAction.SUBMIT, actor_id="author")

    result = materializer.materialize(RELEASE_ID, tmp_path / "out")

    assert _admitted_ids(result) == set()
    assert _exclusion(result, RECORD_ID) == "review"

def test_an_approved_record_is_excluded(store, materializer, tmp_path):
    """``approved`` is the status immediately before ``published``.

    It is built through the real lifecycle (draft -> review -> approved) rather
    than hand-written, so this pins the status the service actually produces —
    a gate that admitted anything "past review" would let unreleased numbers
    reach students through the narrowest possible margin.
    """
    service = AdminLifecycleService(store)
    service.create(_record(status=RecordStatus.DRAFT), actor_id="author")
    service.transition(RECORD_ID, LifecycleAction.SUBMIT, actor_id="author")
    approved = service.transition(
        RECORD_ID, LifecycleAction.APPROVE, actor_id="reviewer"
    )
    assert approved.status is RecordStatus.APPROVED

    result = materializer.materialize(RELEASE_ID, tmp_path / "out")

    assert _admitted_ids(result) == set()
    assert _exclusion(result, RECORD_ID) == "approved"
    assert result.records_path.read_bytes() == b""

def test_a_published_record_is_admitted_with_its_provenance(store, materializer, tmp_path):
    service = AdminLifecycleService(store)
    service.create(_record(status=RecordStatus.DRAFT), actor_id="author")
    service.transition(RECORD_ID, LifecycleAction.SUBMIT, actor_id="author")
    service.transition(RECORD_ID, LifecycleAction.APPROVE, actor_id="reviewer")
    published = service.transition(RECORD_ID, LifecycleAction.PUBLISH, actor_id="publisher")

    result = materializer.materialize(RELEASE_ID, tmp_path / "out")

    assert _admitted_ids(result) == {RECORD_ID}
    assert result.state is MaterializationState.MATERIALIZED
    # The index input carries the fact's provenance and effective window, so a
    # chunk built from it can still be traced to its source page (ticket 06).
    line = json.loads(result.records_path.read_text(encoding="utf-8").strip())
    assert line["record_id"] == RECORD_ID
    assert line["version"] == published.version
    assert line["status"] == "published"
    assert line["source_document_id"] == DOC_ID
    assert line["source_page"] == 12
    assert line["effective_from"] == "2026-06-01"

# ---------------------------------------------------------------------------
# archived (temporal policy) and superseded (derived relation)
# ---------------------------------------------------------------------------

def test_an_archived_record_is_excluded_by_the_default_temporal_policy(
    store, materializer, tmp_path
):
    _write(store, admission_year=2025, effective_from=date(2025, 1, 1),
           effective_to=date(2025, 12, 31))
    store.edit_record(RECORD_ID, status=RecordStatus.ARCHIVED)

    result = materializer.materialize(RELEASE_ID, tmp_path / "out")

    assert _admitted_ids(result) == set()
    assert _exclusion(result, RECORD_ID) == "archived"
    assert result.manifest["temporal_policy"] == {"as_of": None}

def test_an_archived_record_is_admitted_when_the_question_is_about_its_window(
    store, tmp_path
):
    _write(store, admission_year=2025, effective_from=date(2025, 1, 1),
           effective_to=date(2025, 12, 31))
    store.edit_record(RECORD_ID, status=RecordStatus.ARCHIVED)

    result = PublishedMaterializer(
        store, policy=TemporalPolicy(as_of=date(2025, 6, 1))
    ).materialize(RELEASE_ID, tmp_path / "out")

    assert _admitted_ids(result) == {RECORD_ID}
    assert result.manifest["temporal_policy"] == {"as_of": "2025-06-01"}

def test_an_archived_record_is_still_excluded_when_the_date_is_outside_its_window(
    store, tmp_path
):
    """Naming a date is not a blanket "include archived".

    The record was not the live statement on that date, so admitting it would
    answer a 2026 question with retracted 2025 data.
    """
    _write(store, admission_year=2025, effective_from=date(2025, 1, 1),
           effective_to=date(2025, 12, 31))
    store.edit_record(RECORD_ID, status=RecordStatus.ARCHIVED)

    result = PublishedMaterializer(
        store, policy=TemporalPolicy(as_of=date(2026, 6, 1))
    ).materialize(RELEASE_ID, tmp_path / "out")

    assert _admitted_ids(result) == set()
    assert _exclusion(result, RECORD_ID) == "archived"

def test_a_superseded_record_is_excluded_while_its_status_is_still_published(
    store, materializer, tmp_path
):
    """``superseded`` is a relation, not a status.

    The old record is named by another *published* record's ``supersedes``, and
    its own status is still ``published`` — nothing has run the ``supersede``
    action on it yet. That window is exactly what the relation closes; a
    materializer that trusted the status alone would serve both numbers.
    """
    _write(store, record_id=OLD_ID, admission_year=2025,
           effective_from=date(2025, 1, 1), effective_to=date(2025, 12, 31))
    _write(store, record_id=NEW_ID, supersedes=OLD_ID)

    result = materializer.materialize(RELEASE_ID, tmp_path / "out")

    assert _admitted_ids(result) == {NEW_ID}
    assert _exclusion(result, OLD_ID) == "superseded"
    assert store.get_record(OLD_ID).latest.status is RecordStatus.PUBLISHED
    assert store.get_record(OLD_ID).latest.status is not RecordStatus.ARCHIVED

def test_a_draft_naming_an_older_record_does_not_supersede_it(store, materializer, tmp_path):
    """Only published versions are candidates; the relation is the union over them."""
    _write(store, record_id=OLD_ID, admission_year=2025,
           effective_from=date(2025, 1, 1), effective_to=date(2025, 12, 31))
    _write(store, record_id=NEW_ID, status=RecordStatus.DRAFT, supersedes=OLD_ID)

    result = materializer.materialize(RELEASE_ID, tmp_path / "out")

    assert _admitted_ids(result) == {OLD_ID}
    assert _exclusion(result, NEW_ID) == "draft"

# ---------------------------------------------------------------------------
# which version of a record is live
# ---------------------------------------------------------------------------

def test_a_draft_edit_does_not_unpublish_the_version_it_edits(store, materializer, tmp_path):
    """An edit appends a draft; the published version underneath stays live."""
    _write(store)
    store.edit_record(RECORD_ID, status=RecordStatus.DRAFT, text=EDITED_TEXT)

    result = materializer.materialize(RELEASE_ID, tmp_path / "out")

    assert [record.version for record in result.admitted] == [1]
    assert EDITED_TEXT not in result.records_path.read_text(encoding="utf-8")
    assert TEXT in result.records_path.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# cost: one supersession index per materialize, linear version reads
# ---------------------------------------------------------------------------

def test_supersession_index_is_built_once_and_the_per_record_predicate_is_never_called(
    store, tmp_path
):
    for index in range(5):
        _write(store, record_id=f"DIEM_CHUAN:NGANH:nganh{index}:2026")
    counts = _counting(store, "supersession_index", "is_superseded", "superseding_record")

    PublishedMaterializer(store).materialize(RELEASE_ID, tmp_path / "out")

    assert counts["supersession_index"] == 1
    # Both of these rebuild the whole index per call, so one call per record
    # would be O(n^2) in the size of the store.
    assert counts["is_superseded"] == 0
    assert counts["superseding_record"] == 0

def test_version_reads_grow_linearly_with_the_store_not_quadratically(tmp_path):
    """The scaling property itself, not the shape of the implementation.

    A materialization reads each stored version once to build the supersession
    index and once more for the live version of each record: 2n. Answering
    supersession per record instead re-reads every stored version for every
    record, so quadrupling the store multiplies the reads by ~16 instead of ~4.
    """
    reads: dict[int, int] = {}
    for size in (60, 240):
        store = CanonicalRecordStore(tmp_path / f"store{size}")
        for index in range(size):
            _write(store, record_id=f"DIEM_CHUAN:NGANH:nganh{index:04d}:2026")
        counts = _counting(store, "get_version")

        result = PublishedMaterializer(store).materialize(
            RELEASE_ID, tmp_path / f"out{size}"
        )

        assert len(result.admitted) == size
        reads[size] = counts["get_version"]

    assert reads[240] <= 6 * reads[60]
    assert reads[240] <= 3 * 240

def test_a_record_directory_with_no_version_files_does_not_abort_the_release(
    store, materializer, tmp_path
):
    """A crash between directory creation and the version write leaves no fact.

    It must not take the whole release down with it: the other records still
    materialize, and it is not counted as a record.
    """
    _write(store)
    (store.root / "records" / "rec-orphan_r").mkdir(parents=True)

    result = materializer.materialize(RELEASE_ID, tmp_path / "out")

    assert _admitted_ids(result) == {RECORD_ID}
    assert result.manifest["stored_record_count"] == 1

# ---------------------------------------------------------------------------
# idempotency, traceability, empty-store honesty
# ---------------------------------------------------------------------------

def _mixed_store(store: CanonicalRecordStore) -> None:
    _write(store, record_id=OLD_ID, admission_year=2025,
           effective_from=date(2025, 1, 1), effective_to=date(2025, 12, 31))
    _write(store, record_id=NEW_ID, supersedes=OLD_ID)
    _write(store, record_id="HOC_PHI:NGANH:ktpm:2026")
    _write(store, record_id="CHI_TIEU:NGANH:ktpm:2026", status=RecordStatus.DRAFT)

def test_two_runs_over_the_same_release_are_byte_identical(store, tmp_path):
    _mixed_store(store)

    first = PublishedMaterializer(store).materialize(RELEASE_ID, tmp_path / "run1")
    second = PublishedMaterializer(store).materialize(RELEASE_ID, tmp_path / "run2")

    assert first.records_path.read_bytes() == second.records_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.manifest["records_sha256"] == second.manifest["records_sha256"]
    assert first.manifest["records"] == second.manifest["records"]

def test_rematerializing_into_the_same_directory_does_not_duplicate_records(store, tmp_path):
    _mixed_store(store)
    out = tmp_path / "out"

    first = PublishedMaterializer(store).materialize(RELEASE_ID, out)
    second = PublishedMaterializer(store).materialize(RELEASE_ID, out)

    lines = [line for line in second.records_path.read_text(encoding="utf-8").splitlines() if line]
    keys = [json.loads(line)["record_id"] for line in lines]
    assert len(lines) == len(second.admitted) == len(first.admitted)
    assert len(keys) == len(set(keys)), "one release must not contribute a record twice"
    assert second.records_path.read_bytes() == first.records_path.read_bytes()
    assert second.manifest["record_count"] == first.manifest["record_count"]

def test_the_output_manifest_records_the_release_id(store, materializer, tmp_path):
    _write(store)

    result = materializer.materialize("R-2026-08-01", tmp_path / "out")

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["release_id"] == "R-2026-08-01"
    assert result.release_id == "R-2026-08-01"

def test_a_blank_release_id_is_refused(store, materializer, tmp_path):
    with pytest.raises(ValueError):
        materializer.materialize("   ", tmp_path / "out")
    assert not (tmp_path / "out").exists()

def test_an_empty_store_does_not_look_like_a_successful_materialization(tmp_path):
    store = CanonicalRecordStore(tmp_path / "empty")

    result = PublishedMaterializer(store).materialize(RELEASE_ID, tmp_path / "out")

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.state is MaterializationState.NOTHING_TO_MATERIALIZE
    assert result.is_materialized is False
    assert manifest["materialization_state"] != "materialized"
    assert manifest["materialization_state"] == "nothing_to_materialize"
    assert manifest["warning"]
    assert manifest["record_count"] == 0
    assert manifest["stored_record_count"] == 0
    assert manifest["records"] == []
    assert result.records_path.read_bytes() == b""

def test_a_store_of_drafts_only_is_not_reported_as_materialized(store, tmp_path):
    for index in range(3):
        _write(store, record_id=f"CHI_TIEU:NGANH:nganh{index}:2026",
               status=RecordStatus.DRAFT)

    result = PublishedMaterializer(store).materialize(RELEASE_ID, tmp_path / "out")

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.state is MaterializationState.NO_PUBLISHED_RECORDS
    assert result.is_materialized is False
    assert manifest["materialization_state"] == "no_published_records"
    assert manifest["warning"]
    assert manifest["stored_record_count"] == 3
    assert manifest["record_count"] == 0
    assert manifest["excluded_counts"]["draft"] == 3

# ---------------------------------------------------------------------------
# the real run: ticket 28's seed store
# ---------------------------------------------------------------------------

def _publish_through_lifecycle(service: AdminLifecycleService, record_id: str) -> None:
    """Take one seeded draft to ``published`` along the real lifecycle.

    ``edit`` first, because the seed writes records with no ``authored_by`` and
    four-eyes refuses to approve a version whose author cannot be verified —
    the state machine is not bypassed here, only fed.
    """
    service.edit(record_id, actor_id="seed-author")
    service.transition(record_id, LifecycleAction.SUBMIT, actor_id="seed-author")
    service.transition(record_id, LifecycleAction.APPROVE, actor_id="seed-reviewer")
    service.transition(record_id, LifecycleAction.PUBLISH, actor_id="seed-publisher")

def test_real_seed_store_publishes_a_subset_and_withholds_the_drafts(tmp_path):
    """Ticket 05's last acceptance item, on ticket 28's real seed data.

    A materializer proved only on a synthetic store is not proved at all: this
    publishes a real subset of the seeded records through the real lifecycle,
    materializes, and shows that the drafts from that same set — and every other
    draft the seed wrote — are absent from the output. It skips when the seed
    store is not on disk rather than passing on an empty store.
    """
    records_dir = SEED_STORE / "records"
    if not records_dir.is_dir():
        pytest.skip(f"ticket 28 seed store is not present at {SEED_STORE}")

    source = CanonicalRecordStore(SEED_STORE)
    seed_ids = list(source.record_ids())
    if not seed_ids:
        pytest.skip(f"ticket 28 seed store at {SEED_STORE} holds no record")

    # Work on a copy: publishing appends versions and makes them immutable, and
    # the seed store belongs to every other ticket that reads it.
    store = CanonicalRecordStore(tmp_path / "canonical_records")
    shutil.copytree(records_dir, store.root / "records")

    # A spread across the whole store, not the first N ids: the seed's ids sort
    # by fact type, so a prefix would exercise exactly one kind of record.
    step = max(1, len(seed_ids) // 40)
    subset = seed_ids[::step][:40]
    publish_ids = subset[::2]
    draft_ids = subset[1::2]
    service = AdminLifecycleService(store)
    for record_id in publish_ids:
        _publish_through_lifecycle(service, record_id)

    result = PublishedMaterializer(store).materialize("R-seed-subset", tmp_path / "out")

    admitted = _admitted_ids(result)
    assert result.state is MaterializationState.MATERIALIZED
    assert result.manifest["release_id"] == "R-seed-subset"
    assert set(publish_ids) <= admitted
    assert set(draft_ids).isdisjoint(admitted)
    assert set(seed_ids).difference(subset).isdisjoint(admitted)
    assert all(record.status is RecordStatus.PUBLISHED for record in result.admitted)
    # The published subset really is a cross-section of the seeded data, not one
    # fact type: a filter that only ever saw CHI_TIEU would prove very little.
    assert len({record.record_type for record in result.admitted}) > 1
    # The drafts are real records with real text; their own bytes are not in the
    # output either, so nothing reached the index through a different route.
    for record_id in draft_ids:
        draft = store.get_record(record_id).latest
        assert draft.status is RecordStatus.DRAFT
        assert draft.record_version_key not in result.records_path.read_text(encoding="utf-8")
    assert result.manifest["excluded_counts"]["draft"] >= len(draft_ids)
