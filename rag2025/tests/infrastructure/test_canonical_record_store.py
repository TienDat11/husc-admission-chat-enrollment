"""TDD: rag2025/src/infrastructure/canonical_record_store.py — ticket 03.

Acceptance (``issues/03-canonical-store-provenance.md``):
- writing a record creates a new version and every prior version stays readable;
- editing a record that has a published version never overwrites that version;
- querying by ``record_id`` returns the latest version and the prior versions;
- every record resolves to its ``source_document_id`` and page;
- Vietnamese text reads back byte-identical to what was written (bytes
  compared, not normalized strings — NFC and NFD input forms both).

Everything is asserted through the store's public surface and the bytes on
disk, never through the store's private attributes.
"""
from __future__ import annotations

import json
import sys
import unicodedata
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
from src.infrastructure.canonical_record_store import (  # noqa: E402
    CanonicalRecordStore,
    RecordNotFoundError,
    RecordVersionConflictError,
)

RECORD_ID = "DIEM_CHUAN:NGANH:cntt:2026"
OLD_ID = "DIEM_CHUAN:NGANH:cntt:2025"
DOC_ID = "de-an-tuyen-sinh-2026"
URL = "https://tuyensinh.husc.edu.vn/de-an-2026.pdf"
TEXT_V1 = "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 25,5 điểm."
TEXT_V2 = "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 26,0 điểm (đính chính)."
TEXT_V3 = "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 26,5 điểm (đính chính lần 2)."

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
        "text": TEXT_V1,
    }
    fields.update(overrides)
    return CanonicalRecord(**fields)

@pytest.fixture
def store(tmp_path: Path) -> CanonicalRecordStore:
    return CanonicalRecordStore(tmp_path / "canonical")

def test_writing_a_record_creates_a_new_version_and_prior_versions_stay_readable(store):
    texts = {1: TEXT_V1, 2: TEXT_V2, 3: TEXT_V3}
    for version, text in texts.items():
        store.write_record(_record(version=version, text=text))

    assert store.versions(RECORD_ID) == (1, 2, 3)
    for version, text in texts.items():
        # Read through a fresh store: the versions live on disk, not in memory.
        reopened = CanonicalRecordStore(store.root)
        assert reopened.get_version(RECORD_ID, version).text == text
        assert reopened.get_version(RECORD_ID, version).version == version

def test_editing_a_record_that_has_a_published_version_does_not_overwrite_it(store):
    store.write_record(_record(version=1))
    published_path = store.version_path(RECORD_ID, 1)
    published_bytes = published_path.read_bytes()

    edited = store.edit_record(
        RECORD_ID,
        text=TEXT_V2,
        status=RecordStatus.DRAFT,
        published_at=None,
    )

    assert edited.version == 2
    assert edited.status is RecordStatus.DRAFT

    # The published version is still there, still published, byte-identical.
    published = store.get_version(RECORD_ID, 1)
    assert published.status is RecordStatus.PUBLISHED
    assert published.text == TEXT_V1
    assert published_path.read_bytes() == published_bytes

    # ...and it is read-only on disk, so even a direct write cannot replace it.
    with pytest.raises(PermissionError):
        published_path.write_bytes(b"{}")

    # The store refuses to re-write a version that already exists.
    with pytest.raises(RecordVersionConflictError):
        store.write_record(_record(version=1, text=TEXT_V3))
    assert store.get_version(RECORD_ID, 1).text == TEXT_V1

def test_query_by_record_id_returns_the_latest_version_and_the_prior_versions(store):
    for version, text in ((1, TEXT_V1), (2, TEXT_V2), (3, TEXT_V3)):
        store.write_record(_record(version=version, text=text))

    chain = store.get_record(RECORD_ID)

    assert chain.record_id == RECORD_ID
    assert chain.latest.version == 3
    assert chain.latest.text == TEXT_V3
    # Newest-first: the version `latest` succeeded comes first.
    assert [r.version for r in chain.prior] == [2, 1]
    assert [r.text for r in chain.prior] == [TEXT_V2, TEXT_V1]

def test_a_record_with_one_version_has_no_prior_versions(store):
    store.write_record(_record(version=1))

    chain = store.get_record(RECORD_ID)

    assert chain.latest.version == 1
    assert chain.prior == ()
    with pytest.raises(RecordNotFoundError):
        store.get_record("CHI_TIEU:NGANH:cntt:2026")

def test_every_record_resolves_to_its_source_document_and_page(store):
    store.write_record(_record(version=1, source_page=12))
    store.write_record(
        _record(
            version=2,
            source_document_id="dinh-chinh-diem-chuan-2026",
            source_url="https://tuyensinh.husc.edu.vn/dinh-chinh.pdf",
            source_page=3,
            text=TEXT_V2,
        )
    )

    # Provenance comes off disk: a store opened on the same root resolves it too.
    reopened = CanonicalRecordStore(store.root)
    original = reopened.resolve_provenance(f"{RECORD_ID}@1")
    assert (original.source_document_id, original.source_url, original.source_page) == (
        DOC_ID,
        URL,
        12,
    )
    correction = reopened.resolve_provenance(f"{RECORD_ID}@2")
    assert (
        correction.source_document_id,
        correction.source_page,
        correction.version,
    ) == ("dinh-chinh-diem-chuan-2026", 3, 2)

    with pytest.raises(RecordNotFoundError):
        store.resolve_provenance(f"{RECORD_ID}@99")
    with pytest.raises(ValueError):
        store.resolve_provenance(RECORD_ID)  # not a `record_id@version` key

def test_a_version_that_is_not_the_next_one_is_refused(store):
    # Nothing stored yet: the first version is 1, so a later version is refused
    # and nothing is written.
    with pytest.raises(RecordVersionConflictError):
        store.write_record(_record(version=2))
    assert store.versions(RECORD_ID) == ()
    assert store.next_version(RECORD_ID) == 1

    store.write_record(_record(version=1))

    # A gap (skipping version 2) is refused too — versions stay contiguous.
    with pytest.raises(RecordVersionConflictError):
        store.write_record(_record(version=3))
    assert store.versions(RECORD_ID) == (1,)
    assert store.next_version(RECORD_ID) == 2

def test_record_ids_lists_every_stored_record(store):
    other_id = "HOC_PHI:NGANH:ktpm:2026"
    store.write_record(_record(version=1))
    store.write_record(_record(record_id=other_id, version=1, text=TEXT_V2))

    reopened = CanonicalRecordStore(store.root)

    assert reopened.record_ids() == tuple(sorted([RECORD_ID, other_id]))
    assert reopened.get_record(other_id).latest.text == TEXT_V2

@pytest.mark.parametrize("form", ["NFC", "NFD"])
def test_vietnamese_text_reads_back_byte_identical(store, form):
    text = unicodedata.normalize(
        form,
        "Học phí ngành Kỹ thuật phần mềm năm 2026: 18.000.000 đồng/năm, "
        "ưu đãi cho thí sinh đạt giải học sinh giỏi.",
    )
    raw = text.encode("utf-8")
    other_form = "NFD" if form == "NFC" else "NFC"
    # Guard the test itself: the two forms must be different bytes, otherwise
    # a store that normalized Unicode would still pass.
    assert unicodedata.normalize(other_form, text).encode("utf-8") != raw

    record_id = "HOC_PHI:NGANH:ktpm:2026"
    store.write_record(
        _record(
            record_id=record_id,
            record_type=RecordType.HOC_PHI,
            text=text,
            source_page=31,
        )
    )

    on_disk = store.version_path(record_id, 1).read_bytes()
    # Stored as raw UTF-8 — never \uXXXX escapes.
    assert raw in on_disk
    # The stored text is the exact string written, in the exact form written.
    assert json.loads(on_disk)["text"].encode("utf-8") == raw
    # Read back through the store: same bytes, both for this version and the
    # latest-version query.
    assert store.get_version(record_id, 1).text.encode("utf-8") == raw
    assert store.get_record(record_id).latest.text.encode("utf-8") == raw

def test_record_ids_with_colons_do_not_collide_on_disk(store):
    # ':' is illegal in an NTFS filename, and a lossy sanitizer (the
    # `re.sub(r"[^a-zA-Z0-9_]+", "_")` idiom) would collapse these three
    # distinct ids onto one path and silently overwrite history.
    ids = [
        "DIEM_CHUAN:NGANH:cntt:2026",
        "DIEM_CHUAN_NGANH_cntt_2026",
        "DIEM CHUAN NGANH cntt 2026",
    ]
    for index, record_id in enumerate(ids):
        store.write_record(_record(record_id=record_id, version=1, text=f"bản ghi {index}"))

    # Distinct paths, and the exact ids survive the encoding round-trip.
    assert len({store.version_path(record_id, 1) for record_id in ids}) == 3
    assert store.record_ids() == tuple(sorted(ids))
    for index, record_id in enumerate(ids):
        assert store.get_record(record_id).latest.text == f"bản ghi {index}"

    # Win32 strips a trailing dot from a path component, so these two ids must
    # still not share a directory.
    dotted, plain = "DIEM_CHUAN:NGANH:cntt:2026.", "DIEM_CHUAN:NGANH:cntt:2026"
    store.write_record(_record(record_id=dotted, version=1, text="bản ghi có dấu chấm"))
    assert store.version_path(dotted, 1) != store.version_path(plain, 1)
    assert store.get_record(dotted).latest.text == "bản ghi có dấu chấm"
    assert store.get_record(plain).latest.text == "bản ghi 0"

def test_a_published_record_that_names_an_older_one_supersedes_it(store):
    store.write_record(_record(record_id=OLD_ID, version=1, admission_year=2025))
    store.write_record(_record(version=1, supersedes=OLD_ID))

    superseder = store.superseding_record(OLD_ID)

    assert superseder is not None
    # The answer names the superseding *version*, not just the record.
    assert superseder.record_version_key == f"{RECORD_ID}@1"
    assert superseder.text == TEXT_V1
    assert store.is_superseded(OLD_ID) is True
    # The newer record supersedes something; nothing supersedes it.
    assert store.is_superseded(RECORD_ID) is False

def test_an_unpublished_record_naming_an_older_one_does_not_supersede_it(store):
    store.write_record(_record(record_id=OLD_ID, version=1, admission_year=2025))
    store.write_record(
        _record(
            version=1,
            status=RecordStatus.DRAFT,
            supersedes=OLD_ID,
            published_at=None,
        )
    )

    # An unpublished edit must not retract live admissions data.
    assert store.superseding_record(OLD_ID) is None
    assert store.is_superseded(OLD_ID) is False

    # Publishing that same change is what makes the relation real.
    store.write_record(
        _record(version=2, status=RecordStatus.PUBLISHED, supersedes=OLD_ID, text=TEXT_V2)
    )

    superseder = store.superseding_record(OLD_ID)
    assert superseder is not None
    assert superseder.record_version_key == f"{RECORD_ID}@2"
    assert store.is_superseded(OLD_ID) is True

def test_supersession_queries_reject_an_unknown_record(store):
    store.write_record(_record(version=1))

    # Same strict contract as the module's other lookups: an id with no stored
    # version is an error, never silently "nothing supersedes this".
    with pytest.raises(RecordNotFoundError):
        store.superseding_record("CHI_TIEU:NGANH:cntt:2026")
    with pytest.raises(RecordNotFoundError):
        store.is_superseded("CHI_TIEU:NGANH:cntt:2026")

def test_the_superseder_is_picked_deterministically_when_several_name_it(store):
    other_id = "HOC_PHI:NGANH:ktpm:2026"
    store.write_record(_record(record_id=OLD_ID, version=1, admission_year=2025))
    # Two different records both name the old one.
    store.write_record(_record(record_id=other_id, version=1, supersedes=OLD_ID, text=TEXT_V3))
    store.write_record(_record(version=1, supersedes=OLD_ID))

    # Highest version wins — version first, so 1@other_id beats 1@RECORD_ID on
    # the record_id tie-break, and RECORD_ID@2 beats both.
    assert store.superseding_record(OLD_ID).record_version_key == f"{other_id}@1"
    assert store.supersession_index()[OLD_ID].record_version_key == f"{other_id}@1"

    store.write_record(_record(version=2, supersedes=OLD_ID, text=TEXT_V2))

    assert store.superseding_record(OLD_ID).record_version_key == f"{RECORD_ID}@2"
    # The batch index is the same answer, so a materializer gating a whole
    # release cannot see a different truth from a per-record caller.
    assert store.supersession_index()[OLD_ID].record_version_key == f"{RECORD_ID}@2"

def test_a_reference_may_use_the_full_record_version_key(store):
    store.write_record(_record(record_id=OLD_ID, version=1, admission_year=2025))
    store.write_record(_record(version=1, supersedes=f"{OLD_ID}@1"))

    assert store.superseding_record(OLD_ID).record_version_key == f"{RECORD_ID}@1"
    # A malformed reference (non-numeric version) supersedes nothing.
    store.write_record(_record(version=2, supersedes=f"{OLD_ID}@khong-phai-so", text=TEXT_V2))
    assert store.superseding_record(OLD_ID).record_version_key == f"{RECORD_ID}@1"

def test_a_later_version_cannot_silently_retract_a_published_supersession(store):
    store.write_record(_record(record_id=OLD_ID, version=1, admission_year=2025))
    store.write_record(_record(version=1, supersedes=OLD_ID))

    # An unpublished edit that drops the reference: the live published
    # statement is still v1, so the old record stays superseded. Otherwise an
    # unpublished draft would re-materialize retracted admissions data.
    store.write_record(
        _record(version=2, status=RecordStatus.DRAFT, supersedes=None, published_at=None)
    )
    assert store.superseding_record(OLD_ID).record_version_key == f"{RECORD_ID}@1"

    # The relation is the union over published versions, so even a published v3
    # that omits the reference does not retract it — v1 is still published and
    # still says it. Un-superseding is a lifecycle decision (ticket 12), not a
    # side effect of an edit.
    store.write_record(_record(version=3, supersedes=None, text=TEXT_V3))

    assert store.superseding_record(OLD_ID).record_version_key == f"{RECORD_ID}@1"
    assert store.is_superseded(OLD_ID) is True

def test_self_supersession_is_ignored_and_a_cycle_terminates(store):
    # A record naming itself is not its own superseder.
    store.write_record(_record(record_id=OLD_ID, version=1, admission_year=2025, supersedes=OLD_ID))
    assert store.superseding_record(OLD_ID) is None
    assert OLD_ID not in store.supersession_index()

    # A cycle: each published record names the other. Nothing is traversed, so
    # both queries answer from the same single pass — no hang, stable answer.
    store.write_record(_record(version=1, supersedes=OLD_ID))
    store.write_record(_record(record_id=OLD_ID, version=2, admission_year=2025, supersedes=RECORD_ID))

    assert store.superseding_record(OLD_ID).record_version_key == f"{RECORD_ID}@1"
    assert store.superseding_record(RECORD_ID).record_version_key == f"{OLD_ID}@2"
    assert store.is_superseded(OLD_ID) is True
    assert store.is_superseded(RECORD_ID) is True
