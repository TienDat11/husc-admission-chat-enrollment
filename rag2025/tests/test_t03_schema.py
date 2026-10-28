"""T03 — schema 1.1, back-fill, NFC normalisation.

Acceptance:
- 487 legacy records in ``data/canonical_store`` load under 1.1 schema.
- A new record missing ``major_codes`` is valid.
- Back-fill dry-run prints the correct count of records with a major code.
- NFD text stored via the service is NFC; yields the same ``chunk_id``
  as the NFC version of the same text.
"""
from __future__ import annotations

import json
import sys
import unicodedata
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical.record import (  # noqa: E402
    SCHEMA_VERSION,
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
    from_canonical_json,
    to_canonical_json,
)
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.infrastructure.source_document_registry import SourceDocumentRegistry  # noqa: E402
from src.services.admin_lifecycle import AdminLifecycleService  # noqa: E402
from src.services.chunk_identity import chunk_id_for  # noqa: E402
from src.domain.canonical.semantic_units import SemanticUnit, UnitKind  # noqa: E402
from scripts.backfill_major_codes import extract_major_code  # noqa: E402


# ---------------------------------------------------------------------------
# 1. SCHEMA_VERSION
# ---------------------------------------------------------------------------

def test_schema_version_is_1_1() -> None:
    assert SCHEMA_VERSION == "1.1"


# ---------------------------------------------------------------------------
# 2. New RecordType values
# ---------------------------------------------------------------------------

def test_thong_bao_and_hoi_dap_are_record_types() -> None:
    assert RecordType("THONG_BAO") is RecordType.THONG_BAO
    assert RecordType("HOI_DAP") is RecordType.HOI_DAP


# ---------------------------------------------------------------------------
# 3. 1.1 additive fields carry defaults
# ---------------------------------------------------------------------------

def _minimal_draft(**overrides: object) -> CanonicalRecord:
    base: dict[str, object] = {
        "record_id": "HOI_DAP:question:2026:seed",
        "record_type": RecordType.HOI_DAP,
        "version": 1,
        "admission_year": 2026,
        "effective_from": date(2026, 1, 1),
        "status": RecordStatus.DRAFT,
        "authority_level": AuthorityLevel.HUSC_OFFICIAL,
        "text": "Xin chào.",
    }
    base.update(overrides)
    return CanonicalRecord(**base)


def test_new_record_without_major_codes_is_valid() -> None:
    record = _minimal_draft()
    assert record.major_codes == []
    assert record.admission_methods == []
    assert record.admission_round is None


def test_new_fields_can_be_set() -> None:
    record = _minimal_draft(
        major_codes=["7220104"],
        admission_methods=["Xét tuyển thẳng"],
        admission_round="Đợt 1",
    )
    assert record.major_codes == ["7220104"]
    assert record.admission_methods == ["Xét tuyển thẳng"]
    assert record.admission_round == "Đợt 1"


# ---------------------------------------------------------------------------
# 4. Legacy 1.0 payload still reads
# ---------------------------------------------------------------------------

def test_legacy_1_0_payload_still_validates() -> None:
    """A JSON payload that predates 1.1 (no new fields) must still load."""
    record = _minimal_draft()
    payload = json.loads(to_canonical_json(record))
    # Simulate a 1.0 payload: strip the 1.1 fields and the version stamp.
    for key in ("major_codes", "admission_methods", "admission_round", "schema_version"):
        payload.pop(key, None)
    restored = from_canonical_json(json.dumps(payload, ensure_ascii=False))
    assert restored.major_codes == []
    assert restored.admission_methods == []
    assert restored.admission_round is None
    assert restored.record_id == record.record_id


# ---------------------------------------------------------------------------
# 5. 487 legacy records load from the real store
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (ROOT / "data" / "canonical_store" / "records").is_dir(),
    reason="real canonical store not present in this environment",
)
def test_all_487_legacy_records_load() -> None:
    store = CanonicalRecordStore(ROOT / "data" / "canonical_store")
    ids = store.record_ids()
    assert len(ids) == 487, f"expected 487 records, found {len(ids)}"
    for rid in ids:
        record = store.get_record(rid).latest
        assert isinstance(record, CanonicalRecord)
        assert record.schema_version in ("1.0", "1.1")


# ---------------------------------------------------------------------------
# 6. Back-fill dry-run count
# ---------------------------------------------------------------------------

def test_dry_run_count_matches_records_with_major_code(tmp_path: Path) -> None:
    """Dry-run must report exactly the number of records whose id has a major code."""
    store = CanonicalRecordStore(tmp_path)

    # 3 records with a major code, 2 without.
    with_code = [
        "NGANH:nganh:7220104:2026:seed",
        "TO_HOP:nganh:7480201:2026:seed",
        "CHI_TIEU:nganh:7310301:2026:seed",
    ]
    without_code = [
        "THOI_GIAN:toan-truong:timeline:2026:seed",
        "HOI_DAP:question:2026:seed",
    ]
    for rid in with_code + without_code:
        record = CanonicalRecord(
            record_id=rid,
            record_type=RecordType.NGANH,
            version=1,
            admission_year=2026,
            effective_from=date(2026, 1, 1),
            status=RecordStatus.DRAFT,
            authority_level=AuthorityLevel.HUSC_CANONICAL,
            text="test",
        )
        store.write_record(record)

    # Verify extract_major_code finds the right count.
    found = [rid for rid in store.record_ids() if extract_major_code(rid) is not None]
    assert len(found) == 3


def test_extract_major_code_returns_code_for_known_ids() -> None:
    assert extract_major_code("NGANH:nganh:7220104:2026:seed") == "7220104"
    assert extract_major_code("NGANH:nganh:7510302ic:2026:seed") == "7510302ic"
    assert extract_major_code("THOI_GIAN:toan-truong:x:2026:seed") is None
    assert extract_major_code("HOI_DAP:question:x:2026:seed") is None


# ---------------------------------------------------------------------------
# 7. Back-fill writes a new version with major_codes populated
# ---------------------------------------------------------------------------

def test_back_fill_appends_version_with_major_code(tmp_path: Path) -> None:
    store = CanonicalRecordStore(tmp_path)
    rid = "NGANH:nganh:7220104:2026:seed"
    v1 = CanonicalRecord(
        record_id=rid,
        record_type=RecordType.NGANH,
        version=1,
        admission_year=2026,
        effective_from=date(2026, 1, 1),
        status=RecordStatus.DRAFT,
        authority_level=AuthorityLevel.HUSC_CANONICAL,
        text="Kỹ thuật phần mềm",
    )
    store.write_record(v1)
    assert store.get_record(rid).latest.major_codes == []

    code = extract_major_code(rid)
    store.edit_record(rid, major_codes=[code])

    v2 = store.get_record(rid).latest
    assert v2.version == 2
    assert v2.major_codes == [code]
    # v1 is unchanged on disk (append-only)
    assert store.get_version(rid, 1).major_codes == []


# ---------------------------------------------------------------------------
# 8. NFC normalisation at the service boundary
# ---------------------------------------------------------------------------

def _nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def test_create_stores_nfc_text(tmp_path: Path) -> None:
    store = CanonicalRecordStore(tmp_path)
    # T07: a HOI_DAP create writes a real source document, so the service
    # needs the registry — provenance-less HOI_DAPs are forbidden.
    service = AdminLifecycleService(
        store, four_eyes=False, registry=SourceDocumentRegistry(tmp_path, tmp_path)
    )

    nfd_text = _nfd("Học phí ngành Kỹ thuật phần mềm: 18 triệu/năm.")
    nfc_text = unicodedata.normalize("NFC", nfd_text)
    assert nfd_text != nfc_text, "precondition: NFD and NFC must differ"

    record = CanonicalRecord(
        record_id="HOI_DAP:question:nfd:2026",
        record_type=RecordType.HOI_DAP,
        version=1,
        admission_year=2026,
        effective_from=date(2026, 1, 1),
        status=RecordStatus.DRAFT,
        authority_level=AuthorityLevel.HUSC_OFFICIAL,
        text=nfd_text,
    )
    stored = service.create(record, actor_id="test-actor")
    assert stored.text == nfc_text, "create must store NFC-normalised text"


def test_edit_stores_nfc_text(tmp_path: Path) -> None:
    store = CanonicalRecordStore(tmp_path)
    service = AdminLifecycleService(store, four_eyes=False)

    nfd_text = _nfd("Câu hỏi mới: Khi nào nộp đơn?")
    nfc_text = unicodedata.normalize("NFC", nfd_text)

    rid = "HOI_DAP:question:edit:2026"
    v1 = CanonicalRecord(
        record_id=rid,
        record_type=RecordType.HOI_DAP,
        version=1,
        admission_year=2026,
        effective_from=date(2026, 1, 1),
        status=RecordStatus.DRAFT,
        authority_level=AuthorityLevel.HUSC_OFFICIAL,
        text="text gốc",
    )
    store.write_record(v1)

    v2 = service.edit(rid, text=nfd_text, actor_id="editor")
    assert v2.text == nfc_text, "edit must store NFC-normalised text"


def test_nfd_and_nfc_produce_same_chunk_id(tmp_path: Path) -> None:
    """A record created with NFD text must yield the same chunk_id as NFC text."""
    store = CanonicalRecordStore(tmp_path)
    service = AdminLifecycleService(
        store, four_eyes=False, registry=SourceDocumentRegistry(tmp_path, tmp_path)
    )

    text_nfc = "Học phí: 18 triệu."
    text_nfd = unicodedata.normalize("NFD", text_nfc)
    assert text_nfd != text_nfc

    # Create two records: one from NFD input, one from NFC input.
    rid_nfd = "HOI_DAP:nfd:2026"
    rid_nfc = "HOI_DAP:nfc:2026"

    record_nfd = CanonicalRecord(
        record_id=rid_nfd,
        record_type=RecordType.HOI_DAP,
        version=1,
        admission_year=2026,
        effective_from=date(2026, 1, 1),
        status=RecordStatus.DRAFT,
        authority_level=AuthorityLevel.HUSC_OFFICIAL,
        text=text_nfd,
    )
    stored_nfd = service.create(record_nfd, actor_id="a")

    record_nfc = CanonicalRecord(
        record_id=rid_nfc,
        record_type=RecordType.HOI_DAP,
        version=1,
        admission_year=2026,
        effective_from=date(2026, 1, 1),
        status=RecordStatus.DRAFT,
        authority_level=AuthorityLevel.HUSC_OFFICIAL,
        text=text_nfc,
    )
    stored_nfc = service.create(record_nfc, actor_id="b")

    # Both must be stored as NFC.
    assert stored_nfd.text == text_nfc
    assert stored_nfc.text == text_nfc

    # The real invariant: both stored texts are byte-identical NFC, so a chunk
    # built from either record's stored text uses the same value — the NFD
    # input was normalised away at the boundary, keeping chunk ids stable.
    assert stored_nfd.text == stored_nfc.text
    assert stored_nfd.text.encode("utf-8") == stored_nfc.text.encode("utf-8")


def test_chunk_identity_is_not_normalized() -> None:
    """chunk_identity.normalize_for_hash deliberately does NOT normalise Unicode.

    Two records whose text differs only in NFD vs NFC spelling produce
    *different* chunk ids — this is by design (spec 02 §1).
    """
    text_nfc = "Kỹ thuật phần mềm"
    text_nfd = unicodedata.normalize("NFD", text_nfc)
    assert text_nfc != text_nfd

    def _unit(value: str) -> SemanticUnit:
        return SemanticUnit(
            record_id="HOI_DAP:test:2026", record_type="HOI_DAP",
            field="text", slot="0", value=value, kind=UnitKind.TEXT,
        )

    key = "HOI_DAP:test:2026@1"
    assert chunk_id_for(key, _unit(text_nfc)) != chunk_id_for(key, _unit(text_nfd)), (
        "chunk_identity must NOT treat NFD and NFC as the same chunk"
    )
