"""Ticket 28 — the seeded canonical record store holds real, traceable facts.

Every assertion here is about what a consumer of the store can observe: the
records that exist, the provenance they carry, the fact-type coverage they
reach, the report file, and whether a stored record leads back to text that is
really printed on the page it names. Nothing asserts internal wiring.

The store is seeded twice into a throwaway directory, so the idempotence claim
is proven against the real corpus rather than a stub.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypdf import PdfReader

RAG_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = RAG_ROOT.parent
for entry in (RAG_ROOT, RAG_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import seed_canonical_records as seed_mod  # noqa: E402
from src.domain.canonical import RecordStatus  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402

# The six fact kinds the ticket names (điểm chuẩn, chỉ tiêu, học phí, tổ hợp,
# mã ngành, mốc thời gian) as the canonical schema spells them, and the coverage
# floor the ticket sets.
REQUIRED_FACT_TYPES = (
    "DIEM_CHUAN",
    "CHI_TIEU",
    "HOC_PHI",
    "TO_HOP",
    "NGANH",
    "THOI_GIAN",
)
MIN_NGANH_PER_FACT_TYPE = 20
# Warning codes that mean "a printed cell could not be read faithfully".
MISSING_CELL_CODES = frozenset(
    {
        "unparsed_table_cell",
        "blank_table_cell",
        "unattributed_table_cell",
        "ambiguous_table_cell",
        "table_row_without_cells",
    }
)


def _flat(text: str) -> str:
    """Page and record text with runs of whitespace collapsed to one space."""
    return re.sub(r"\s+", " ", text)


def _snapshot(store_root: Path) -> dict[str, list[int]]:
    """Every record id in the store with its version numbers."""
    store = CanonicalRecordStore(store_root)
    return {record_id: store.versions(record_id) for record_id in store.record_ids()}


@pytest.fixture(scope="module")
def seeded(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    store_root = tmp_path_factory.mktemp("canonical_store")
    first = seed_mod.seed(store_root=store_root, repo_root=REPO_ROOT)
    before = _snapshot(store_root)
    second = seed_mod.seed(store_root=store_root, repo_root=REPO_ROOT)
    after = _snapshot(store_root)
    return SimpleNamespace(
        store_root=store_root,
        store=CanonicalRecordStore(store_root),
        first=first,
        second=second,
        before=before,
        after=after,
        report=first.report,
    )


def _records(seeded: SimpleNamespace):
    for record_id in seeded.store.record_ids():
        for version in seeded.store.versions(record_id):
            yield seeded.store.get_version(record_id, version)


def test_store_is_not_empty_and_every_record_carries_real_provenance(
    seeded: SimpleNamespace,
) -> None:
    """No record without a source document, and no page that the PDF lacks."""
    records = list(_records(seeded))
    assert records, "the seed wrote no records at all"

    page_counts: dict[str, int] = {}
    for record in records:
        document_id = record.source_document_id
        assert document_id, f"{record.record_id}: record has no source_document_id"
        if document_id.endswith(".pdf"):
            assert record.source_page is not None, (
                f"{record.record_id}: a PDF-derived record must name its page"
            )
            if document_id not in page_counts:
                page_counts[document_id] = len(
                    PdfReader(str(REPO_ROOT / document_id)).pages
                )
            assert 1 <= record.source_page <= page_counts[document_id], (
                f"{record.record_id}: page {record.source_page} is outside "
                f"{document_id} (1..{page_counts[document_id]})"
            )
        else:
            assert record.source_page is None, (
                f"{record.record_id}: the chunk corpus carries no page numbers, so a "
                "page here would be invented"
            )

    documents = {record.source_document_id for record in records}
    assert len(documents) >= 3, f"expected several source documents, got {documents}"
    assert seeded.report["provenance"]["pdf_records_carry_page"] > 0
    assert seeded.report["provenance"]["corpus_records_have_no_page"] > 0


def test_six_fact_types_each_reach_twenty_nganh(seeded: SimpleNamespace) -> None:
    """The coverage the ticket demands, measured — not asserted optimistically."""
    coverage = seeded.report["fact_type_coverage"]
    by_type = seeded.report["record_counts"]["by_fact_type"]

    assert set(REQUIRED_FACT_TYPES) <= set(by_type), (
        f"missing fact types: {sorted(set(REQUIRED_FACT_TYPES) - set(by_type))}"
    )
    shortfalls = {
        fact_type: coverage[fact_type]["nganh_count"]
        for fact_type in REQUIRED_FACT_TYPES
        if coverage[fact_type]["nganh_count"] < MIN_NGANH_PER_FACT_TYPE
    }
    assert not shortfalls, (
        f"fact types below {MIN_NGANH_PER_FACT_TYPE} ngành: {shortfalls} "
        f"(counts: { {k: coverage[k]['nganh_count'] for k in REQUIRED_FACT_TYPES} })"
    )
    for fact_type in REQUIRED_FACT_TYPES:
        assert by_type[fact_type] >= 1


def test_seeding_twice_creates_no_record_and_no_version(seeded: SimpleNamespace) -> None:
    """Idempotence, proven by the store's own record set and version counts."""
    assert seeded.second.report["record_counts"]["written_this_run"] == 0
    assert (
        seeded.second.report["record_counts"]["already_present"]
        == seeded.first.report["record_counts"]["total"]
    )
    assert seeded.after == seeded.before, "a second run changed the store"
    assert all(
        set(versions) == {1} for versions in seeded.after.values()
    ), "a record gained a version on the second run"
    assert len(seeded.before) == seeded.first.report["record_counts"]["total"]


def test_every_record_lands_in_draft_and_none_is_published(
    seeded: SimpleNamespace,
) -> None:
    """The seed never publishes: a human decides what reaches students."""
    records = list(_records(seeded))
    statuses = {record.status for record in records}
    assert statuses == {RecordStatus.DRAFT}, f"unexpected statuses: {statuses}"
    assert all(record.published_at is None for record in records)
    assert seeded.report["record_counts"]["published"] == 0
    assert seeded.report["record_counts"]["by_status"] == {"draft": len(seeded.before)}


def test_an_unreadable_cell_warns_and_leaves_the_field_missing(
    seeded: SimpleNamespace,
) -> None:
    """A cell that cannot be read is reported and left empty — never guessed."""
    warnings = seeded.report["warnings"]
    assert warnings, "no warning was emitted for the cells that could not be read"
    assert {item["code"] for item in warnings} & MISSING_CELL_CODES

    flagged = [
        record
        for record in _records(seeded)
        if record.missing_fields
    ]
    assert flagged, (
        "no record declares a missing field, so an unreadable cell was either "
        "filled in or silently dropped"
    )
    for record in flagged:
        values = record.values or {}
        for field_name in record.missing_fields:
            assert values.get(field_name) is None, (
                f"{record.record_id}: '{field_name}' is declared missing but carries "
                f"{values.get(field_name)!r}"
            )
        assert record.text, f"{record.record_id}: a record with a missing field has no text"


def test_hand_verified_fixture_pages_agree_with_the_extraction(
    seeded: SimpleNamespace,
) -> None:
    """Ticket 04's hand-transcribed anchors are really printed on the pages named.

    The fixture pages are the independent evidence for this seed's page numbers;
    the checks that compare extracted *values* against them (học phí page 55,
    tổ hợp page 24) must agree too.
    """
    checks = {check["name"]: check for check in seeded.report["cross_checks"]}
    fixture_checks = {
        name: check
        for name, check in checks.items()
        if name.startswith("ticket04_fixture_pages_")
    }
    assert len(fixture_checks) == 5, f"expected all 5 fixtures, got {sorted(fixture_checks)}"
    for name, check in fixture_checks.items():
        assert check["anchors_checked"] > 0
        assert check["ok"], f"{name}: anchors not found on their page: {check['anchors_not_found']}"

    for name in (
        "pdf_hoc_phi_vs_ticket04_fixture_p55",
        "pdf_to_hop_pairing_vs_ticket04_fixture_p24",
    ):
        assert checks[name]["ok"], f"{name}: {checks[name]['mismatches']}"


def test_report_is_machine_readable_with_counts_by_type_and_document(
    seeded: SimpleNamespace,
) -> None:
    path = Path(seeded.first.report_path)
    assert path.is_file()
    report = json.loads(path.read_text(encoding="utf-8"))

    counts = report["record_counts"]
    assert counts["total"] == len(seeded.before)
    assert sum(counts["by_fact_type"].values()) == counts["total"]
    assert sum(counts["by_source_document"].values()) == counts["total"]
    assert set(counts["by_source_document"]) == {
        record.source_document_id for record in _records(seeded)
    }
    assert report["schema_version"] == "1.0"


def test_provenance_round_trip_reaches_the_printed_page(seeded: SimpleNamespace) -> None:
    """From a stored record back to text that is really on the page it names.

    Every PDF-derived record must resolve to its own page and find its own text
    there: the ngành token it names, or — for the school-wide records that name
    no ngành — the printed sentence it was extracted from.
    """
    pdf_pages: dict[tuple[str, int], str] = {}
    checked = 0
    for record in _records(seeded):
        document_id = record.source_document_id
        if not document_id.endswith(".pdf"):
            continue
        provenance = seeded.store.resolve_provenance(record.record_version_key)
        assert provenance.source_document_id == document_id
        assert provenance.source_page == record.source_page

        key = (document_id, provenance.source_page)
        if key not in pdf_pages:
            page = PdfReader(str(REPO_ROOT / document_id)).pages[key[1] - 1]
            pdf_pages[key] = _flat(page.extract_text() or "")
        page_text = pdf_pages[key]

        values = record.values or {}
        tokens = [
            str(values.get("ma_nganh") or ""),
            str(values.get("ten_nganh") or ""),
            str(values.get("ten_nganh_theo_toa_do") or ""),
        ]
        if not any(token and token in page_text for token in tokens):
            printed = _flat(record.raw_source_text or "")
            assert printed[:40] and printed[:40] in page_text, (
                f"{record.record_id}: nothing this record claims to state is printed "
                f"on page {key[1]} of {document_id}"
            )
        checked += 1
    assert checked >= 20, f"only {checked} records were round-trip verified"


def test_corpus_records_resolve_to_a_chunk_that_exists(
    seeded: SimpleNamespace,
) -> None:
    """A corpus-derived record names a chunk the corpus file really holds."""
    checked = 0
    for record in _records(seeded):
        document_id, _, chunk_id = record.source_document_id.partition("#")
        if document_id.endswith(".pdf"):
            continue
        assert chunk_id, f"{record.record_id}: corpus record names no chunk"
        path = REPO_ROOT / document_id
        assert path.is_file(), f"{record.record_id}: {document_id} does not exist"
        assert chunk_id in path.read_text(encoding="utf-8"), (
            f"{record.record_id}: chunk {chunk_id} is not in {document_id}"
        )
        checked += 1
    assert checked >= 20, f"only {checked} corpus records were round-trip verified"


def test_default_root_is_the_store_the_app_reads() -> None:
    """Seeding with no ``--store-root`` must land where the app looks.

    The seeder once defaulted to ``data/canonical_records`` while
    ``configure_canonical_record_store`` read ``RAGSettings.CANONICAL_STORE_ROOT``
    (``data/canonical_store``). Both paths existed, so nothing failed loudly:
    the seeder reported 487 records and every admin route served an empty store.
    Asserting the default against the setting is the guard — it fails the moment
    the two drift apart again.
    """
    from config.settings import RAGSettings

    configured = Path(RAGSettings().CANONICAL_STORE_ROOT)
    assert Path(seed_mod.DEFAULT_STORE_ROOT) == configured, (
        "seeding without --store-root writes to "
        f"{seed_mod.DEFAULT_STORE_ROOT}, but the app reads {configured}"
    )


def test_a_seeded_store_is_readable_through_the_app_wiring(
    seeded: SimpleNamespace,
) -> None:
    """The records a run seeds are the records the app serves.

    Proves the two halves meet: seed a store, hand that same root to the real
    startup wiring, and read the records back through the store it attaches to
    ``app.state``. This is the seam that was broken — each side worked in
    isolation while the app saw nothing.
    """
    from fastapi import FastAPI

    from config.settings import RAGSettings
    from rag2025.src.app.lifecycle import configure_canonical_record_store

    app = FastAPI()
    attached = configure_canonical_record_store(
        app, settings=RAGSettings(CANONICAL_STORE_ROOT=seeded.store_root)
    )
    seeded_ids = set(seeded.store.record_ids())
    assert seeded_ids, "the fixture seeded nothing, so this proves nothing"
    assert set(attached.record_ids()) == seeded_ids
    assert attached.root == Path(seeded.store_root)
