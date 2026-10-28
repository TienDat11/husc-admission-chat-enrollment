"""Source-document registry against the REAL canonical store — ticket 39, criterion 2.

The three sibling unit files (``tests/domain/test_source_document.py``,
``tests/infrastructure/test_source_document_registry.py``,
``tests/services/test_source_document_service.py``) prove the registry's
mechanics on synthetic stores. They cannot answer the ticket's actual
acceptance question:

    after the registry lands, do the 188 records that cite ``new_data/*.pdf``
    on the real store still resolve, and does the cited count not decrease?

This file measures the real store at
``rag2025/data/canonical_records/records/`` and locks those numbers in as
assertions (487 record dirs, 188 ``new_data/*.pdf`` citations split 176/11/1
across the three PDFs, 299 ``*.jsonl#slug`` chunk-corpus pointers). If the
store legitimately changes underneath (records added, citation counts move
with them), the assertions are supposed to fail loudly — that is the point of
a permanent measurement, and the ticket's numbers then get re-measured and
re-recorded, not silently relaxed.
**Nothing here writes to the real store.** The canonical root is opened
read-only through ``CanonicalRecordStore`` (its constructor creates nothing;
only ``write_record`` touches disk), and every registry is a ``tmp_path``
directory — the same explicit-path discipline ``SourceDocumentRegistry``
documents ("nothing here defaults to a production location"). The criterion-2
mutation (registry missing one of the three PDFs) is therefore exercised by
retiring and re-registering inside a throwaway registry, so the real store's
state is never the thing at risk.

Skip guards: the store is committed, but the three PDFs are gitignored, so a
machine may legitimately lack either (Ghi chú 3's ``registered, file_absent``
is exactly that deployment reality). When the records dir or any cited PDF is
absent, every test here skips cleanly rather than failing — the suite stays
green everywhere, and ``-rs`` shows why.

Repo-root note (load-bearing): ``PdfText`` resolves
``repo_root / document_id`` (``scripts/seed_canonical_records.py:249-251``)
with ``repo_root`` = the *parent* of ``rag2025`` (``seed_canonical_records.py:47-48``),
and the router derives the same anchor (``src/routers/admin_source_documents.py:65``).
The cited PDFs live at ``<project root>/new_data/``, NOT ``rag2025/new_data/``;
using the wrong root would register all three as ``file_absent`` and make this
file measure nothing.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

RAG2025_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = RAG2025_ROOT.parent
if str(RAG2025_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG2025_ROOT))

from src.domain.source_document import (  # noqa: E402
    SourceDocumentKind,
    classify_document_id,
)
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.infrastructure.source_document_registry import (  # noqa: E402
    SourceDocumentRegistry,
)
from src.services.source_document_service import (  # noqa: E402
    SourceDocumentService,
)

# The store and the repo-root anchor it was seeded against.
STORE_ROOT = RAG2025_ROOT / "data" / "canonical_records"
RECORDS_DIR = STORE_ROOT / "records"

# The three cited PDFs, verbatim repo-relative ids
# (``scripts/seed_canonical_records.py:90-92``).
TTTS_PDF = "new_data/THONG_TIN_TUYEN_SINH_202611.pdf"
HD2026_PDF = "new_data/2304_BGDDT-GDDH_HD tuyển sinh ĐH, CĐ 2026.pdf"
TT06_PDF = "new_data/Thông tư 06_2026_BGD-ĐT_Quy chế tuyển sinh ĐH năm 2026.pdf"
CITED_PDFS = (TTTS_PDF, HD2026_PDF, TT06_PDF)

# -- the numbers this file exists to lock (measured 2026-09-13 on the real store)
EXPECTED_RECORD_DIRS = 487
EXPECTED_PDF_CITATIONS = 188
EXPECTED_CHUNK_POINTER_VERSIONS = 299
EXPECTED_SCANNED_VERSIONS = 487
EXPECTED_DISTINCT_CITED_IDS = 67
EXPECTED_PDF_SPLIT: dict[str, int] = {
    TTTS_PDF: 176,
    HD2026_PDF: 11,
    TT06_PDF: 1,
}

STORE_DIR_PRESENT = RECORDS_DIR.is_dir()
PDFS_PRESENT = all((PROJECT_ROOT / document_id).is_file() for document_id in CITED_PDFS)

pytestmark = pytest.mark.skipif(
    not (STORE_DIR_PRESENT and PDFS_PRESENT),
    reason=(
        "real canonical store unavailable: need "
        f"{RECORDS_DIR} and the three gitignored new_data PDFs "
        f"under {PROJECT_ROOT} (records dir present: {STORE_DIR_PRESENT}; "
        f"PDFs present: {PDFS_PRESENT})"
    ),
)


@dataclass(frozen=True)
class StoreMeasurement:
    """One read-only pass over the real store, shared by every test here.

    ``pdf_rvks`` maps each cited PDF id to the sorted record-version keys of
    the versions citing it — the population the ticket's criterion 2 and its
    mutation are about. ``chunk_pointer_versions`` counts the
    ``*.jsonl#slug`` citations (the Ghi chú 4 backlog the registry
    deliberately does *not* hold).
    """

    record_dir_count: int
    record_id_count: int
    scanned_versions: int
    pdf_citation_versions: int
    chunk_pointer_versions: int
    distinct_cited_ids: int
    pdf_rvks: dict[str, list[str]] = field(default_factory=dict)


@pytest.fixture(scope="module")
def measured() -> StoreMeasurement:
    """Measure the real store once; tests assert against the ticket's numbers."""
    store = CanonicalRecordStore(STORE_ROOT)
    record_dir_count = sum(
        1 for path in RECORDS_DIR.iterdir() if path.is_dir()
    )
    pdf_rvks: dict[str, list[str]] = {}
    pointer_versions = 0
    scanned = 0
    distinct_ids: set[str] = set()
    for record_id in store.record_ids():
        for version in store.versions(record_id):
            record = store.get_version(record_id, version)
            scanned += 1
            document_id = (record.source_document_id or "").strip()
            if not document_id:
                continue
            distinct_ids.add(document_id)
            kind = classify_document_id(document_id)
            if kind is SourceDocumentKind.PDF_DOCUMENT:
                pdf_rvks.setdefault(document_id, []).append(
                    f"{record_id}@{version}"
                )
            elif kind is SourceDocumentKind.CHUNK_CORPUS:
                pointer_versions += 1
    return StoreMeasurement(
        record_dir_count=record_dir_count,
        record_id_count=len(store.record_ids()),
        scanned_versions=scanned,
        pdf_citation_versions=sum(len(v) for v in pdf_rvks.values()),
        chunk_pointer_versions=pointer_versions,
        distinct_cited_ids=len(distinct_ids),
        pdf_rvks={key: sorted(value) for key, value in pdf_rvks.items()},
    )


def _service(tmp_path: Path) -> SourceDocumentService:
    """A service whose registry lives in ``tmp_path`` and whose store is the real one.

    No ledger: the audit semantics are the unit file's job (criteria 3/4);
    this file's contract is that registry writes leave the *real* store
    untouched, and a tmp registry with ``ledger=None`` cannot write anywhere
    but tmp.
    """
    registry = SourceDocumentRegistry(tmp_path, PROJECT_ROOT)
    store = CanonicalRecordStore(STORE_ROOT)
    return SourceDocumentService(registry, store)


# -- criterion 2: the ticket's numbers, re-measured ---------------------------


def test_real_store_matches_the_tickets_measured_shape(
    measured: StoreMeasurement,
) -> None:
    """487 dirs, 188 PDF citations, 176/11/1 split, 299 corpus pointers.

    These are the exact figures ticket 39 criterion 2 quotes. A failure here
    means the store changed since the ticket was written — re-measure and
    update the ticket, never weaken the assertion to match drift.
    """
    assert measured.record_dir_count == EXPECTED_RECORD_DIRS
    assert measured.record_id_count == EXPECTED_RECORD_DIRS
    assert measured.scanned_versions == EXPECTED_SCANNED_VERSIONS
    assert measured.pdf_citation_versions == EXPECTED_PDF_CITATIONS
    assert measured.chunk_pointer_versions == EXPECTED_CHUNK_POINTER_VERSIONS
    assert measured.distinct_cited_ids == EXPECTED_DISTINCT_CITED_IDS
    assert {
        key: len(value) for key, value in measured.pdf_rvks.items()
    } == EXPECTED_PDF_SPLIT


# -- criterion 2: backfill leaves all 188 citations resolving, count intact ---


def test_backfill_from_real_store_keeps_every_citation_resolving(
    tmp_path: Path, measured: StoreMeasurement
) -> None:
    """The ticket's sentence, executed: after backfill, ``resolve_provenance``
    still returns for **all 188** cited versions, their document ids sit in
    the registry, and the cited count has not decreased."""
    service = _service(tmp_path)
    store = CanonicalRecordStore(STORE_ROOT)

    # Pre-state: provenance resolves for all 188, and an empty registry
    # vouches for none of them (the read path predates the registry).
    rvks = sorted(rvk for group in measured.pdf_rvks.values() for rvk in group)
    assert len(rvks) == EXPECTED_PDF_CITATIONS
    cited_before = 0
    for rvk in rvks:
        prov = store.resolve_provenance(rvk)
        assert prov.source_document_id
        cited_before += 1
    assert cited_before == EXPECTED_PDF_CITATIONS
    assert all(
        service.is_registered(doc) is False for doc in CITED_PDFS
    )

    report = service.backfill()
    assert report.scanned_versions == EXPECTED_SCANNED_VERSIONS
    assert report.citations == EXPECTED_PDF_CITATIONS
    assert report.registered_count == len(CITED_PDFS)
    assert set(report.registered) == set(CITED_PDFS)
    # The PDFs are present on this machine (skip-guard), so the entries carry
    # the hash of the real file (Ghi chú 3), not Ghi chú 3's file_absent state:
    # backfill measured the PDFs on disk, it did not register bare names.
    assert report.file_absent == ()
    for cited in CITED_PDFS:
        entry = service.lookup(cited)
        assert entry is not None
        assert entry.file_present is True
        path = PROJECT_ROOT / cited
        assert entry.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report.skipped_kinds == {
        SourceDocumentKind.CHUNK_CORPUS.value: EXPECTED_CHUNK_POINTER_VERSIONS
    }

    # Post-state: same population, every one still resolves and is now vouched for.
    cited_after = 0
    for rvk in rvks:
        prov = store.resolve_provenance(rvk)
        assert prov.source_document_id
        assert service.is_registered(prov.source_document_id)
        cited_after += 1
    assert cited_after == cited_before == EXPECTED_PDF_CITATIONS

    # Re-running changes nothing (backfill is safe to repeat).
    second = service.backfill()
    assert second.registered == ()
    assert set(second.already_registered) == set(CITED_PDFS)
    assert second.citations == EXPECTED_PDF_CITATIONS


def test_blind_report_after_backfill_is_exactly_the_corpus_pointer_backlog(
    tmp_path: Path
) -> None:
    """With all three PDFs registered, the only blind citations the real store
    reports are the 299 ``*.jsonl#slug`` pointers — zero unregistered file
    paths. This is Ghi chú 4's chosen semantics made observable."""
    service = _service(tmp_path)
    service.backfill()
    report = service.blind_citations(None)
    assert report.counts_by_kind == {
        SourceDocumentKind.CHUNK_CORPUS.value: EXPECTED_CHUNK_POINTER_VERSIONS
    }
    assert report.scanned_versions == EXPECTED_SCANNED_VERSIONS
    assert report.cited_versions == EXPECTED_SCANNED_VERSIONS
    assert all(
        classify_document_id(document_id) is SourceDocumentKind.CHUNK_CORPUS
        for document_id in report.counts_by_document_id
    )
    assert all(
        document_id not in report.counts_by_document_id
        for document_id in CITED_PDFS
    )


# -- criterion 2 mutation: registry missing one PDF -> exactly its records fail -


@pytest.mark.parametrize(
    ("document_id", "affected"),
    sorted(EXPECTED_PDF_SPLIT.items(), key=lambda item: item[1]),
    ids=["tt06-1", "hd2026-11", "ttts-176"],
)
def test_missing_one_pdf_fails_exactly_its_records_then_restores(
    tmp_path: Path, document_id: str, affected: int
) -> None:
    """The ticket's mutation: "registry bắt buộc hash-của-file-tồn-tại mà bỏ sót
    một trong ba PDF → các record tương ứng fail kiểm".

    A registry that *omits* a PDF is exactly a registry whose entry for it is
    retired — the record is still there, but ``is_registered`` no longer
    vouches for it. So: backfill fully, retire one PDF, and the ``affected``
    records citing it must fail the existence check while their provenance
    *still resolves* (criterion 2's read path is untouched — it is the
    registry check that goes red). Then re-register and the state is restored:
    the blind report returns to corpus-pointers-only.
    """
    service = _service(tmp_path)
    store = CanonicalRecordStore(STORE_ROOT)
    report = service.backfill()
    assert set(report.registered) == set(CITED_PDFS)
    assert service.is_registered(document_id) is True

    rvks = sorted(
        rvk
        for cited_id, group in _pdf_rvks().items()
        if cited_id == document_id
        for rvk in group
    )
    assert len(rvks) == affected

    # MUTATE: the registry stops vouching for this one document.
    service.retire(
        document_id,
        actor_id="t39.real-store-mutation",
        reason="criterion-2 mutation check: registry missing one PDF",
    )
    assert service.is_registered(document_id) is False

    # The affected records fail the existence check...
    for rvk in rvks:
        prov = store.resolve_provenance(rvk)
        assert prov.source_document_id == document_id
        assert service.is_registered(prov.source_document_id) is False
    blind = service.blind_citations(None)
    assert blind.counts_by_document_id.get(document_id) == affected
    assert blind.counts_by_kind["pdf_document"] == affected
    assert {
        item.record_version_key
        for item in blind.items
        if item.source_document_id == document_id
    } == set(rvks)

    # ...while the other two PDFs' records are unaffected: still registered,
    # and the corpus-pointer backlog is the only other blindness.
    for other in CITED_PDFS:
        if other == document_id:
            continue
        assert service.is_registered(other) is True
    assert (
        sum(blind.counts_by_document_id.values())
        == EXPECTED_CHUNK_POINTER_VERSIONS + affected
    )

    # RESTORE: re-registering revives the entry; the check goes green again.
    service.register(document_id, actor_id="t39.real-store-mutation")
    assert service.is_registered(document_id) is True
    restored = service.blind_citations(None)
    assert document_id not in restored.counts_by_document_id
    assert restored.counts_by_kind == {
        SourceDocumentKind.CHUNK_CORPUS.value: EXPECTED_CHUNK_POINTER_VERSIONS
    }
    for rvk in rvks:
        assert service.is_registered(
            store.resolve_provenance(rvk).source_document_id
        )


def _pdf_rvks() -> dict[str, list[str]]:
    """Cited-PDF -> record-version keys, measured fresh from the store.

    The parametrized mutation tests cannot take the module-scoped ``measured``
    fixture for the rvk sets *and* stay honest: the sets are data, and this
    helper re-derives them from the same read path the property uses, rather
    than hand-copied literals.
    """
    store = CanonicalRecordStore(STORE_ROOT)
    out: dict[str, list[str]] = {}
    for record_id in store.record_ids():
        for version in store.versions(record_id):
            record = store.get_version(record_id, version)
            document_id = (record.source_document_id or "").strip()
            if document_id in CITED_PDFS:
                out.setdefault(document_id, []).append(
                    f"{record_id}@{version}"
                )
    return {key: sorted(value) for key, value in out.items()}


# -- criterion 6: the id still means the same file to the PARSER --------------


def test_registered_id_still_parses_the_same_pdf(tmp_path: Path) -> None:
    """Criterion 6, executed on the real parse path — not on a path string.

    The criterion's sentence is that ``PdfText.__init__`` keeps resolving
    ``repo_root / document_id`` *after* the registry lands, so the parser opens
    the same file as before. Comparing ``resolve_path`` against a literal path
    would not prove that: the file could be unreadable, or the registry could
    hand back a path the parser never sees. This test therefore feeds the
    registry's answer into the actual parser and asserts the parsed document.

    The mutation the criterion names — "registry returns an opaque uuid and the
    parse path takes that id as a path" — fails here at ``is_file()`` /
    ``parse_pdf`` (the uuid names nothing), and the pre-existing behaviour it
    must preserve is pinned by the same call being made with the verbatim id.

    One PDF only: ``PdfText`` parses eagerly (measured ~9.7s for the 77-page
    TTTS file, the largest of the three), and the criterion is about the id's
    meaning, which one document establishes. The other two ids are covered for
    registration and hashing by criterion 2's tests above.
    """
    from scripts.seed_canonical_records import PdfText  # noqa: PLC0415

    service = _service(tmp_path)
    service.register(TTTS_PDF, actor_id="t39.criterion-6")

    # The registry's answer, then the parser's own expression, must agree.
    registered_path = service.resolve_path(TTTS_PDF)
    assert registered_path == (PROJECT_ROOT / TTTS_PDF).resolve()
    assert registered_path.is_file()

    # The criterion's real subject: the parser opens that file and yields text.
    pdf = PdfText(PROJECT_ROOT, TTTS_PDF, tmp_path / "parse_cache")
    assert pdf.path == registered_path
    assert pdf.parser == "pypdf"
    assert pdf.page_count > 0
    assert pdf.pages, "parser produced no pages"
    assert any(text.strip() for text in pdf.pages.values()), (
        "parser produced pages but no extractable text"
    )

    # And the record that cites this PDF still resolves to the same page anchor
    # (criterion 7's field, read through the path criterion 6 protects).
    prov = CanonicalRecordStore(STORE_ROOT).resolve_provenance(
        f"{_first_rvk_citing(TTTS_PDF)}"
    )
    assert prov.source_document_id == TTTS_PDF
    assert prov.source_page is not None and prov.source_page >= 1


def _first_rvk_citing(document_id: str) -> str:
    """Any record-version key citing ``document_id`` — data, not a literal."""
    rvks = _pdf_rvks().get(document_id, [])
    assert rvks, f"no record cites {document_id!r}; the store changed"
    return rvks[0]
