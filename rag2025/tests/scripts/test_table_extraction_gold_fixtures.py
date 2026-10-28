"""Gold fixtures for table extraction (ticket 04) — regression net, not a parser change.

The JSON files in ``tests/fixtures/tables/`` are hand-transcribed from real
documents under ``new_data/`` (every cell was read off the printed page, never
generated) and are fed back through ``scripts/chunk_table_aware_v2.py``. Nothing
in the parser is modified by these tests: they lock the behaviour it has today.

What is asserted, per fixture:

* no cell silently disappears — the number of non-empty cells across emitted
  data rows equals the declared count, and every declared cell is still present;
* the row<->header relationship survives — every emitted chunk re-emits the
  fixture's declared header line verbatim, and each emitted data row keeps its
  declared positional cells (so each cell traces to its column header);
* every row carries page provenance, and (when the source PDF is present) each
  row's cells are found on the page the fixture claims;
* a row whose leading cells are empty while something above fills them is a
  merged/rowspan artifact: the chunker itself reports it, as `rowspan_attribution`
  (the owning row's identity) or `rowspan_unresolved` (with a warning), and its
  content must still be intact.

Note on the source documents: the PDFs wrap a table header across 2-3 pipe rows,
so the chunker's first pipe line is a header *fragment*. "Header" below therefore
means the declared first pipe line, re-emitted verbatim, with column attribution
hand-mapped to the header printed on the page (``column_headers``).
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
import tiktoken

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "tables"
SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "chunk_table_aware_v2.py"
REPO_ROOT = Path(__file__).resolve().parents[3]

ENCODING = "cl100k_base"
SEPARATOR_RE = re.compile(r"^\|[\s\-|:]+\|$")

FIXTURE_PATHS = sorted(FIXTURE_DIR.glob("*.json"))
FIXTURE_IDS = [p.stem for p in FIXTURE_PATHS]

BASE_META = {
    "source": "test",
    "source_type": "web_crawl",
    "year": "2026",
    "data_year": 2026,
    "info_type": "thong_tin_chung",
    "school": "HUSC",
    "audience": "thi_sinh",
    "unit": "HUSC",
}

@pytest.fixture(scope="module")
def cta():
    spec = importlib.util.spec_from_file_location("chunk_table_aware_v2", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding(ENCODING)


@pytest.fixture(scope="module")
def fixtures() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in FIXTURE_PATHS}


def _fields(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _pipe_lines(text: str) -> list[str]:
    return [l.strip() for l in text.split("\n") if l.strip().startswith("|")]


def _body_lines(text: str) -> list[str]:
    """Pipe lines minus markdown separator lines."""
    return [l for l in _pipe_lines(text) if not SEPARATOR_RE.match(l)]


def _cells(lines: list[str]) -> list[str]:
    return [c for l in lines if not SEPARATOR_RE.match(l) for c in _fields(l) if c]


def _norm(s: str) -> str:
    return " ".join(s.split())


def _declared_body(fx: dict) -> list[str]:
    return [l for l in fx["markdown_lines"] if not SEPARATOR_RE.match(l.strip())]


def _declared_rows(fx: dict) -> list[str]:
    return _declared_body(fx)[1:]


def _emit(cta, fx: dict, enc) -> list[dict]:
    """Run the fixture through the parser exactly the way the ingest path does."""
    doc_id = fx["fixture_id"]
    if fx["chunk_path"] == "merge_small_table":
        chunks: list[dict] = []
        for fragment in fx["fragments"]:
            table = cta.MarkdownTable(fragment["markdown_lines"])
            chunks.extend(cta.chunk_table(table, BASE_META, doc_id, enc, len(chunks)))
        return cta._merge_consecutive_table_chunks(chunks, BASE_META, doc_id, enc)
    table = cta.MarkdownTable(fx["markdown_lines"])
    return cta.chunk_table(table, BASE_META, doc_id, enc, 0)


def _output_rows(chunks: list[dict]) -> list[str]:
    return [row for c in chunks for row in _body_lines(c["text"])[1:]]


def _rowspan_records(chunks: list[dict]) -> tuple[dict[int, str], dict[int, str]]:
    """The chunker's own merged-cell report: row -> owning identity (attributed),
    row -> reason (unresolved)."""
    attributed: dict[int, str] = {}
    unresolved: dict[int, str] = {}
    for chunk in chunks:
        for record in chunk["metadata"].get("rowspan_attribution") or []:
            attributed[record["row"]] = record["identity"]
        for record in chunk["metadata"].get("rowspan_unresolved") or []:
            unresolved[record["row"]] = record["reason"]
    return attributed, unresolved


def _leading_region_rows(fx: dict) -> set[int]:
    """The declared rows that lean on a cell above them in their LEADING cells —
    the columns before the row's first value. That is the region the chunker can
    attribute: a blank interior cell (p57's unwritten 'Phương thức tuyển sinh')
    belongs to the row that owns it and must stay blank."""
    rows = [_fields(r) for r in _declared_rows(fx)]
    in_domain = set()
    for row in fx["ambiguous_rows"]:
        cells = rows[row["index"]]
        first = min((j for j, c in enumerate(cells) if c), default=0)
        if any(c["col"] < first for c in row["columns"]):
            in_domain.add(row["index"])
    return in_domain


# ──────────────────────────────────────────────────────────────────────────
# Fixture provenance
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_fixture_declares_real_document_and_row_pages(fixtures, fixture_id):
    """Every row is tied to a real document in the repo and to a page of it."""
    fx = fixtures[fixture_id]

    assert (REPO_ROOT / fx["source_document"]).is_file(), "fixture must cite a real document"
    assert (REPO_ROOT / fx["staging_file"]).is_file(), "fixture must cite the parser's real input"
    assert (REPO_ROOT / fx["corpus_file"]).is_file(), "fixture must cite the real parser output"

    declared_rows = _declared_rows(fx)
    assert declared_rows, "a fixture must declare at least one data row"
    assert len(fx["rows"]) == len(declared_rows)
    assert len(fx["column_headers"]) >= max(len(_fields(r)) for r in declared_rows)

    for row in fx["rows"]:
        assert isinstance(row["page"], int) and row["page"] > 0
        assert row["page"] in fx["source_pages"]
        assert row["anchor"].strip(), "each row needs an anchor identifying it on the page"


def test_fixture_set_covers_both_ambiguous_and_clean_tables(fixtures):
    """The merged-cell guard must have a real positive and a real negative."""
    ambiguous = [fid for fid, fx in fixtures.items() if fx["ambiguous_rows"]]
    clean = [fid for fid, fx in fixtures.items() if not fx["ambiguous_rows"]]

    assert ambiguous, "no fixture exercises inherited (merged/rowspan) attribution"
    assert clean, "no fixture is free of inherited attribution to act as a control"


# ──────────────────────────────────────────────────────────────────────────
# No cell disappears
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_every_expected_cell_survives_extraction(cta, enc, fixtures, fixture_id):
    """Each hand-transcribed cell is still somewhere in the emitted chunks."""
    fx = fixtures[fixture_id]
    chunks = _emit(cta, fx, enc)
    emitted = set(_norm(c) for c in _cells([l for c in chunks for l in _pipe_lines(c["text"])]))

    missing = [c for c in fx["expected_cells"] if _norm(c) not in emitted]
    assert missing == [], f"{fixture_id}: cells lost during extraction: {missing}"


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_output_cell_count_matches_expected(cta, enc, fixtures, fixture_id):
    """A dropped or duplicated row changes the count and fails here."""
    fx = fixtures[fixture_id]
    chunks = _emit(cta, fx, enc)
    out_rows = _output_rows(chunks)

    assert len(out_rows) == len(_declared_rows(fx)), "row count changed"

    emitted_count = sum(len([c for c in _fields(r) if c]) for r in out_rows)
    assert emitted_count == fx["expected_data_cell_count"], (
        f"{fixture_id}: emitted {emitted_count} data cells, "
        f"expected {fx['expected_data_cell_count']}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Row <-> header relationship
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_every_chunk_reemits_the_declared_header(cta, enc, fixtures, fixture_id):
    """A chunk that loses its header loses the meaning of every cell in it."""
    fx = fixtures[fixture_id]
    chunks = _emit(cta, fx, enc)

    assert chunks, "fixture produced no chunks at all"
    for chunk in chunks:
        assert _pipe_lines(chunk["text"])[0] == fx["markdown_lines"][0]


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_every_row_keeps_its_positional_cells(cta, enc, fixtures, fixture_id):
    """Cells stay in their columns: no shift, no merge across rows."""
    fx = fixtures[fixture_id]
    chunks = _emit(cta, fx, enc)
    declared = [_fields(r) for r in _declared_rows(fx)]
    emitted = [_fields(r) for r in _output_rows(chunks)]

    assert len(emitted) == len(declared)
    for index, (want, got) in enumerate(zip(declared, emitted)):
        assert got == want, f"{fixture_id}: data row {index} changed columns"


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_every_cell_traces_to_its_column_header(cta, enc, fixtures, fixture_id):
    """Each emitted data cell matches the fixture's declared column attribution.

    ``cell_attribution`` in the fixture is the canonical form: for every data row
    it names the column index, the column header and the cell text. A cell that
    slid to another column, or was dropped, changes this mapping.

    The header line is held to a bounds check only: these PDFs wrap a header
    across pipe rows, so a header *fragment* can land in a column whose declared
    name belongs to a neighbouring column (e.g. 'đào tạo' split off from
    'Tên ngành/Chương trình đào tạo'). Naming that column would be inventing a
    header the page does not have.
    """
    fx = fixtures[fixture_id]
    chunks = _emit(cta, fx, enc)
    headers = fx["column_headers"]

    for chunk in chunks:
        for col, cell in enumerate(_fields(_pipe_lines(chunk["text"])[0])):
            if cell:
                assert col < len(headers), (
                    f"{fixture_id}: header cell {cell!r} outside declared columns"
                )

    emitted = [
        {"row": i, "cells": [{"col": j, "text": c} for j, c in enumerate(_fields(row)) if c]}
        for i, row in enumerate(_output_rows(chunks))
    ]
    declared = [
        {"row": a["row"], "cells": [{"col": c["col"], "text": c["text"]} for c in a["cells"]]}
        for a in fx["cell_attribution"]
    ]
    assert emitted == declared, f"{fixture_id}: column attribution changed"


# ──────────────────────────────────────────────────────────────────────────
# Page provenance
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_every_row_carries_page_provenance(cta, enc, fixtures, fixture_id):
    """Each emitted row is bound to the page the fixture claims for it."""
    fx = fixtures[fixture_id]
    chunks = _emit(cta, fx, enc)
    out_rows = _output_rows(chunks)

    assert len(out_rows) == len(fx["rows"])
    for index, row in enumerate(out_rows):
        page = fx["rows"][index]["page"]
        assert page in fx["source_pages"]
        assert any(_fields(row)), f"{fixture_id}: row {index} on page {page} is empty"


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_declared_pages_really_contain_the_row_cells(fixtures, fixture_id):
    """The page claimed for a row is verified against the source PDF itself."""
    fx = fixtures[fixture_id]
    pdf_path = REPO_ROOT / fx["source_document"]
    if not pdf_path.is_file():
        pytest.skip(f"source document not present: {fx['source_document']}")
    pypdf = pytest.importorskip("pypdf")

    reader = pypdf.PdfReader(str(pdf_path))
    page_text = {
        page: _norm(reader.pages[page - 1].extract_text(extraction_mode="layout") or "")
        for page in fx["source_pages"]
    }

    for index, declared in enumerate(_declared_rows(fx)):
        page = fx["rows"][index]["page"]
        for cell in _fields(declared):
            if cell:
                assert _norm(cell) in page_text[page], (
                    f"{fixture_id}: row {index} cell {cell!r} is not on page {page}"
                )


def test_pypdf_tier_reproduces_the_fixture_cells(fixtures, tmp_path):
    """The fixtures' cells come from the document through the pipeline's tier 0.

    docling is not installed, so tier 0 (pypdf) is the tier under test; a fixture
    cell that only existed in a hand-made table would not be found here.
    """
    pipeline = pytest.importorskip("src.services.pdf_pipeline")
    pdf_path = REPO_ROOT / "new_data/THONG_TIN_TUYEN_SINH_202611.pdf"
    if not pdf_path.is_file():
        pytest.skip("source document not present")

    result = pipeline.parse_pdf(pdf_path, tmp_path, force="pypdf")
    assert result.parser == "pypdf"
    assert result.page_count > 0
    text = _norm(result.text)

    for fixture_id, fx in fixtures.items():
        if fx["source_document"] != "new_data/THONG_TIN_TUYEN_SINH_202611.pdf":
            continue
        assert _norm(fx["provenance_anchor"]) in text
        missing = [c for c in fx["expected_cells"] if _norm(c) not in text]
        assert missing == [], f"{fixture_id}: not in pypdf text: {missing}"


# ──────────────────────────────────────────────────────────────────────────
# Merged / rowspan cells: reported by the parser, never silently dropped
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_ambiguous_merged_cells_are_reported_by_the_chunker_itself(cta, enc, fixtures, fixture_id):
    """Rows leaning on inherited attribution are accounted for by the chunker, and
    keep their cells.

    This is a read of the production report (`rowspan_attribution` /
    `rowspan_unresolved`), not a second detector: ticket 32 moved the guard into
    ``chunk_table_aware_v2``, so a harness-local scan would stay green while the
    emitted corpus silently lost the attribution again.
    """
    fx = fixtures[fixture_id]
    chunks = _emit(cta, fx, enc)
    attributed, unresolved = _rowspan_records(chunks)

    expected = _leading_region_rows(fx)
    assert expected <= set(attributed) | set(unresolved), (
        f"{fixture_id}: rows {sorted(expected - (set(attributed) | set(unresolved)))} "
        f"lean on a merged cell and were neither attributed nor reported"
    )
    assert not (set(attributed) & set(unresolved)), "a row cannot be both resolved and unresolved"
    for index, reason in unresolved.items():
        assert reason.strip(), f"{fixture_id}: unresolved row {index} carries no reason"

    # The leaning rows are reported, not dropped: their content is still there.
    out_rows = [_fields(r) for r in _output_rows(chunks)]
    declared = [_fields(r) for r in _declared_rows(fx)]
    for index in sorted(set(attributed) | set(unresolved)):
        got = out_rows[index]
        for col, cell in enumerate(declared[index]):
            if cell:
                assert got[col] == cell, (
                    f"{fixture_id}: leaning row {index} lost cell {cell!r} "
                    f"from column {col}"
                )
