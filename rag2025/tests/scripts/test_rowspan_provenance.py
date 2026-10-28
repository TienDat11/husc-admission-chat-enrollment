"""Ticket 32: rowspan / merged-cell provenance on the production chunker path.

A PDF table that merges an identity cell vertically ("Số TT" / "mã ngành" /
"tên ngành") prints the value once; the continuation rows reach
``scripts/chunk_table_aware_v2.py`` with those cells empty. Emitted as-is, such a
row asserts a quota / cutoff score / tuition figure for a programme the chunk
never names. These tests pin what the production path now does instead: it
attaches the owning row's identity to the chunk (``rowspan_attribution``) or
reports the row it cannot attribute (``rowspan_unresolved`` + a `logger.warning`
raised by the parser itself, not by this harness) — never both, never silently.
A row that cannot be attributed is *not* dropped: its printed cells stay exactly
as the page has them (ticket 04's landed net asserts they survive) and it is
labelled as unattributed in the chunk metadata instead.

Fixtures come from real documents (ticket 04 hand-transcription):
``ttts2026_p12_chi_tieu_dh_luat`` (page 12), ``ttts2026_p12_chi_tieu_dh_luat_kinh_te``
(page 12), ``ttts2026_p24_to_hop_mon_merge`` (page 24, negative control),
``ttts2026_p55_hoc_phi_khoa_hoc`` (page 55), ``ttts2026_p57_diem_trung_tuyen_kinh_te``
(page 57) — all from ``new_data/THONG_TIN_TUYEN_SINH_202611.pdf``, parsed through
``rag2025/data/staging/pdf/pdf_2.md``.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
import tiktoken
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "chunk_table_aware_v2.py"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "tables"
STAGING_PDF_MD = REPO_ROOT / "rag2025" / "data" / "staging" / "pdf" / "pdf_2.md"

ENCODING = "cl100k_base"
SEPARATOR_RE = re.compile(r"^\|[\s\-|:]+\|$")

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

# The document every table fixture is transcribed from (page per fixture below).
SOURCE_PDF = "new_data/THONG_TIN_TUYEN_SINH_202611.pdf"
FIXTURE_PAGES = {
    "ttts2026_p12_chi_tieu_dh_luat": 12,
    "ttts2026_p12_chi_tieu_dh_luat_kinh_te": 12,
    "ttts2026_p24_to_hop_mon_merge": 24,
    "ttts2026_p55_hoc_phi_khoa_hoc": 55,
    "ttts2026_p57_diem_trung_tuyen_kinh_te": 57,
}

# Rows the production path must attribute, and rows it must leave unresolved
# (leading-cell regions that no anchor owns). Taken from ticket 04's per-row /
# per-column scan of pages 12 and 57 — see the issue for why each row sits here.
P12_SCHOOL = "I.  TRƯỜNG ĐẠI HỌC LUẬT – Ký hiệu trường DHA"
EXPECTED = {
    "ttts2026_p12_chi_tieu_dh_luat": {
        "attributed": {4: P12_SCHOOL, 5: P12_SCHOOL, 7: "1  Luật / 7380101", 8: "1  Luật / 7380101"},
        "unresolved": {1},
    },
    "ttts2026_p12_chi_tieu_dh_luat_kinh_te": {
        "attributed": {4: "2  Luật Kinh tế / 7380107", 5: "2  Luật Kinh tế / 7380107"},
        "unresolved": set(),
    },
    "ttts2026_p57_diem_trung_tuyen_kinh_te": {
        "attributed": {},
        "unresolved": {0, 2},
    },
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
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(FIXTURE_DIR.glob("*.json"))}


class _Captured:
    """Warnings the production code logs (loguru), not warnings raised by tests."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self._sink_id = 0

    def __enter__(self) -> "_Captured":
        self._sink_id = logger.add(lambda m: self.messages.append(str(m)), level="WARNING", format="{message}")
        return self

    def __exit__(self, *exc) -> None:
        logger.remove(self._sink_id)


def _fields(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _row_lines(fixture: dict) -> list[str]:
    """The fixture's data rows, verbatim (header and separator removed)."""
    body = [l for l in fixture["markdown_lines"] if not SEPARATOR_RE.match(l.strip())]
    return [l.strip() for l in body[1:]]


def _rowspan_metadata(chunks: list[dict]) -> tuple[dict[int, dict], dict[int, dict]]:
    """Every rowspan record the emitted chunks carry, keyed by data-row index."""
    attributed: dict[int, dict] = {}
    unresolved: dict[int, dict] = {}
    for chunk in chunks:
        meta = chunk["metadata"]
        for entry in meta.get("rowspan_attribution", []):
            assert entry["row"] not in attributed, f"row {entry['row']} attributed twice"
            attributed[entry["row"]] = {**entry, "chunk": chunk}
        for entry in meta.get("rowspan_unresolved", []):
            assert entry["row"] not in unresolved, f"row {entry['row']} reported unresolved twice"
            unresolved[entry["row"]] = {**entry, "chunk": chunk}
    return attributed, unresolved


def _emit(cta, fixture: dict, enc) -> list[dict]:
    """Run a fixture through the parser exactly the way the ingest path does."""
    doc_id = fixture["fixture_id"]
    chunks: list[dict] = []
    fragments = fixture["fragments"] if fixture.get("chunk_path") == "merge_small_table" else [
        {"markdown_lines": fixture["markdown_lines"]}
    ]
    for fragment in fragments:
        table = cta.MarkdownTable(fragment["markdown_lines"])
        chunks.extend(cta.chunk_table(table, BASE_META, doc_id, enc, len(chunks)))
    if fixture.get("chunk_path") == "merge_small_table":
        return cta._merge_consecutive_table_chunks(chunks, BASE_META, doc_id, enc)
    return chunks


def _data_rows(fixture: dict) -> list[list[str]]:
    body = [l for l in fixture["markdown_lines"] if not SEPARATOR_RE.match(l.strip())]
    return [_fields(l) for l in body[1:]]


def _leaning_rows(fixture: dict) -> set[int]:
    """Rows whose leading cells are empty while a row above (or the header, for
    the first row) fills one of them — the merged-cell shape, detected here
    independently of the parser so the test can prove nothing is missed."""
    body = [l for l in fixture["markdown_lines"] if not SEPARATOR_RE.match(l.strip())]
    filled = {j for j, c in enumerate(_fields(body[0])) if c}
    leaning: set[int] = set()
    for index, cells in enumerate(_data_rows(fixture)):
        content = [j for j, c in enumerate(cells) if c]
        if content and content[0] > 0 and any(j in filled for j in range(content[0])):
            leaning.add(index)
        filled |= set(content)
    return leaning


# ──────────────────────────────────────────────────────────────────────────
# Fixture provenance: real document + real page
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture_id", sorted(EXPECTED))
def test_fixture_comes_from_a_real_document_page(fixtures, fixture_id):
    fixture = fixtures[fixture_id]

    assert (REPO_ROOT / fixture["source_document"]).is_file()
    assert fixture["source_document"] == SOURCE_PDF
    assert fixture["source_pages"] == [FIXTURE_PAGES[fixture_id]]
    assert (REPO_ROOT / fixture["staging_file"]).is_file()


# ──────────────────────────────────────────────────────────────────────────
# Every leaning row: attributed, or reported — never both, never neither
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture_id", sorted(EXPECTED))
def test_every_leaning_row_is_attributed_or_reported_never_both(cta, enc, fixtures, fixture_id):
    """The core contract: no row that leans on a merged cell is emitted silently.

    ``attributed`` and ``unresolved`` are disjoint (a row is never in both
    states) and together cover exactly the rows whose leading cells lean on a
    row above (no leaning row is ignored by the production path).
    """
    fixture = fixtures[fixture_id]
    with _Captured() as captured:
        chunks = _emit(cta, fixture, enc)

    attributed, unresolved = _rowspan_metadata(chunks)

    assert set(attributed) & set(unresolved) == set(), "a row must not be attributed and reported at once"
    assert set(attributed) | set(unresolved) == _leaning_rows(fixture)

    expected = EXPECTED[fixture_id]
    assert set(attributed) == set(expected["attributed"])
    assert set(unresolved) == expected["unresolved"]
    assert len([m for m in captured.messages if "merged/rowspan" in m]) == len(unresolved), (
        "exactly one production warning per row that could not be attributed"
    )
    for index, identity in expected["attributed"].items():
        assert attributed[index]["identity"] == identity
    for index in expected["unresolved"]:
        assert unresolved[index]["reason"].strip(), "an unresolved row must say why"
        assert unresolved[index]["columns"], "an unresolved row must name the empty columns"


@pytest.mark.parametrize("fixture_id", ["ttts2026_p12_chi_tieu_dh_luat", "ttts2026_p57_diem_trung_tuyen_kinh_te"])
def test_reported_rows_are_warned_by_the_parser_itself(cta, enc, fixtures, fixture_id):
    """The warning lives on the production path: the parser logs it, and it names
    the document and the row that could not be attributed."""
    fixture = fixtures[fixture_id]
    with _Captured() as captured:
        chunks = _emit(cta, fixture, enc)

    _, unresolved = _rowspan_metadata(chunks)

    assert unresolved, "this fixture is expected to have unattributable rows"
    for index in unresolved:
        matching = [m for m in captured.messages if f"row {index} " in m and fixture_id in m]
        assert matching, f"no production warning for row {index}: {captured.messages}"
        assert "merged/rowspan" in matching[0]


def test_p24_contiguous_table_is_the_negative_control(cta, enc, fixtures):
    """Negative control: a table whose rows are contiguous warns zero times and
    carries no rowspan metadata — after the merge step that this table really
    goes through, so the check is not vacuous."""
    fixture = fixtures["ttts2026_p24_to_hop_mon_merge"]
    assert _leaning_rows(fixture) == set()

    with _Captured() as captured:
        chunks = _emit(cta, fixture, enc)

    assert captured.messages == [], f"false positives: {captured.messages}"
    assert chunks, "the fixture must still produce chunks"
    for chunk in chunks:
        assert chunk["metadata"], "metadata was stripped, so 'no rowspan keys' would mean nothing"
        assert not any(k.startswith("rowspan_") for k in chunk["metadata"])


# ──────────────────────────────────────────────────────────────────────────
# The size gate must not re-silence a reported row
# ──────────────────────────────────────────────────────────────────────────

# Constructed boundary case, not a transcribed page: the shape mirrors the real
# programme-grouped tables (header carrying "ngành", one group per programme
# prefix) but is sized so the trailing group's chunk lands between MIN_TOKENS
# (25) and the soft floor (settings.SOFT_MIN_TOKENS = 50), which is where
# `_enforce_size_gates` folds it into the neighbouring group's chunk.
SOFT_FLOOR_TABLE = [
    "| TT | Mã ngành | Tổ hợp môn |",
    "| --- | --- | --- |",
] + [f"| 1 | 7380101 | {i}. Toán, Vật lí, Hóa học |" for i in range(1, 7)] + [
    "|  |  | 3. Ngữ văn |",
]


def test_soft_floor_merge_keeps_the_absorbed_chunks_provenance(cta, enc):
    """A sub-floor chunk is folded into its neighbour, which then carries its
    rows too — so the absorbed chunk's rowspan report has to survive the merge.
    Dropping it would put the row back into the corpus with neither attribution
    nor warning: the silent state this ticket exists to remove."""
    table = cta.MarkdownTable(SOFT_FLOOR_TABLE)
    pre = cta._chunk_table(table, BASE_META, "size_gate", enc, 0)
    absorbed = [c for c in pre if c["metadata"].get("rowspan_unresolved")]
    assert len(absorbed) == 1, "the boundary case must produce one leaning chunk"
    assert cta.count_tokens(absorbed[0]["text"], enc) < cta.settings.SOFT_MIN_TOKENS
    assert absorbed[0]["metadata"]["rowspan_unresolved"][0]["row"] == 6

    chunks = cta.chunk_table(table, BASE_META, "size_gate", enc, 0)

    assert len(chunks) == len(pre) - 1, "the sub-floor chunk must have been absorbed"
    merged = [c for c in chunks if len(c["text"]) > len(absorbed[0]["text"])]
    assert len(merged) == 1, "exactly one chunk survives the merge"
    kept = merged[0]["metadata"].get("rowspan_unresolved")
    assert kept == absorbed[0]["metadata"]["rowspan_unresolved"], (
        "the absorbed chunk's rowspan report did not survive the soft-floor merge"
    )
    assert "3. Ngữ văn" in merged[0]["text"], "the absorbed rows are in this chunk"


# ──────────────────────────────────────────────────────────────────────────
# The positives: rows 2..n carry the owning programme
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fixture_id, rows",
    [
        ("ttts2026_p12_chi_tieu_dh_luat", (4, 5, 7, 8)),
        ("ttts2026_p12_chi_tieu_dh_luat_kinh_te", (4, 5)),
    ],
)
def test_continuation_rows_carry_the_owning_programme(cta, enc, fixtures, fixture_id, rows):
    """Page 12: the combo rows of a programme block inherit that block's Số TT,
    tên ngành AND mã ngành — the label a student lookup needs to be right."""
    fixture = fixtures[fixture_id]
    chunks = _emit(cta, fixture, enc)
    attributed, _ = _rowspan_metadata(chunks)

    for index in rows:
        entry = attributed[index]
        assert entry["anchor_row"] < index
        # The identity is the anchor row's own cells, verbatim — not a value
        # from another column and not another programme's numbers.
        anchor_cells = _data_rows(fixture)[entry["anchor_row"]]
        assert entry["cells"] == {str(j): anchor_cells[j] for j in sorted(int(k) for k in entry["cells"])}
        # ...and the row it belongs to is emitted, unmodified, in that same chunk.
        assert _row_lines(fixture)[index] in entry["chunk"]["text"]


def test_the_programme_code_is_inherited_not_dropped(cta, enc, fixtures):
    """Mã ngành is the strongest programme identifier on the page: a row that
    leans on a merged cell must come out with it, not just with the name."""
    fixture = fixtures["ttts2026_p12_chi_tieu_dh_luat"]
    chunks = _emit(cta, fixture, enc)
    attributed, _ = _rowspan_metadata(chunks)

    assert attributed[7]["cells"]["2"] == "7380101"
    assert attributed[7]["identity"] == "1  Luật / 7380101"


def test_attribution_addresses_rows_the_chunk_carries(cta, enc, fixtures):
    """Traceability: for every attribution a chunk carries, both the leaning row
    and its anchor row are rows of that same chunk (row indices are table-level —
    the chunk re-emits the header, so the table is the frame of reference), and
    the inherited cells are the anchor row's own cells."""
    for fixture_id in EXPECTED:
        fixture = fixtures[fixture_id]
        row_lines = _row_lines(fixture)
        for chunk in _emit(cta, fixture, enc):
            for entry in chunk["metadata"].get("rowspan_attribution", []):
                for index in (entry["row"], entry["anchor_row"]):
                    assert row_lines[index] in chunk["text"], (
                        f"{fixture_id}: attribution for row {entry['row']} is not in the "
                        f"chunk carrying it"
                    )


# ──────────────────────────────────────────────────────────────────────────
# page 57: two programmes printed with the same Số TT
# ──────────────────────────────────────────────────────────────────────────


def test_p57_shares_no_attribution_between_two_programmes(cta, enc, fixtures):
    """Page 57 prints Số TT "5" for both ``7310106 Kinh tế quốc tế`` and
    ``7510605 Logistics& Quản lý chuỗi cung ứng``. No lineage is invented: the
    continuation rows stay unattributed and are warned about, and the blank
    "Phương thức tuyển sinh" cell is never filled with the row above's THPT."""
    fixture = fixtures["ttts2026_p57_diem_trung_tuyen_kinh_te"]
    rows = fixture["markdown_lines"]
    with _Captured() as captured:
        chunks = _emit(cta, fixture, enc)

    attributed, unresolved = _rowspan_metadata(chunks)
    text = "\n".join(c["text"] for c in chunks)

    assert attributed == {}
    assert set(unresolved) == {0, 2}
    assert len([m for m in captured.messages if "merged/rowspan" in m]) == 2
    # every declared row is still emitted verbatim: nothing was invented for the
    # empty cells, and nothing was dropped either.
    for line in rows:
        if not SEPARATOR_RE.match(line.strip()):
            assert line.strip() in text


# ──────────────────────────────────────────────────────────────────────────
# Real wrapped cells (pages 12 / 48 of the prospectus, via the staging file):
# a fragment or a number must never become a programme label
# ──────────────────────────────────────────────────────────────────────────


def test_wrapped_cell_fragments_are_never_inherited(cta, enc):
    """``new_data/THONG_TIN_TUYEN_SINH_202611.pdf`` page 15 (arriving here as
    ``data/staging/pdf/pdf_2.md`` lines 552-555): the programme name "1 - Đầu tư;
    Kinh tế và quản lý du lịch; Kinh tế và quản lý tài nguyên môi trường" is
    wrapped across a markdown row boundary, so the combo row below it has only
    the fragments "lịch; Kinh" | "tế và quản" | "lý tài" above it. Inheriting
    those would label the row with half a sentence."""
    lines = [
        "| 1  - Đầu tư; Kinh tế và quản lý du  |               |          |     |                           |      |     | 170    |",
        "| ----------------------------------- | ------------- | -------- | --- | ------------------------- | ---- | --- | ------ |",
        "| lịch;  Kinh                         | tế  và  quản  | lý  tài  |     |                           |      |     |        |",
        "|                                     |               |          |     | 1. Toán, Vật lí, Hóa học  | A00  |     |        |",
    ]
    table = cta.MarkdownTable(lines)

    with _Captured() as captured:
        chunks = cta.chunk_table(table, BASE_META, "pdf_2", enc, 0)

    attributed, unresolved = _rowspan_metadata(chunks)
    assert attributed == {}, "a wrapped cell is not an identity"
    assert set(unresolved) == {1}
    assert "wrapped cell fragments" in unresolved[1]["reason"]
    assert unresolved[1]["skipped"] == {"0": "lịch;  Kinh", "1": "tế  và  quản", "2": "lý  tài"}
    assert any(f"row {1} " in m and "pdf_2" in m for m in captured.messages)


def test_a_quota_number_is_never_inherited_as_a_label(cta, enc):
    """``new_data/THONG_TIN_TUYEN_SINH_202611.pdf`` page 61 (arriving here as
    ``data/staging/pdf/pdf_2.md`` lines 2749-2752): the anchor row carries Số TT,
    mã ngành, tên ngành *and* two quota figures (30 / 4). Only the identity
    columns may be inherited — a quota must never become a label."""
    lines = [
        "|     |          |                  |     |     |     | 21,00*  |     |     |     |        |     |",
        "| --- | -------- | ---------------- | --- | --- | --- | ------- | --- | --- | --- | ------ | --- |",
        "| 5   | 7229042  | Quản lý văn hóa  |     | 30  | 4   |         |     |     |     | 16,00  |     |",
        "|     |          |                  |     |     |     | 16,00   |     | 30  | 30  |        |     |",
    ]
    table = cta.MarkdownTable(lines)

    chunks = cta.chunk_table(table, BASE_META, "pdf_2", enc, 0)

    attributed, unresolved = _rowspan_metadata(chunks)
    assert set(attributed) == {1}
    assert attributed[1]["identity"] == "5 / 7229042 / Quản lý văn hóa"
    assert set(attributed[1]["cells"].values()) == {"5", "7229042", "Quản lý văn hóa"}
    assert unresolved == {}


# ──────────────────────────────────────────────────────────────────────────
# End to end through the materializer entry point
# ──────────────────────────────────────────────────────────────────────────


def test_materializer_emits_provenance_and_warnings(cta, enc):
    """``process_file`` is what re-materializes the corpus: running it on the
    real staging document must both warn and carry the attribution in chunk
    metadata — no test-only mechanism involved."""
    with _Captured() as captured:
        chunks = cta.process_file(STAGING_PDF_MD, enc, 24)

    assert len([m for m in captured.messages if "merged/rowspan" in m]) > 0
    luat = [
        c for c in chunks
        if "1  Luật" in c["text"] and "7380101" in c["text"] and c["metadata"].get("rowspan_attribution")
    ]
    assert luat, "the Luật table lost its attribution"
    identities = [e["identity"] for c in luat for e in c["metadata"]["rowspan_attribution"]]
    assert "1  Luật / 7380101" in identities
