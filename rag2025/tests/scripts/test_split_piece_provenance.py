"""Ticket 33: a split piece carries only the provenance of the rows it holds.

Ticket 32 attaches each table's rowspan provenance to the chunk that carries the
rows it names. The size gate then cuts an oversized chunk into pieces — and the
pre-ticket-33 gate copied the whole provenance dict into *every* piece, so a
piece could name a row that landed in a sibling piece. On
``data/staging/pdf/pdf_2.md`` that was 55 of the 149 entries carried by split
pieces (13 pieces, 6 oversized parents): provenance that lies about which rows
its own text contains.

The rule now: a piece carries exactly the entries of the rows it holds. Row
indices stay TABLE-scoped (ticket 32's frame — the chunk re-emits the table
header, so a row index never identifies a row inside a chunk), and the pieces of
one split hold disjoint row sets, so every entry has exactly one home and none
is dropped.

The tests below run the real ``process_file`` path over ``pdf_2.md`` — the
document the defect was measured on — and pin: the scoping on a chunk that
really splits, conservation across a split, an unchanged unsplit chunk, and the
one-sided direction of error (no piece names a row it does not contain).
"""

from __future__ import annotations

import copy
import importlib.util
from collections import Counter
from pathlib import Path

import pytest
import tiktoken

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "chunk_table_aware_v2.py"
STAGING_PDF_MD = REPO_ROOT / "rag2025" / "data" / "staging" / "pdf" / "pdf_2.md"

ENCODING = "cl100k_base"

BASE_META = {
    "source": "test",
    "source_type": "pdf_extract",
    "year": "2026",
    "data_year": 2026,
    "info_type": "diem_chuan",
    "school": "HUSC",
    "audience": "thi_sinh",
    "unit": "HUSC",
}

PROVENANCE_KEYS = ("rowspan_attribution", "rowspan_unresolved")


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


def _entries(meta: dict) -> list[dict]:
    """Every rowspan record a chunk's metadata carries."""
    return [entry for key in PROVENANCE_KEYS for entry in meta.get(key, [])]


def _run_process_file(cta, enc) -> tuple[list[tuple], list[dict]]:
    """``process_file`` over the real document, recording every table's chunks.

    Returns ``(calls, chunks)`` where each call is ``(table, pre_gate, post_gate)``:
    the table the production path chunked, the chunks ``_chunk_table`` produced
    (before the size gates) and the chunks the gates returned.
    """
    calls: list[tuple] = []
    original = cta.chunk_table

    def recording_chunk_table(table, base_meta, doc_id, enc_, start_idx):
        pre_gate = cta._chunk_table(table, base_meta, doc_id, enc_, start_idx)
        post_gate = original(table, base_meta, doc_id, enc_, start_idx)
        calls.append((table, pre_gate, post_gate))
        return post_gate

    cta.chunk_table = recording_chunk_table
    try:
        chunks = cta.process_file(STAGING_PDF_MD, enc, 2)
    finally:
        cta.chunk_table = original
    return calls, chunks


@pytest.fixture(scope="module")
def pdf2(cta, enc) -> tuple[list[tuple], list[dict]]:
    """The real split corpus: `pdf_2.md` chunked exactly as the ingest path does."""
    return _run_process_file(cta, enc)


def _split_pieces(calls: list[tuple]) -> list[tuple]:
    """``(table, piece)`` for every piece the size gate produced."""
    out = []
    for table, _pre_gate, post_gate in calls:
        for piece in post_gate:
            if "split_index" in piece["metadata"]:
                out.append((table, piece))
    return out


def _oversized_provenance_parents(cta, enc, calls: list[tuple]) -> list[dict]:
    """The real chunks the size gate had to cut, the ones that carry provenance."""
    hard_max = cta.settings.HARD_MAX_TOKENS
    return [
        parent
        for _table, pre_gate, _post in calls
        for parent in pre_gate
        if _entries(parent["metadata"]) and cta.count_tokens(parent["text"], enc) > hard_max
    ]


# ──────────────────────────────────────────────────────────────────────────
# The real split case: pdf_2.md through process_file
# ──────────────────────────────────────────────────────────────────────────


def test_the_real_document_really_splits_provenance_carrying_chunks(cta, enc, pdf2):
    """Guard against a vacuous suite: the document the defect was measured on
    must still produce split pieces that carry rowspan provenance."""
    calls, _chunks = pdf2
    pieces = _split_pieces(calls)

    assert pieces, "pdf_2.md no longer produces split pieces — this suite proves nothing"
    carrying = [(table, piece) for table, piece in pieces if _entries(piece["metadata"])]
    assert carrying, "no split piece carries provenance — the scoping has nothing to prove"
    for table, piece in carrying:
        for entry in _entries(piece["metadata"]):
            assert 0 <= entry["row"] < len(table.data_rows)


def test_no_split_piece_names_a_row_its_text_does_not_contain(cta, enc, pdf2):
    """The direction of error, on the real corpus: an entry may only name a row
    the piece actually contains.

    Checked against the piece's own text — the table's row line has to be in it —
    so the proof does not depend on the row-index bookkeeping under test. Before
    ticket 33 the gate copied every entry into every piece and 55 entries named a
    row that had fallen to a sibling piece.
    """
    calls, _chunks = pdf2
    pieces = _split_pieces(calls)
    assert pieces

    checked = 0
    for table, piece in pieces:
        lines = set(piece["text"].split("\n"))
        for entry in _entries(piece["metadata"]):
            assert table.data_rows[entry["row"]] in lines, (
                f"piece {piece['id']} names row {entry['row']}, which its text does not contain"
            )
            checked += 1
    assert checked, "no entry was checked"


def test_split_corpus_carries_each_entry_exactly_once(cta, enc, pdf2):
    """Conservation, corpus-wide: the pieces of the real splits carry exactly the
    entries the unsplit chunks produced — no entry duplicated into a piece that
    does not own it, and none dropped.

    The unsplit entry set is read off ``_chunk_table`` (the chunking before the
    gates) and the piece set off the chunks the gates returned.
    """
    calls, chunks = pdf2

    unsplit = Counter()
    for _table, pre_gate, _post_gate in calls:
        for parent in pre_gate:
            unsplit.update((e["row"], e.get("anchor_row")) for e in _entries(parent["metadata"]))

    emitted = Counter()
    for chunk in chunks:
        emitted.update((e["row"], e.get("anchor_row")) for e in _entries(chunk["metadata"]))

    assert emitted == unsplit, (
        f"split corpus lost or duplicated provenance: {unsplit - emitted} missing, "
        f"{emitted - unsplit} extra"
    )
    assert sum(emitted.values()) > 0


def test_each_oversized_chunk_keeps_its_entries_across_its_own_pieces(cta, enc, pdf2):
    """Conservation per split: for every real oversized chunk, the union of its
    pieces' entries is its own entry set — the same records, intact (identity and
    inherited cells included), not merely the same count."""
    calls, _chunks = pdf2
    parents = _oversized_provenance_parents(cta, enc, calls)

    for parent in parents:
        # soft_min_tokens=0 isolates the split gate from the merge gate, which
        # ticket 32 already made loss-free.
        pieces = cta._enforce_size_gates(
            [copy.deepcopy(parent)], BASE_META, "probe", enc, soft_min_tokens=0,
        )
        assert len(pieces) > 1, "an oversized chunk must be cut into pieces"

        before = {e["row"]: e for e in _entries(parent["metadata"])}
        after: dict[int, dict] = {}
        for piece in pieces:
            for entry in _entries(piece["metadata"]):
                assert entry["row"] not in after, "an entry survived in two pieces"
                after[entry["row"]] = entry
        assert after == before, "the pieces do not carry the parent's entries intact"

    assert parents, "pdf_2.md no longer has an oversized provenance-carrying chunk"


def test_the_real_pieces_no_longer_duplicate_provenance(cta, enc, pdf2):
    """Non-vacuity on real data: the scoping really removes entries from pieces
    that do not own them.

    Before ticket 33 the gate copied a parent's whole provenance dict into every
    piece, so the pieces held ``entries x pieces`` records; scoping leaves
    exactly one copy of each. Both totals are read from the production gate on
    the document's real oversized chunks.
    """
    calls, _chunks = pdf2
    parents = _oversized_provenance_parents(cta, enc, calls)
    assert parents, "pdf_2.md no longer has an oversized provenance-carrying chunk"

    unfiltered = filtered = parent_total = 0
    for parent in parents:
        pieces = cta._enforce_size_gates(
            [copy.deepcopy(parent)], BASE_META, "probe", enc, soft_min_tokens=0,
        )
        parent_total += len(_entries(parent["metadata"]))
        unfiltered += len(_entries(parent["metadata"])) * len(pieces)
        filtered += sum(len(_entries(piece["metadata"])) for piece in pieces)

    assert filtered == parent_total, "every entry survives in exactly one piece"
    assert unfiltered > filtered, (
        f"scoping dropped nothing: {unfiltered} copies before, {filtered} after"
    )


# ──────────────────────────────────────────────────────────────────────────
# Unsplit chunks: metadata unchanged
# ──────────────────────────────────────────────────────────────────────────


def test_unsplit_chunks_keep_their_metadata_byte_for_byte(cta, enc, pdf2):
    """A chunk the gate does not cut must come out with the metadata it went in
    with — ticket 32's output is not to regress, and the row-index handoff the
    gate now reads must not survive into any emitted chunk."""
    calls, chunks = pdf2
    hard_max = cta.settings.HARD_MAX_TOKENS
    checked = 0

    for _table, pre_gate, post_gate in calls:
        emitted = {chunk["text"]: chunk for chunk in post_gate}
        for parent in pre_gate:
            if cta.count_tokens(parent["text"], enc) > hard_max:
                continue
            kept = emitted.get(parent["text"])
            if kept is None:  # folded into a neighbour by the soft floor
                continue
            assert kept["metadata"] == parent["metadata"], (
                f"{kept['id']} is not split but its metadata changed"
            )
            checked += 1

    assert checked, "no unsplit chunk was checked"
    for chunk in chunks:
        assert "_rowspan_row_indices" not in chunk, (
            f"{chunk['id']} leaked the gate's row-index handoff into the corpus"
        )


def test_an_unsplit_provenance_chunk_passes_through_untouched(cta, enc, pdf2):
    """The same guarantee at the gate itself, on a real provenance-carrying chunk:
    the chunk object is returned as-is, handoff popped."""
    calls, _chunks = pdf2
    hard_max = cta.settings.HARD_MAX_TOKENS

    candidates = [
        parent
        for _table, pre_gate, _post in calls
        for parent in pre_gate
        if _entries(parent["metadata"]) and cta.count_tokens(parent["text"], enc) <= hard_max
    ]
    assert candidates, "no unsplit provenance-carrying chunk on pdf_2.md"

    parent = copy.deepcopy(candidates[0])
    assert "_rowspan_row_indices" in parent, "the chunking must hand its rows to the gate"
    metadata_before = copy.deepcopy(parent["metadata"])

    out = cta._enforce_size_gates([parent], BASE_META, "doc", enc)

    assert len(out) == 1
    assert out[0]["metadata"] == metadata_before
    assert "_rowspan_row_indices" not in out[0]


# ──────────────────────────────────────────────────────────────────────────
# A split boundary that falls between an anchor and the row leaning on it
# ──────────────────────────────────────────────────────────────────────────

# Constructed boundary case, not a transcribed page: a merged identity cell
# ("Số TT / mã xét tuyển" printed once, the continuation line reaching the parser
# with those cells empty — ticket 32's shape) sized so the size gate cuts exactly
# between the anchor row and the row that leans on it. Measured on the real
# document: 6 of the 66 entries in split parents sit on such a boundary.
ANCHOR_BOUNDARY_TABLE = [
    "| TT | Mã xét tuyển | Tổ hợp môn | Điểm chuẩn |",
    "| --- | --- | --- | --- |",
    "| 1 | 7380101 | 1. Toán, Vật lí, Hóa học |  |",
    "|  |  |  | 25,00 |",
]


def test_split_between_anchor_and_leaning_row_keeps_the_entry(cta, enc):
    """The pieces of a split hold disjoint rows, so an entry can only live in the
    piece holding the row it labels. When the cut separates that row from its
    anchor, the entry stays with the row it labels: dropping it would leave a
    leaning row in the corpus with neither attribution nor warning, and the row
    it names is in the piece either way."""
    table = cta.MarkdownTable(ANCHOR_BOUNDARY_TABLE)
    parent = cta._chunk_table(table, BASE_META, "boundary", enc, 0)
    assert len(parent) == 1, "the boundary case must start as one chunk"
    parent = parent[0]
    attributed = {e["row"]: e for e in parent["metadata"]["rowspan_attribution"]}
    assert attributed[1]["anchor_row"] == 0, "row 1 must lean on row 0"

    header = ANCHOR_BOUNDARY_TABLE[0]
    separator = ANCHOR_BOUNDARY_TABLE[1]
    anchor_row = ANCHOR_BOUNDARY_TABLE[2]
    # Just enough room for the anchor row, not for the row leaning on it.
    hard_max = cta.count_tokens("\n".join([header, separator, anchor_row]), enc) + 2
    assert cta.count_tokens(parent["text"], enc) > hard_max

    pieces = cta._enforce_size_gates(
        [copy.deepcopy(parent)], BASE_META, "boundary", enc,
        hard_max_tokens=hard_max, soft_min_tokens=0,
    )

    assert len(pieces) == 2, "the cut must land between the anchor and the leaning row"
    anchor_piece, leaning_piece = pieces
    assert anchor_row in anchor_piece["text"]
    assert ANCHOR_BOUNDARY_TABLE[3] in leaning_piece["text"]
    assert not _entries(anchor_piece["metadata"]), (
        "the piece that does not hold row 1 must not name it"
    )
    kept = _entries(leaning_piece["metadata"])
    assert [e["row"] for e in kept] == [1], "the entry stays with the row it labels"
    assert kept[0] == attributed[1], "the entry survives intact"
    assert table.data_rows[kept[0]["row"]] in leaning_piece["text"]
