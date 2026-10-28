"""TDD: rag2025/src/services/chunk_identity.py + canonical_chunk_builder.py — ticket 06.

Acceptance (``issues/06-deterministic-chunk-identity.md``):
- ``chunk_id`` is a hash of the canonical record version plus the semantic unit
  and is stable across runs and processes;
- changing a record's content changes the chunk ids that content feeds, and
  leaves every other chunk id alone — the test that catches an over-broad hash;
- a major code, a subject combination and a timeline milestone are never split
  across chunks, with a negative control proving the check detects a split;
- table chunks keep their repeated header and no rows were merged to reach a
  token target;
- every chunk resolves back to its canonical record version and source page;
- the chunker version stamp is present, stable across runs, and is neither a
  timestamp nor a content hash.

The last acceptance item runs over ticket 05's real materialized records. That
artifact is produced by another ticket, so its test skips loudly when absent.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical.record import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.domain.canonical.semantic_units import SemanticUnit, UnitKind  # noqa: E402
from src.services.canonical_chunk_builder import (  # noqa: E402
    CHUNKS_FILENAME,
    build_chunk_index,
    chunk_spans_split_atoms,
    load_materialized_records,
    records_by_key,
    table_rows_not_whole,
)
from src.services.chunk_identity import (  # noqa: E402
    CHUNKER_VERSION,
    ChunkLineage,
    chunk_id_for,
    chunk_id_hash_input,
    chunker_version,
    resolve_record,
    validate_version_stamp,
)

RELEASE_ID = "R-2026-09-01"

# The real text of a seeded TO_HOP record (ticket 28's store, materialized by
# ticket 05): one major code and five combinations inside one sentence.
TO_HOP_TEXT = (
    "Tổ hợp xét tuyển năm 2026 ngành Quản lý văn hóa (mã ngành 7229042): "
    "C00, C03, C19, X70, D14."
)
TO_HOP_VALUES = {
    "chi_tieu": 30,
    "ma_nganh": "7229042",
    "ten_nganh": "Quản lý văn hóa",
    "to_hop_codes": ["C00", "C03", "C19", "X70", "D14"],
}

# A real small_table from data/chunked/chunked_24.jsonl (ticket 32/33's output),
# used verbatim so the header/row assertions are about the chunker the repo
# actually runs.
REAL_TABLE = "\n".join(
    [
        "| TT  Mã ngành  | Ngành đào tạo                  |     |     |     | Khối ngành  |",
        "| ------------- | ------------------------------ | --- | --- | --- | ----------- |",
        "| 20  7140221   | Sư phạm Âm nhạc                |     |     |     | I           |",
        "| 21  7140246   | Sư phạm Công nghệ              |     |     |     | I           |",
        "| 22  7140247   | Sư phạm Khoa học tự nhiên      |     |     |     | I           |",
        "| 23  7140248   | Giáo dục Kinh tế và Pháp luật  |     |     |     | I           |",
        "| 24  7140249   | Sư phạm Lịch sử - Địa lý       |     |     |     | I           |",
        "| 25  7480104   | Hệ thống thông tin             |     |     |     | V           |",
    ]
)

# Ticket 05's real materialization, produced from ticket 28's seed store through
# the real lifecycle (edit → submit → approve → publish).
REAL_MATERIALIZATION = (
    ROOT / "data" / "canonical_store" / "t05_real_run" / "out" / "materialized_records.jsonl"
)


def _record(**overrides) -> CanonicalRecord:
    fields = {
        "record_id": "TO_HOP:NGANH:7229042:2026",
        "record_type": RecordType.TO_HOP,
        "version": 1,
        "admission_year": 2026,
        "effective_from": "2026-01-01",
        "status": RecordStatus.PUBLISHED,
        "authority_level": AuthorityLevel.HUSC_CANONICAL,
        "source_document_id": "THONG_TIN_TUYEN_SINH_202611.pdf",
        "source_page": 25,
        "text": TO_HOP_TEXT,
    }
    fields.update(overrides)
    return CanonicalRecord(**fields)


def _unit(value: str, field: str = "text", slot: str = "0", kind: UnitKind = UnitKind.TEXT) -> SemanticUnit:
    return SemanticUnit(
        record_id="r",
        record_type="TO_HOP",
        field=field,
        slot=slot,
        value=value,
        kind=kind,
    )


# ── the hash construction ──────────────────────────────────────────────────


def test_chunk_id_is_a_sha256_of_the_version_key_and_the_unit():
    import hashlib

    unit = _unit("tổ hợp xét tuyển: A00", field="to_hop_codes", slot="A00", kind=UnitKind.ATOM)
    key = "TO_HOP:NGANH:7229042:2026@1"
    chunk_id = chunk_id_for(key, unit)
    assert len(chunk_id) == 64
    assert chunk_id == hashlib.sha256(chunk_id_hash_input(key, unit)).hexdigest()
    # The version key and the unit's structural key are both in the preimage.
    assert key.encode("utf-8") in chunk_id_hash_input(key, unit)
    assert unit.key.encode("utf-8") in chunk_id_hash_input(key, unit)


def test_chunk_id_changes_with_the_record_version_key():
    unit = _unit("mã ngành: 7229042")
    assert chunk_id_for("R@1", unit) != chunk_id_for("R@2", unit)


def test_chunk_id_does_not_depend_on_the_record_around_it():
    """The hash covers the version key and one unit — nothing else."""
    record = _record(values=TO_HOP_VALUES)
    other = _record(
        record_id="TO_HOP:NGANH:7229042:2026",
        version=1,
        text="Hoàn toàn khác.",
        values={"ma_nganh": "7229042"},
    )
    unit = _unit("mã ngành: 7229042", field="ma_nganh", kind=UnitKind.ATOM)
    assert chunk_id_for(record.record_version_key, unit) == chunk_id_for(
        other.record_version_key, unit
    )


def test_chunk_id_is_whitespace_insensitive_but_case_sensitive():
    assert chunk_id_for("R@1", _unit("A00  và\nC03")) == chunk_id_for("R@1", _unit("A00 và C03"))
    assert chunk_id_for("R@1", _unit("A00")) != chunk_id_for("R@1", _unit("a00"))


def test_chunk_ids_are_identical_in_a_fresh_process(tmp_path):
    """Cross-process determinism, with a different PYTHONHASHSEED in the child."""
    record = _record(values=TO_HOP_VALUES)
    here = build_chunk_index([record], tmp_path / "here", RELEASE_ID)
    expected = sorted(here.chunk_ids)

    payload = tmp_path / "records.jsonl"
    from src.domain.canonical.record import to_canonical_json

    payload.write_bytes(to_canonical_json(record) + b"\n")
    script = (
        "import sys, json;"
        f"sys.path.insert(0, {str(ROOT)!r});"
        f"sys.path.insert(0, {str(ROOT / 'src')!r});"
        "from src.services.canonical_chunk_builder import build_chunk_index;"
        f"r = build_chunk_index({str(payload)!r}, {str(tmp_path / 'there')!r}, {RELEASE_ID!r});"
        "print(json.dumps(sorted(r.chunk_ids)))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=True,
    )
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == expected


# ── idempotency and scoped change ──────────────────────────────────────────


def test_two_runs_over_the_same_records_produce_the_same_chunk_id_set(tmp_path):
    records = [_record(), _record(record_id="CHI_TIEU:NGANH:7220104:2026", record_type=RecordType.CHI_TIEU, text="Chỉ tiêu ngành Hán Nôm: 32.")]
    first = build_chunk_index(records, tmp_path / "a", RELEASE_ID)
    second = build_chunk_index(records, tmp_path / "b", RELEASE_ID)

    assert first.chunk_ids == second.chunk_ids  # set equality, not counts
    assert len(first.chunk_ids) == len(first.chunks), "ids must be unique within a corpus"
    assert (tmp_path / "a" / CHUNKS_FILENAME).read_bytes() == (
        tmp_path / "b" / CHUNKS_FILENAME
    ).read_bytes()


def test_record_order_does_not_reach_the_output(tmp_path):
    first = _record()
    second = _record(record_id="CHI_TIEU:NGANH:7220104:2026", record_type=RecordType.CHI_TIEU, text="Chỉ tiêu: 32.")
    forward = build_chunk_index([first, second], tmp_path / "a", RELEASE_ID)
    backward = build_chunk_index([second, first], tmp_path / "b", RELEASE_ID)
    assert (tmp_path / "a" / CHUNKS_FILENAME).read_bytes() == (
        tmp_path / "b" / CHUNKS_FILENAME
    ).read_bytes()
    assert forward.chunk_ids == backward.chunk_ids


def test_editing_one_field_changes_only_that_field_s_chunk(tmp_path):
    """The over-broad-hash catcher: same version key, one value edited."""
    record = _record(values=TO_HOP_VALUES)
    before = build_chunk_index([record], tmp_path / "a", RELEASE_ID)

    edited_values = {**TO_HOP_VALUES, "to_hop_codes": ["C00", "C03", "C20", "X70", "D14"]}
    edited_text = TO_HOP_TEXT.replace("C19", "C20")
    edited = _record(values=edited_values, text=edited_text)
    after = build_chunk_index([edited], tmp_path / "b", RELEASE_ID)

    changed = before.chunk_ids ^ after.chunk_ids
    by_unit = {
        c["metadata"]["unit_key"]: c["id"] for c in before.chunks
    }
    changed_units = {
        key for key, chunk_id in by_unit.items() if chunk_id in changed
    }
    # Exactly the edited combination's unit and the sentence that repeats it.
    assert changed_units == {"TO_HOP|to_hop_codes|C19", "TO_HOP|text|0"}
    assert len(before.chunk_ids) == len(after.chunk_ids)


def test_a_content_change_in_one_record_leaves_every_other_record_stable(tmp_path):
    stable = _record(
        record_id="CHI_TIEU:NGANH:7220104:2026",
        record_type=RecordType.CHI_TIEU,
        text="Chỉ tiêu tuyển sinh năm 2026 ngành Hán Nôm (mã ngành 7220104): 32.",
        values={"ma_nganh": "7220104", "chi_tieu": 32, "ten_nganh": "Hán Nôm"},
    )
    before_edit = _record(values=TO_HOP_VALUES)
    before = build_chunk_index([before_edit, stable], tmp_path / "a", RELEASE_ID)

    # A new version of the first record with different content — the shape a real
    # admin edit takes (history is append-only, so the version increments).
    after_edit = _record(version=2, text=TO_HOP_TEXT + " Bổ sung: X70.")
    after = build_chunk_index([after_edit, stable], tmp_path / "b", RELEASE_ID)

    def ids_of(result, record_id):
        return {c["id"] for c in result.chunks if c["metadata"]["record_id"] == record_id}

    assert ids_of(before, "TO_HOP:NGANH:7229042:2026").isdisjoint(
        ids_of(after, "TO_HOP:NGANH:7229042:2026")
    ), "a new record version must retire the old chunk ids"
    assert ids_of(before, "CHI_TIEU:NGANH:7220104:2026") == ids_of(
        after, "CHI_TIEU:NGANH:7220104:2026"
    ), "an unrelated record's chunk ids must be untouched"


# ── atomicity, at chunk level ──────────────────────────────────────────────


def test_no_major_code_combination_or_milestone_is_split_across_chunks(tmp_path):
    body = " ".join(
        [
            "Thí sinh nộp hồ sơ trước 01/07/2026.",
            "Tổ hợp xét tuyển gồm A00 và C03.",
            "Mã ngành 7229042 áp dụng cho ngành Quản lý văn hóa.",
        ]
        * 20
    )
    record = _record(text=body, values=None)
    result = build_chunk_index([record], tmp_path, RELEASE_ID)

    assert len(result.chunks) > 1, "the budget must force a split for this to test anything"
    assert chunk_spans_split_atoms(record, result.chunks) == ()
    for atom in ("A00", "C03", "7229042", "01/07/2026"):
        assert any(atom in chunk["text"] for chunk in result.chunks)


def test_the_atom_audit_detects_a_split():
    """Negative control: a span that stops mid-atom must be reported."""
    record = _record(text="Mã ngành 7229042 và tổ hợp A00.")
    cut_chunk = [
        {
            "text": "Mã ngành 72290",
            "metadata": {"unit_source_span": [0, 14]},
        }
    ]
    cut = chunk_spans_split_atoms(record, cut_chunk)
    assert "7229042" in cut
    assert "A00" in cut


# ── tables ─────────────────────────────────────────────────────────────────


def test_table_chunks_keep_the_repeated_header(tmp_path):
    record = _record(text="Danh mục ngành đào tạo năm 2026.\n\n" + REAL_TABLE)
    result = build_chunk_index([record], tmp_path, RELEASE_ID)
    tables = [c for c in result.chunks if c["metadata"]["chunk_type"] == "table"]

    assert tables, "the table must reach the chunker, not the prose splitter"
    for chunk in tables:
        lines = chunk["text"].split("\n")
        assert "Mã ngành" in lines[0] and "Khối ngành" in lines[0]
        assert any(set(line.strip()) <= set("|-: ") for line in lines[1:2])
    assert all(c["metadata"]["unit_field"].startswith("table:") for c in tables)
    assert not [c for c in result.chunks if c["metadata"]["unit_field"] == "text"
                and "|" in c["text"]], "table rows must not also be indexed as prose"


def test_no_row_was_merged_away_or_reordered_for_a_token_target(tmp_path):
    record = _record(text="Danh mục ngành đào tạo năm 2026.\n\n" + REAL_TABLE)
    result = build_chunk_index([record], tmp_path, RELEASE_ID)
    tables = [c for c in result.chunks if c["metadata"]["chunk_type"] == "table"]

    assert table_rows_not_whole(record, result.chunks) == ()

    source_rows = [
        tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        for line in REAL_TABLE.split("\n")[2:]
    ]
    seen: list[int] = []
    for chunk in tables:
        for line in chunk["text"].split("\n")[1:]:
            row = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
            if row in source_rows:
                seen.append(source_rows.index(row))
    assert sorted(seen) == list(range(len(source_rows))), (
        "every source row appears exactly once, in source order — nothing merged "
        "in, nothing dropped"
    )
    assert seen == sorted(seen)


def test_two_tables_are_never_merged_into_one_chunk(tmp_path):
    """The row-merge the ticket forbids: rows glued across different headers.

    Two tables with different headers, each small enough that a token-budget
    merge would be tempting. Every emitted table chunk must carry exactly one
    table's header and only that table's rows.
    """
    second_table = "\n".join(
        [
            "| Mã tổ hợp | Môn xét tuyển        |",
            "| --------- | -------------------- |",
            "| A00       | Toán, Lý, Hóa        |",
            "| C03       | Văn, Toán, Sử        |",
        ]
    )
    record = _record(text=f"{REAL_TABLE}\n\n{second_table}")
    result = build_chunk_index([record], tmp_path, RELEASE_ID)
    tables = [c for c in result.chunks if c["metadata"]["chunk_type"] == "table"]

    assert len({c["metadata"]["unit_field"] for c in tables}) == 2, (
        "each table must be its own unit group"
    )
    first_rows = {tuple(c.strip() for c in line.strip().strip("|").split("|"))
                  for line in REAL_TABLE.split("\n")[2:]}
    second_rows = {tuple(c.strip() for c in line.strip().strip("|").split("|"))
                   for line in second_table.split("\n")[2:]}
    for chunk in tables:
        lines = [line for line in chunk["text"].split("\n") if line.strip()]
        data_lines = [
            line for line in lines[1:] if not set(line.strip()) <= set("|-: ")
        ]
        rows = {tuple(c.strip() for c in line.strip().strip("|").split("|"))
                for line in data_lines}
        assert rows, f"a table chunk with no rows: {chunk['text']!r}"
        # Every row of a chunk comes from ONE of the two tables — never a mix.
        assert rows <= first_rows or rows <= second_rows, (
            f"rows from two different tables were merged into one chunk: {rows}"
        )
    assert table_rows_not_whole(record, result.chunks) == ()


def test_a_row_cut_across_chunks_is_detected():
    """Negative control for the table audit."""
    record = _record(text=REAL_TABLE)
    fragment = [
        {"text": "| 20  7140221   | Sư phạm Âm nhạc |\n", "metadata": {}},
        {"text": "| 21  7140246   | Sư phạm Công nghệ |\n", "metadata": {}},
    ]
    missing = table_rows_not_whole(record, fragment)
    assert len(missing) >= 4
    assert ("22  7140247", "Sư phạm Khoa học tự nhiên", "", "", "", "I") in missing


# ── lineage ────────────────────────────────────────────────────────────────


def test_every_chunk_resolves_to_its_record_version_and_page(tmp_path):
    record = _record(values=TO_HOP_VALUES)
    result = build_chunk_index([record], tmp_path, RELEASE_ID)

    assert result.chunks
    for chunk in result.chunks:
        lineage = ChunkLineage.from_dict(chunk["lineage"])
        assert lineage.record_version_key == "TO_HOP:NGANH:7229042:2026@1"
        assert lineage.source_page == 25
        assert lineage.chunker_version == CHUNKER_VERSION
        assert lineage.chunk_id == chunk["id"]
        resolved = resolve_record(lineage, records_by_key(result.records))
        assert resolved.record_id == "TO_HOP:NGANH:7229042:2026"
        assert resolved.source_page == 25


def test_lineage_survives_a_jsonl_round_trip(tmp_path):
    result = build_chunk_index([_record(values=TO_HOP_VALUES)], tmp_path, RELEASE_ID)
    written = [
        json.loads(line)
        for line in (tmp_path / CHUNKS_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert [c["id"] for c in written] == [c["id"] for c in result.chunks]
    assert [ChunkLineage.from_dict(c["lineage"]) for c in written] == list(result.lineage)


def test_a_chunk_from_a_corpus_record_has_no_fabricated_page(tmp_path):
    """A JSONL-sourced record has no page; the lineage must say so, not invent one."""
    record = _record(
        source_document_id="rag2025/data/chunked/husc_2026_timeline_hocba_v1.jsonl#qa_1",
        source_page=None,
    )
    result = build_chunk_index([record], tmp_path, RELEASE_ID)
    assert all(c["lineage"]["source_page"] is None for c in result.chunks)


def test_an_unresolvable_lineage_raises(tmp_path):
    result = build_chunk_index([_record()], tmp_path, RELEASE_ID)
    lineage = result.lineage[0]
    with pytest.raises(KeyError, match="not among the records"):
        resolve_record(lineage, {})


# ── the version stamp ──────────────────────────────────────────────────────


def test_the_stamp_is_present_on_every_chunk_and_in_the_manifest(tmp_path):
    result = build_chunk_index([_record()], tmp_path, RELEASE_ID)
    assert result.manifest["chunker_version"] == CHUNKER_VERSION
    assert result.manifest["release_id"] == RELEASE_ID
    assert all(c["metadata"]["chunker_version"] == CHUNKER_VERSION for c in result.chunks)
    assert all(c["lineage"]["chunker_version"] == CHUNKER_VERSION for c in result.chunks)
    assert chunker_version() == CHUNKER_VERSION


def test_the_stamp_is_stable_across_runs_and_content_changes(tmp_path):
    """A content change must NOT move the stamp; the logic did not change."""
    first = build_chunk_index([_record()], tmp_path / "a", RELEASE_ID)
    same_code = build_chunk_index([_record()], tmp_path / "b", RELEASE_ID)
    other_content = build_chunk_index(
        [_record(version=7, text="Nội dung hoàn toàn khác.", values={"chi_tieu": 99})],
        tmp_path / "c",
        RELEASE_ID,
    )
    assert first.manifest["chunker_version"] == same_code.manifest["chunker_version"]
    assert first.manifest["chunker_version"] == other_content.manifest["chunker_version"]
    assert first.manifest["chunker_version"] == chunker_version()


def test_the_stamp_cannot_be_a_timestamp_or_a_content_digest():
    """Structural guard, not a comment: the two wrong shapes are rejected."""
    assert validate_version_stamp("1.0") == "1.0"
    for wrong in (
        "2026-09-13",
        "20260913T101500",
        "1757721600",
        "9f2c1b7d4e5a6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8",
        "chunker-v1",
    ):
        with pytest.raises(ValueError, match="not a semantic version"):
            validate_version_stamp(wrong)
    # And the constant actually in use passes that guard.
    assert validate_version_stamp(CHUNKER_VERSION) == CHUNKER_VERSION


# ── ingest compatibility ───────────────────────────────────────────────────


def test_chunks_are_in_the_shape_the_ingest_layer_reads(tmp_path):
    result = build_chunk_index([_record(values=TO_HOP_VALUES)], tmp_path, RELEASE_ID)
    for chunk in result.chunks:
        assert chunk["id"] and chunk["text"]
        assert chunk["metadata"]["ingest_file"] == CHUNKS_FILENAME
        assert chunk["metadata"]["data_year"] == 2026
        assert chunk["metadata"]["record_id"]
        assert isinstance(chunk["breadcrumbs"], list)
        assert isinstance(chunk["faq_type"], str)


def test_a_blank_release_id_is_refused(tmp_path):
    with pytest.raises(ValueError, match="release_id"):
        build_chunk_index([_record()], tmp_path, "  ")


def test_a_missing_materialization_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="ticket 05"):
        build_chunk_index(tmp_path / "nope", tmp_path, RELEASE_ID)


# ── the published-only boundary (the second gate) ──────────────────────────


@pytest.mark.parametrize(
    "status",
    [
        RecordStatus.DRAFT,
        RecordStatus.REVIEW,
        RecordStatus.APPROVED,
        RecordStatus.ARCHIVED,
    ],
)
def test_a_record_the_materializer_would_not_serve_is_refused_by_the_chunker(
    tmp_path, status
):
    """The second gate, exercised on its own.

    No ``PublishedMaterializer`` is anywhere in this path — that is the point of
    the test. The materializer is the gate that decides what a release
    *contains*; this asserts the chunker independently refuses the same record,
    so a caller that reaches ``build_chunk_index`` directly (ticket 15's job
    takes records as an argument; ticket 26 wires it into production) cannot put
    an unpublished version in the index even though nothing else stopped it.
    """
    with pytest.raises(ValueError, match=f"not published: .*{status.value}"):
        build_chunk_index([_record(status=status)], tmp_path, RELEASE_ID)

    # Refused *before* any artifact exists, not built and then rejected.
    assert list(tmp_path.iterdir()) == []


def test_one_unpublished_record_refuses_the_whole_batch(tmp_path):
    """All-or-nothing: a published record must not carry a draft in beside it."""
    published = _record()
    draft = _record(
        record_id="CHI_TIEU:NGANH:7220104:2026",
        record_type=RecordType.CHI_TIEU,
        status=RecordStatus.DRAFT,
        text="Chỉ tiêu ngành Hán Nôm năm 2026: 32.",
    )

    with pytest.raises(ValueError, match="refuses 1 record version"):
        build_chunk_index([published, draft], tmp_path, RELEASE_ID)

    assert list(tmp_path.iterdir()) == []


def test_the_refusal_names_the_offending_record_and_status(tmp_path):
    """A loud failure is one a caller can act on: which record, which status, what to do."""
    with pytest.raises(ValueError) as caught:
        build_chunk_index([_record(status=RecordStatus.REVIEW)], tmp_path, RELEASE_ID)

    message = str(caught.value)
    assert "TO_HOP:NGANH:7229042:2026@1" in message
    assert "(review)" in message
    assert "PublishedMaterializer" in message


def test_a_file_of_draft_records_is_refused_like_a_sequence(tmp_path):
    """The path argument is not a way around the gate: same check, same refusal."""
    from src.domain.canonical.record import to_canonical_json

    source = tmp_path / "drafts"
    source.mkdir()
    (source / "materialized_records.jsonl").write_bytes(
        b"".join(
            to_canonical_json(_record(status=RecordStatus.DRAFT, version=version)) + b"\n"
            for version in (1, 2)
        )
    )

    with pytest.raises(ValueError, match="refuses 2 record version"):
        build_chunk_index(source, tmp_path / "out", RELEASE_ID)

    assert not (tmp_path / "out").exists()


def test_a_status_that_only_spells_published_is_not_published(tmp_path):
    """The gate is enum identity, and it must still refuse loudly, not crash.

    ``RecordStatus`` is a ``str`` enum, so ``"published" == RecordStatus.PUBLISHED``
    is true — a gate written as a text comparison would admit a record whose
    status is a bare string. This one compares identity, so such a record is
    refused. ``model_copy`` is how the repo builds a variant record without
    re-validating (``src/domain/canonical/lifecycle.py`` does exactly that), so
    the object below is one a caller can really hold. Matching on ``ValueError``
    also pins the failure mode: the refusal must come from the gate with the
    offending record named, not from an ``AttributeError`` raised three frames
    deeper while formatting a status that has no ``.value``.
    """
    forged = _record().model_copy(update={"status": "published"})

    with pytest.raises(ValueError, match="not published: .*published"):
        build_chunk_index([forged], tmp_path, RELEASE_ID)


def test_the_published_path_emits_the_status_the_ingest_gate_reads(tmp_path):
    """The two gates must agree on the artifact, or a release cannot be served.

    The ingest boundary refuses any chunk whose ``metadata.status`` is not
    ``"published"``. If this side stopped stamping that key — a rename, a
    refactor that drops it — every legitimate corpus would be refused at ingest:
    the guard's own failure mode. Pinned here as the contract between the two.
    """
    result = build_chunk_index([_record(values=TO_HOP_VALUES)], tmp_path, RELEASE_ID)

    assert result.chunks
    assert {c["metadata"]["status"] for c in result.chunks} == {"published"}


# ── real data ──────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not REAL_MATERIALIZATION.is_file(),
    reason=(
        "ticket 05's real materialization is absent "
        f"({REAL_MATERIALIZATION}) — run PublishedMaterializer first"
    ),
)
def test_real_materialized_records_produce_a_deterministic_corpus(tmp_path):
    """Ticket 05's own output, end to end: idempotent, scoped, fully traced."""
    records = load_materialized_records(REAL_MATERIALIZATION)
    assert records, "the materialization must carry records"

    first = build_chunk_index(records, tmp_path / "a", "R-t06-real")
    second = build_chunk_index(records, tmp_path / "b", "R-t06-real")

    assert first.chunk_ids == second.chunk_ids
    assert (tmp_path / "a" / CHUNKS_FILENAME).read_bytes() == (
        tmp_path / "b" / CHUNKS_FILENAME
    ).read_bytes()
    assert len(first.chunk_ids) == len(first.chunks)
    assert first.manifest["record_count"] == len(records)

    # Every chunk is traceable, and the pages that exist are the real ones.
    by_key = records_by_key(records)
    pages = 0
    for chunk in first.chunks:
        lineage = ChunkLineage.from_dict(chunk["lineage"])
        resolved = resolve_record(lineage, by_key)
        assert resolved.record_type.value == lineage.record_type
        if lineage.source_page is not None:
            pages += 1
            assert resolved.source_page == lineage.source_page
    assert pages == sum(1 for c in first.chunks if c["metadata"]["source_page"] is not None)

    # Atomicity holds on the real corpus, per record.
    for record in records:
        assert chunk_spans_split_atoms(record, first.chunks_of(record.record_id)) == ()
