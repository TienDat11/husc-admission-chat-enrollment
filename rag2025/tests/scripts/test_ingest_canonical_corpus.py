"""TDD: rag2025/scripts/ingest_lancedb.py — canonical corpus ingestion (ticket 15).

The canonical track (tickets 02/03/05/06/28) reaches the served table here.
Two properties this path must hold, both asserted against a real LanceDB table:

* **Additive.** The served table holds 1679 legacy rows and zero canonical ones.
  A canonical release is 158 chunks, so a wholesale replace would take coverage
  1679 → 158 — a ~90% drop arriving as a side effect of a mechanism fix. The
  canonical sweep is scoped by ``metadata.ingest_file`` (canonical rows say
  ``chunks.jsonl``, legacy rows name their own file), so legacy rows are not
  merely spared by luck: they are outside the sweep's scope by construction.
  The row count is asserted here, not just "no errors".
* **Refused when stale.** A corpus whose stamp (ticket 06) is not the stamp the
  running code produces describes a chunker this process does not have.
  Ingesting it would put chunk output in the index that no current code path
  can reproduce, so the run aborts and the table is left untouched.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import lancedb
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
from src.services.chunk_identity import version_stamp  # noqa: E402
from src.services.reindex_job import (  # noqa: E402
    CHUNK_MANIFEST_FILENAME,
    CHUNKS_FILENAME,
    CorpusStamp,
    ReindexJob,
)

DIM = 4
RELEASE_ID = "R-2026-09-01"
TEXT_A = (
    "Điểm chuẩn ngành Công nghệ thông tin (mã ngành 7480201) năm 2026 là 25,5 điểm.\n"
    "\n"
    "| Ngành | Mã ngành | Điểm chuẩn |\n"
    "| --- | --- | --- |\n"
    "| Công nghệ thông tin | 7480201 | 25,5 |\n"
)
TEXT_B = "Học phí ngành Công nghệ thông tin năm 2026 là 15.000.000 đồng một năm học."

def _record(record_id: str, text: str) -> CanonicalRecord:
    return CanonicalRecord(
        record_id=record_id,
        record_type=RecordType.DIEM_CHUAN,
        version=1,
        admission_year=2026,
        effective_from=date(2026, 6, 1),
        effective_to=date(2026, 12, 31),
        status=RecordStatus.PUBLISHED,
        authority_level=AuthorityLevel.HUSC_OFFICIAL,
        source_document_id="de-an-tuyen-sinh-2026",
        source_url="https://tuyensinh.husc.edu.vn/de-an-2026.pdf",
        source_page=12,
        published_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        text=text,
    )

def _embed(texts):
    return [[float(len(t) % 97)] * DIM for t in texts]

@pytest.fixture()
def temp_db():
    d = tempfile.mkdtemp(dir="D:/tmp", prefix="test_canonical_ingest_")
    yield lancedb.connect(d)
    shutil.rmtree(d, ignore_errors=True)

@pytest.fixture()
def ingest_mod(temp_db, monkeypatch):
    """``ingest_lancedb`` with settings pointing at the temp DB."""
    import importlib

    mod = importlib.import_module("scripts.ingest_lancedb")

    class FakeSettings:
        LANCEDB_URI = temp_db.uri
        LANCEDB_TABLE = "husc_chunks"
        EMBEDDING_MODEL = "fake"
        EMBEDDING_DIM = DIM

    monkeypatch.setattr(mod, "RAGSettings", FakeSettings)
    return mod

@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """A real canonical corpus, built by the ticket-15 job."""
    corpus_dir = tmp_path / "canonical_corpus"
    ReindexJob(
        records=[_record("DIEM_CHUAN:cntt:2026", TEXT_A), _record("HOC_PHI:cntt:2026", TEXT_B)],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=_embed,
    ).run()
    return corpus_dir

def _table_names(db) -> List[str]:
    """``list_tables()`` returns a ``ListTablesResponse`` on lancedb 0.29."""
    listed = db.list_tables()
    return list(listed.tables if hasattr(listed, "tables") else listed)

def _rows(table, columns: List[str]) -> List[Dict[str, Any]]:
    return table.to_arrow().select(columns).to_pylist()

def _legacy_row(row_id: str, text: str, ingest_file: str) -> Dict[str, Any]:
    return {
        "id": row_id,
        "doc_id": row_id,
        "release_id": "legacy",
        "text": text,
        "summary": None,
        "vector": [0.25] * DIM,
        "data_year": 2026,
        "scope": "admissions",
        "doc_type": "chunk",
        "status": "current",
        "effective_from": 0,
        "effective_to": 99991231,
        "is_tombstone": False,
        "superseded_by": None,
        "metadata_json": json.dumps({"ingest_file": ingest_file, "chunk_id": row_id}),
        "is_reference": False,
        "crawl_date": None,
    }

def _seed_legacy(table, rows: List[Dict[str, Any]]) -> None:
    table.merge_insert("id").when_not_matched_insert_all().execute(rows)

def _corpus_chunk_count(corpus: Path) -> int:
    return len(
        [
            line
            for line in (corpus / CHUNKS_FILENAME).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )

# ── the initial ingestion ──────────────────────────────────────────────────

def test_first_canonical_ingest_creates_the_table_with_exactly_the_corpus_rows(
    ingest_mod, temp_db, corpus: Path
) -> None:
    """From an empty footprint: every corpus chunk, no more and no fewer."""
    assert "husc_chunks" not in _table_names(temp_db)

    assert ingest_mod.ingest_canonical_corpus(corpus) is True

    table = temp_db.open_table("husc_chunks")
    assert table.count_rows() == _corpus_chunk_count(corpus)
    assert {r["release_id"] for r in _rows(table, ["release_id"])} == {RELEASE_ID}

def test_a_canonical_row_records_one_release_in_both_views(
    ingest_mod, temp_db, corpus: Path
) -> None:
    """The two places a release id lives must not disagree on a row we wrote."""
    ingest_mod.ingest_canonical_corpus(corpus)

    for row in _rows(temp_db.open_table("husc_chunks"), ["release_id", "metadata_json"]):
        assert json.loads(row["metadata_json"])["release_id"] == row["release_id"] == RELEASE_ID

def test_the_served_table_carries_the_stamp_that_built_it(
    ingest_mod, temp_db, corpus: Path
) -> None:
    """So the next run can tell whether the table is still current — no memory needed."""
    ingest_mod.ingest_canonical_corpus(corpus)

    assert ingest_mod.served_corpus_stamp(temp_db.open_table("husc_chunks")) == CorpusStamp.current()
    assert CorpusStamp.from_manifest(corpus) == CorpusStamp.current()

# ── additive, not a cutover ────────────────────────────────────────────────

def test_legacy_rows_survive_and_coverage_is_additive(
    ingest_mod, temp_db, corpus: Path
) -> None:
    """THE coverage assertion: 1679 legacy rows must not become 158 chunks.

    Asserted as an exact resulting count, and as byte-equality of every legacy
    row — a mechanism fix must not drop ~90% of served coverage as a side
    effect.
    """
    legacy = [
        _legacy_row("legacy-a", "Nội dung legacy A", "chunked_24.jsonl"),
        _legacy_row("legacy-b", "Nội dung legacy B", "chunked_24.jsonl"),
        _legacy_row("legacy-c", "Nội dung legacy C", "chunked_60.jsonl"),
    ]
    db0 = temp_db
    table0 = db0.create_table("husc_chunks", data=legacy)
    before = _rows(table0, ["id", "text", "release_id", "vector", "metadata_json"])
    canonical_count = _corpus_chunk_count(corpus)
    assert table0.count_rows() == len(legacy)

    assert ingest_mod.ingest_canonical_corpus(corpus) is True

    table = temp_db.open_table("husc_chunks")
    assert table.count_rows() == len(legacy) + canonical_count
    after = {
        r["id"]: r
        for r in _rows(table, ["id", "text", "release_id", "vector", "metadata_json"])
        if r["id"] in {row["id"] for row in before}
    }
    assert after == {row["id"]: row for row in before}
    assert {r["release_id"] for r in _rows(table, ["release_id"])} == {"legacy", RELEASE_ID}

def test_a_legacy_row_carrying_the_canonical_filename_is_out_of_scope_by_name(
    ingest_mod, temp_db, corpus: Path
) -> None:
    """Scope is the recorded ``ingest_file``, not a guess about row shape.

    A legacy row that never recorded one is untouchable (581 of 2068 production
    rows are in exactly that position); a row that recorded the canonical
    filename is the canonical track's to sweep.
    """
    unnamed = _legacy_row("unnamed", "row from another ingest script", "other.jsonl")
    unnamed["metadata_json"] = json.dumps({"chunk_id": "unnamed"})
    owned = _legacy_row("owned", "row this track wrote and then dropped", "chunks.jsonl")
    table = temp_db.create_table("husc_chunks", data=[unnamed, owned])

    assert ingest_mod.ingest_canonical_corpus(corpus) is True

    ids = {r["id"] for r in _rows(temp_db.open_table("husc_chunks"), ["id"])}
    assert "unnamed" in ids
    assert "owned" not in ids

def test_canonical_rows_absent_from_the_corpus_are_swept(
    ingest_mod, temp_db, corpus: Path
) -> None:
    """A chunk that is no longer in the corpus must not keep being served."""
    ingest_mod.ingest_canonical_corpus(corpus)
    table = temp_db.open_table("husc_chunks")
    before_ids = {r["id"] for r in _rows(table, ["id"])}

    import shutil as _shutil

    slim = corpus.parent / "slim"
    _shutil.copytree(corpus, slim)
    lines = [
        line
        for line in (slim / CHUNKS_FILENAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dropped = json.loads(lines[-1])["id"]
    (slim / CHUNKS_FILENAME).write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    assert ingest_mod.ingest_canonical_corpus(slim) is True

    after_ids = {r["id"] for r in _rows(temp_db.open_table("husc_chunks"), ["id"])}
    assert dropped not in after_ids
    assert after_ids == before_ids - {dropped}

# ── idempotency ────────────────────────────────────────────────────────────

def test_rerunning_the_same_corpus_does_not_duplicate(
    ingest_mod, temp_db, corpus: Path
) -> None:
    ingest_mod.ingest_canonical_corpus(corpus)
    first = _rows(temp_db.open_table("husc_chunks"), ["id", "text", "vector"])

    assert ingest_mod.ingest_canonical_corpus(corpus) is True

    table = temp_db.open_table("husc_chunks")
    assert table.count_rows() == len(first)
    assert _rows(table, ["id", "text", "vector"]) == first

# ── refusal ────────────────────────────────────────────────────────────────

def test_a_stale_corpus_is_refused_and_the_table_is_untouched(
    ingest_mod, temp_db, corpus: Path
) -> None:
    """Ingesting a corpus from a chunker this code does not have is a silent
    divergence — refused, with the serving table left exactly as it was."""
    legacy = [_legacy_row("legacy-a", "Nội dung legacy A", "chunked_24.jsonl")]
    table0 = temp_db.create_table("husc_chunks", data=legacy)
    before = _rows(table0, ["id", "text", "metadata_json"])

    stale = corpus.parent / "stale"
    shutil.copytree(corpus, stale)
    manifest = json.loads((stale / CHUNK_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest["chunker_version"] = "0.9"
    (stale / CHUNK_MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")

    assert ingest_mod.ingest_canonical_corpus(stale) is False

    table = temp_db.open_table("husc_chunks")
    assert table.count_rows() == len(legacy)
    assert _rows(table, ["id", "text", "metadata_json"]) == before

def test_a_corpus_with_no_manifest_is_refused(ingest_mod, temp_db, corpus: Path) -> None:
    """No manifest means no stamp, and no stamp means stale — never "assume current"."""
    unstamped = corpus.parent / "unstamped"
    shutil.copytree(corpus, unstamped)
    (unstamped / CHUNK_MANIFEST_FILENAME).unlink()

    assert ingest_mod.ingest_canonical_corpus(unstamped) is False
    assert "husc_chunks" not in _table_names(temp_db)

def test_a_chunk_without_a_vector_is_refused(ingest_mod, temp_db, corpus: Path) -> None:
    """A canonical chunk that reaches the corpus must reach the index."""
    broken = corpus.parent / "broken"
    shutil.copytree(corpus, broken)
    vectors = [
        line
        for line in (broken / "vectors.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    (broken / "vectors.jsonl").write_text("\n".join(vectors[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no vector"):
        ingest_mod.canonical_corpus_rows(
            broken, embedding_dim=DIM, stamp=CorpusStamp.current()
        )

def test_the_manifest_stamp_is_ticket_06s(ingest_mod, corpus: Path) -> None:
    """Not a constant invented by the ingest layer."""
    manifest = json.loads((corpus / CHUNK_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    for key, value in version_stamp().items():
        assert manifest[key] == value

# ── the published-only boundary (the second gate) ──────────────────────────


def _restatus(corpus: Path, name: str, status: Any) -> Path:
    """A copy of the corpus with every chunk's ``metadata.status`` replaced.

    Writes what the chunker's own guard would never emit, which is the shape a
    corpus from older code, a hand-assembled one, or a file edited between the
    build and the ingest step actually has.
    """
    edited = corpus.parent / name
    shutil.copytree(corpus, edited)
    path = edited / CHUNKS_FILENAME
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for chunk in chunks:
        if status is None:
            chunk["metadata"].pop("status", None)
        else:
            chunk["metadata"]["status"] = status
    path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n", encoding="utf-8"
    )
    return edited


@pytest.mark.parametrize("status", ["draft", "review", "approved", "archived"])
def test_a_corpus_holding_an_unpublished_chunk_is_refused(
    ingest_mod, temp_db, corpus: Path, status: str
) -> None:
    """The boundary's own gate: the chunk says what it is, and the ingest reads it.

    No chunker call is involved here — ``canonical_corpus_rows`` reads the
    status off the chunk in front of it. That is what makes this an independent
    second gate rather than the same one reached twice: a corpus that arrives by
    any other route (older code, a hand-assembled directory, a file edited
    between the build and the ingest) is still refused.
    """
    poisoned = _restatus(corpus, f"unpublished-{status}", status)

    with pytest.raises(ValueError, match=f"is not published \\(status={status}\\)"):
        ingest_mod.canonical_corpus_rows(
            poisoned, embedding_dim=DIM, stamp=CorpusStamp.current()
        )


def test_a_chunk_with_no_status_at_all_is_refused(ingest_mod, temp_db, corpus: Path) -> None:
    """Fail closed: a chunk that does not say it is published is not published."""
    stripped = _restatus(corpus, "statusless", None)

    with pytest.raises(ValueError, match="is not published \\(status=<missing>\\)"):
        ingest_mod.canonical_corpus_rows(
            stripped, embedding_dim=DIM, stamp=CorpusStamp.current()
        )


def test_an_unpublished_corpus_never_reaches_the_served_table(
    ingest_mod, temp_db, corpus: Path
) -> None:
    """End to end: the refusal happens before a row is written, not after."""
    legacy = [_legacy_row("legacy-a", "Nội dung legacy A", "chunked_24.jsonl")]
    table0 = temp_db.create_table("husc_chunks", data=legacy)
    before = _rows(table0, ["id", "text", "metadata_json"])

    poisoned = _restatus(corpus, "unpublished-e2e", "draft")
    with pytest.raises(ValueError, match="is not published"):
        ingest_mod.ingest_canonical_corpus(poisoned)

    table = temp_db.open_table("husc_chunks")
    assert table.count_rows() == len(legacy)
    assert _rows(table, ["id", "text", "metadata_json"]) == before


def test_the_refusal_names_the_offending_chunk(ingest_mod, corpus: Path) -> None:
    """A loud failure names which chunk, so the corpus can be rebuilt, not guessed at."""
    poisoned = _restatus(corpus, "unpublished-named", "draft")
    first_id = json.loads(
        (poisoned / CHUNKS_FILENAME).read_text(encoding="utf-8").splitlines()[0]
    )["id"]

    with pytest.raises(ValueError) as caught:
        ingest_mod.canonical_corpus_rows(
            poisoned, embedding_dim=DIM, stamp=CorpusStamp.current()
        )

    assert first_id in str(caught.value)
    assert "build_chunk_index" in str(caught.value)


def test_the_published_corpus_passes_the_gate_unchanged(ingest_mod, corpus: Path) -> None:
    """The guard must not refuse a legitimate corpus — that would be the worse bug.

    The corpus fixture is built by the real ticket-15 job from published records,
    so this is the production shape, and its rows must come through with the
    status the chunker stamped.
    """
    rows = ingest_mod.canonical_corpus_rows(
        corpus, embedding_dim=DIM, stamp=CorpusStamp.current()
    )

    assert len(rows) == _corpus_chunk_count(corpus)
    for row in rows:
        assert json.loads(row["metadata_json"])["status"] == "published"

# ── the premise, against the served table ──────────────────────────────────

def test_the_served_table_holds_no_canonical_scoped_row_yet() -> None:
    """Why the first canonical release is additive rather than a cutover.

    The sweep removes canonical-track rows the corpus no longer contains. If the
    served table already held rows recorded under the canonical filename, the
    first release would sweep them — and take coverage with it. It holds none:
    every one of the 1679 rows names a legacy file, so the sweep's scope is
    empty on the first run by fact rather than by assumption. Skips when the
    live table is unavailable, like the suite's other live-table checks.
    """
    from config.settings import RAGSettings

    import importlib

    mod = importlib.import_module("scripts.ingest_lancedb")
    settings = RAGSettings()
    try:
        db = lancedb.connect(settings.LANCEDB_URI)
        listed = db.list_tables()
        names = list(listed.tables if hasattr(listed, "tables") else listed)
    except Exception as exc:  # noqa: BLE001 - a missing live table is not a failure
        pytest.skip(f"live LanceDB unavailable: {exc}")
    if "husc_chunks" not in names:
        pytest.skip("live husc_chunks table unavailable")

    table = db.open_table("husc_chunks")
    rows = _rows(table, ["id", "release_id", "metadata_json"])
    if not rows:
        pytest.skip("live husc_chunks is empty")

    canonical_scoped = [
        row["id"]
        for row in rows
        if json.loads(row["metadata_json"] or "{}").get("ingest_file")
        == mod.CANONICAL_INGEST_FILE
    ]
    assert canonical_scoped == [], (
        f"{len(canonical_scoped)} served row(s) already sit in the canonical "
        "sweep's scope; the first canonical release would delete them"
    )
    # And nothing served today carries the stamp, so the corpus is stale by
    # definition — the ticket's "đã xảy ra" case, on the real table.
    assert CorpusStamp.from_rows(rows) is None
