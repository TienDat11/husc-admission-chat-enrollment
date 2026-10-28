"""TDD: rag2025/src/services/reindex_job.py — ticket 15.

Acceptance (``issues/15-incremental-reindex-outbox.md``):
- only changed chunks are re-embedded, and the count equals the change count —
  asserted as an exact number, never as "less than before";
- related graph edges are updated and unrelated edges are untouched, compared
  byte for byte;
- the job states are enforced, and ``saved`` is never ``active``;
- a failed job leaves the serving corpus byte-identical;
- re-running the same version produces no duplicates;
- **a stamp bump alone — with an EMPTY affected-id set — forces a full
  re-index.** This is the gap the ticket documents: a parser/chunker change
  alters chunk output for records whose canonical content did not change, so the
  record-level trigger never fires. The test simulates exactly that (a chunker
  whose metadata differs while the text does not) and asserts the served corpus
  picks the change up with no affected ids at all.

Everything is asserted through the job's public surface and the bytes it wrote.
"""
from __future__ import annotations

import copy
import hashlib
import json
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
from src.services import reindex_job as reindex_job_module  # noqa: E402
from src.services.canonical_chunk_builder import build_chunk_index  # noqa: E402
from src.services.chunk_identity import version_stamp  # noqa: E402
from src.services.reindex_job import (  # noqa: E402
    CHUNK_MANIFEST_FILENAME,
    CHUNKS_FILENAME,
    GRAPH_EDGES_FILENAME,
    VECTORS_FILENAME,
    CorpusStamp,
    JobState,
    JobStateError,
    ReindexError,
    ReindexJob,
    ReindexMode,
    plan_reindex,
    stamp_is_stale,
)
from src.services.reindex_outbox import ReindexOutbox  # noqa: E402

RELEASE_ID = "R-2026-09-01"
RECORD_A = "DIEM_CHUAN:NGANH:cntt:2026"
RECORD_B = "HOC_PHI:NGANH:cntt:2026"

TEXT_A = (
    "Điểm chuẩn ngành Công nghệ thông tin (mã ngành 7480201) năm 2026 là 25,5 "
    "điểm, tổ hợp xét tuyển A00.\n"
    "\n"
    "| Ngành | Mã ngành | Điểm chuẩn |\n"
    "| --- | --- | --- |\n"
    "| Công nghệ thông tin | 7480201 | 25,5 |\n"
    "| Khoa học máy tính | 7480101 | 24,0 |\n"
)
TEXT_A_V2 = TEXT_A.replace("25,5", "26,0").replace("26,0 điểm", "26,0 điểm (đính chính)")
TEXT_B = "Học phí ngành Công nghệ thông tin năm 2026 là 15.000.000 đồng một năm học."

def _record(record_id: str, text: str, version: int = 1, **overrides) -> CanonicalRecord:
    fields = {
        "record_id": record_id,
        "record_type": RecordType.DIEM_CHUAN,
        "version": version,
        "admission_year": 2026,
        "effective_from": date(2026, 6, 1),
        "effective_to": date(2026, 12, 31),
        "status": RecordStatus.PUBLISHED,
        "authority_level": AuthorityLevel.HUSC_OFFICIAL,
        "source_document_id": "de-an-tuyen-sinh-2026",
        "source_url": "https://tuyensinh.husc.edu.vn/de-an-2026.pdf",
        "source_page": 12,
        "published_at": datetime(2026, 9, 1, tzinfo=timezone.utc),
        "text": text,
    }
    fields.update(overrides)
    return CanonicalRecord(**fields)

class CountingEmbedder:
    """Records every batch it was asked to embed, so the count is exact."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.batches: list[list[str]] = []

    @property
    def calls(self) -> int:
        return len(self.batches)

    @property
    def embedded_texts(self) -> list[str]:
        return [text for batch in self.batches for text in batch]

    def __call__(self, texts):
        texts = list(texts)
        self.batches.append(texts)
        return [[float(len(text) % 97), *([0.5] * (self.dim - 1))] for text in texts]

@pytest.fixture
def embedder() -> CountingEmbedder:
    return CountingEmbedder()

@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    return tmp_path / "corpus"

def _run(
    corpus_dir: Path,
    embedder: CountingEmbedder,
    records: list[CanonicalRecord],
    **kwargs,
):
    job = ReindexJob(
        records=records,
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
        **kwargs,
    )
    return job.run()

def _snapshot(corpus_dir: Path) -> dict[str, str]:
    """SHA-256 of every file in the corpus — the byte-level serving identity."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(corpus_dir.iterdir())
        if path.is_file()
    }

def _edge_lines(corpus_dir: Path) -> list[str]:
    return [
        line
        for line in (corpus_dir / GRAPH_EDGES_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

def _lines_for_records(corpus_dir: Path, record_ids: set[str]) -> list[str]:
    """The raw edge lines contributed by ``record_ids`` — compared as bytes."""
    return sorted(
        line
        for line in _edge_lines(corpus_dir)
        if set(json.loads(line)["record_ids"]) & record_ids
    )

def _lines_not_for_records(corpus_dir: Path, record_ids: set[str]) -> list[str]:
    return sorted(
        line
        for line in _edge_lines(corpus_dir)
        if not set(json.loads(line)["record_ids"]) & record_ids
    )

def _chunks(corpus_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (corpus_dir / CHUNKS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

def _vectors(corpus_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (corpus_dir / VECTORS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

def _expected_chunk_ids(records: list[CanonicalRecord], tmp_path: Path) -> set[str]:
    """What ticket 06 produces for ``records`` — the change count, derived not guessed."""
    result = build_chunk_index(records, tmp_path / "expected", RELEASE_ID)
    return set(result.chunk_ids)

# ── the first ingestion, from an empty footprint ───────────────────────────

def test_first_run_on_an_empty_corpus_is_full(
    corpus_dir: Path, embedder: CountingEmbedder, tmp_path: Path
) -> None:
    """No corpus yet: every chunk is new, and the run must not be a no-op."""
    records = [_record(RECORD_A, TEXT_A), _record(RECORD_B, TEXT_B)]
    job = ReindexJob(
        records=records,
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
    )

    result = job.run()

    assert result.plan.mode is ReindexMode.FULL
    assert result.plan.stale is True
    assert result.plan.reason == "corpus-unstamped"
    assert set(result.written_chunk_ids) == _expected_chunk_ids(records, tmp_path)
    assert result.embedded_count == len(result.written_chunk_ids)
    assert job.state is JobState.ACTIVE
    assert job.serving is True

def test_first_run_publishes_a_stamped_corpus(
    corpus_dir: Path, embedder: CountingEmbedder
) -> None:
    _run(corpus_dir, embedder, [_record(RECORD_A, TEXT_A)])

    manifest = json.loads((corpus_dir / CHUNK_MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert CorpusStamp.from_manifest(corpus_dir) == CorpusStamp.current()
    assert manifest["chunker_version"] == version_stamp()["chunker_version"]
    assert manifest["unit_rules_version"] == version_stamp()["unit_rules_version"]
    assert manifest["release_id"] == RELEASE_ID
    assert (corpus_dir / VECTORS_FILENAME).is_file()
    assert (corpus_dir / GRAPH_EDGES_FILENAME).is_file()

def test_a_corpus_without_a_stamp_is_stale_not_unknown(
    corpus_dir: Path, embedder: CountingEmbedder
) -> None:
    """The served table today has no stamp at all — that must mean rebuild."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / CHUNKS_FILENAME).write_text(
        json.dumps(
            {
                "id": "stale",
                "text": "row from a corpus nobody can trace",
                "metadata": {"record_id": RECORD_A, "unit_key": "x"},
                "content_hash": "deadbeef",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    plan = ReindexJob(
        records=[_record(RECORD_A, TEXT_A)],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
    ).plan()

    assert plan.mode is ReindexMode.FULL
    assert plan.stale is True
    assert stamp_is_stale(None) is True

# ── the stale-corpus trigger: the ticket's real point ──────────────────────

def test_stamp_mismatch_alone_forces_full_with_an_empty_affected_id_set() -> None:
    """The exact scenario that was silently missed: no ids, stale corpus."""
    plan = plan_reindex(
        built_stamp=CorpusStamp(
            chunker_version="0.9", unit_rules_version="1.0", chunk_manifest_version=1
        ),
        current_stamp=CorpusStamp.current(),
        affected_record_ids=[],
    )

    assert plan.mode is ReindexMode.FULL
    assert plan.stale is True
    assert plan.affected_record_ids == ()
    assert "stamp-mismatch" in plan.reason

def test_a_matching_stamp_with_no_affected_ids_is_a_noop() -> None:
    """The counterfactual: same empty id set, current stamp — nothing to do."""
    plan = plan_reindex(
        built_stamp=CorpusStamp.current(),
        current_stamp=CorpusStamp.current(),
        affected_record_ids=[],
    )

    assert plan.mode is ReindexMode.INCREMENTAL
    assert plan.stale is False
    assert plan.is_noop is True

def test_a_partial_stamp_is_stale() -> None:
    """A stamp missing ``unit_rules_version`` must not compare equal."""
    assert CorpusStamp.from_mapping({"chunker_version": "1.0"}) is None
    assert stamp_is_stale(None) is True

def test_rows_without_a_stamp_are_not_a_stamp() -> None:
    """The served table's 1679 legacy rows: no stamp, so stale."""
    assert CorpusStamp.from_rows([{"metadata_json": "{}"}]) is None
    assert CorpusStamp.from_rows([]) is None
    stamped = json.dumps({"corpus_stamp": CorpusStamp.current().to_dict()})
    assert CorpusStamp.from_rows([{"metadata_json": stamped}]) == CorpusStamp.current()
    assert (
        CorpusStamp.from_rows(
            [
                {"metadata_json": stamped},
                {
                    "metadata_json": json.dumps(
                        {
                            "corpus_stamp": {
                                "chunker_version": "0.9",
                                "unit_rules_version": "1.0",
                                "chunk_manifest_version": 1,
                            }
                        }
                    )
                },
            ]
        )
        is None
    )

@pytest.fixture
def metadata_only_chunker(monkeypatch):
    """A chunker whose OUTPUT METADATA differs while the chunk text does not.

    Ticket 32/33 changed ``scripts/chunk_table_aware_v2.py`` so chunk metadata
    gained ``rowspan_attribution``/``rowspan_unresolved`` while the text of most
    chunks stayed byte-identical. This double reproduces exactly that: the ids
    do not move, the content hashes do not move, and only the record-level
    trigger would therefore never fire.
    """
    real = reindex_job_module.build_chunk_index
    state: dict[str, dict] = {"extra": {}}

    class _Result:
        def __init__(self, chunks: list[dict]) -> None:
            self.chunks = tuple(chunks)

    def fake(records, output_root, release_id):
        produced = real(records, output_root, release_id)
        chunks = []
        for chunk in produced.chunks:
            updated = copy.deepcopy(chunk)
            updated["metadata"].update(state["extra"])
            chunks.append(updated)
        return _Result(chunks)

    monkeypatch.setattr(reindex_job_module, "build_chunk_index", fake)
    return state

def test_stamp_bump_with_empty_affected_ids_rebuilds_the_whole_corpus(
    corpus_dir: Path,
    embedder: CountingEmbedder,
    metadata_only_chunker,
    monkeypatch,
) -> None:
    """A parser change reaches the corpus with NO record-level trigger at all."""
    records = [_record(RECORD_A, TEXT_A), _record(RECORD_B, TEXT_B)]
    _run(corpus_dir, embedder, records)
    before = _chunks(corpus_dir)
    assert all("rowspan_attribution" not in c["metadata"] for c in before)

    # The parser changed: the stamp bumps, and every record — including the ones
    # no outbox entry names — now produces chunks with the new metadata.
    monkeypatch.setattr(
        reindex_job_module,
        "version_stamp",
        lambda: {
            "chunker_version": "1.1",
            "unit_rules_version": "1.0",
            "chunk_manifest_version": 1,
        },
    )
    metadata_only_chunker["extra"] = {"rowspan_attribution": "merged"}

    embedder.batches.clear()
    result = ReindexJob(
        records=records,
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
        affected_record_ids=[],
    ).run()

    assert result.plan.mode is ReindexMode.FULL
    assert result.plan.affected_record_ids == ()
    after = _chunks(corpus_dir)
    assert set(c["id"] for c in after) == set(c["id"] for c in before)
    assert all(c["metadata"]["rowspan_attribution"] == "merged" for c in after)
    # The text did not change, so nothing had to be re-embedded — the rebuild is
    # a re-chunk, not a re-embed of the whole corpus.
    assert result.embedded_count == 0
    assert CorpusStamp.from_manifest(corpus_dir).chunker_version == "1.1"

# ── incremental re-index: only what changed ────────────────────────────────

def test_only_changed_chunks_are_re_embedded_and_the_count_equals_the_change(
    corpus_dir: Path, embedder: CountingEmbedder, tmp_path: Path
) -> None:
    """The count is asserted exactly, not as "less than the corpus"."""
    a_v1, b = _record(RECORD_A, TEXT_A), _record(RECORD_B, TEXT_B)
    _run(corpus_dir, embedder, [a_v1, b])

    a_v2 = _record(RECORD_A, TEXT_A_V2, version=2)
    embedder.batches.clear()
    result = ReindexJob(
        records=[a_v2, b],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
        affected_record_ids=[RECORD_A],
    ).run()

    expected = _expected_chunk_ids([a_v2], tmp_path)
    assert result.plan.mode is ReindexMode.INCREMENTAL
    assert result.plan.affected_record_ids == (RECORD_A,)
    assert result.embedded_count == len(expected)
    assert set(result.embedded_chunk_ids) == expected
    assert len(embedder.embedded_texts) == len(expected)

def test_a_new_version_whose_text_is_unchanged_re_embeds_nothing(
    corpus_dir: Path, embedder: CountingEmbedder, tmp_path: Path
) -> None:
    """Only *changed* chunks — and a re-versioned record with identical text
    changed no chunk's text, only its identity. Ticket 06 keys a chunk id on the
    record version, so every id moves here; re-embedding on that basis would
    make a one-field correction cost the record's whole table."""
    a_v1, b = _record(RECORD_A, TEXT_A), _record(RECORD_B, TEXT_B)
    _run(corpus_dir, embedder, [a_v1, b])

    a_v2 = _record(RECORD_A, TEXT_A, version=2)
    assert a_v2.record_version_key != a_v1.record_version_key
    embedder.batches.clear()
    result = ReindexJob(
        records=[a_v2, b],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
        affected_record_ids=[RECORD_A],
    ).run()

    # New identities, same text: the corpus moves to the new version's ids...
    expected = _expected_chunk_ids([a_v2], tmp_path)
    assert set(result.written_chunk_ids) >= expected
    assert {
        c["metadata"]["record_version_key"]
        for c in _chunks(corpus_dir)
        if c["metadata"]["record_id"] == RECORD_A
    } == {a_v2.record_version_key}
    # ...without spending a single embedding on text that did not change.
    assert result.embedded_count == 0
    assert embedder.batches == []
    assert expected <= set(result.reused_chunk_ids)
    assert {v["chunk_id"] for v in _vectors(corpus_dir)} == set(result.written_chunk_ids)

def test_untouched_records_keep_their_vectors(
    corpus_dir: Path, embedder: CountingEmbedder
) -> None:
    a_v1, b = _record(RECORD_A, TEXT_A), _record(RECORD_B, TEXT_B)
    _run(corpus_dir, embedder, [a_v1, b])
    b_chunk_ids = {v["chunk_id"] for v in _vectors(corpus_dir)} - _expected_chunk_ids(
        [a_v1], corpus_dir.parent
    )

    result = ReindexJob(
        records=[_record(RECORD_A, TEXT_A_V2, version=2), b],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
        affected_record_ids=[RECORD_A],
    ).run()

    assert b_chunk_ids
    assert b_chunk_ids <= set(result.reused_chunk_ids)
    assert not (b_chunk_ids & set(result.embedded_chunk_ids))

def test_the_corpus_holds_the_new_version_and_no_orphan_of_the_old_one(
    corpus_dir: Path, embedder: CountingEmbedder
) -> None:
    a_v1, b = _record(RECORD_A, TEXT_A), _record(RECORD_B, TEXT_B)
    _run(corpus_dir, embedder, [a_v1, b])
    old_ids = {c["id"] for c in _chunks(corpus_dir) if c["metadata"]["record_id"] == RECORD_A}

    a_v2 = _record(RECORD_A, TEXT_A_V2, version=2)
    ReindexJob(
        records=[a_v2, b],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
        affected_record_ids=[RECORD_A],
    ).run()

    chunks = _chunks(corpus_dir)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert not (old_ids & set(ids))
    assert {c["metadata"]["record_version_key"] for c in chunks if c["metadata"]["record_id"] == RECORD_A} == {
        a_v2.record_version_key
    }
    assert {v["chunk_id"] for v in _vectors(corpus_dir)} == set(ids)

def test_a_record_that_is_no_longer_published_leaves_the_corpus(
    corpus_dir: Path, embedder: CountingEmbedder
) -> None:
    a, b = _record(RECORD_A, TEXT_A), _record(RECORD_B, TEXT_B)
    _run(corpus_dir, embedder, [a, b])

    result = ReindexJob(
        records=[b],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
        affected_record_ids=[RECORD_A],
    ).run()

    assert result.plan.mode is ReindexMode.INCREMENTAL
    assert {c["metadata"]["record_id"] for c in _chunks(corpus_dir)} == {RECORD_B}
    assert result.removed_chunk_ids
    assert _lines_for_records(corpus_dir, {RECORD_A}) == []

# ── graph edges ────────────────────────────────────────────────────────────

def test_related_edges_are_updated_and_unrelated_edges_are_byte_identical(
    corpus_dir: Path, embedder: CountingEmbedder
) -> None:
    a_v1, b = _record(RECORD_A, TEXT_A), _record(RECORD_B, TEXT_B)
    _run(corpus_dir, embedder, [a_v1, b])
    unrelated_before = _lines_not_for_records(corpus_dir, {RECORD_A})
    related_before = _lines_for_records(corpus_dir, {RECORD_A})
    assert unrelated_before and related_before

    result = ReindexJob(
        records=[_record(RECORD_A, TEXT_A_V2, version=2), b],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
        affected_record_ids=[RECORD_A],
    ).run()

    # The unrelated lines are the same bytes, not merely an equal set.
    assert _lines_not_for_records(corpus_dir, {RECORD_A}) == unrelated_before
    # The related ones were rebuilt: the record's chunk ids all changed, so
    # every HAS_CHUNK edge moved and the HAS_ATOM edges moved with them.
    related_after = _lines_for_records(corpus_dir, {RECORD_A})
    assert related_after != related_before
    before_keys = {json.loads(line)["edge_key"] for line in related_before}
    after_keys = {json.loads(line)["edge_key"] for line in related_after}
    assert set(result.edge_keys_removed) == before_keys - after_keys
    assert set(result.edge_keys_written) == after_keys
    assert before_keys - after_keys  # something really was replaced
    # Nothing of the untouched record was dropped or rewritten.
    assert not (set(result.edge_keys_removed) & {
        json.loads(line)["edge_key"] for line in unrelated_before
    })
    assert not (set(result.edge_keys_written) & {
        json.loads(line)["edge_key"] for line in unrelated_before
    })

def test_every_edge_names_the_record_that_contributed_it(
    corpus_dir: Path, embedder: CountingEmbedder
) -> None:
    """Provenance is what makes a scoped re-index possible at all."""
    _run(corpus_dir, embedder, [_record(RECORD_A, TEXT_A), _record(RECORD_B, TEXT_B)])

    lines = _edge_lines(corpus_dir)
    assert lines
    for line in lines:
        edge = json.loads(line)
        assert edge["edge_key"] == f"{edge['head']}|{edge['relation']}|{edge['tail']}"
        # One contributing record and one contributing chunk per edge: an edge
        # with two owners could not be dropped when only one of them changes.
        assert len(edge["record_ids"]) == 1
        assert len(edge["chunk_ids"]) == 1
        assert edge["head"] == f"RECORD:{edge['record_ids'][0]}" or edge["head"] == (
            f"CHUNK:{edge['chunk_ids'][0]}"
        )
        assert edge["tail"] == f"CHUNK:{edge['chunk_ids'][0]}" or edge["relation"] == "HAS_ATOM"
    assert _lines_for_records(corpus_dir, {RECORD_B})

# ── job state machine ──────────────────────────────────────────────────────

def test_saved_is_not_a_job_state() -> None:
    """A job that merely persisted is not a job that is serving."""
    assert "saved" not in {state.value for state in JobState}
    assert not hasattr(JobState, "SAVED")
    assert JobState.ACTIVE.is_serving is True
    for state in (JobState.PENDING, JobState.RUNNING, JobState.FAILED):
        assert state.is_serving is False

def test_a_job_that_fails_during_the_build_is_failed_and_not_serving(
    corpus_dir: Path, embedder: CountingEmbedder
) -> None:
    def exploding(texts):
        raise RuntimeError("embedding provider down")

    job = ReindexJob(
        records=[_record(RECORD_A, TEXT_A)],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=exploding,
    )

    with pytest.raises(RuntimeError):
        job.run()

    assert job.state is JobState.FAILED
    assert job.serving is False
    assert job.saved is False
    assert isinstance(job.error, RuntimeError)

def test_a_job_whose_publish_fails_is_saved_but_not_active(
    corpus_dir: Path, embedder: CountingEmbedder
) -> None:
    """The distinction the ticket insists on: persisted is not serving."""
    def failing_publish(staged_dir: Path, target: Path) -> None:
        raise OSError("swap failed mid-flight")

    job = ReindexJob(
        records=[_record(RECORD_A, TEXT_A)],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
        publish=failing_publish,
    )

    with pytest.raises(OSError):
        job.run()

    assert job.saved is True
    assert job.staged_dir is not None
    assert (job.staged_dir / CHUNKS_FILENAME).is_file()
    assert job.state is JobState.FAILED
    assert job.serving is False

def test_a_failed_job_leaves_the_serving_corpus_byte_identical(
    corpus_dir: Path, embedder: CountingEmbedder
) -> None:
    a, b = _record(RECORD_A, TEXT_A), _record(RECORD_B, TEXT_B)
    _run(corpus_dir, embedder, [a, b])
    before = _snapshot(corpus_dir)

    def failing_publish(staged_dir: Path, target: Path) -> None:
        raise OSError("swap failed mid-flight")

    job = ReindexJob(
        records=[_record(RECORD_A, TEXT_A_V2, version=2), b],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
        affected_record_ids=[RECORD_A],
        publish=failing_publish,
    )
    with pytest.raises(OSError):
        job.run()

    assert _snapshot(corpus_dir) == before

def test_a_failed_job_does_not_advance_the_outbox_cursor(
    corpus_dir: Path, embedder: CountingEmbedder, tmp_path: Path
) -> None:
    """A failure retries with the same entries instead of losing them."""
    outbox = ReindexOutbox(tmp_path / "outbox" / "reindex_outbox.jsonl")
    outbox.record_publication(_record(RECORD_A, TEXT_A), release_id=RELEASE_ID)

    def exploding(texts):
        raise RuntimeError("embedding provider down")

    job = ReindexJob(
        records=[_record(RECORD_A, TEXT_A)],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=exploding,
        outbox=outbox,
    )
    with pytest.raises(RuntimeError):
        job.run()

    assert outbox.cursor() == 0
    assert outbox.pending_affected_record_ids() == (RECORD_A,)

def test_a_cursor_that_cannot_advance_still_reports_the_corpus_as_serving(
    corpus_dir: Path, embedder: CountingEmbedder, tmp_path: Path
) -> None:
    """The corpus is published; only the cursor is behind. Not a failed run."""
    outbox = ReindexOutbox(tmp_path / "outbox" / "reindex_outbox.jsonl")
    outbox.record_publication(_record(RECORD_A, TEXT_A), release_id=RELEASE_ID)
    outbox.cursor_path.mkdir(parents=True)  # the cursor file cannot be written

    job = ReindexJob(
        records=[_record(RECORD_A, TEXT_A)],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
        outbox=outbox,
    )
    result = job.run()

    assert result.outbox_cursor_advanced is False
    assert job.state is JobState.ACTIVE
    assert job.serving is True
    assert (corpus_dir / CHUNKS_FILENAME).is_file()
    # The entries are still there, so nothing was lost.
    assert outbox.pending_affected_record_ids() == (RECORD_A,)

def test_a_job_runs_once(corpus_dir: Path, embedder: CountingEmbedder) -> None:
    job = ReindexJob(
        records=[_record(RECORD_A, TEXT_A)],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
    )
    job.run()

    with pytest.raises(JobStateError):
        job.run()

def test_an_embedder_that_returns_the_wrong_number_of_vectors_is_refused(
    corpus_dir: Path,
) -> None:
    """A corpus whose vectors and chunks disagree must never be published."""
    job = ReindexJob(
        records=[_record(RECORD_A, TEXT_A)],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=lambda texts: [[0.0] * 8],
    )

    with pytest.raises(ReindexError):
        job.run()
    assert job.state is JobState.FAILED

def test_a_full_reindex_will_not_publish_an_empty_corpus(
    corpus_dir: Path, embedder: CountingEmbedder
) -> None:
    """A forced full rebuild with nothing to build must not wipe what is served.

    This is the sharp edge of the stale trigger: the stamp bump that forces FULL
    is exactly the moment the record set might be empty (a release that
    materialized nothing). Publishing an empty corpus over a live one would turn
    a staleness repair into a coverage outage.
    """
    _run(corpus_dir, embedder, [_record(RECORD_A, TEXT_A)])
    before = _snapshot(corpus_dir)

    manifest_path = corpus_dir / CHUNK_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunker_version"] = "0.9"  # stale → forces FULL
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    stale_snapshot = _snapshot(corpus_dir)

    job = ReindexJob(
        records=[],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
    )
    with pytest.raises(ReindexError):
        job.run()

    assert job.state is JobState.FAILED
    assert job.serving is False
    assert _snapshot(corpus_dir) == stale_snapshot
    # The chunks and vectors — the part a reader serves — are untouched.
    assert stale_snapshot[CHUNKS_FILENAME] == before[CHUNKS_FILENAME]
    assert stale_snapshot[VECTORS_FILENAME] == before[VECTORS_FILENAME]

def test_a_release_id_is_required(corpus_dir: Path, embedder: CountingEmbedder) -> None:
    with pytest.raises(ValueError):
        ReindexJob(
            records=[],
            corpus_dir=corpus_dir,
            release_id="  ",
            embedder=embedder,
        )

# ── idempotency ────────────────────────────────────────────────────────────

def test_rerunning_the_same_version_does_not_duplicate(
    corpus_dir: Path, embedder: CountingEmbedder
) -> None:
    a_v2, b = _record(RECORD_A, TEXT_A_V2, version=2), _record(RECORD_B, TEXT_B)
    _run(corpus_dir, embedder, [a_v2, b])
    first_ids = [c["id"] for c in _chunks(corpus_dir)]

    embedder.batches.clear()
    result = ReindexJob(
        records=[a_v2, b],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
        affected_record_ids=[RECORD_A],
    ).run()

    second_ids = [c["id"] for c in _chunks(corpus_dir)]
    assert len(second_ids) == len(set(second_ids))
    assert sorted(second_ids) == sorted(first_ids)
    assert result.embedded_count == 0
    assert embedder.batches == []
    assert {v["chunk_id"] for v in _vectors(corpus_dir)} == set(second_ids)

def test_a_repeated_publication_is_a_noop_run(
    corpus_dir: Path, embedder: CountingEmbedder, tmp_path: Path
) -> None:
    outbox = ReindexOutbox(tmp_path / "outbox" / "reindex_outbox.jsonl")
    record = _record(RECORD_A, TEXT_A)
    _run(corpus_dir, embedder, [record])
    outbox.record_publication(record, release_id=RELEASE_ID)
    ReindexJob(
        records=[record],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
        outbox=outbox,
    ).run()
    before = _snapshot(corpus_dir)

    # The publish path is retried: the same version is appended again.
    outbox.record_publication(record, release_id=RELEASE_ID)
    embedder.batches.clear()
    job = ReindexJob(
        records=[record],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
        outbox=outbox,
    )
    result = job.run()

    assert result.plan.is_noop is True
    assert embedder.batches == []
    assert _snapshot(corpus_dir) == before
    assert job.state is JobState.ACTIVE

def test_the_outbox_drives_the_affected_id_set(
    corpus_dir: Path, embedder: CountingEmbedder, tmp_path: Path
) -> None:
    """End to end: publish → outbox entry → the job re-indexes exactly that record."""
    a_v1, b = _record(RECORD_A, TEXT_A), _record(RECORD_B, TEXT_B)
    _run(corpus_dir, embedder, [a_v1, b])

    outbox = ReindexOutbox(tmp_path / "outbox" / "reindex_outbox.jsonl")
    a_v2 = _record(RECORD_A, TEXT_A_V2, version=2)
    outbox.record_publication(a_v2, release_id=RELEASE_ID)

    embedder.batches.clear()
    result = ReindexJob(
        records=[a_v2, b],
        corpus_dir=corpus_dir,
        release_id=RELEASE_ID,
        embedder=embedder,
        outbox=outbox,
    ).run()

    assert result.plan.affected_record_ids == (RECORD_A,)
    assert set(result.embedded_chunk_ids) == _expected_chunk_ids(
        [a_v2], corpus_dir.parent
    )
    assert outbox.pending_affected_record_ids() == ()
