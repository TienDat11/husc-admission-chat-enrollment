"""TDD: ingest_lancedb.py emits full 17-field CHUNK_SCHEMA to husc_chunks (M3-N2b).

REAL LanceDB table under D:/tmp — no mocks on table ops.
Monkeypatch RAGSettings + EmbeddingService so zero model load.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import lancedb
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure.lancedb_schema import CHUNK_SCHEMA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> Path:
    d = Path("D:/tmp")
    d.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def temp_db(tmp_workspace: Path):
    d = tempfile.mkdtemp(dir="D:/tmp", prefix="test_ingest_schema_")
    db = lancedb.connect(d)
    yield db
    try:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, chunks: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks),
        encoding="utf-8",
    )


def _patch_ingest(mod, monkeypatch, temp_db, chunk_file, table_name="husc_chunks"):
    """Common monkeypatch setup for ingest tests."""
    monkeypatch.setattr(mod, "MANIFEST_PATH", Path(tempfile.mktemp(dir="D:/tmp", suffix=".json")))
    monkeypatch.setattr(mod, "TARGET_TABLE", table_name)
    monkeypatch.setattr(mod, "_trigger_graph_sync", lambda *_: None)  # no real graph build in tests

    def _fake_resolve(chunks_dir, incremental):
        return [chunk_file]

    monkeypatch.setattr(mod, "_resolve_target_files", _fake_resolve)
    monkeypatch.setattr(mod, "_refresh_manifest_for_all_files", lambda *_: None)

    class FakeSettings:
        LANCEDB_URI = temp_db.uri
        LANCEDB_TABLE = table_name
        LANCEDB_UNIFIED_TABLE = table_name
        EMBEDDING_MODEL = "fake"
        EMBEDDING_DIM = 1024

    monkeypatch.setattr(mod, "RAGSettings", FakeSettings)

    vec_fixed = np.full(1024, 0.1)

    class FakeEmbeddingService:
        def __init__(self, settings):
            pass

        def encode_documents(self, texts):
            return [vec_fixed.copy() for _ in texts]

    monkeypatch.setattr(mod, "EmbeddingService", FakeEmbeddingService)


# ---------------------------------------------------------------------------
# Test 1 — ingest writes full schema to husc_chunks
# ---------------------------------------------------------------------------


def test_ingest_writes_full_schema_to_husc_chunks(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Ingest 1 chunk with metadata → row has all 17 fields with correct defaults."""
    import rag2025.scripts.ingest_lancedb as mod

    chunk_file = tmp_workspace / "chunked_full.jsonl"
    _write_jsonl(chunk_file, [
        {
            "id": "c1",
            "text": "Thông tin tuyển sinh 2026",
            "metadata": {
                "data_year": 2026,
                "is_reference": True,
                "crawl_date": "2026-07-01",
                "source": "test.jsonl",
            },
        },
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    result = mod.ingest_lancedb(incremental=True)
    assert result is True

    table = temp_db.open_table("husc_chunks")
    assert table.count_rows() == 1

    rows = table.to_arrow().to_pylist()
    row = rows[0]

    # Core identity — id is the T1.5 globally-unique hash; doc_id keeps the
    # original chunk_id ("c1") for document-level grouping.
    assert row["id"] == mod._unique_id("test.jsonl", chunk_file.name, "c1")
    assert row["doc_id"] == "c1"
    assert row["text"] == "Thông tin tuyển sinh 2026"
    assert len(row["vector"]) == 1024

    # Backfill defaults
    assert row["release_id"] == "legacy"
    assert row["scope"] == "admissions"
    assert row["status"] == "current"
    assert row["effective_from"] == 0
    assert row["effective_to"] == 99991231
    assert row["is_tombstone"] is False

    # Metadata-derived
    assert row["data_year"] == 2026
    assert row["is_reference"] is True
    assert row["crawl_date"] == "2026-07-01"
    assert row["doc_type"] == "chunk"  # no faq_type → "chunk"


# ---------------------------------------------------------------------------
# Test 2 — backfill defaults when metadata missing
# ---------------------------------------------------------------------------


def test_ingest_backfill_defaults_when_metadata_missing(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Chunk without data_year/is_reference → data_year=0, needs_review=true, is_reference=False."""
    import rag2025.scripts.ingest_lancedb as mod

    chunk_file = tmp_workspace / "chunked_nometa.jsonl"
    _write_jsonl(chunk_file, [
        {"id": "c2", "text": "Nội dung không có metadata"},
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    result = mod.ingest_lancedb(incremental=True)
    assert result is True

    table = temp_db.open_table("husc_chunks")
    rows = table.to_arrow().to_pylist()
    row = rows[0]

    assert row["data_year"] == 0
    assert row["is_reference"] is False
    assert row["crawl_date"] is None

    # metadata_json must contain needs_review: true
    meta = json.loads(row["metadata_json"])
    assert meta.get("needs_review") is True


# ---------------------------------------------------------------------------
# Test 3 — _load_existing_ids no TypeError (F14 fix)
# ---------------------------------------------------------------------------


def test_load_existing_ids_no_typeerror(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """_load_existing_ids must NOT raise TypeError on LanceDB 0.29.2."""
    import rag2025.scripts.ingest_lancedb as mod

    # Create a table with some rows
    table = temp_db.create_table("test_ids", data=[
        {"id": "a1", "text": "hello", "vector": [0.1] * 1024},
        {"id": "a2", "text": "world", "vector": [0.2] * 1024},
    ])

    # This must NOT raise TypeError
    ids = mod._load_existing_ids(table)
    assert ids == {"a1", "a2"}


# ---------------------------------------------------------------------------
# Test 4 — idempotent on husc_chunks
# ---------------------------------------------------------------------------


def test_ingest_idempotent_on_husc_chunks(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Ingest same JSONL twice → count_rows() does not increase."""
    import rag2025.scripts.ingest_lancedb as mod

    chunk_file = tmp_workspace / "chunked_idem.jsonl"
    _write_jsonl(chunk_file, [
        {"id": "i1", "text": "Idempotent test", "metadata": {"data_year": 2026}},
        {"id": "i2", "text": "Second chunk", "metadata": {"data_year": 2026}},
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    mod.ingest_lancedb(incremental=True)
    table = temp_db.open_table("husc_chunks")
    assert table.count_rows() == 2

    # Second ingest — same content
    mod.ingest_lancedb(incremental=True)
    table2 = temp_db.open_table("husc_chunks")
    assert table2.count_rows() == 2  # MUST stay 2


# ---------------------------------------------------------------------------
# Test 5 — content-hash skips re-embed (M3-N1b)
# ---------------------------------------------------------------------------


def test_content_hash_skips_reembed(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Ingest same text twice → second ingest calls encode_documents 0 times."""
    import rag2025.scripts.ingest_lancedb as mod

    chunk_file = tmp_workspace / "chunked_hash.jsonl"
    _write_jsonl(chunk_file, [
        {"id": "h1", "text": "Nội dung giống nhau", "metadata": {"data_year": 2026}},
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    # Spy on encode_documents
    call_count = {"n": 0}
    original_encode = mod.EmbeddingService.encode_documents

    def spy_encode(self, texts):
        call_count["n"] += len(texts)
        return original_encode(self, texts)

    monkeypatch.setattr(mod.EmbeddingService, "encode_documents", spy_encode)

    # First ingest
    mod.ingest_lancedb(incremental=True)
    first_count = call_count["n"]
    assert first_count == 1  # 1 chunk embedded

    # Second ingest — same text
    call_count["n"] = 0
    mod.ingest_lancedb(incremental=True)
    second_count = call_count["n"]

    # MUST be 0 (skip re-embed)
    assert second_count == 0, f"Expected 0 re-embeds, got {second_count}"


# ---------------------------------------------------------------------------
# Test 6 — content-hash re-embeds on text change (M3-N1b)
# ---------------------------------------------------------------------------


def test_content_hash_reembeds_on_text_change(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Change text → re-embed happens (call count >= 1)."""
    import rag2025.scripts.ingest_lancedb as mod

    chunk_file = tmp_workspace / "chunked_hash2.jsonl"
    _write_jsonl(chunk_file, [
        {"id": "h2", "text": "Nội dung cũ", "metadata": {"data_year": 2026}},
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    call_count = {"n": 0}
    original_encode = mod.EmbeddingService.encode_documents

    def spy_encode(self, texts):
        call_count["n"] += len(texts)
        return original_encode(self, texts)

    monkeypatch.setattr(mod.EmbeddingService, "encode_documents", spy_encode)

    # First ingest
    mod.ingest_lancedb(incremental=True)
    assert call_count["n"] == 1

    # Change text
    _write_jsonl(chunk_file, [
        {"id": "h2", "text": "Nội dung mới", "metadata": {"data_year": 2026}},
    ])

    call_count["n"] = 0
    mod.ingest_lancedb(incremental=True)

    # MUST re-embed (count >= 1)
    assert call_count["n"] >= 1, f"Expected re-embed on text change, got {call_count['n']}"


# ---------------------------------------------------------------------------
# Test 7 — T1.4/G4 full-overwrite tags pre-ingest version BEFORE drop
# ---------------------------------------------------------------------------


def test_full_overwrite_tags_preingest_before_drop(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Full-overwrite (incremental=False on an existing table) must tag the
    pre-overwrite version BEFORE create_table(mode='overwrite'), so a mistaken
    default run that drops rows (581/2068) is recoverable via rollback."""
    import rag2025.scripts.ingest_lancedb as mod

    # Pre-existing table simulating production husc_chunks with old rows.
    temp_db.create_table(
        "husc_chunks",
        data=[{"id": "old1", "text": "old row", "vector": [0.1] * 1024}],
    )

    chunk_file = tmp_workspace / "chunked_overwrite.jsonl"
    _write_jsonl(chunk_file, [
        {"id": "n1", "text": "New content", "metadata": {"data_year": 2026}},
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    tagged = []

    def fake_tag(table, release_id):
        tagged.append(release_id)
        return table.version

    import src.infrastructure.lancedb_versioning as lv

    monkeypatch.setattr(lv, "tag_release", fake_tag)

    result = mod.ingest_lancedb(incremental=False)
    assert result is True

    assert any(t.startswith("pre-ingest-") for t in tagged), (
        f"expected a pre-ingest-* tag before overwrite, got {tagged}"
    )

    table = temp_db.open_table("husc_chunks")
    assert table.count_rows() == 1  # overwrite replaced the old row


# ---------------------------------------------------------------------------
# Test 8 — T1.5-bl vector-reuse accounting (reuse vs re-embed)
# ---------------------------------------------------------------------------


def _ingest_style_row(
    doc_id: str, text: str, data_year: int, metadata_json: str = "{}"
) -> Dict[str, Any]:
    """A pre-existing row shaped exactly like what ingest_lancedb writes, so a
    hand-seeded table is schema-compatible with the incremental merge path."""
    return {
        "id": doc_id,
        "doc_id": doc_id,
        "release_id": "legacy",
        "text": text,
        "summary": None,
        "vector": [0.1] * 1024,
        "data_year": data_year,
        "scope": "admissions",
        "doc_type": "chunk",
        "status": "current",
        "effective_from": 0,
        "effective_to": 99991231,
        "is_tombstone": False,
        "superseded_by": None,
        "metadata_json": metadata_json,
        "is_reference": False,
        "crawl_date": None,
    }


def test_vector_reuse_accounting_reuses_existing_vectors(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Pins the T1.5-bl vector-reuse accounting LOGIC behind the production
    'reuse 1634 / re-embed 45' claim. In an overwrite rebuild every id is
    re-hashed (G9) so the id-keyed guardrail never matches — a chunk whose
    content-hash already exists in the CURRENT table must reuse its stored
    vector instead of re-embedding. Absolute numbers are live-corpus-specific,
    so this synthetic fixture pins the mechanism: 1 reused + 1 embedded.

    NOTE: pins EXISTING behavior — green immediately is valid (TDD exception)."""
    import rag2025.scripts.ingest_lancedb as mod

    reused_text = "Nội dung cũ"
    new_text = "Nội dung hoàn toàn mới"

    # Pre-existing table simulating a prior corpus that already embedded the
    # chunk whose text we are about to re-ingest.
    temp_db.create_table(
        "husc_chunks",
        data=[
            {
                **_ingest_style_row("old1", reused_text, 2025),
                "vector": [0.5] * 1024,
                "metadata_json": json.dumps(
                    {"content_hash": mod._content_hash(reused_text)},
                    ensure_ascii=False,
                ),
            }
        ],
    )

    chunk_file = tmp_workspace / "chunked_reuse.jsonl"
    _write_jsonl(chunk_file, [
        {"id": "r1", "text": reused_text, "metadata": {"data_year": 2026}},
        {"id": "r2", "text": new_text, "metadata": {"data_year": 2026}},
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    encoded_texts: List[str] = []
    original_encode = mod.EmbeddingService.encode_documents

    def spy_encode(self, texts):
        encoded_texts.extend(texts)
        return original_encode(self, texts)

    monkeypatch.setattr(mod.EmbeddingService, "encode_documents", spy_encode)

    result = mod.ingest_lancedb(incremental=False)  # overwrite rebuild
    assert result is True

    # Only the genuinely-new chunk may hit the embedding service.
    assert encoded_texts == [new_text]

    table = temp_db.open_table("husc_chunks")
    assert table.count_rows() == 2

    by_doc = {r["doc_id"]: r for r in table.to_arrow().to_pylist()}
    # r1 reused the stored [0.5] vector; r2 was freshly embedded ([0.1]).
    assert np.allclose(np.asarray(by_doc["r1"]["vector"]), np.full(1024, 0.5))
    assert np.allclose(np.asarray(by_doc["r2"]["vector"]), np.full(1024, 0.1))


# ---------------------------------------------------------------------------
# Test 9 — empty-records guard preserves the table
# ---------------------------------------------------------------------------


def test_empty_records_guard_preserves_table(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Pins the empty-records guard at ingest_lancedb.py:493-497: if no record
    can be written, ingest must abort (return False) and leave the existing
    table untouched — never a silent full-overwrite that destroys rows.

    NOTE: pins EXISTING behavior — green immediately is valid (TDD exception)."""
    import rag2025.scripts.ingest_lancedb as mod

    temp_db.create_table(
        "husc_chunks",
        data=[{"id": "old1", "text": "existing", "vector": [0.1] * 1024}],
    )

    chunk_file = tmp_workspace / "chunked_empty_records.jsonl"
    _write_jsonl(chunk_file, [
        {"id": "e1", "text": "new chunk", "metadata": {"data_year": 2026}},
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    # Embedding returns no vectors → every chunk is skipped as vector-less →
    # records is empty → the guard must abort.
    monkeypatch.setattr(
        mod.EmbeddingService, "encode_documents", lambda self, texts: []
    )

    result = mod.ingest_lancedb(incremental=False)
    assert result is False

    table = temp_db.open_table("husc_chunks")
    assert table.count_rows() == 1  # preserved, not overwritten


# ---------------------------------------------------------------------------
# Test 10 — list_tables() .tables defensive branch
# ---------------------------------------------------------------------------


class _ListTablesWrapper:
    """Delegates to the real LanceDB connection but stubs list_tables() so the
    ingest defensive branch at ingest_lancedb.py:329-331 is exercised for both
    the ListTablesResponse-object (.tables field) and plain-list shapes."""

    def __init__(self, db: Any, response: Any):
        self._db = db
        self._response = response

    def list_tables(self):
        return self._response

    def open_table(self, name):
        return self._db.open_table(name)

    def create_table(self, name, data=None, mode="create"):
        return self._db.create_table(name, data, mode=mode)


def _list_tables_response_cases():
    class _ListTablesResponse:
        tables = ["husc_chunks"]

    return [
        pytest.param(["husc_chunks"], id="plain-list"),
        pytest.param(_ListTablesResponse(), id="object-with-tables-field"),
    ]


@pytest.mark.parametrize("list_tables_response", _list_tables_response_cases())
def test_list_tables_branch_detects_table_both_shapes(
    list_tables_response: Any,
    tmp_workspace: Path,
    temp_db: Any,
    monkeypatch: Any,
):
    """Pins the defensive branch at ingest_lancedb.py:329-331: lancedb 0.29
    list_tables() returns a ListTablesResponse object with a `.tables` field
    (not a plain list), so `target in response` is always False. Both response
    shapes must resolve table_exists=True so the incremental MERGE path runs
    instead of a destructive create_table.

    NOTE: pins EXISTING behavior — green immediately is valid (TDD exception)."""
    import rag2025.scripts.ingest_lancedb as mod

    temp_db.create_table(
        "husc_chunks",
        data=[_ingest_style_row("old1", "old row", 2025)],
    )

    chunk_file = tmp_workspace / "chunked_listtables.jsonl"
    _write_jsonl(chunk_file, [
        {"id": "n1", "text": "New content", "metadata": {"data_year": 2026}},
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    def fake_connect(uri):
        return _ListTablesWrapper(temp_db, list_tables_response)

    monkeypatch.setattr(mod.lancedb, "connect", fake_connect)

    result = mod.ingest_lancedb(incremental=True)
    assert result is True

    table = temp_db.open_table("husc_chunks")
    # Merged the new row into the existing table (no overwrite/no re-create).
    assert table.count_rows() == 2


# ---------------------------------------------------------------------------
# Test 11 — is_reference string parsing
# ---------------------------------------------------------------------------


def test_is_reference_string_parsing(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Pins the defensive parse at ingest_lancedb.py:483-485: bool passes
    through unchanged; strings ('True'/'false'/'yes'/'1'/'0'/'no'/None) are
    normalized via str(...).lower() in ('1','true','yes').

    NOTE: pins EXISTING behavior — green immediately is valid (TDD exception)."""
    import rag2025.scripts.ingest_lancedb as mod

    cases = [
        ("bool-true", {"is_reference": True}, True),
        ("bool-false", {"is_reference": False}, False),
        ("str-True", {"is_reference": "True"}, True),
        ("str-false", {"is_reference": "false"}, False),
        ("str-yes", {"is_reference": "yes"}, True),
        ("str-1", {"is_reference": "1"}, True),
        ("str-0", {"is_reference": "0"}, False),
        ("str-no", {"is_reference": "no"}, False),
        ("none", {"is_reference": None}, False),
        ("missing", {}, False),
    ]

    chunk_file = tmp_workspace / "chunked_isref.jsonl"
    _write_jsonl(chunk_file, [
        {"id": cid, "text": f"text-{cid}", "metadata": meta}
        for cid, meta, _ in cases
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    result = mod.ingest_lancedb(incremental=True)
    assert result is True

    table = temp_db.open_table("husc_chunks")
    rows = table.to_arrow().to_pylist()
    assert len(rows) == len(cases)

    by_doc = {r["doc_id"]: r["is_reference"] for r in rows}
    for cid, _meta, expected in cases:
        assert bool(by_doc[cid]) is expected, (
            f"case {cid}: expected {expected}, got {by_doc[cid]}"
        )


# ---------------------------------------------------------------------------
# Test 12 — `id: null` fallback never collides
# ---------------------------------------------------------------------------


def test_id_null_fallback_does_not_collide(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Pins the fallback at ingest_lancedb.py:172: `id: null` must not
    stringify to 'None' (two such rows would collide on _unique_id and silently
    overwrite in a rebuild). Fallback order: chunk_id, then a stable per-row
    index chunk_{len(all_chunks)}.

    NOTE: pins EXISTING behavior — green immediately is valid (TDD exception)."""
    import rag2025.scripts.ingest_lancedb as mod

    chunk_file = tmp_workspace / "chunked_nullid.jsonl"
    _write_jsonl(chunk_file, [
        {"id": None, "text": "Row A"},
        {"id": None, "text": "Row B"},
        {"id": None, "chunk_id": "manual-id", "text": "Row C"},
        {"id": "explicit", "text": "Row D"},
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    result = mod.ingest_lancedb(incremental=True)
    assert result is True

    table = temp_db.open_table("husc_chunks")
    rows = table.to_arrow().to_pylist()
    assert len(rows) == 4

    doc_ids = {r["doc_id"] for r in rows}
    assert doc_ids == {"chunk_0", "chunk_1", "manual-id", "explicit"}

    # Row ids are globally-unique hashes — the two null-id rows must not collide.
    row_ids = {r["id"] for r in rows}
    assert len(row_ids) == 4


# ---------------------------------------------------------------------------
# Test 13 — T7: vector-reuse key is date-sensitive
# ---------------------------------------------------------------------------


def test_content_hash_date_variants_distinct():
    """T7 RED: two texts differing ONLY by a date must NOT collide in the
    content→vector reuse key. `_normalize_for_hash` stripped \\d{2}/\\d{2}/\\d{4}
    before hashing, so 'ngày 12/01/2026' and 'ngày 01/01/2026' hashed to the
    same key — a date-only variant would silently reuse the other text's
    embedding. Date tokens must be PRESERVED in the hashed text."""
    import rag2025.scripts.ingest_lancedb as mod

    a = "Hạn nộp hồ sơ là ngày 12/01/2026"
    b = "Hạn nộp hồ sơ là ngày 01/01/2026"
    assert mod._content_hash(a) != mod._content_hash(b)


def test_content_hash_byte_identical_reuse_same_key():
    """T7 pin: byte-identical texts must still map to the SAME reuse key, and
    boilerplate whitespace normalization must be preserved (only the date
    stripping is dropped). Pins EXISTING behavior — green immediately is valid
    (TDD exception)."""
    import rag2025.scripts.ingest_lancedb as mod

    text = "Nội dung tuyển sinh 2026"
    assert mod._content_hash(text) == mod._content_hash(text)
    # Whitespace collapse remains intact alongside date preservation.
    assert mod._content_hash("  Nội  dung  ") == mod._content_hash("Nội dung")


def test_vector_reuse_preserved_across_hash_scheme_change(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """T7: the fix must NOT trigger re-embeds for the existing corpus (measured
    0 date-only-variant collision pairs). A row persisted under the OLD scheme
    (date-stripped content_hash in metadata_json) whose text contains a date
    must still be REUSED by a byte-identical re-ingest. The reuse map must key
    from the immutable `text` column (recompute), not the persisted hash, so a
    date-sensitive `_content_hash` never orphans an already-embedded row.

    Pins behavior that is green under the CURRENT code (both old/new hashes
    coincide there) but RED under a wrong fix that only edits the hash without
    rekeying the reuse map — that regression would re-embed every date-bearing
    chunk."""
    import hashlib
    import re as _re

    import rag2025.scripts.ingest_lancedb as mod

    text = "Hạn nộp hồ sơ là ngày 12/01/2026"

    # Reproduce the OLD scheme (date-stripping) hash a pre-fix row would carry.
    old_scheme_hash = hashlib.sha256(
        " ".join(_re.sub(r"\d{2}/\d{2}/\d{4}", "", text).split()).encode("utf-8")
    ).hexdigest()
    assert old_scheme_hash != mod._content_hash(text)  # schemes must differ now

    temp_db.create_table(
        "husc_chunks",
        data=[
            {
                **_ingest_style_row("legacy1", text, 2026),
                "vector": [0.5] * 1024,
                "metadata_json": json.dumps(
                    {"content_hash": old_scheme_hash},
                    ensure_ascii=False,
                ),
            }
        ],
    )

    chunk_file = tmp_workspace / "chunked_scheme_migration.jsonl"
    _write_jsonl(chunk_file, [
        {"id": "n1", "text": text, "metadata": {"data_year": 2026}},
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    encoded_texts: List[str] = []
    original_encode = mod.EmbeddingService.encode_documents

    def spy_encode(self, texts):
        encoded_texts.extend(texts)
        return original_encode(self, texts)

    monkeypatch.setattr(mod.EmbeddingService, "encode_documents", spy_encode)

    result = mod.ingest_lancedb(incremental=False)  # overwrite rebuild
    assert result is True

    # Reuse, not re-embed — the date-bearing chunk already has a vector.
    assert encoded_texts == [], f"expected vector reuse, got re-embeds: {encoded_texts}"

    table = temp_db.open_table("husc_chunks")
    rows = table.to_arrow().to_pylist()
    assert len(rows) == 1
    assert np.allclose(np.asarray(rows[0]["vector"]), np.full(1024, 0.5))


# ---------------------------------------------------------------------------
# Test 14 — STEP 8a: header/title + breadcrumbs inherit into metadata_json
# (metadata-only, no text mutation → no re-embed)
# ---------------------------------------------------------------------------


def test_heading_and_breadcrumbs_land_on_text_section_row(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """STEP 8a: a chunk whose JSONL row carries a top-level `breadcrumbs` list
    and a metadata `name` (heading) must have BOTH inherited onto the persisted
    text_section row's metadata_json — as metadata, NOT by mutating `text`
    (heading injection into text would change content_hash and force a full
    ~2h re-embed of the existing corpus).

    NOTE: RED under the current ingest path — `_load_chunks` drops the
    top-level `breadcrumbs` field and the record writer keeps no `title`.
    The fix is metadata-only, so `text` stays byte-identical."""
    import rag2025.scripts.ingest_lancedb as mod

    chunk_file = tmp_workspace / "chunked_title.jsonl"
    _write_jsonl(chunk_file, [
        {
            "id": "t1",
            "text": "Nội dung thông báo tuyển sinh",
            "metadata": {"name": "Thông báo tuyển sinh 2026", "data_year": 2026},
            "breadcrumbs": ["TB123", "Thông báo tuyển sinh 2026"],
        },
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    result = mod.ingest_lancedb(incremental=True)
    assert result is True

    table = temp_db.open_table("husc_chunks")
    rows = table.to_arrow().to_pylist()
    assert len(rows) == 1

    meta = json.loads(rows[0]["metadata_json"])
    assert meta.get("breadcrumbs") == ["TB123", "Thông báo tuyển sinh 2026"]
    assert meta.get("title") == "Thông báo tuyển sinh 2026"
    # text is UNCHANGED — no heading injected, existing embeddings stay valid.
    assert rows[0]["text"] == "Nội dung thông báo tuyển sinh"


# ---------------------------------------------------------------------------
# Test 15 — STEP 8b: ingest-time size-gate flags oversized/undersized rows
# (LOG-AND-FLAG, never silently drop, count surfaced at end of run)
# ---------------------------------------------------------------------------


def test_oversized_record_is_flagged_not_dropped(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """STEP 8b: a synthetic record exceeding HARD_MAX_TOKENS (512) must be
    WRITTEN and flagged `oversized` in metadata_json — never silently dropped,
    never silently accepted without a marker."""
    import rag2025.scripts.ingest_lancedb as mod

    long_text = " ".join(["đại học sư phạm huế tuyển sinh"] * 400)  # 6000 tokens
    assert mod._count_tokens(long_text) > 512

    chunk_file = tmp_workspace / "chunked_oversize.jsonl"
    _write_jsonl(chunk_file, [
        {"id": "o1", "text": long_text, "metadata": {"data_year": 2026}},
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    result = mod.ingest_lancedb(incremental=True)
    assert result is True

    table = temp_db.open_table("husc_chunks")
    rows = table.to_arrow().to_pylist()
    assert len(rows) == 1  # NOT dropped
    meta = json.loads(rows[0]["metadata_json"])
    assert meta.get("size_flag") == "oversized"


def test_undersized_record_is_flagged_not_dropped(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """STEP 8b: a synthetic record below SOFT_MIN_TOKENS (50) must be WRITTEN
    and flagged `undersized` in metadata_json — never silently dropped."""
    import rag2025.scripts.ingest_lancedb as mod

    short_text = "Ngắn"  # 3 tokens
    assert mod._count_tokens(short_text) < 50

    chunk_file = tmp_workspace / "chunked_undersize.jsonl"
    _write_jsonl(chunk_file, [
        {"id": "u1", "text": short_text, "metadata": {"data_year": 2026}},
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    result = mod.ingest_lancedb(incremental=True)
    assert result is True

    table = temp_db.open_table("husc_chunks")
    rows = table.to_arrow().to_pylist()
    assert len(rows) == 1  # NOT dropped
    meta = json.loads(rows[0]["metadata_json"])
    assert meta.get("size_flag") == "undersized"
