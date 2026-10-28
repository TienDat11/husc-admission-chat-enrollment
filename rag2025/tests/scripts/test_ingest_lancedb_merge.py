"""TDD: rag2025/scripts/ingest_lancedb.py — incremental merge behaviour (B5).

REAL LanceDB table under D:/tmp — no mocks on table.add / merge_insert.
Monkeypatch RAGSettings + EmbeddingService so zero model load.

PATCH STRATEGY for hard-coded paths (lines 128-129 of ingest_lancedb.py):
  chunks_dir = Path(__file__).parent.parent / "data" / "chunked"   ← local var
  MANIFEST_PATH = Path(__file__).parent.parent / ...                ← module-level

  We monkeypatch TWO module-level helpers so the function never touches
  rag2025/data/chunked/ at all:
    1. mod._resolve_target_files → return list of temp JSONL paths we control
    2. mod._refresh_manifest_for_all_files → no-op (avoids writing real manifest)
  MANIFEST_PATH is also patched to point into tmp so _load_manifest()
  does not collide with any real .ingest_manifest.json.

  This guarantees ZERO writes to rag2025/data/chunked/ or rag2025/data/lancedb/.

  For RAGSettings: replace mod.RAGSettings with a plain class holding
  LANCEDB_URI / LANCEDB_TABLE / EMBEDDING_MODEL / EMBEDDING_DIM as class
  attrs. Pydantic BaseSettings resolves fields at __init__ time, so
  setattr on the class won't work — full replacement is required.
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> Path:
    """Create D:/tmp workspace for real LanceDB tables."""
    d = Path("D:/tmp")
    d.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def temp_db(tmp_workspace: Path):
    """Return (lancedb.DB, lancedb.Table) in an isolated temp dir."""
    d = tempfile.mkdtemp(dir="D:/tmp", prefix="test_ingest_merge_")
    db = lancedb.connect(d)
    yield db
    # cleanup
    try:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

INITIAL_CHUNKS = [
    {"id": "X", "chunk_id": "X", "text": "Nội dung A", "source": "file1.jsonl", "metadata": {}},
    {"id": "Y", "chunk_id": "Y", "text": "Nội dung Y", "source": "file1.jsonl", "metadata": {}},
    {"id": "Z", "chunk_id": "Z", "text": "Nội dung Z", "source": "file1.jsonl", "metadata": {}},
]


def _write_jsonl(path: Path, chunks: List[Dict]) -> None:
    """Write chunks as one-JSON-per-line file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks),
        encoding="utf-8",
    )


def _row_id(mod, chunk_file: Path, cid: str) -> str:
    """Row id under the T1.5 globally-unique scheme.

    Test chunks carry metadata={} and a top-level "source" key, so
    `_load_chunks` resolves source to `file.name` (the ingest_file) via
    `obj.get("metadata", {}).get("source", file.name)`.
    """
    return mod._unique_id(chunk_file.name, chunk_file.name, cid)


# ---------------------------------------------------------------------------
# Test 1 — idempotent re-ingest same content → no growth
# ---------------------------------------------------------------------------

def test_idempotent_reingest_same_content_no_growth(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Ingest {X,Y,Z} → re-ingest identical file → count_rows() == 3.

    PATCH STRATEGY:
      - Replace mod.RAGSettings with a plain class (Pydantic fields aren't
        class-level attrs, so setattr won't work).
      - monkeypatch mod.MANIFEST_PATH → tmp_path / ".ingest_manifest.json"
      - monkeypatch mod._resolve_target_files → return our temp JSONL path
      - monkeypatch mod._refresh_manifest_for_all_files → no-op
      - monkeypatch mod.EmbeddingService.encode_documents → fixed vectors
    """
    import rag2025.scripts.ingest_lancedb as mod

    # --- setup patches ---
    fake_manifest = tmp_workspace / ".ingest_manifest.json"
    monkeypatch.setattr(mod, "MANIFEST_PATH", fake_manifest)
    monkeypatch.setattr(mod, "TARGET_TABLE", "test_chunks")
    monkeypatch.setattr(mod, "_trigger_graph_sync", lambda *_: None)

    chunk_file = tmp_workspace / "chunked_test.jsonl"
    _write_jsonl(chunk_file, INITIAL_CHUNKS)

    def _fake_resolve(chunks_dir, incremental):
        return [chunk_file]

    monkeypatch.setattr(mod, "_resolve_target_files", _fake_resolve)
    monkeypatch.setattr(mod, "_refresh_manifest_for_all_files", lambda *_: None)

    # Patch _load_existing_ids — lancedb 0.29.2 to_pandas() doesn't accept columns=
    def _fake_load_ids(table):
        try:
            return {str(v) for v in table.to_pandas()["id"].tolist()}
        except Exception:
            return set()

    monkeypatch.setattr(mod, "_load_existing_ids", _fake_load_ids)

    # Replace RAGSettings entirely (Pydantic BaseSettings needs __new__ trickery)
    class FakeSettings:
        LANCEDB_URI = temp_db.uri
        LANCEDB_TABLE = "test_chunks"
        EMBEDDING_MODEL = "fake"
        EMBEDDING_DIM = 1024

    monkeypatch.setattr(mod, "RAGSettings", FakeSettings)

    vec_fixed = np.full(1024, 0.5)

    class FakeEmbeddingService:
        def __init__(self, settings):
            pass

        def encode_documents(self, texts):
            return [vec_fixed.copy() for _ in texts]

    monkeypatch.setattr(mod, "EmbeddingService", FakeEmbeddingService)

    # --- first ingest ---
    result = mod.ingest_lancedb(incremental=True)
    assert result is True

    table = temp_db.open_table("test_chunks")
    assert table.count_rows() == 3

    # --- second ingest (same file) ---
    result2 = mod.ingest_lancedb(incremental=True)
    assert result2 is True

    table2 = temp_db.open_table("test_chunks")
    assert table2.count_rows() == 3  # MUST stay 3


# ---------------------------------------------------------------------------
# Test 2 — update same ID changes content → NO duplicate (RED expected)
# ---------------------------------------------------------------------------

def test_update_same_id_changes_content_no_duplicate(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Change text of chunk X → re-ingest. Assert (a) count==3, (b) exactly 1 row id=X, (c) text=="Nội dung B".

    BUG B5: incremental mode only APPENDS new IDs, never updates existing ones.
    Expected: RED — current code will have 4 rows after re-ingest, and row X still has old text.
    """
    import rag2025.scripts.ingest_lancedb as mod

    fake_manifest = tmp_workspace / ".ingest_manifest2.json"
    monkeypatch.setattr(mod, "MANIFEST_PATH", fake_manifest)
    monkeypatch.setattr(mod, "TARGET_TABLE", "test_update")
    monkeypatch.setattr(mod, "_trigger_graph_sync", lambda *_: None)

    chunk_file = tmp_workspace / "chunked_update.jsonl"
    _write_jsonl(chunk_file, INITIAL_CHUNKS)

    def _fake_resolve(chunks_dir, incremental):
        return [chunk_file]

    monkeypatch.setattr(mod, "_resolve_target_files", _fake_resolve)
    monkeypatch.setattr(mod, "_refresh_manifest_for_all_files", lambda *_: None)

    def _fake_load_ids(table):
        try:
            return {str(v) for v in table.to_pandas()["id"].tolist()}
        except Exception:
            return set()

    monkeypatch.setattr(mod, "_load_existing_ids", _fake_load_ids)

    class FakeSettings:
        LANCEDB_URI = temp_db.uri
        LANCEDB_TABLE = "test_update"
        EMBEDDING_MODEL = "fake"
        EMBEDDING_DIM = 1024

    monkeypatch.setattr(mod, "RAGSettings", FakeSettings)

    vec_fixed = np.full(1024, 0.5)

    class FakeEmbeddingService:
        def __init__(self, settings):
            pass

        def encode_documents(self, texts):
            return [vec_fixed.copy() for _ in texts]

    monkeypatch.setattr(mod, "EmbeddingService", FakeEmbeddingService)

    # First ingest
    mod.ingest_lancedb(incremental=True)
    table = temp_db.open_table("test_update")
    assert table.count_rows() == 3

    # Modify file: change X's text
    modified = [
        {"id": "X", "chunk_id": "X", "text": "Nội dung B", "source": "file1.jsonl", "metadata": {}},
        {"id": "Y", "chunk_id": "Y", "text": "Nội dung Y", "source": "file1.jsonl", "metadata": {}},
        {"id": "Z", "chunk_id": "Z", "text": "Nội dung Z", "source": "file1.jsonl", "metadata": {}},
    ]
    _write_jsonl(chunk_file, modified)

    # Re-ingest
    mod.ingest_lancedb(incremental=True)

    table2 = temp_db.open_table("test_update")
    count = table2.count_rows()
    assert count == 3, f"Expected 3 rows after update, got {count} — duplicate created"

    df = table2.to_pandas()
    x_rows = df[df["id"] == _row_id(mod, chunk_file, "X")]
    assert len(x_rows) == 1, f"Expected exactly 1 row with id={_row_id(mod, chunk_file, 'X')}, got {len(x_rows)}"
    assert x_rows.iloc[0]["text"] == "Nội dung B", f"Expected 'Nội dung B', got '{x_rows.iloc[0]['text']}'"


# ---------------------------------------------------------------------------
# Test 3 — orphan row deleted when missing from source (RED expected)
# ---------------------------------------------------------------------------

def test_orphan_row_deleted_when_missing_from_source(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Table has {X,Y,Z} from file1.jsonl → file now only has {X,Y} → re-ingest. Assert count==2 and Z gone.

    BUG B5: incremental mode never deletes orphan rows.
    Expected: RED — Z remains in table.
    """
    import rag2025.scripts.ingest_lancedb as mod

    fake_manifest = tmp_workspace / ".ingest_manifest3.json"
    monkeypatch.setattr(mod, "MANIFEST_PATH", fake_manifest)
    monkeypatch.setattr(mod, "TARGET_TABLE", "test_orphan")
    monkeypatch.setattr(mod, "_trigger_graph_sync", lambda *_: None)

    chunk_file = tmp_workspace / "chunked_orphan.jsonl"
    _write_jsonl(chunk_file, INITIAL_CHUNKS)

    def _fake_resolve(chunks_dir, incremental):
        return [chunk_file]

    monkeypatch.setattr(mod, "_resolve_target_files", _fake_resolve)
    monkeypatch.setattr(mod, "_refresh_manifest_for_all_files", lambda *_: None)

    def _fake_load_ids(table):
        try:
            return {str(v) for v in table.to_pandas()["id"].tolist()}
        except Exception:
            return set()

    monkeypatch.setattr(mod, "_load_existing_ids", _fake_load_ids)

    class FakeSettings:
        LANCEDB_URI = temp_db.uri
        LANCEDB_TABLE = "test_orphan"
        EMBEDDING_MODEL = "fake"
        EMBEDDING_DIM = 1024

    monkeypatch.setattr(mod, "RAGSettings", FakeSettings)

    vec_fixed = np.full(1024, 0.5)

    class FakeEmbeddingService:
        def __init__(self, settings):
            pass

        def encode_documents(self, texts):
            return [vec_fixed.copy() for _ in texts]

    monkeypatch.setattr(mod, "EmbeddingService", FakeEmbeddingService)

    # First ingest with all 3
    mod.ingest_lancedb(incremental=True)
    table = temp_db.open_table("test_orphan")
    assert table.count_rows() == 3

    # Remove Z from source file
    reduced = [
        {"id": "X", "chunk_id": "X", "text": "Nội dung A", "source": "file1.jsonl", "metadata": {}},
        {"id": "Y", "chunk_id": "Y", "text": "Nội dung Y", "source": "file1.jsonl", "metadata": {}},
    ]
    _write_jsonl(chunk_file, reduced)

    # Re-ingest
    mod.ingest_lancedb(incremental=True)

    table2 = temp_db.open_table("test_orphan")
    count = table2.count_rows()
    assert count == 2, f"Expected 2 rows after orphan removal, got {count}"

    ids = set(table2.to_pandas()["id"].tolist())
    assert _row_id(mod, chunk_file, "Z") not in ids, f"Orphan Z still present in table: {ids}"


# ---------------------------------------------------------------------------
# Test 4 — orphan delete does not touch surviving rows
# ---------------------------------------------------------------------------

def test_orphan_delete_does_not_touch_surviving_rows(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Same scenario as test 3. X/Y keep original text → verify their text unchanged.

    May PASS immediately if orphan deletion were implemented correctly.
    Regression guard.
    """
    import rag2025.scripts.ingest_lancedb as mod

    fake_manifest = tmp_workspace / ".ingest_manifest4.json"
    monkeypatch.setattr(mod, "MANIFEST_PATH", fake_manifest)
    monkeypatch.setattr(mod, "TARGET_TABLE", "test_survive")
    monkeypatch.setattr(mod, "_trigger_graph_sync", lambda *_: None)

    chunk_file = tmp_workspace / "chunked_survive.jsonl"
    _write_jsonl(chunk_file, INITIAL_CHUNKS)

    def _fake_resolve(chunks_dir, incremental):
        return [chunk_file]

    monkeypatch.setattr(mod, "_resolve_target_files", _fake_resolve)
    monkeypatch.setattr(mod, "_refresh_manifest_for_all_files", lambda *_: None)

    def _fake_load_ids(table):
        try:
            return {str(v) for v in table.to_pandas()["id"].tolist()}
        except Exception:
            return set()

    monkeypatch.setattr(mod, "_load_existing_ids", _fake_load_ids)

    class FakeSettings:
        LANCEDB_URI = temp_db.uri
        LANCEDB_TABLE = "test_survive"
        EMBEDDING_MODEL = "fake"
        EMBEDDING_DIM = 1024

    monkeypatch.setattr(mod, "RAGSettings", FakeSettings)

    vec_fixed = np.full(1024, 0.5)

    class FakeEmbeddingService:
        def __init__(self, settings):
            pass

        def encode_documents(self, texts):
            return [vec_fixed.copy() for _ in texts]

    monkeypatch.setattr(mod, "EmbeddingService", FakeEmbeddingService)

    mod.ingest_lancedb(incremental=True)

    # Reduce source: remove Z only
    reduced = [
        {"id": "X", "chunk_id": "X", "text": "Nội dung A", "source": "file1.jsonl", "metadata": {}},
        {"id": "Y", "chunk_id": "Y", "text": "Nội dung Y", "source": "file1.jsonl", "metadata": {}},
    ]
    _write_jsonl(chunk_file, reduced)

    mod.ingest_lancedb(incremental=True)

    table = temp_db.open_table("test_survive")
    df = table.to_pandas()
    x_row = df[df["id"] == _row_id(mod, chunk_file, "X")].iloc[0]
    y_row = df[df["id"] == _row_id(mod, chunk_file, "Y")].iloc[0]
    assert x_row["text"] == "Nội dung A", f"X text changed: {x_row['text']}"
    assert y_row["text"] == "Nội dung Y", f"Y text changed: {y_row['text']}"


# ---------------------------------------------------------------------------
# Test 5 — orphan delete scoped by source (PASS expected)
# ---------------------------------------------------------------------------

def test_orphan_delete_scoped_by_source_does_not_touch_other_file(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Have chunks from file1.jsonl AND file2.jsonl. Only file1 loses a chunk.
    Chunks from file2 remain intact, total count correct.

    Current code doesn't delete orphans at all, so this may PASS trivially.
    Regression guard for future correct implementation.
    """
    import rag2025.scripts.ingest_lancedb as mod

    fake_manifest = tmp_workspace / ".ingest_manifest5.json"
    monkeypatch.setattr(mod, "MANIFEST_PATH", fake_manifest)
    monkeypatch.setattr(mod, "TARGET_TABLE", "test_scoped")
    monkeypatch.setattr(mod, "_trigger_graph_sync", lambda *_: None)

    file1 = tmp_workspace / "chunked_file1.jsonl"
    file2 = tmp_workspace / "chunked_file2.jsonl"

    # Initial: 2 from file1, 1 from file2
    initial_f1 = [
        {"id": "F1A", "chunk_id": "F1A", "text": "File1 A", "source": "file1.jsonl", "metadata": {}},
        {"id": "F1B", "chunk_id": "F1B", "text": "File1 B", "source": "file1.jsonl", "metadata": {}},
    ]
    initial_f2 = [
        {"id": "F2A", "chunk_id": "F2A", "text": "File2 A", "source": "file2.jsonl", "metadata": {}},
    ]
    _write_jsonl(file1, initial_f1)
    _write_jsonl(file2, initial_f2)

    def _fake_resolve(chunks_dir, incremental):
        return sorted([file1, file2])

    monkeypatch.setattr(mod, "_resolve_target_files", _fake_resolve)
    monkeypatch.setattr(mod, "_refresh_manifest_for_all_files", lambda *_: None)

    def _fake_load_ids(table):
        try:
            return {str(v) for v in table.to_pandas()["id"].tolist()}
        except Exception:
            return set()

    monkeypatch.setattr(mod, "_load_existing_ids", _fake_load_ids)

    class FakeSettings:
        LANCEDB_URI = temp_db.uri
        LANCEDB_TABLE = "test_scoped"
        EMBEDDING_MODEL = "fake"
        EMBEDDING_DIM = 1024

    monkeypatch.setattr(mod, "RAGSettings", FakeSettings)

    vec_fixed = np.full(1024, 0.5)

    class FakeEmbeddingService:
        def __init__(self, settings):
            pass

        def encode_documents(self, texts):
            return [vec_fixed.copy() for _ in texts]

    monkeypatch.setattr(mod, "EmbeddingService", FakeEmbeddingService)

    mod.ingest_lancedb(incremental=True)
    table = temp_db.open_table("test_scoped")
    assert table.count_rows() == 3

    # Remove F1B from file1 only
    reduced_f1 = [
        {"id": "F1A", "chunk_id": "F1A", "text": "File1 A", "source": "file1.jsonl", "metadata": {}},
    ]
    _write_jsonl(file1, reduced_f1)

    mod.ingest_lancedb(incremental=True)

    table2 = temp_db.open_table("test_scoped")
    df = table2.to_pandas()
    f2a_rows = df[df["id"] == _row_id(mod, file2, "F2A")]
    assert len(f2a_rows) == 1, f"F2A should survive, found {len(f2a_rows)} rows"


# ---------------------------------------------------------------------------
# Test 6 — vector updated when content changes (RED expected)
# ---------------------------------------------------------------------------

def test_vector_updated_when_content_changes(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Two different fake embeddings: [0.1]*1024 for "Nội dung A", [0.9]*1024 for "Nội dung B".
    After updating chunk X's text and re-ingesting, vector for X must differ.

    BUG B5: incremental mode appends new record with new vector but keeps old row.
    So either there are 2 rows for X (duplicate), or if update worked the vector would change.
    Either way the assertion fails on current code.
    """
    import rag2025.scripts.ingest_lancedb as mod

    fake_manifest = tmp_workspace / ".ingest_manifest6.json"
    monkeypatch.setattr(mod, "MANIFEST_PATH", fake_manifest)
    monkeypatch.setattr(mod, "TARGET_TABLE", "test_vec")
    monkeypatch.setattr(mod, "_trigger_graph_sync", lambda *_: None)

    chunk_file = tmp_workspace / "chunked_vec.jsonl"
    _write_jsonl(chunk_file, INITIAL_CHUNKS)

    def _fake_resolve(chunks_dir, incremental):
        return [chunk_file]

    monkeypatch.setattr(mod, "_resolve_target_files", _fake_resolve)
    monkeypatch.setattr(mod, "_refresh_manifest_for_all_files", lambda *_: None)

    def _fake_load_ids(table):
        try:
            return {str(v) for v in table.to_pandas()["id"].tolist()}
        except Exception:
            return set()

    monkeypatch.setattr(mod, "_load_existing_ids", _fake_load_ids)

    class FakeSettings:
        LANCEDB_URI = temp_db.uri
        LANCEDB_TABLE = "test_vec"
        EMBEDDING_MODEL = "fake"
        EMBEDDING_DIM = 1024

    monkeypatch.setattr(mod, "RAGSettings", FakeSettings)

    vec_a = np.full(1024, 0.1)
    vec_b = np.full(1024, 0.9)

    class FakeEmbeddingService:
        def __init__(self, settings):
            pass

        def encode_documents(self, texts):
            result = []
            for t in texts:
                if "Nội dung A" in t:
                    result.append(vec_a.copy())
                else:
                    result.append(vec_b.copy())
            return result

    monkeypatch.setattr(mod, "EmbeddingService", FakeEmbeddingService)

    # First ingest
    mod.ingest_lancedb(incremental=True)
    table = temp_db.open_table("test_vec")
    df_before = table.to_pandas()
    x_before = df_before[df_before["id"] == _row_id(mod, chunk_file, "X")].iloc[0]
    vec_before = np.array(x_before["vector"])

    # Change X's text
    modified = [
        {"id": "X", "chunk_id": "X", "text": "Nội dung B", "source": "file1.jsonl", "metadata": {}},
        {"id": "Y", "chunk_id": "Y", "text": "Nội dung Y", "source": "file1.jsonl", "metadata": {}},
        {"id": "Z", "chunk_id": "Z", "text": "Nội dung Z", "source": "file1.jsonl", "metadata": {}},
    ]
    _write_jsonl(chunk_file, modified)

    mod.ingest_lancedb(incremental=True)

    table2 = temp_db.open_table("test_vec")
    df_after = table2.to_pandas()
    x_rows = df_after[df_after["id"] == _row_id(mod, chunk_file, "X")]

    # Must be exactly 1 row for X (no duplicate)
    assert len(x_rows) == 1, f"Expected 1 row for X, got {len(x_rows)} — duplicate created"

    vec_after = np.array(x_rows.iloc[0]["vector"])
    assert not np.allclose(vec_before, vec_after), (
        f"Vector for X did not change after content update. "
        f"Before sum={vec_before.sum():.1f}, after sum={vec_after.sum():.1f}"
    )


# ---------------------------------------------------------------------------
# Test 7 — T1.2 cross-file text dedup (44.6% dup fix)
# ---------------------------------------------------------------------------

def test_cross_file_text_dedup_keeps_single_canonical(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Two files with the same normalized text (case/whitespace differ) under
    different ids → only the canonical row (first by ingest_file) is stored."""
    import rag2025.scripts.ingest_lancedb as mod

    fake_manifest = tmp_workspace / ".ingest_manifest_dd.json"
    monkeypatch.setattr(mod, "MANIFEST_PATH", fake_manifest)
    monkeypatch.setattr(mod, "TARGET_TABLE", "test_dedup")
    monkeypatch.setattr(mod, "_trigger_graph_sync", lambda *_: None)

    file1 = tmp_workspace / "chunked_dd1.jsonl"
    file2 = tmp_workspace / "chunked_dd2.jsonl"
    _write_jsonl(file1, [
        {"id": "a1", "chunk_id": "a1", "text": "  Nội Dung Tuyển Sinh 2026  ", "metadata": {}},
    ])
    _write_jsonl(file2, [
        {"id": "b1", "chunk_id": "b1", "text": "nội dung tuyển sinh 2026", "metadata": {}},
    ])

    def _fake_resolve(chunks_dir, incremental):
        return [file1, file2]

    monkeypatch.setattr(mod, "_resolve_target_files", _fake_resolve)
    monkeypatch.setattr(mod, "_refresh_manifest_for_all_files", lambda *_: None)

    def _fake_load_ids(table):
        try:
            return {str(v) for v in table.to_pandas()["id"].tolist()}
        except Exception:
            return set()

    monkeypatch.setattr(mod, "_load_existing_ids", _fake_load_ids)

    class FakeSettings:
        LANCEDB_URI = temp_db.uri
        LANCEDB_TABLE = "test_dedup"
        EMBEDDING_MODEL = "fake"
        EMBEDDING_DIM = 1024

    monkeypatch.setattr(mod, "RAGSettings", FakeSettings)

    vec_fixed = np.full(1024, 0.5)

    class FakeEmbeddingService:
        def __init__(self, settings):
            pass

        def encode_documents(self, texts):
            return [vec_fixed.copy() for _ in texts]

    monkeypatch.setattr(mod, "EmbeddingService", FakeEmbeddingService)

    mod.ingest_lancedb(incremental=True)
    table = temp_db.open_table("test_dedup")
    assert table.count_rows() == 1, (
        f"Expected 1 row after cross-file text dedup, got {table.count_rows()}"
    )
    df = table.to_pandas()
    meta = json.loads(df.iloc[0]["metadata_json"])
    assert meta.get("ingest_file") == "chunked_dd1.jsonl", (
        "canonical row should be first by ingest_file (chunked_dd1.jsonl)"
    )


# ---------------------------------------------------------------------------
# Test 8 — T1.5 globally-unique ids (G9 collision fix)
# ---------------------------------------------------------------------------

def test_unique_ids_across_documents(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Same numeric chunk_id ('13') in two different files → distinct row ids.

    Without the fix, both rows share id='13' and the last write silently
    overwrites the first (collision). After the fix each row id is a hash of
    source|ingest_file|chunk_id → both survive.
    """
    import rag2025.scripts.ingest_lancedb as mod

    fake_manifest = tmp_workspace / ".ingest_manifest_uid.json"
    monkeypatch.setattr(mod, "MANIFEST_PATH", fake_manifest)
    monkeypatch.setattr(mod, "TARGET_TABLE", "test_uid")
    monkeypatch.setattr(mod, "_trigger_graph_sync", lambda *_: None)

    file1 = tmp_workspace / "chunked_uid1.jsonl"
    file2 = tmp_workspace / "chunked_uid2.jsonl"
    _write_jsonl(file1, [
        {"id": "13", "chunk_id": "13", "text": "Doc A chunk 13", "metadata": {}},
    ])
    _write_jsonl(file2, [
        {"id": "13", "chunk_id": "13", "text": "Doc B chunk 13", "metadata": {}},
    ])

    def _fake_resolve(chunks_dir, incremental):
        return [file1, file2]

    monkeypatch.setattr(mod, "_resolve_target_files", _fake_resolve)
    monkeypatch.setattr(mod, "_refresh_manifest_for_all_files", lambda *_: None)

    def _fake_load_ids(table):
        try:
            return {str(v) for v in table.to_pandas()["id"].tolist()}
        except Exception:
            return set()

    monkeypatch.setattr(mod, "_load_existing_ids", _fake_load_ids)

    class FakeSettings:
        LANCEDB_URI = temp_db.uri
        LANCEDB_TABLE = "test_uid"
        EMBEDDING_MODEL = "fake"
        EMBEDDING_DIM = 1024

    monkeypatch.setattr(mod, "RAGSettings", FakeSettings)

    vec_fixed = np.full(1024, 0.5)

    class FakeEmbeddingService:
        def __init__(self, settings):
            pass

        def encode_documents(self, texts):
            return [vec_fixed.copy() for _ in texts]

    monkeypatch.setattr(mod, "EmbeddingService", FakeEmbeddingService)

    mod.ingest_lancedb(incremental=True)
    table = temp_db.open_table("test_uid")
    assert table.count_rows() == 2, f"Expected 2 rows, got {table.count_rows()}"
    ids = {r["id"] for r in table.to_arrow().select(["id"]).to_pylist()}
    assert len(ids) == 2, f"id collision — only {len(ids)} distinct ids for 2 rows: {ids}"
    assert not ({"13"} & ids), "raw numeric chunk_id must not be used as row id"
    assert ids == {
        _row_id(mod, file1, "13"),
        _row_id(mod, file2, "13"),
    }, f"unexpected id scheme: {ids}"

# ---------------------------------------------------------------------------
# Test 9 — T1.2 dedup is scoped per canonical record (ticket 34)
# ---------------------------------------------------------------------------

def test_distinct_canonical_records_with_identical_text_both_survive(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Two DIFFERENT canonical records carrying byte-identical text must both
    survive ingest, each with its own `metadata_json.record_version_key`.

    Ticket 06's real run lost 37/158 chunks exactly this way: an atom like
    "mã ngành: 7420201" recurs in every record naming that major, and the
    text-only dedup collapsed them into one — deleting the other record's
    chunk_id + lineage so ticket 10 could not cite it.

    RED pre-fix: the two rows collapse to 1 and one record_version_key is gone.
    """
    import rag2025.scripts.ingest_lancedb as mod

    fake_manifest = tmp_workspace / ".ingest_manifest_rvk.json"
    monkeypatch.setattr(mod, "MANIFEST_PATH", fake_manifest)
    monkeypatch.setattr(mod, "TARGET_TABLE", "test_rvk")
    monkeypatch.setattr(mod, "_trigger_graph_sync", lambda *_: None)

    rvk_a = "CHI_TIEU:nganh:7220104:2026:canon-lietke29@1"
    rvk_b = "CHI_TIEU:nganh:7420201:2026:canon-lietke30@1"
    shared_text = "mã ngành: 7420201"

    chunk_file = tmp_workspace / "chunked_canonical_rvk.jsonl"
    _write_jsonl(chunk_file, [
        {
            "id": "a-atom",
            "chunk_id": "a-atom",
            "text": shared_text,
            "metadata": {
                "record_version_key": rvk_a,
                "record_id": "rec-a",
                "release_id": "R-2026-09-01",
            },
        },
        {
            "id": "b-atom",
            "chunk_id": "b-atom",
            "text": shared_text,
            "metadata": {
                "record_version_key": rvk_b,
                "record_id": "rec-b",
                "release_id": "R-2026-09-01",
            },
        },
    ])

    def _fake_resolve(chunks_dir, incremental):
        return [chunk_file]

    monkeypatch.setattr(mod, "_resolve_target_files", _fake_resolve)
    monkeypatch.setattr(mod, "_refresh_manifest_for_all_files", lambda *_: None)

    class FakeSettings:
        LANCEDB_URI = temp_db.uri
        LANCEDB_TABLE = "test_rvk"
        EMBEDDING_MODEL = "fake"
        EMBEDDING_DIM = 1024

    monkeypatch.setattr(mod, "RAGSettings", FakeSettings)

    vec_fixed = np.full(1024, 0.5)

    class FakeEmbeddingService:
        def __init__(self, settings):
            pass

        def encode_documents(self, texts):
            return [vec_fixed.copy() for _ in texts]

    monkeypatch.setattr(mod, "EmbeddingService", FakeEmbeddingService)

    mod.ingest_lancedb(incremental=True)
    table = temp_db.open_table("test_rvk")
    assert table.count_rows() == 2, (
        f"both canonical records must survive identical text; got "
        f"{table.count_rows()} row(s)"
    )
    rows = table.to_arrow().to_pylist()
    rvks = {json.loads(r["metadata_json"])["record_version_key"] for r in rows}
    assert rvks == {rvk_a, rvk_b}, (
        f"each surviving row must keep its OWN record_version_key: {rvks}"
    )

# ---------------------------------------------------------------------------
# Test 9b — T1.2 dedup is scoped per semantic unit (ticket 06 identity)
# ---------------------------------------------------------------------------


def test_two_units_of_one_record_with_identical_text_both_survive(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Two units of the SAME record carrying byte-identical text must both
    survive ingest, each with its own `metadata_json.unit_key`.

    Test 9 scopes the dedup key to the record, which fixes the cross-record
    case but leaves the same collapse one level down. A record that lists two
    majors, or two years, holds several facts that legitimately repeat a value
    — "mã ngành: 7420201" appears as its own citable unit under each — and
    `canonical_chunk_builder` gives each unit its own `chunk_id`
    (``sha256(record_version_key + unit_key)``) precisely so they are separately
    citable. A key of ``(record_version_key, text)`` throws that away: the
    second unit's chunk_id and lineage vanish, and ticket 10 cannot cite it.

    RED pre-fix: the two rows collapse to 1 and one unit_key is gone.
    """
    import rag2025.scripts.ingest_lancedb as mod

    fake_manifest = tmp_workspace / ".ingest_manifest_unit.json"
    monkeypatch.setattr(mod, "MANIFEST_PATH", fake_manifest)
    monkeypatch.setattr(mod, "TARGET_TABLE", "test_unit")
    monkeypatch.setattr(mod, "_trigger_graph_sync", lambda *_: None)

    # One record version, two semantic units, one repeated value.
    rvk = "CHI_TIEU:nganh:7420201:2026:canon-lietke29@1"
    shared_text = "mã ngành: 7420201"

    chunk_file = tmp_workspace / "chunked_canonical_unit.jsonl"
    _write_jsonl(chunk_file, [
        {
            "id": "unit-a",
            "chunk_id": "unit-a",
            "text": shared_text,
            "metadata": {
                "record_version_key": rvk,
                "unit_key": "CHI_TIEU|ma_nganh|0",
                "record_id": "rec-unit",
                "release_id": "R-2026-09-01",
            },
        },
        {
            "id": "unit-b",
            "chunk_id": "unit-b",
            "text": shared_text,
            "metadata": {
                "record_version_key": rvk,
                "unit_key": "CHI_TIEU|ma_nganh|1",
                "record_id": "rec-unit",
                "release_id": "R-2026-09-01",
            },
        },
    ])

    def _fake_resolve(chunks_dir, incremental):
        return [chunk_file]

    monkeypatch.setattr(mod, "_resolve_target_files", _fake_resolve)
    monkeypatch.setattr(mod, "_refresh_manifest_for_all_files", lambda *_: None)

    class FakeSettings:
        LANCEDB_URI = temp_db.uri
        LANCEDB_TABLE = "test_unit"
        EMBEDDING_MODEL = "fake"
        EMBEDDING_DIM = 1024

    monkeypatch.setattr(mod, "RAGSettings", FakeSettings)

    vec_fixed = np.full(1024, 0.5)

    class FakeEmbeddingService:
        def __init__(self, settings):
            pass

        def encode_documents(self, texts):
            return [vec_fixed.copy() for _ in texts]

    monkeypatch.setattr(mod, "EmbeddingService", FakeEmbeddingService)

    mod.ingest_lancedb(incremental=True)
    table = temp_db.open_table("test_unit")
    assert table.count_rows() == 2, (
        f"both units of one record must survive identical text; got "
        f"{table.count_rows()} row(s)"
    )
    rows = table.to_arrow().to_pylist()
    unit_keys = {json.loads(r["metadata_json"])["unit_key"] for r in rows}
    assert unit_keys == {"CHI_TIEU|ma_nganh|0", "CHI_TIEU|ma_nganh|1"}, (
        f"each surviving row must keep its OWN unit_key: {unit_keys}"
    )


# ---------------------------------------------------------------------------
# Test 10 — top-level release_id agrees with metadata_json (ticket 34)
# ---------------------------------------------------------------------------

def test_release_id_column_agrees_with_metadata_and_falls_back_to_legacy(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """The top-level `release_id` column must be the chunk's REAL release id,
    matching `metadata_json.release_id`; only a chunk with no release metadata
    falls back to "legacy".

    Ticket 07 records the served corpus's release id from this column, so a
    hardcoded "legacy" made its acceptance item false on every canonical chunk.
    """
    import rag2025.scripts.ingest_lancedb as mod

    fake_manifest = tmp_workspace / ".ingest_manifest_rel.json"
    monkeypatch.setattr(mod, "MANIFEST_PATH", fake_manifest)
    monkeypatch.setattr(mod, "TARGET_TABLE", "test_release")
    monkeypatch.setattr(mod, "_trigger_graph_sync", lambda *_: None)

    chunk_file = tmp_workspace / "chunked_release.jsonl"
    _write_jsonl(chunk_file, [
        {
            "id": "with-rel",
            "chunk_id": "with-rel",
            "text": "Học phí năm 2026",
            "metadata": {"release_id": "R-2026-09-01"},
        },
        {
            "id": "no-rel",
            "chunk_id": "no-rel",
            "text": "Điểm chuẩn năm 2026",
            "metadata": {},
        },
    ])

    def _fake_resolve(chunks_dir, incremental):
        return [chunk_file]

    monkeypatch.setattr(mod, "_resolve_target_files", _fake_resolve)
    monkeypatch.setattr(mod, "_refresh_manifest_for_all_files", lambda *_: None)

    class FakeSettings:
        LANCEDB_URI = temp_db.uri
        LANCEDB_TABLE = "test_release"
        EMBEDDING_MODEL = "fake"
        EMBEDDING_DIM = 1024

    monkeypatch.setattr(mod, "RAGSettings", FakeSettings)

    vec_fixed = np.full(1024, 0.5)

    class FakeEmbeddingService:
        def __init__(self, settings):
            pass

        def encode_documents(self, texts):
            return [vec_fixed.copy() for _ in texts]

    monkeypatch.setattr(mod, "EmbeddingService", FakeEmbeddingService)

    mod.ingest_lancedb(incremental=True)
    table = temp_db.open_table("test_release")
    rows = {r["doc_id"]: r for r in table.to_arrow().to_pylist()}
    assert set(rows) == {"with-rel", "no-rel"}

    # Half 1 — a chunk carrying release metadata stamps it in BOTH places.
    with_rel = rows["with-rel"]
    assert with_rel["release_id"] == "R-2026-09-01", (
        f"top-level column must carry the real release id, got "
        f"{with_rel['release_id']!r}"
    )
    assert json.loads(with_rel["metadata_json"])["release_id"] == "R-2026-09-01"
    assert with_rel["release_id"] == json.loads(with_rel["metadata_json"])["release_id"]

    # Half 2 — a chunk with no release metadata still yields "legacy".
    no_rel = rows["no-rel"]
    assert no_rel["release_id"] == "legacy", (
        f"absent release metadata must fall back to 'legacy', got "
        f"{no_rel['release_id']!r}"
    )
