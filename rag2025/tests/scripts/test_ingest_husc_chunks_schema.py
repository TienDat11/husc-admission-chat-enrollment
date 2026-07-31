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

    # Core identity
    assert row["id"] == "c1"
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
