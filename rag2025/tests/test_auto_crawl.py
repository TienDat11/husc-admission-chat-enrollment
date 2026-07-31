"""Tests for auto-crawl pipeline (Phase E)."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.infrastructure.auto_crawl import (
    CrawlerState,
    Checkpoint,
    IngestPipeline,
    PipelineStepError,
    check_supersession,
    compute_hash,
    hash_dedup,
    load_hash_log,
    save_hash_log,
    snapshot_before_ingest,
    rollback_on_failure,
    resume_pipeline,
    create_scheduled_task,
)


# --- CrawlerState ---


def test_crawler_state_defaults():
    state = CrawlerState()
    assert state.max_id == 0
    assert state.year_partition == "2026"
    assert state.hash_log == {}


def test_crawler_state_save_load(tmp_path):
    path = str(tmp_path / "crawler_state.json")
    state = CrawlerState(max_id=42, last_crawl="2026-05-18T00:00:00Z", hash_log={"abc": "url1"})
    state.save(path)

    loaded = CrawlerState.load(path)
    assert loaded.max_id == 42
    assert loaded.last_crawl == "2026-05-18T00:00:00Z"
    assert loaded.hash_log == {"abc": "url1"}


def test_crawler_state_load_missing():
    state = CrawlerState.load("/nonexistent/path.json")
    assert state.max_id == 0


def test_crawler_state_load_corrupted(tmp_path):
    path = str(tmp_path / "bad.json")
    Path(path).write_text("NOT JSON", encoding="utf-8")
    state = CrawlerState.load(path)
    assert state.max_id == 0


# --- Checkpoint ---


def test_checkpoint_save_load(tmp_path):
    path = str(tmp_path / "checkpoint.json")
    ckpt = Checkpoint(run_id="test_run", last_success_step=3, failed_step=0)
    ckpt.save(path)

    loaded = Checkpoint.load(path)
    assert loaded is not None
    assert loaded.run_id == "test_run"
    assert loaded.last_success_step == 3


def test_checkpoint_load_missing():
    assert Checkpoint.load("/nonexistent/path.json") is None


def test_checkpoint_clear(tmp_path):
    path = str(tmp_path / "checkpoint.json")
    ckpt = Checkpoint(run_id="test")
    ckpt.save(path)
    assert Path(path).exists()

    ckpt.clear(path)
    assert not Path(path).exists()


# --- Hash Dedup ---


def test_compute_hash_deterministic():
    h1 = compute_hash("hello world")
    h2 = compute_hash("hello world")
    assert h1 == h2
    assert len(h1) == 64  # SHA256 hex length


def test_compute_hash_different_content():
    h1 = compute_hash("hello")
    h2 = compute_hash("world")
    assert h1 != h2


def test_hash_dedup_new_content():
    log = {}
    assert hash_dedup("new content", "http://example.com/1", log) is True
    assert "new content" in [v for v in log.values()] or compute_hash("new content") in log


def test_hash_dedup_duplicate():
    log = {}
    hash_dedup("content A", "http://example.com/1", log)
    assert hash_dedup("content A", "http://example.com/2", log) is False


def test_hash_log_save_load(tmp_path):
    path = str(tmp_path / "hash_log.json")
    log = {"hash1": "url1", "hash2": "url2"}
    save_hash_log(log, path)

    loaded = load_hash_log(path)
    assert loaded == log


# --- Supersession ---


def test_check_supersession_marks_old():
    new_meta = {"section": "hoc_phi", "data_year": 2026, "chunk_id": "new_1"}
    existing = [
        {"section": "hoc_phi", "data_year": 2026, "version": 1, "is_superseded": False},
        {"section": "hoc_phi", "data_year": 2025, "version": 1, "is_superseded": False},
    ]

    result = check_supersession(new_meta, existing)
    # Same section + same year → superseded
    assert result[0]["is_superseded"] is True
    assert result[0]["superseded_by"] == "new_1"
    # Different year → not superseded
    assert result[1]["is_superseded"] is False
    # New chunk version incremented
    assert new_meta["version"] == 2


def test_check_supersession_no_match():
    new_meta = {"section": "diem_chuan", "data_year": 2026, "chunk_id": "new_2"}
    existing = [
        {"section": "hoc_phi", "data_year": 2026, "version": 1},
    ]
    result = check_supersession(new_meta, existing)
    assert not result[0].get("is_superseded", False)


# --- Snapshot / Rollback ---


def test_snapshot_creates_dir(tmp_path):
    with patch("src.infrastructure.auto_crawl.DATA_DIRS", [str(tmp_path / "chunked")]):
        chunked = tmp_path / "chunked"
        chunked.mkdir()
        (chunked / "test.jsonl").write_text("test", encoding="utf-8")

        snap = snapshot_before_ingest("test_run")
        assert Path(snap).exists()
        assert (Path(snap) / "manifest.json").exists()


def test_rollback_restores(tmp_path):
    # Setup: create source dir with content
    src = tmp_path / "chunked"
    src.mkdir()
    (src / "data.jsonl").write_text('{"test": true}', encoding="utf-8")

    # Create snapshot
    snap_dir = tmp_path / "snapshots" / "rollback_test"
    snap_dir.mkdir(parents=True)
    snap_chunked = snap_dir / "chunked"
    snap_chunked.mkdir()
    (snap_chunked / "data.jsonl").write_text('{"test": true}', encoding="utf-8")

    manifest = {"run_id": "rollback_test", "targets": [str(src)]}
    (snap_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Corrupt source
    (src / "data.jsonl").write_text("CORRUPTED", encoding="utf-8")

    # Rollback
    with patch("src.infrastructure.auto_crawl.DATA_DIRS", [str(src)]):
        rollback_on_failure(str(snap_dir))

    # Verify restored
    content = (src / "data.jsonl").read_text(encoding="utf-8")
    assert content == '{"test": true}'


# --- Pipeline ---


def test_pipeline_empty_run():
    pipeline = IngestPipeline(CrawlerState())
    result = pipeline.run()

    assert result["status"] == "success"
    assert result["chunks_produced"] == 0


def test_pipeline_with_mock_crawl():
    pages = [("http://example.com/1", "new content about HUSC")]

    def mock_crawl():
        return pages

    def mock_chunk(content, meta):
        return [{"id": "c1", "text": content, "metadata": meta}]

    def mock_embed(chunks):
        return len(chunks)

    def mock_graph(chunks):
        return len(chunks)

    pipeline = IngestPipeline(CrawlerState())
    pipeline._hash_log = {}  # Isolate from disk state
    result = pipeline.run(
        crawl_fn=mock_crawl,
        chunk_fn=mock_chunk,
        embed_fn=mock_embed,
        graph_fn=mock_graph,
    )

    assert result["status"] == "success"
    assert result["chunks_produced"] == 1
    assert "crawl" in result["steps"]


def test_pipeline_dedup_skips():
    content = "duplicate content"
    pages = [
        ("http://example.com/1", content),
        ("http://example.com/2", content),  # Same content, different URL
    ]

    def mock_crawl():
        return pages

    pipeline = IngestPipeline(CrawlerState())
    pipeline._hash_log = {}  # Isolate from disk state
    result = pipeline.run(crawl_fn=mock_crawl)

    assert result["skipped_duplicates"] == 1


def test_pipeline_nonempty_crawl_requires_chunk_and_embed_callbacks():
    pipeline = IngestPipeline(CrawlerState())
    pipeline._hash_log = {}

    result = pipeline.run(crawl_fn=lambda: [("http://example.com/1", "new content")])

    assert result["status"] == "failed at step 3"
    assert "chunk_fn" in result["error"]


def test_pipeline_chunks_require_embed_callback():
    pipeline = IngestPipeline(CrawlerState())
    pipeline._hash_log = {}

    result = pipeline.run(
        crawl_fn=lambda: [("http://example.com/1", "new content")],
        chunk_fn=lambda content, meta: [{"id": "c1", "text": content, "metadata": meta}],
    )

    assert result["status"] == "failed at step 6"
    assert "embed_fn" in result["error"]


def test_pipeline_chunks_require_embed_callback_before_supersession(monkeypatch):
    pipeline = IngestPipeline(CrawlerState())
    pipeline._hash_log = {}
    supersession_called = False

    def mark_supersession(chunks):
        nonlocal supersession_called
        supersession_called = True
        return chunks

    monkeypatch.setattr(pipeline, "_step_supersession", mark_supersession)
    result = pipeline.run(
        crawl_fn=lambda: [("http://example.com/1", "new content")],
        chunk_fn=lambda content, meta: [{"id": "c1", "text": content, "metadata": meta}],
    )

    assert result["status"] == "failed at step 6"
    assert supersession_called is False


def test_pipeline_step_error():
    def failing_crawl():
        raise ConnectionError("Network down")

    pipeline = IngestPipeline(CrawlerState())
    result = pipeline.run(crawl_fn=failing_crawl)

    assert "failed" in result["status"]
    assert result.get("error") is not None


def test_pipeline_metadata_assigned():
    def mock_crawl():
        return [("http://example.com/1", "test content")]

    def mock_chunk(content, meta):
        return [{"id": "c1", "text": content, "metadata": meta}]

    def mock_embed(chunks):
        return len(chunks)

    pipeline = IngestPipeline(CrawlerState())
    result = pipeline.run(crawl_fn=mock_crawl, chunk_fn=mock_chunk, embed_fn=mock_embed)

    assert result["status"] == "success"
    assert "metadata" in result["steps"]


def test_pipeline_step_error_class():
    err = PipelineStepError(3, "chunking failed")
    assert err.step == 3
    assert "Step 3 failed" in str(err)


# --- Resume ---


def test_resume_no_checkpoint():
    with patch("src.infrastructure.auto_crawl.Checkpoint.load", return_value=None):
        result = resume_pipeline()
    assert result is None


# --- Scheduler ---


def test_create_scheduled_task_returns_bool():
    # On non-Windows or without admin rights, should return False gracefully
    result = create_scheduled_task("TestTask")
    assert isinstance(result, bool)
