"""Tests for snapshot_legacy_chunks (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import importlib.util
import json
import time
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "snapshot_legacy_chunks.py"


@pytest.fixture
def snapshot_module():
    spec = importlib.util.spec_from_file_location("snapshot_legacy_chunks", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def populated_source(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "chunked_1.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
    (src / "chunked_2.jsonl").write_text('{"id":"b"}\n', encoding="utf-8")
    (src / ".ingest_manifest.json").write_text('{"chunked_1.jsonl":"hash1"}', encoding="utf-8")
    return src


def test_snapshot_creates_timestamped_dir(snapshot_module, populated_source, tmp_path):
    target = tmp_path / "tgt"
    out = snapshot_module.snapshot_chunks(populated_source, target, "20260525T120000Z")
    assert out == target / "20260525T120000Z"
    assert out.exists() and out.is_dir()


def test_snapshot_copies_all_jsonl_files(snapshot_module, populated_source, tmp_path):
    out = snapshot_module.snapshot_chunks(populated_source, tmp_path / "tgt", "20260525T120000Z")
    assert (out / "chunked_1.jsonl").exists()
    assert (out / "chunked_2.jsonl").exists()


def test_snapshot_copies_manifest(snapshot_module, populated_source, tmp_path):
    out = snapshot_module.snapshot_chunks(populated_source, tmp_path / "tgt", "20260525T120000Z")
    assert (out / ".ingest_manifest.json").exists()


def test_snapshot_writes_meta_json(snapshot_module, populated_source, tmp_path):
    out = snapshot_module.snapshot_chunks(populated_source, tmp_path / "tgt", "20260525T120000Z")
    meta_path = out / "_snapshot_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["file_count"] == 3
    assert meta["total_bytes"] > 0
    assert isinstance(meta["files"], list)
    assert all("sha256" in f and len(f["sha256"]) == 64 for f in meta["files"])


def test_snapshot_idempotent_unique_timestamps(snapshot_module, populated_source, tmp_path):
    out1 = snapshot_module.snapshot_chunks(populated_source, tmp_path / "tgt", None)
    time.sleep(1.1)
    out2 = snapshot_module.snapshot_chunks(populated_source, tmp_path / "tgt", None)
    assert out1.name != out2.name
    assert out1.exists() and out2.exists()


def test_snapshot_fails_on_existing_dir(snapshot_module, populated_source, tmp_path):
    snapshot_module.snapshot_chunks(populated_source, tmp_path / "tgt", "20260525T120000Z")
    with pytest.raises(FileExistsError):
        snapshot_module.snapshot_chunks(populated_source, tmp_path / "tgt", "20260525T120000Z")


def test_snapshot_handles_missing_manifest(snapshot_module, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "chunked_1.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
    out = snapshot_module.snapshot_chunks(src, tmp_path / "tgt", "20260525T120000Z")
    assert (out / "chunked_1.jsonl").exists()
    assert not (out / ".ingest_manifest.json").exists()
    meta = json.loads((out / "_snapshot_meta.json").read_text(encoding="utf-8"))
    assert meta["file_count"] == 1
