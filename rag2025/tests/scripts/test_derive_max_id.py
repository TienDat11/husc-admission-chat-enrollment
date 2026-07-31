"""TDD: derive_max_id from LanceDB husc_chunks (M3-N1a).

Self-healing crawl state: read max thongbao-id from DB instead of trusting
crawler_state.json (which has max_id=0, F7).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import lancedb
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def temp_db():
    d = tempfile.mkdtemp(dir="D:/tmp", prefix="test_derive_max_")
    db = lancedb.connect(d)
    yield db
    try:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Test 1 — union of id-pattern and metadata notification_id
# ---------------------------------------------------------------------------


def test_derive_max_id_union_of_id_and_metadata(temp_db: Any):
    """Rows with id-pattern max=73 AND metadata notification_id=74 → return 74."""
    from rag2025.scripts.crawl_incremental import derive_max_id

    temp_db.create_table("husc_chunks", data=[
        {"id": "husc_thongbao_id40_2026", "metadata_json": "{}", "vector": [0.1] * 1024},
        {"id": "husc_thongbao_id73_2026", "metadata_json": "{}", "vector": [0.1] * 1024},
        {"id": "other_chunk", "metadata_json": json.dumps({"notification_id": 74}), "vector": [0.1] * 1024},
        {"id": "chunked_1_x", "metadata_json": "{}", "vector": [0.1] * 1024},  # noise, no match
    ])

    result = derive_max_id(temp_db.uri, "husc_chunks")
    assert result == 74


# ---------------------------------------------------------------------------
# Test 2 — empty table returns 0
# ---------------------------------------------------------------------------


def test_derive_max_id_empty_table_returns_zero(temp_db: Any):
    """Empty table → 0."""
    from rag2025.scripts.crawl_incremental import derive_max_id

    temp_db.create_table("husc_chunks", data=[
        {"id": "x", "metadata_json": "{}", "vector": [0.1] * 1024},
    ])
    # Delete all rows
    table = temp_db.open_table("husc_chunks")
    table.delete("id = 'x'")

    result = derive_max_id(temp_db.uri, "husc_chunks")
    assert result == 0


# ---------------------------------------------------------------------------
# Test 3 — missing table returns 0, no raise
# ---------------------------------------------------------------------------


def test_derive_max_id_missing_table_returns_zero(temp_db: Any):
    """Table doesn't exist → 0, no exception."""
    from rag2025.scripts.crawl_incremental import derive_max_id

    result = derive_max_id(temp_db.uri, "nonexistent_table")
    assert result == 0


# ---------------------------------------------------------------------------
# Test 4 — crawl_fn starts after derived max
# ---------------------------------------------------------------------------


def test_crawl_fn_starts_after_derived_max(temp_db: Any, monkeypatch: Any):
    """crawl_fn must NOT call httpx.get for id <= derived_max."""
    import rag2025.scripts.crawl_incremental as mod

    # Patch derive_max_id to return 74
    monkeypatch.setattr(mod, "derive_max_id", lambda uri, table="husc_chunks": 74)

    # Patch CrawlerState.load to return max_id=0 (simulating F7)
    class FakeState:
        max_id = 0
        last_crawl = None
        hash_log = {}

        @staticmethod
        def load():
            return FakeState()

        def save(self):
            pass

    monkeypatch.setattr(mod, "CrawlerState", FakeState)

    # Spy on httpx.get
    called_ids = []

    def fake_get(url, **kwargs):
        # Extract id from URL
        tid = int(url.split("id=")[1])
        called_ids.append(tid)

        class FakeResp:
            status_code = 404
            text = "<html></html>"

        return FakeResp()

    monkeypatch.setattr(mod.httpx, "get", fake_get)

    # Run crawl_fn
    mod.crawl_fn()

    # Assert: NO call for id <= 74
    assert all(tid > 74 for tid in called_ids), f"crawl_fn called ids <= 74: {called_ids}"
    # First call should be id=75
    if called_ids:
        assert called_ids[0] == 75, f"First call should be id=75, got {called_ids[0]}"
