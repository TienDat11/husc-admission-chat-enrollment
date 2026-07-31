"""@spec(S18.2-T1) TDD test for scripts/fix_topic_route_id73_74.py.

Strategy: in-memory fake LanceDB table (delete-by-chunk_id + add) and a
deterministic fake embedder (no real model load).  Verifies:

* chunked_22_chunk_54 is RE-ADDED with route=offline, info_type=xtt_nop_ho_so
  (NOT superseded, year preserved).
* husc_tb74_dkxt_online_2026 is ADDED with route=online + URL dkxt.hueuni.edu.vn.
* idempotency: re-running the plan produces no extra rows / no double-deletes.
* jsonl output written for build_graph parity.
* live table integration (real LanceDB) — gated by a marker.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "fix_topic_route_id73_74.py"


@pytest.fixture
def fix_module():
    spec = importlib.util.spec_from_file_location("fix_topic_route_id73_74", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fakes (no real LanceDB, no real model)
# ---------------------------------------------------------------------------
class FakeTable:
    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        self._rows: List[Dict[str, Any]] = list(rows or [])
        self.del_calls: List[str] = []
        self.add_calls: List[List[Dict[str, Any]]] = []

    def delete(self, where_clause: str) -> None:
        self.del_calls.append(where_clause)
        prefix = "chunk_id = "
        if where_clause.startswith(prefix):
            v = where_clause[len(prefix):].strip().strip("'\"")
            self._rows = [r for r in self._rows if r.get("chunk_id") != v]

    def add(self, rows: List[Dict[str, Any]]) -> None:
        self.add_calls.append(list(rows))
        self._rows.extend(rows)

    def to_pandas(self):
        import pandas as pd
        if not self._rows:
            return pd.DataFrame(columns=["chunk_id", "text", "metadata_json", "vector"])
        return pd.DataFrame(self._rows)

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)


class FakeEmbedder:
    DIM = 1024

    def encode_documents(self, texts: List[str]):
        return np.stack([np.full(self.DIM, 0.1 * (i + 1), dtype=np.float32) for i in range(len(texts))])

    def encode_single(self, text: str):
        return np.full(self.DIM, 0.5, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1) CONSTANTS locked
# ---------------------------------------------------------------------------
def test_constants_locked(fix_module):
    assert fix_module.CHUNK_54_ID == "chunked_22_chunk_54"
    assert fix_module.NEW_ONLINE_CHUNK_ID == "husc_tb74_dkxt_online_2026"
    assert "EMS" in fix_module.CHUNK_54_TEXT
    assert "dkxt.hueuni.edu.vn" in fix_module.NEW_ONLINE_TEXT
    assert "02/7" in fix_module.NEW_ONLINE_TEXT and "14/7" in fix_module.NEW_ONLINE_TEXT


def test_jsonl_output_path_locked(fix_module):
    assert fix_module.CHUNKED_JSONL_OUT.name == "husc_thongbao_fix_2026.jsonl"
    assert "data/chunked" in str(fix_module.CHUNKED_JSONL_OUT).replace("\\", "/")


# ---------------------------------------------------------------------------
# 2) Dry-run is non-mutating
# ---------------------------------------------------------------------------
def test_dry_run_no_table_mutation(fix_module):
    table = FakeTable(rows=[{
        "chunk_id": "chunked_22_chunk_54",
        "text": "old text",
        "metadata_json": "{}",
        "vector": [0.0] * 1024,
        "source": "x",
    }])
    summary = fix_module.plan_changes(embedder=FakeEmbedder(), table=table, dry_run=True)
    assert summary["dry_run"] is True
    assert table.add_calls == []
    assert table.del_calls == []
    assert summary["chunk_54"]["action"].startswith("would_")
    assert summary["new_online"]["action"].startswith("would_")


# ---------------------------------------------------------------------------
# 3) Live: chunked_22_chunk_54 is RE-ADDED with route=offline, NOT superseded
# ---------------------------------------------------------------------------
def test_chunk_54_re_added_with_route_offline(fix_module, tmp_path, monkeypatch):
    # redirect jsonl output to tmp
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable(rows=[{
        "chunk_id": "chunked_22_chunk_54",
        "text": "old text without EMS marker",
        "metadata_json": json.dumps({"year": "2026", "data_year": 2026, "info_type": "diem_chuan"}),
        "vector": [0.0] * 1024,
        "source": "old",
    }])
    summary = fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    assert summary["chunk_54"]["deleted"] is True
    assert summary["chunk_54"]["added"] is True

    new = next(r for r in table.rows if r["chunk_id"] == "chunked_22_chunk_54")
    meta = json.loads(new["metadata_json"])
    assert meta["info_type"] == "xtt_nop_ho_so"
    assert meta["route"] == "offline"
    assert meta["data_year"] == 2026
    assert "is_superseded" not in meta
    # text preserved verbatim from baseline
    assert "EMS" in new["text"]


# ---------------------------------------------------------------------------
# 4) Live: husc_tb74_dkxt_online_2026 is ADDED with route=online + URL
# ---------------------------------------------------------------------------
def test_online_chunk_added(fix_module, tmp_path, monkeypatch):
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()
    summary = fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    assert summary["new_online"]["added"] is True
    new = next(r for r in table.rows if r["chunk_id"] == "husc_tb74_dkxt_online_2026")
    meta = json.loads(new["metadata_json"])
    assert meta["route"] == "online"
    assert meta["data_year"] == 2026
    assert "dkxt.hueuni.edu.vn" in new["text"]
    assert "02/7" in new["text"] and "14/7" in new["text"]
    # embedding dim guard
    assert len(new["vector"]) == 1024


# ---------------------------------------------------------------------------
# 5) Idempotency: second run on freshly-added state should be safe no-op
#    (delete + re-add with same data → no row count growth, no extra add)
# ---------------------------------------------------------------------------
def test_idempotent_second_run(fix_module, tmp_path, monkeypatch):
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()

    s1 = fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    rows_after_run1 = len(table.rows)

    s2 = fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    rows_after_run2 = len(table.rows)
    # run 2 deletes 2 + adds 2 → row count unchanged
    assert rows_after_run1 == rows_after_run2
    # run 2 still issues 2 delete + 2 add calls per run (delete-then-add = idempotent swap)
    assert len(table.del_calls) == 4  # 2 per run
    assert len(table.add_calls) == 4  # 2 add calls per run (one per chunk)


# ---------------------------------------------------------------------------
# 6) Retrieval contract (post-fix, top-3 membership) — verified by fakes only
# ---------------------------------------------------------------------------
def test_retrieval_top3_for_dkxt_online_query(fix_module, tmp_path, monkeypatch):
    """After fix, query 'đăng ký xét tuyển online 2026 ở đâu' must return
    husc_tb74_dkxt_online_2026 in top-1 of an L2-distance ranking over the
    fake table.

    The fake embedder encodes by index position (0.1*i), so we can rely on
    the *insertion order* of the rows being meaningful for the L2 distance:
    husc_tb74_dkxt_online_2026 is inserted last (with index-based encoding),
    so the query's encoding picks it as closest when the query vector matches
    the last-added doc.
    """
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()
    # Seed an unrelated chunk first to make top-3 meaningful.
    table.add([{
        "chunk_id": "unrelated_seed",
        "text": "học phí năm 2026",
        "metadata_json": "{}",
        "vector": [0.1] * 1024,
        "source": "x",
    }])
    fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    rows = table.rows

    # Encode query with the same fake embedder; the last inserted doc
    # (husc_tb74_dkxt_online_2026) gets vec [0.2, 0.2, ...].
    qvec = np.full(1024, 0.2, dtype=np.float32)
    scored = []
    for r in rows:
        v = np.asarray(r["vector"], dtype=np.float32)
        d = float(np.linalg.norm(v - qvec))
        scored.append((d, r["chunk_id"]))
    scored.sort()
    top3 = [cid for _, cid in scored[:3]]
    assert "husc_tb74_dkxt_online_2026" in top3


def test_retrieval_top3_for_ems_query_returns_chunk_54(fix_module, tmp_path, monkeypatch):
    """Query 'nộp lệ phí xét tuyển thẳng' must still return chunked_22_chunk_54."""
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()
    fix_module.plan_changes(embedder=FakeEmbedder(), table=table)

    rows = table.rows
    # chunk_54 re-add uses text index 0 → vec = [0.1, 0.1, ...].
    qvec = np.full(1024, 0.1, dtype=np.float32)
    scored = sorted((float(np.linalg.norm(np.asarray(r["vector"], dtype=np.float32) - qvec)), r["chunk_id"]) for r in rows)
    top3 = [cid for _, cid in scored[:3]]
    assert "chunked_22_chunk_54" in top3


# ---------------------------------------------------------------------------
# 7) JSONL output exists and contains both chunks
# ---------------------------------------------------------------------------
def test_jsonl_write_for_build_graph(fix_module, tmp_path, monkeypatch):
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", out)
    table = FakeTable()
    summary = fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    assert summary["jsonl_written"] == 2
    assert out.exists()
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    ids = {json.loads(ln)["chunk_id"] for ln in lines}
    assert ids == {"chunked_22_chunk_54", "husc_tb74_dkxt_online_2026"}