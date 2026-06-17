"""TDD test for scripts/supersede_stale_aggregates.py (S18 — QA-driven data-ops).

Context (bug 2026-06-09): two STALE aggregate ranking/overview chunks still
live in LanceDB alongside their corrected ``*_v2`` twins. They carry the wrong
total (1969 vs the correct 2079) and still list the discontinued major
``7480107TD``. The generation model retrieves both and mixes them, producing
the wrong "tổng chỉ tiêu" and a phantom "3 ngành cùng 20 chỉ tiêu".

Fix: flip ``is_superseded=True`` on the two stale chunks so the retriever's
temporal filter (lancedb_retrieval.py:331) drops them. The corrected ``*_v2``
chunks are NEVER touched.

Strategy mirrors test_supersede_generic_timeline.py: a fake table records
add/delete and a fake embedder returns deterministic vectors.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "supersede_stale_aggregates.py"
)


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("supersede_stale_aggregates", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeTable:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows: List[Dict[str, Any]] = list(rows)
        self.add_calls: List[List[Dict[str, Any]]] = []
        self.del_calls: List[str] = []

    def add(self, rows: List[Dict[str, Any]]) -> None:
        self.add_calls.append(list(rows))
        self._rows.extend(rows)

    def delete(self, where_clause: str) -> None:
        self.del_calls.append(where_clause)
        prefix = "chunk_id = "
        if where_clause.startswith(prefix):
            value = where_clause[len(prefix):].strip().strip("'\"")
            self._rows = [r for r in self._rows if r.get("chunk_id") != value]

    def to_pandas(self, columns: Optional[List[str]] = None):
        import pandas as pd

        if columns is None:
            columns = ["chunk_id", "id", "text", "metadata_json", "vector"]
        data: Dict[str, list] = {col: [r.get(col) for r in self._rows] for col in columns}
        return pd.DataFrame(data)

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)


class FakeEmbedder:
    DIM = 8

    def encode_documents(self, texts: List[str]):
        return np.stack([np.full(self.DIM, 0.5 * (i + 1), dtype=np.float32) for i, _ in enumerate(texts)])

    def encode_single(self, text: str):
        return np.full(self.DIM, 0.5, dtype=np.float32)


def _row(chunk_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "text": text,
        "source": "old-source",
        "faq_type": "aggregate",
        "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
        "vector": [0.1] * 8,
    }


# ---------------------------------------------------------------------------
# 1. Stale ranking chunk flipped to superseded.
# ---------------------------------------------------------------------------
def test_stale_ranking_chunk_superseded(mod):
    target = "bang_xep_hang_chi_tieu_2026"
    table = FakeTable(rows=[
        _row(target, "Bảng xếp hạng ... Tổng: 1969 chỉ tiêu ... 7480107TD", {"data_year": 2026}),
        # the corrected twin MUST remain untouched
        _row("bang_xep_hang_chi_tieu_2026_v2", "Bảng xếp hạng 28 ngành ... ~2.079", {"data_year": 2026}),
    ])
    summary = mod.supersede_stale_aggregates(table=table, embedder=FakeEmbedder())

    assert any(target in c for c in table.del_calls), table.del_calls
    new_row = table.add_calls[0][0]
    meta = json.loads(new_row["metadata_json"])
    assert meta["is_superseded"] is True
    assert meta["superseded_by"] == "bang_xep_hang_chi_tieu_2026_v2"
    assert "1969" in new_row["text"] or "Tổng" in new_row["text"]  # text preserved
    assert summary[target]["is_superseded_now"] is True


# ---------------------------------------------------------------------------
# 2. The corrected *_v2 twin is NEVER touched.
# ---------------------------------------------------------------------------
def test_v2_twins_never_touched(mod):
    table = FakeTable(rows=[
        _row("bang_xep_hang_chi_tieu_2026", "old 1969", {"data_year": 2026}),
        _row("bang_xep_hang_chi_tieu_2026_v2", "new 2079", {"data_year": 2026}),
        _row("tuyensinh_overview_2026_v2", "new overview", {"data_year": 2026}),
    ])
    mod.supersede_stale_aggregates(table=table, embedder=FakeEmbedder())
    # No *_v2 id may appear in any delete clause.
    for c in table.del_calls:
        assert "_v2" not in c, f"v2 twin must never be deleted: {c}"


# ---------------------------------------------------------------------------
# 3. Idempotent — already-superseded is a no-op.
# ---------------------------------------------------------------------------
def test_already_superseded_noop(mod):
    target = "tuyensinh_overview_2026"
    table = FakeTable(rows=[
        _row(target, "old overview 1969", {"is_superseded": True, "data_year": 2026}),
    ])
    summary = mod.supersede_stale_aggregates(table=table, embedder=FakeEmbedder())
    assert table.add_calls == []
    assert table.del_calls == []
    assert summary[target]["already_superseded"] is True


# ---------------------------------------------------------------------------
# 4. TARGET_CHUNK_IDS locked — exactly the 2 stale aggregates, no *_v2.
# ---------------------------------------------------------------------------
def test_target_ids_locked(mod):
    assert "bang_xep_hang_chi_tieu_2026" in mod.TARGET_CHUNK_IDS
    assert "tuyensinh_overview_2026" in mod.TARGET_CHUNK_IDS
    for tid in mod.TARGET_CHUNK_IDS:
        assert not tid.endswith("_v2"), f"must not supersede a corrected twin: {tid}"


# ---------------------------------------------------------------------------
# 5. Missing chunk is reported, not crashed.
# ---------------------------------------------------------------------------
def test_missing_chunk_reported(mod):
    table = FakeTable(rows=[])  # neither target present
    summary = mod.supersede_stale_aggregates(table=table, embedder=FakeEmbedder())
    for tid in mod.TARGET_CHUNK_IDS:
        assert summary[tid]["found"] is False
        assert summary[tid]["is_superseded_now"] is False
