"""TDD test for scripts/supersede_generic_timeline.py (S17.1 — TDD data-ops).

Strategy: inject a fake table that records ``add`` / ``delete`` calls and a fake
embedder that returns deterministic vectors. Tests verify:

  * The 2 generic chunk_ids get a fresh row with
    ``is_superseded=True``, ``superseded_by="qa_dang_ky_xet_tuyen_2026_v2"``,
    ``superseded_reason="generic placeholder timeline; specific id=74 dates ingested"``
    AND the old row is deleted.
  * Already-superseded rows are left untouched (idempotent re-run = no-op).
  * The vector of the superseded row matches the re-embed output.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "supersede_generic_timeline.py"


@pytest.fixture
def supersede_module():
    spec = importlib.util.spec_from_file_location("supersede_generic_timeline", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fake table + embedder (no real LanceDB / no real model)
# ---------------------------------------------------------------------------
class FakeTable:
    """Mimics the LanceDB table — records + add + delete + to_pandas."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows: List[Dict[str, Any]] = list(rows)
        self.add_calls: List[List[Dict[str, Any]]] = []
        self.del_calls: List[str] = []

    def add(self, rows: List[Dict[str, Any]]) -> None:
        self.add_calls.append(list(rows))
        self._rows.extend(rows)

    def delete(self, where_clause: str) -> None:
        self.del_calls.append(where_clause)
        # Best-effort parse: chunk_id = 'value'  → remove rows where chunk_id matches
        prefix = "chunk_id = "
        if where_clause.startswith(prefix):
            value = where_clause[len(prefix):].strip().strip("'\"")
            self._rows = [r for r in self._rows if r.get("chunk_id") != value]

    def to_pandas(self, columns: Optional[List[str]] = None):  # noqa: D401
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
        rows = []
        for i, _t in enumerate(texts):
            vec = np.full(self.DIM, 0.5 * (i + 1), dtype=np.float32)
            rows.append(vec)
        return np.stack(rows)

    def encode_single(self, text: str):
        return np.full(self.DIM, 0.5, dtype=np.float32)


def _row(chunk_id: str, text: str, vec_index: int = 0, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "text": text,
        "source": "old-source",
        "faq_type": "old",
        "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
        "vector": [0.1 * (vec_index + 1)] * 8,
    }


# ---------------------------------------------------------------------------
# 1. A generic chunk gets flipped: is_superseded=True + superseded_by set.
# ---------------------------------------------------------------------------
def test_generic_chunk_flagged_as_superseded(supersede_module):
    target_id = "qa_lich_tuyen_sinh_2026"
    initial = _row(
        target_id,
        "Đợt 1 tháng 5-6, Đợt 2 tháng 7-8 [CẦN HUSC CUNG CẤP ngày cụ thể]",
        metadata={"data_year": 2026, "info_type": "qa_legacy"},
    )
    table = FakeTable(rows=[initial])
    embedder = FakeEmbedder()

    summary = supersede_module.supersede_generic_timeline(
        table=table,
        embedder=embedder,
    )

    # Delete called for the old row
    assert any(target_id in c for c in table.del_calls), table.del_calls
    # Add called with a new row carrying the flag
    assert len(table.add_calls) == 1
    new_rows = table.add_calls[0]
    assert len(new_rows) == 1
    new_row = new_rows[0]

    meta = json.loads(new_row["metadata_json"])
    assert meta["is_superseded"] is True
    assert meta["superseded_by"] == "qa_dang_ky_xet_tuyen_2026_v2"
    assert meta["superseded_reason"] == (
        "generic placeholder timeline; specific id=74 dates ingested"
    )
    # text/source preserved
    assert "Đợt 1 tháng 5-6" in new_row["text"]
    # vector length matches
    assert len(new_row["vector"]) == FakeEmbedder.DIM
    # summary records the supersede
    assert summary[target_id]["is_superseded_now"] is True
    assert summary[target_id]["already_superseded"] is False


# ---------------------------------------------------------------------------
# 2. Already-superseded chunk is a no-op.
# ---------------------------------------------------------------------------
def test_already_superseded_is_noop(supersede_module):
    target_id = "qa_xet_hoc_ba_quy_trinh_2026"
    initial = _row(
        target_id,
        "Generic placeholder text",
        metadata={
            "is_superseded": True,
            "superseded_by": "qa_dang_ky_xet_tuyen_2026_v2",
            "data_year": 2026,
        },
    )
    table = FakeTable(rows=[initial])
    embedder = FakeEmbedder()

    summary = supersede_module.supersede_generic_timeline(
        table=table,
        embedder=embedder,
    )

    # no add, no delete
    assert table.add_calls == []
    assert table.del_calls == []
    assert summary[target_id]["already_superseded"] is True
    assert summary[target_id]["is_superseded_now"] is False


# ---------------------------------------------------------------------------
# 3. TARGET_CHUNK_IDS includes both generic ids (locked by TDD).
# ---------------------------------------------------------------------------
def test_target_chunk_ids_locked(supersede_module):
    assert "qa_lich_tuyen_sinh_2026" in supersede_module.TARGET_CHUNK_IDS
    assert "qa_xet_hoc_ba_quy_trinh_2026" in supersede_module.TARGET_CHUNK_IDS
    assert "qa_dang_ky_xet_tuyen_2026_v2" in supersede_module.SUPERSEDED_BY
