"""TDD test for scripts/dedup_exact_duplicate_ids.py (S18 — remove exact dup rows).

Context (2026-06-10): 4 chunk_ids exist as TWO byte-identical rows in LanceDB
(same text + same metadata_json). A duplicate row wastes a top-k slot and
biases retrieval. This script keeps exactly ONE row per chunk_id when the
duplicates are byte-identical, and REFUSES to touch a chunk_id whose copies
differ (safety — divergent copies need human review, not blind deletion).

Strategy mirrors the other data-ops tests: a fake table records add/delete
and exposes to_pandas().
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "dedup_exact_duplicate_ids.py"
)


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("dedup_exact_duplicate_ids", SCRIPT_PATH)
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

    def to_pandas(self):
        import pandas as pd
        cols = ["chunk_id", "id", "text", "source", "faq_type", "metadata_json", "vector"]
        return pd.DataFrame({c: [r.get(c) for r in self._rows] for c in cols})

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)


def _row(cid: str, text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "id": cid, "chunk_id": cid, "text": text,
        "source": "s", "faq_type": "f",
        "metadata_json": json.dumps(meta or {}, ensure_ascii=False),
        "vector": [0.1] * 8,
    }


# ---------------------------------------------------------------------------
# 1. Exact-identical duplicate → collapsed to ONE row.
# ---------------------------------------------------------------------------
def test_identical_duplicate_collapsed(mod):
    table = FakeTable(rows=[
        _row("dup_a", "same text", {"data_year": 2026}),
        _row("dup_a", "same text", {"data_year": 2026}),
        _row("unique_b", "other", {"data_year": 2026}),
    ])
    summary = mod.dedup_exact_duplicates(table=table)

    # deleted by chunk_id then re-added exactly once.
    assert any("dup_a" in c for c in table.del_calls)
    # final state: dup_a appears exactly once, unique_b untouched.
    ids = [r["chunk_id"] for r in table.rows]
    assert ids.count("dup_a") == 1
    assert ids.count("unique_b") == 1
    assert summary["dup_a"]["removed"] == 1
    assert summary["dup_a"]["kept"] == 1


# ---------------------------------------------------------------------------
# 2. Unique chunk_id is never touched.
# ---------------------------------------------------------------------------
def test_unique_id_untouched(mod):
    table = FakeTable(rows=[_row("solo", "x", {"y": 1})])
    summary = mod.dedup_exact_duplicates(table=table)
    assert table.del_calls == []
    assert table.add_calls == []
    assert "solo" not in summary  # only duplicated ids appear in summary


# ---------------------------------------------------------------------------
# 3. DIVERGENT copies (same id, different text) are REFUSED (safety).
# ---------------------------------------------------------------------------
def test_divergent_copies_refused(mod):
    table = FakeTable(rows=[
        _row("conflict", "version one", {"data_year": 2026}),
        _row("conflict", "version TWO — different!", {"data_year": 2026}),
    ])
    summary = mod.dedup_exact_duplicates(table=table)
    # Must NOT delete divergent copies — needs human review.
    assert table.del_calls == []
    assert summary["conflict"]["removed"] == 0
    assert summary["conflict"]["divergent"] is True


# ---------------------------------------------------------------------------
# 4. Triple-identical → keep 1, remove 2.
# ---------------------------------------------------------------------------
def test_triplicate_collapsed(mod):
    table = FakeTable(rows=[
        _row("trip", "t", {"a": 1}),
        _row("trip", "t", {"a": 1}),
        _row("trip", "t", {"a": 1}),
    ])
    summary = mod.dedup_exact_duplicates(table=table)
    ids = [r["chunk_id"] for r in table.rows]
    assert ids.count("trip") == 1
    assert summary["trip"]["removed"] == 2
    assert summary["trip"]["kept"] == 1


# ---------------------------------------------------------------------------
# 5. Idempotent — running on a clean table is a no-op.
# ---------------------------------------------------------------------------
def test_idempotent_noop(mod):
    table = FakeTable(rows=[_row("a", "x"), _row("b", "y")])
    summary = mod.dedup_exact_duplicates(table=table)
    assert table.del_calls == []
    assert table.add_calls == []
    assert summary == {}
