"""TDD test for scripts/dedup_boilerplate_chunks.py (S17.2 — boilerplate dedup).

Strategy: inject a fake LanceDB-shaped table (rows list + ``add`` + ``delete``
+ ``to_pandas``) and a fake embedder. Tests verify:

  * EXACT-duplicate short (<80 char) chunks are flagged
    ``is_superseded=True`` with ``superseded_by`` = kept canonical chunk_id.
  * Each duplicate group keeps EXACTLY ONE canonical (the FIRST/lowest-sort
    chunk_id) and the REST are superseded.
  * Chunks with len >= 80 are NEVER touched (real content).
  * Single-occurrence short chunks are NEVER touched.
  * Re-running on an already-superseded table is a NO-OP (idempotent).
  * Already-superseded rows in a duplicate group are skipped (not double-
    re-superseded).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dedup_boilerplate_chunks.py"


@pytest.fixture
def dedup_module():
    spec = importlib.util.spec_from_file_location("dedup_boilerplate_chunks", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fake table + embedder (no real LanceDB / no real model)
# ---------------------------------------------------------------------------
class FakeTable:
    """Mimics the LanceDB table — records + add + delete + to_pandas."""

    DIM = 8

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        # Each row needs a vector column of the right dim for to_pandas.
        norm = [r for r in rows]
        for i, r in enumerate(norm):
            r.setdefault("vector", [0.1 * (i + 1)] * self.DIM)
        self._rows: List[Dict[str, Any]] = norm
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
            columns = ["chunk_id", "id", "text", "metadata_json", "vector", "source", "faq_type"]
        data: Dict[str, list] = {col: [r.get(col) for r in self._rows] for col in columns}
        return pd.DataFrame(data)

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)

    def by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        for r in self._rows:
            if r.get("chunk_id") == chunk_id:
                return r
        return None


class FakeEmbedder:
    DIM = 8

    def encode_single(self, text: str):
        return np.full(self.DIM, 0.5, dtype=np.float32)

    def encode_documents(self, texts: List[str]):
        rows = []
        for i, _t in enumerate(texts):
            rows.append(np.full(self.DIM, 0.5 * (i + 1), dtype=np.float32))
        return np.stack(rows)


def _row(
    chunk_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "text": text,
        "source": "src",
        "faq_type": "info",
        "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# 1. SHORT exact-duplicates → only one kept, rest marked superseded.
# ---------------------------------------------------------------------------
def test_short_exact_duplicates_are_deduped(dedup_module):
    text = "CÁC PHƯƠNG THỨC TUYỂN SINH ĐẠI HỌC HỆ CHÍNH QUY NĂM 2026"
    assert len(text.strip()) < 80
    rows = [
        _row("chunked_15_chunk_16", text, {"data_year": 2026}),
        _row("chunked_17_chunk_20", text, {"data_year": 2026}),
        _row("chunked_21_chunk_1",  text, {"data_year": 2026}),
        _row("chunked_22_chunk_62", text, {"data_year": 2026}),
    ]
    table = FakeTable(rows=list(rows))

    summary = dedup_module.dedup_boilerplate(
        table=table,
        embedder=FakeEmbedder(),
        short_threshold=80,
    )

    # 1 group, 3 superseded (keep 1 of 4)
    assert summary["duplicate_groups"] == 1
    assert summary["chunks_superseded"] == 3
    # The kept canonical is the first chunk_id (stable sort of source order)
    groups = summary["groups"]
    assert len(groups) == 1
    g = groups[0]
    assert g["kept_chunk_id"] == "chunked_15_chunk_16"
    assert set(g["superseded_chunk_ids"]) == {
        "chunked_17_chunk_20",
        "chunked_21_chunk_1",
        "chunked_22_chunk_62",
    }
    # Exactly 3 delete+add pairs
    assert len(table.del_calls) == 3
    assert len(table.add_calls) == 3
    # The kept row is NOT touched (still original metadata, not is_superseded)
    kept_row = table.by_chunk_id("chunked_15_chunk_16")
    assert kept_row is not None
    kept_meta = json.loads(kept_row["metadata_json"])
    assert kept_meta.get("is_superseded", False) is False
    # Each superseded row's replacement carries the right metadata
    for added in table.add_calls[0]:
        meta = json.loads(added["metadata_json"])
        assert meta["is_superseded"] is True
        assert meta["superseded_by"] == "chunked_15_chunk_16"
        assert "exact-duplicate boilerplate" in meta["superseded_reason"]
        assert "chunked_15_chunk_16" in meta["superseded_reason"]


# ---------------------------------------------------------------------------
# 2. len>=80 chunks are NEVER touched, even if duplicated.
# ---------------------------------------------------------------------------
def test_long_chunks_never_superseded(dedup_module):
    long_text = "A" * 100  # 100 chars, >= 80
    rows = [
        _row("long_a", long_text, {"data_year": 2026}),
        _row("long_b", long_text, {"data_year": 2026}),
        _row("long_c", long_text, {"data_year": 2026}),
    ]
    table = FakeTable(rows=list(rows))

    summary = dedup_module.dedup_boilerplate(
        table=table,
        embedder=FakeEmbedder(),
        short_threshold=80,
    )

    # 0 duplicate groups because the long text is excluded from the dedup universe
    assert summary["duplicate_groups"] == 0
    assert summary["chunks_superseded"] == 0
    assert table.add_calls == []
    assert table.del_calls == []
    for cid in ("long_a", "long_b", "long_c"):
        row = table.by_chunk_id(cid)
        assert row is not None
        meta = json.loads(row["metadata_json"])
        assert meta.get("is_superseded", False) is False


# ---------------------------------------------------------------------------
# 3. Single-occurrence short chunks are NEVER touched.
# ---------------------------------------------------------------------------
def test_single_occurrence_short_chunk_untouched(dedup_module):
    rows = [
        _row("solo", "Một câu ngắn duy nhất", {"data_year": 2026}),
    ]
    table = FakeTable(rows=list(rows))

    summary = dedup_module.dedup_boilerplate(
        table=table,
        embedder=FakeEmbedder(),
        short_threshold=80,
    )

    assert summary["duplicate_groups"] == 0
    assert summary["chunks_superseded"] == 0
    assert table.add_calls == []
    assert table.del_calls == []
    row = table.by_chunk_id("solo")
    assert row is not None
    meta = json.loads(row["metadata_json"])
    assert meta.get("is_superseded", False) is False


# ---------------------------------------------------------------------------
# 4. Mixed scenario: multiple groups + len<80 + len>=80 + unique.
# ---------------------------------------------------------------------------
def test_mixed_scenario(dedup_module):
    header = "CÁC PHƯƠNG THỨC TUYỂN SINH ĐẠI HỌC HỆ CHÍNH QUY NĂM 2026"
    short_dup = "Vụ GDĐH\nCục GDNNTX"  # 22 chars, exact-dup
    long_text = ("Đợt 1 tháng 5-6, Đợt 2 tháng 7-8 [CẦN HUSC CUNG CẤP ngày cụ thể] "
                 "để phục vụ thí sinh")  # > 80
    rows = [
        _row("h1", header, {"data_year": 2026}),
        _row("h2", header, {"data_year": 2026}),
        _row("h3", header, {"data_year": 2026}),
        _row("v1", short_dup, {"data_year": 2025}),
        _row("v2", short_dup, {"data_year": 2025}),
        _row("v3", short_dup, {"data_year": 2025}),
        _row("v4", short_dup, {"data_year": 2025}),
        _row("v5", short_dup, {"data_year": 2025}),
        _row("long_only", long_text, {"data_year": 2026}),
        _row("long_dup",  long_text, {"data_year": 2026}),
        _row("solo_short", "Câu ngắn một mình", {"data_year": 2026}),
    ]
    table = FakeTable(rows=list(rows))

    summary = dedup_module.dedup_boilerplate(
        table=table,
        embedder=FakeEmbedder(),
        short_threshold=80,
    )

    # 2 groups: header (3 dups, keep 1, supersede 2) + v_dup (5 dups, keep 1, supersede 4)
    assert summary["duplicate_groups"] == 2
    assert summary["chunks_superseded"] == 2 + 4

    # The 2 long chunks remain untouched
    for cid in ("long_only", "long_dup"):
        row = table.by_chunk_id(cid)
        meta = json.loads(row["metadata_json"])
        assert meta.get("is_superseded", False) is False

    # The solo short chunk remains untouched
    row = table.by_chunk_id("solo_short")
    meta = json.loads(row["metadata_json"])
    assert meta.get("is_superseded", False) is False

    # h1 + v1 are the canonicals (kept)
    kept_h = table.by_chunk_id("h1")
    assert kept_h is not None
    meta = json.loads(kept_h["metadata_json"])
    assert meta.get("is_superseded", False) is False
    kept_v = table.by_chunk_id("v1")
    assert kept_v is not None
    meta = json.loads(kept_v["metadata_json"])
    assert meta.get("is_superseded", False) is False


# ---------------------------------------------------------------------------
# 5. Idempotent: re-running on an already-deduped table is a no-op.
# ---------------------------------------------------------------------------
def test_idempotent_rerun_is_noop(dedup_module):
    text = "HEADER NGẮN TRÙNG LẶP"
    # Pre-seeded: 1 kept + 2 already superseded
    rows = [
        _row("h1", text, {"data_year": 2026}),
        _row(
            "h2", text,
            {
                "data_year": 2026,
                "is_superseded": True,
                "superseded_by": "h1",
                "superseded_reason": "exact-duplicate boilerplate (dedup); canonical copy = h1",
            },
        ),
        _row(
            "h3", text,
            {
                "data_year": 2026,
                "is_superseded": True,
                "superseded_by": "h1",
                "superseded_reason": "exact-duplicate boilerplate (dedup); canonical copy = h1",
            },
        ),
    ]
    table = FakeTable(rows=list(rows))

    summary = dedup_module.dedup_boilerplate(
        table=table,
        embedder=FakeEmbedder(),
        short_threshold=80,
    )

    # No additional supersedes performed
    assert summary["chunks_superseded"] == 0
    assert table.add_calls == []
    assert table.del_calls == []


# ---------------------------------------------------------------------------
# 6. Constants are locked: short_threshold = 80.
# ---------------------------------------------------------------------------
def test_constants_locked(dedup_module):
    assert dedup_module.SHORT_THRESHOLD == 80
    # The default is_superseded reason mentions the canonical copy
    assert "exact-duplicate boilerplate" in dedup_module.SUPERSEDED_REASON_PREFIX
