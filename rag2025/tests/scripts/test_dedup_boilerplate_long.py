"""TDD test for LONG-text exact-duplicate dedup in dedup_boilerplate_chunks.py (T1.6).

Extends the S17.2 short-chunk dedup to long boilerplate (>= LONG_THRESHOLD chars,
normalized by lowercase + whitespace-collapse). Tests verify:

  * Long (>=200 char) rows whose NORMALIZED text (lowercase + collapsed
    whitespace) collides are deduped: keep the FIRST/lowest-sort chunk_id as
    canonical, supersede the rest.
  * Rows that differ only by case / whitespace are treated as duplicates
    (normalized match, mirroring sweep_lancedb_noise._normalize_text).
  * Long rows below LONG_THRESHOLD (e.g. 100 chars) are NEVER touched.
  * Unique long rows are NEVER touched.
  * Short (<80 char) exact-duplicates are STILL deduped (existing behavior).
  * Re-running on an already-deduped table is a NO-OP (idempotent).
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
        prefix = "chunk_id = "
        if where_clause.startswith(prefix):
            value = where_clause[len(prefix):].strip().strip("'\"")
            self._rows = [r for r in self._rows if r.get("chunk_id") != value]

    def to_pandas(self, columns: Optional[List[str]] = None):
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
        return np.stack(
            [np.full(self.DIM, 0.5 * (i + 1), dtype=np.float32) for i, _ in enumerate(texts)]
        )


class FakeFlatTable:
    """Mimics the LIVE flat husc_chunks schema (no chunk_id column; update-based)."""

    DIM = 8

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = [dict(r) for r in rows]
        for i, r in enumerate(self._rows):
            r.setdefault("vector", [0.1 * (i + 1)] * self.DIM)
            r.setdefault("doc_id", r.get("id"))
            r.setdefault("superseded_by", None)
        self.update_calls: List[tuple] = []
        self.add_calls: List[List[Dict[str, Any]]] = []
        self.del_calls: List[str] = []

    def update(self, where: str, values: Dict[str, Any]) -> None:
        self.update_calls.append((where, dict(values)))
        prefix = "id = "
        if where.startswith(prefix):
            val = where[len(prefix):].strip().strip("'\"")
            for r in self._rows:
                if str(r.get("id")) == val:
                    r.update(values)

    def add(self, rows: List[Dict[str, Any]]) -> None:
        self.add_calls.append(list(rows))

    def delete(self, where_clause: str) -> None:
        self.del_calls.append(where_clause)

    def to_pandas(self, columns: Optional[List[str]] = None):
        import pandas as pd

        if columns is None:
            columns = ["id", "doc_id", "text", "vector", "data_year", "metadata_json", "superseded_by"]
        data: Dict[str, list] = {col: [r.get(col) for r in self._rows] for col in columns}
        return pd.DataFrame(data)

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)

    def by_id(self, cid: str) -> Optional[Dict[str, Any]]:
        for r in self._rows:
            if str(r.get("id")) == cid:
                return r
        return None


def _flat_row(chunk_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Flat-schema row: id is the key; NO chunk_id column."""
    return {
        "id": chunk_id,
        "doc_id": chunk_id,
        "text": text,
        "data_year": 2026,
        "superseded_by": None,
        "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
        "vector": [0.2] * 8,
    }


def _row(chunk_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "text": text,
        "source": "src",
        "faq_type": "info",
        "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
    }


# A long boilerplate header repeated across notifications (>= LONG_THRESHOLD).
LONG_BOILERPLATE = (
    "ĐẠI HỌC HUẾ  TRƯỜNG ĐẠI HỌC KHOA HỌC  CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM  "
    "Độc lập - Tự do - Hạnh phúc  Số: 123/HUSC/TB  V/v tuyển sinh đại học hệ chính "
    "quy năm 2026. Căn cứ Quy chế tuyển sinh hiện hành và Thông báo của Hội đồng "
    "tuyển sinh, Trường Đại học Khoa học, Đại học Huế thông báo kế hoạch xét tuyển, "
    "thời gian nhận hồ sơ, lệ phí xét tuyển và các mốc thời gian quan trọng để thí "
    "sinh và phụ huynh được biết và thực hiện đúng quy định."
)
assert len(LONG_BOILERPLATE) >= 200


# ---------------------------------------------------------------------------
# 1. LONG normalized-exact duplicates are deduped (case + whitespace variants).
# ---------------------------------------------------------------------------
def test_long_normalized_duplicates_deduped(dedup_module):
    # Two texts differ only by case / whitespace → normalized collision.
    text_a = "  " + LONG_BOILERPLATE + "\n\n"
    text_b = LONG_BOILERPLATE.lower().replace("  ", " ").strip()
    rows = [
        _row("chunked_12_chunk_5", text_a, {"data_year": 2026}),
        _row("chunked_20_chunk_9", text_b, {"data_year": 2026}),
    ]
    table = FakeTable(rows=list(rows))

    summary = dedup_module.dedup_boilerplate(table=table, embedder=FakeEmbedder())

    assert summary["long_duplicate_groups"] == 1
    assert summary["long_chunks_superseded"] == 1
    assert summary["duplicate_groups"] == 1
    assert summary["chunks_superseded"] == 1
    # Canonical = lowest chunk_id (stable sort)
    g = summary["groups"][0]
    assert g["kept_chunk_id"] == "chunked_12_chunk_5"
    assert g["superseded_chunk_ids"] == ["chunked_20_chunk_9"]
    # The kept row is untouched
    kept = table.by_chunk_id("chunked_12_chunk_5")
    meta = json.loads(kept["metadata_json"])
    assert meta.get("is_superseded", False) is False
    # The superseded replacement carries the long reason
    superseded = table.by_chunk_id("chunked_20_chunk_9")
    assert superseded is not None
    smeta = json.loads(superseded["metadata_json"])
    assert smeta["is_superseded"] is True
    assert smeta["superseded_by"] == "chunked_12_chunk_5"
    assert "chunked_12_chunk_5" in smeta["superseded_reason"]


# ---------------------------------------------------------------------------
# 2. Unique long chunks are NEVER touched.
# ---------------------------------------------------------------------------
def test_unique_long_chunk_untouched(dedup_module):
    rows = [
        _row("solo_long", "Đây là một nội dung dài duy nhất. " + LONG_BOILERPLATE, {"data_year": 2026}),
    ]
    table = FakeTable(rows=list(rows))

    summary = dedup_module.dedup_boilerplate(table=table, embedder=FakeEmbedder())

    assert summary["long_duplicate_groups"] == 0
    assert summary["chunks_superseded"] == 0
    assert table.add_calls == []
    assert table.del_calls == []
    meta = json.loads(table.by_chunk_id("solo_long")["metadata_json"])
    assert meta.get("is_superseded", False) is False


# ---------------------------------------------------------------------------
# 3. Mid-length text (>=80 but < the plan's original 200 guess) IS normalized-deduped.
# ---------------------------------------------------------------------------
def test_mid_length_normalized_duplicates_deduped(dedup_module):
    mid_text = "M" * 100  # 100 chars — >= the 80-char normalized-dedup floor
    rows = [
        _row("mid_a", mid_text, {"data_year": 2026}),
        _row("mid_b", mid_text, {"data_year": 2026}),
    ]
    table = FakeTable(rows=list(rows))

    summary = dedup_module.dedup_boilerplate(table=table, embedder=FakeEmbedder())

    assert summary["long_duplicate_groups"] == 1
    assert summary["chunks_superseded"] == 1
    assert len(table.del_calls) == 1
    assert len(table.add_calls) == 1
    meta = json.loads(table.by_chunk_id("mid_b")["metadata_json"])
    assert meta["is_superseded"] is True
    assert meta["superseded_by"] == "mid_a"
    # kept canonical untouched
    kept = json.loads(table.by_chunk_id("mid_a")["metadata_json"])
    assert kept.get("is_superseded", False) is False


# ---------------------------------------------------------------------------
# 4. Short exact-duplicates are STILL deduped (regression).
# ---------------------------------------------------------------------------
def test_short_dups_still_deduped(dedup_module):
    short_text = "CÁC PHƯƠNG THỨC TUYỂN SINH ĐẠI HỌC HỆ CHÍNH QUY NĂM 2026"
    assert len(short_text.strip()) < 80
    rows = [
        _row("c1", short_text, {"data_year": 2026}),
        _row("c2", short_text, {"data_year": 2026}),
    ]
    table = FakeTable(rows=list(rows))

    summary = dedup_module.dedup_boilerplate(table=table, embedder=FakeEmbedder())

    assert summary["chunks_superseded"] == 1
    assert summary["duplicate_groups"] == 1
    assert summary["long_duplicate_groups"] == 0
    meta = json.loads(table.by_chunk_id("c2")["metadata_json"])
    assert meta["is_superseded"] is True
    assert meta["superseded_by"] == "c1"


# ---------------------------------------------------------------------------
# 5. Mixed short + long: both passes run.
# ---------------------------------------------------------------------------
def test_mixed_short_and_long(dedup_module):
    short_text = "ĐIỀU KIỆN XÉT TUYỂN NĂM 2026"  # < 80
    long_a = LONG_BOILERPLATE
    long_b = LONG_BOILERPLATE.lower().replace("  ", " ")
    rows = [
        _row("s1", short_text, {"data_year": 2026}),
        _row("s2", short_text, {"data_year": 2026}),
        _row("s3", short_text, {"data_year": 2026}),
        _row("l1", long_a, {"data_year": 2026}),
        _row("l2", long_b, {"data_year": 2026}),
    ]
    table = FakeTable(rows=list(rows))

    summary = dedup_module.dedup_boilerplate(table=table, embedder=FakeEmbedder())

    # 1 short group (3 rows → supersede 2) + 1 long group (2 rows → supersede 1)
    assert summary["duplicate_groups"] == 2
    assert summary["chunks_superseded"] == 3
    assert summary["long_duplicate_groups"] == 1
    assert summary["long_chunks_superseded"] == 1


# ---------------------------------------------------------------------------
# 6. Idempotent: re-running on an already-deduped long table is a no-op.
# ---------------------------------------------------------------------------
def test_idempotent_rerun_is_noop(dedup_module):
    rows = [
        _row("l1", LONG_BOILERPLATE, {"data_year": 2026}),
        _row(
            "l2",
            LONG_BOILERPLATE,
            {
                "data_year": 2026,
                "is_superseded": True,
                "superseded_by": "l1",
                "superseded_reason": "exact-duplicate long-text (dedup); canonical copy = l1",
            },
        ),
    ]
    table = FakeTable(rows=list(rows))

    summary = dedup_module.dedup_boilerplate(table=table, embedder=FakeEmbedder())

    assert summary["chunks_superseded"] == 0
    assert table.add_calls == []
    assert table.del_calls == []


# ---------------------------------------------------------------------------
# 7. Constants are locked: LONG_THRESHOLD = 200.
# ---------------------------------------------------------------------------
def test_long_threshold_locked(dedup_module):
    # normalized-dedup floor == 80 (live letterhead fragments are 80-199 chars)
    assert dedup_module.LONG_THRESHOLD == 80
    assert dedup_module.SHORT_THRESHOLD == 80


# ---------------------------------------------------------------------------
# 8. LIVE flat schema: long duplicates are superseded via in-place UPDATE.
#    Row count is preserved and the existing vector is NOT re-embedded.
# ---------------------------------------------------------------------------
def test_flat_schema_uses_update(dedup_module):
    rows = [
        _flat_row("l1", LONG_BOILERPLATE, {"data_year": 2026}),
        _flat_row("l2", LONG_BOILERPLATE.lower().replace("  ", " "), {"data_year": 2026}),
    ]
    table = FakeFlatTable(rows=list(rows))
    orig_vec = list(table.rows[1]["vector"])

    summary = dedup_module.dedup_boilerplate(table=table, embedder=FakeEmbedder())

    # Flat path uses update, NOT delete+add
    assert table.update_calls, "flat path must use table.update()"
    assert table.del_calls == []
    assert table.add_calls == []
    assert summary["long_duplicate_groups"] == 1
    assert summary["long_chunks_superseded"] == 1
    assert len(table.rows) == 2  # count preserved
    sup = table.by_id("l2")
    meta = json.loads(sup["metadata_json"])
    assert meta["is_superseded"] is True
    assert meta["superseded_by"] == "l1"
    # existing vector preserved (text unchanged → no re-embed)
    assert list(sup["vector"]) == orig_vec
    # kept canonical untouched
    kept = table.by_id("l1")
    assert json.loads(kept["metadata_json"]).get("is_superseded", False) is False


# ---------------------------------------------------------------------------
# 9. LIVE flat schema: a row whose id is duplicated in the table is SKIPPED
#    (delete/update by id would corrupt the sibling).
# ---------------------------------------------------------------------------
def test_flat_schema_dup_id_row_skipped(dedup_module):
    # a1 + dup_id share normalized long text → dup group. dup_id's id is ALSO
    # used by another row (sibling) with different text → dup_id is duplicated
    # in the table and must be SKIPPED (update by id would corrupt the sibling).
    rows = [
        _flat_row("a1", LONG_BOILERPLATE, {"data_year": 2026}),
        _flat_row("dup_id", LONG_BOILERPLATE.lower().replace("  ", " "), {"data_year": 2026}),
        _flat_row("dup_id", "Một đoạn nội dung khác hoàn toàn..." + ("x" * 200), {"data_year": 2026}),
    ]
    table = FakeFlatTable(rows=list(rows))

    summary = dedup_module.dedup_boilerplate(table=table, embedder=FakeEmbedder())

    # dup_id row is skipped → no supersede happened
    assert summary["long_chunks_superseded"] == 0
    assert table.update_calls == []
    # sibling untouched
    sibling = table.by_id("dup_id")
    meta = json.loads(sibling["metadata_json"])
    assert meta.get("is_superseded", False) is False
