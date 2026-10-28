"""TDD test for scripts/supersede_slug_noise.py (T1.7 — slug-header + needs_review hygiene).

Strategy mirrors test_supersede_stale_aggregates.py: a fake table records
add/delete and a fake embedder returns deterministic vectors.

Slug-noise chunks are diacritic-stripped markdown headers like::

    ### thi sinh bogddt tt 08 dieu 2 xet tuyen thang 2025

(the chunk_id/header is ASCII-stripped Vietnamese). Curated ``husc_*`` rows that
ALSO start with ``### `` (e.g. ``husc_summary_general_2025``) are REAL content and
must NEVER be superseded. Rows flagged ``needs_review=True`` with incomplete
metadata are superseded too.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "supersede_slug_noise.py"


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("supersede_slug_noise", SCRIPT_PATH)
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

    def by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        for r in self._rows:
            if r.get("chunk_id") == chunk_id:
                return r
        return None


class FakeEmbedder:
    DIM = 8

    def encode_documents(self, texts: List[str]):
        return np.stack([np.full(self.DIM, 0.5 * (i + 1), dtype=np.float32) for i, _ in enumerate(texts)])

    def encode_single(self, text: str):
        return np.full(self.DIM, 0.5, dtype=np.float32)


class FakeFlatTable:
    """Mimics the LIVE flat husc_chunks schema (no chunk_id column; update-based)."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = [dict(r) for r in rows]
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
            columns = ["id", "doc_id", "text", "metadata_json", "superseded_by", "vector"]
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
    return {
        "id": chunk_id,
        "doc_id": chunk_id,
        "text": text,
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
        "vector": [0.1] * 8,
    }


SLUG_TEXT = "### thi sinh bogddt tt 08 dieu 2 xet tuyen thang 2025\n\nTrong tuyển sinh đại học năm 2025 trên phạm vi toàn quốc..."
CURATED_TEXT = "### husc summary general 2025\n\nNăm 2025, Trường Đại học Khoa học - Đại học Huế tuyển sinh tổng cộng 25 ngành..."
REGULAR_TEXT = "Thông báo tuyển sinh đại học hệ chính quy năm 2026 của Trường Đại học Khoa học - Đại học Huế..."


# ---------------------------------------------------------------------------
# 1. Slug-noise rows are superseded.
# ---------------------------------------------------------------------------
def test_slug_noise_rows_superseded(mod):
    table = FakeTable(rows=[
        _row("thi_sinh_bogddt_tt_08_dieu_2_2025", SLUG_TEXT, {"data_year": 2025}),
        _row("all_bogddt_tt_08_header_2025", "### all bogddt tt 08 header 2025\n\nQuy chế tuyển sinh...", {"data_year": 2025}),
    ])
    summary = mod.supersede_slug_noise(table=table, embedder=FakeEmbedder())

    assert len(table.del_calls) == 2
    assert len(table.add_calls) == 2
    for cid in ("thi_sinh_bogddt_tt_08_dieu_2_2025", "all_bogddt_tt_08_header_2025"):
        row = table.by_chunk_id(cid)
        meta = json.loads(row["metadata_json"])
        assert meta["is_superseded"] is True
        assert "slug" in meta["superseded_reason"].lower()
    assert summary["slug_superseded"] == 2


# ---------------------------------------------------------------------------
# 2. Curated husc_* rows that start with "### " are NEVER touched.
# ---------------------------------------------------------------------------
def test_curated_husc_rows_untouched(mod):
    table = FakeTable(rows=[
        _row("husc_summary_general_2025", CURATED_TEXT, {"data_year": 2025}),
        _row("husc_news_0001_diem_chuan_2025", "### husc news 0001 diem chuan 2025\n\nĐiểm chuẩn...", {"data_year": 2025}),
    ])
    summary = mod.supersede_slug_noise(table=table, embedder=FakeEmbedder())

    assert table.del_calls == []
    assert table.add_calls == []
    assert summary["slug_superseded"] == 0
    for cid in ("husc_summary_general_2025", "husc_news_0001_diem_chuan_2025"):
        meta = json.loads(table.by_chunk_id(cid)["metadata_json"])
        assert meta.get("is_superseded", False) is False


# ---------------------------------------------------------------------------
# 3. needs_review rows with incomplete metadata are superseded.
# ---------------------------------------------------------------------------
def test_needs_review_rows_superseded(mod):
    table = FakeTable(rows=[
        _row("thongbao_76_0", "Trang chủ Thông báo Tuyển sinh...", {"needs_review": True}),
        _row("thongbao_77_1", "Đăng ký thông tin xét tuyển...", {"needs_review": True, "data_year": None}),
        _row("thongbao_80_0", "Chunk bình thường...", {"data_year": 2026}),
    ])
    summary = mod.supersede_slug_noise(table=table, embedder=FakeEmbedder())

    assert summary["needs_review_superseded"] == 2
    assert summary["slug_superseded"] == 0
    for cid in ("thongbao_76_0", "thongbao_77_1"):
        meta = json.loads(table.by_chunk_id(cid)["metadata_json"])
        assert meta["is_superseded"] is True
        assert "needs_review" in meta["superseded_reason"].lower()
    # the regular chunk is untouched
    meta = json.loads(table.by_chunk_id("thongbao_80_0")["metadata_json"])
    assert meta.get("is_superseded", False) is False


# ---------------------------------------------------------------------------
# 4. Regular non-slug, non-needs_review chunks are NEVER touched.
# ---------------------------------------------------------------------------
def test_regular_rows_untouched(mod):
    table = FakeTable(rows=[
        _row("chunked_1_chunk_2", REGULAR_TEXT, {"data_year": 2026}),
    ])
    summary = mod.supersede_slug_noise(table=table, embedder=FakeEmbedder())

    assert table.del_calls == []
    assert table.add_calls == []
    assert summary["slug_superseded"] == 0
    assert summary["needs_review_superseded"] == 0


# ---------------------------------------------------------------------------
# 5. Idempotent: already-superseded rows are a no-op.
# ---------------------------------------------------------------------------
def test_already_superseded_noop(mod):
    table = FakeTable(rows=[
        _row("thi_sinh_bogddt_tt_08_dieu_2_2025", SLUG_TEXT,
             {"is_superseded": True, "superseded_reason": "already done", "data_year": 2025}),
    ])
    summary = mod.supersede_slug_noise(table=table, embedder=FakeEmbedder())

    assert table.add_calls == []
    assert table.del_calls == []
    assert summary["slug_superseded"] == 0


# ---------------------------------------------------------------------------
# 6. Predicate helpers behave correctly (text + id based).
# ---------------------------------------------------------------------------
def test_predicate_helpers(mod):
    assert mod._is_slug_noise(SLUG_TEXT, "thi_sinh_bogddt_tt_08_dieu_2_2025") is True
    assert mod._is_slug_noise(CURATED_TEXT, "husc_summary_general_2025") is False
    assert mod._is_slug_noise(REGULAR_TEXT, "chunked_1_chunk_2") is False
    assert mod._is_slug_noise("### 2 17963d0e 0001 chunk 0\n\nDanh muc...", "2_17963d0e_0001_chunk_0") is True
    # body on the SAME line after the ASCII slug header (no newline) is still noise
    assert mod._is_slug_noise(
        "### thi sinh Khoa Hoc cong thuc xet tuyen 2025  Trong tuyển sinh đại học năm 2025 trên phạm vi...",
        "thi_sinh_Khoa_Hoc_cong_thuc_xet_tuyen_2025",
    ) is True


# ---------------------------------------------------------------------------
# 7. Constants locked: keep curated prefixes, thresholds.
# ---------------------------------------------------------------------------
def test_constants_locked(mod):
    assert "husc_" in mod.KEEP_PREFIXES
    assert mod.SLUG_HEADER_RE.match("### thi sinh bogddt tt 08 dieu 2 xet tuyen thang 2025") is not None


# ---------------------------------------------------------------------------
# 8. LIVE flat schema: slug + needs_review rows superseded via in-place UPDATE
#    (row count preserved, no delete/add).
# ---------------------------------------------------------------------------
def test_flat_schema_uses_update(mod):
    rows = [
        _flat_row("thi_sinh_bogddt_tt_08_dieu_2_2025", SLUG_TEXT, {"data_year": 2025}),
        _flat_row("thongbao_76_0", "Trang chủ Thông báo Tuyển sinh...", {"needs_review": True}),
        _flat_row("husc_summary_general_2025", CURATED_TEXT, {"data_year": 2025}),
    ]
    table = FakeFlatTable(rows=list(rows))

    summary = mod.supersede_slug_noise(table=table, embedder=FakeEmbedder())

    assert table.update_calls, "flat path must use table.update()"
    assert table.del_calls == []
    assert table.add_calls == []
    assert summary["slug_superseded"] == 1
    assert summary["needs_review_superseded"] == 1
    assert len(table.rows) == 3  # count preserved
    # slug row superseded
    slug_meta = json.loads(table.by_id("thi_sinh_bogddt_tt_08_dieu_2_2025")["metadata_json"])
    assert slug_meta["is_superseded"] is True
    # needs_review row superseded
    nr_meta = json.loads(table.by_id("thongbao_76_0")["metadata_json"])
    assert nr_meta["is_superseded"] is True
    # curated husc row untouched
    cur_meta = json.loads(table.by_id("husc_summary_general_2025")["metadata_json"])
    assert cur_meta.get("is_superseded", False) is False


# ---------------------------------------------------------------------------
# 9. LIVE flat schema: a row whose id is duplicated in the table is SKIPPED.
# ---------------------------------------------------------------------------
def test_flat_schema_dup_id_skipped(mod):
    rows = [
        _flat_row("dup_id", SLUG_TEXT, {"data_year": 2025}),
        _flat_row("dup_id", "Một nội dung khác với cùng id...", {"data_year": 2026}),
    ]
    table = FakeFlatTable(rows=list(rows))

    summary = mod.supersede_slug_noise(table=table, embedder=FakeEmbedder())

    assert summary["slug_superseded"] == 0
    assert table.update_calls == []
    # sibling untouched
    meta = json.loads(table.by_id("dup_id")["metadata_json"])
    assert meta.get("is_superseded", False) is False
