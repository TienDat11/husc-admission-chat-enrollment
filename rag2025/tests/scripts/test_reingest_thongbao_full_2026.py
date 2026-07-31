"""@spec(S18.2-T2) TDD test for scripts/reingest_thongbao_full_2026.py.

Strategy: fake LanceDB + fake embedder. Verifies:

* id71 fragment chunks (chunked_20_chunk_0..5) get deleted and replaced
  by a single husc_tb71_vneid_2026 chunk containing all 5 VneID steps +
  thisinh.thitotnghiepthpt.edu.vn + 01/5-20/5/2026.
* id73: 2 NEW chunks (husc_tb73_lephi_2026 + husc_tb73_doituong_xtt_2026)
  added WITHOUT touching the 66 existing id73 chunks — i.e. regression
  guard on the 20 chunks that currently MATCH "xét tuyển thẳng".
* Idempotent re-run: row count unchanged.
* Embedding dim = 1024 for every added chunk.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "reingest_thongbao_full_2026.py"


@pytest.fixture
def fix_module():
    spec = importlib.util.spec_from_file_location("reingest_thongbao_full_2026", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fakes
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

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)


class FakeEmbedder:
    DIM = 1024

    def encode_documents(self, texts: List[str]):
        return np.stack([np.full(self.DIM, 0.1 * (i + 1), dtype=np.float32) for i in range(len(texts))])

    def encode_single(self, text: str):
        return np.full(self.DIM, 0.5, dtype=np.float32)


def _seed_id71_id73(table: FakeTable, xtt_count: int = 20) -> None:
    """Seed 6 id71 fragments + N id73 chunks (incl. 'xét tuyển thẳng')."""
    for i in range(6):
        table._rows.append({
            "chunk_id": f"chunked_20_chunk_{i}",
            "text": f"id71 fragment {i}",
            "metadata_json": json.dumps({"data_year": 2026, "info_type": "dang_ky"}),
            "vector": [0.0] * 1024,
            "source": "tuyensinh.husc.edu.vn/thongbao.php?id=71",
        })
    for j in range(xtt_count):
        table._rows.append({
            "chunk_id": f"chunked_22_chunk_{j}",
            "text": f"id73 XTT chunk {j} xét tuyển thẳng",
            "metadata_json": json.dumps({"data_year": 2026, "info_type": "diem_chuan"}),
            "vector": [0.0] * 1024,
            "source": "tuyensinh.husc.edu.vn/thongbao.php?id=73",
        })


# ---------------------------------------------------------------------------
# 1) Constants locked
# ---------------------------------------------------------------------------
def test_constants_locked(fix_module):
    assert fix_module.ID71_FRAGMENT_CHUNK_IDS == [f"chunked_20_chunk_{i}" for i in range(6)]
    assert fix_module.ID71_VNEID_CHUNK_ID == "husc_tb71_vneid_2026"
    assert fix_module.LEPHI_CHUNK_ID == "husc_tb73_lephi_2026"
    assert fix_module.DOITUONG_CHUNK_ID == "husc_tb73_doituong_xtt_2026"
    # VneID body
    assert "VneID" in fix_module.ID71_VNEID_TEXT
    assert "thisinh.thitotnghiepthpt.edu.vn" in fix_module.ID71_VNEID_TEXT
    assert "01/5" in fix_module.ID71_VNEID_TEXT and "20/5" in fix_module.ID71_VNEID_TEXT
    # Lệ phí body
    assert "30.000" in fix_module.LEPHI_TEXT
    assert "EMS" in fix_module.LEPHI_TEXT
    # 8 đối tượng XTT
    assert fix_module.DOITUONG_META["doi_tuong_xtt"] and len(fix_module.DOITUONG_META["doi_tuong_xtt"]) == 8
    txt = fix_module.DOITUONG_TEXT
    for needle in [
        "Anh hùng Lao động",
        "Chiến sĩ thi đua toàn quốc",
        "học sinh giỏi",
        "Khoa học kỹ thuật",
        "nước ngoài",
        "khuyết tật nặng",
        "dân tộc thiểu số",
    ]:
        assert needle in txt, f"missing in doi_tuong text: {needle}"


# ---------------------------------------------------------------------------
# 2) Dry-run is non-mutating
# ---------------------------------------------------------------------------
def test_dry_run_no_mutation(fix_module):
    table = FakeTable()
    _seed_id71_id73(table)
    summary = fix_module.plan_changes(embedder=FakeEmbedder(), table=table, dry_run=True)
    assert summary["dry_run"] is True
    assert table.add_calls == []
    assert table.del_calls == []
    assert summary["id71"]["added"] == "husc_tb71_vneid_2026"
    assert sorted(summary["id73"]["added"]) == ["husc_tb73_doituong_xtt_2026", "husc_tb73_lephi_2026"]


# ---------------------------------------------------------------------------
# 3) Live: id71 fragments replaced
# ---------------------------------------------------------------------------
def test_id71_fragments_replaced_with_vneid(fix_module, tmp_path, monkeypatch):
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()
    _seed_id71_id73(table, xtt_count=20)
    rows_before = len(table.rows)

    summary = fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    # 6 fragments deleted + 1 vneid added + 2 id73 added = -6 + 3 = -3
    # id73 deletes are pre-emptive (no rows exist yet on first run → rows_deleted=6)
    assert summary["rows_deleted"] == 6
    assert summary["rows_added"] == 3

    # No fragment left
    for i in range(6):
        assert not any(r["chunk_id"] == f"chunked_20_chunk_{i}" for r in table.rows)
    # Single consolidated VneID chunk present
    vneid = next(r for r in table.rows if r["chunk_id"] == "husc_tb71_vneid_2026")
    meta = json.loads(vneid["metadata_json"])
    assert meta["info_type"] == "dang_ky_vneid"
    assert meta["data_year"] == 2026
    # 5 bước + URL + mốc ngày
    for needle in ["VneID", "thisinh.thitotnghiepthpt.edu.vn", "01/5", "20/5"]:
        assert needle in vneid["text"]
    # dim
    assert len(vneid["vector"]) == 1024


# ---------------------------------------------------------------------------
# 4) Regression: 20 XTT chunks retained
# ---------------------------------------------------------------------------
def test_id73_xtt_20_chunks_preserved(fix_module, tmp_path, monkeypatch):
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()
    _seed_id71_id73(table, xtt_count=20)

    fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    xtt_chunks = [r for r in table.rows if "xét tuyển thẳng" in r["text"]]
    # The 20 seeded XTT chunks must remain + the 2 new id73 chunks must also
    # contain "xét tuyển thẳng" → at least 22.
    assert len(xtt_chunks) >= 22, f"regressed: only {len(xtt_chunks)} XTT chunks left"


# ---------------------------------------------------------------------------
# 5) Live: id73 new chunks present with 30.000 + 8 đối tượng
# ---------------------------------------------------------------------------
def test_id73_lephi_and_doituong_added(fix_module, tmp_path, monkeypatch):
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()
    _seed_id71_id73(table, xtt_count=20)
    fix_module.plan_changes(embedder=FakeEmbedder(), table=table)

    lephi = next(r for r in table.rows if r["chunk_id"] == "husc_tb73_lephi_2026")
    assert "30.000" in lephi["text"]
    assert "EMS" in lephi["text"]

    doituong = next(r for r in table.rows if r["chunk_id"] == "husc_tb73_doituong_xtt_2026")
    meta = json.loads(doituong["metadata_json"])
    assert len(meta["doi_tuong_xtt"]) == 8
    for needle in ["Anh hùng", "Chiến sĩ thi đua", "học sinh giỏi", "Khoa học kỹ thuật", "khuyết tật"]:
        assert needle in doituong["text"]


# ---------------------------------------------------------------------------
# 6) Idempotent re-run
# ---------------------------------------------------------------------------
def test_idempotent_second_run(fix_module, tmp_path, monkeypatch):
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()
    _seed_id71_id73(table, xtt_count=20)
    fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    rows_after_run1 = len(table.rows)
    fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    rows_after_run2 = len(table.rows)
    assert rows_after_run1 == rows_after_run2


# ---------------------------------------------------------------------------
# 7) JSONL output
# ---------------------------------------------------------------------------
def test_jsonl_write_for_build_graph(fix_module, tmp_path, monkeypatch):
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", out)
    table = FakeTable()
    _seed_id71_id73(table, xtt_count=0)
    summary = fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    assert summary["jsonl_written"] == 3
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3
    ids = {json.loads(ln)["chunk_id"] for ln in lines}
    assert ids == {
        "husc_tb71_vneid_2026",
        "husc_tb73_lephi_2026",
        "husc_tb73_doituong_xtt_2026",
    }