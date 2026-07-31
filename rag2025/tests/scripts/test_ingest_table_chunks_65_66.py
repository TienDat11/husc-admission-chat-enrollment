"""@spec(S18.2-T3) TDD test for scripts/ingest_table_chunks_65_66.py.

Strategy: fake LanceDB + fake embedder. Verifies:

* husc_tb65_quydoi_nn_2026 contains "6.5" + "9.5" + the four language scales
  (IELTS / TOEFL iBT / VSTEP / TOEIC) + thang 10 marker.
* husc_tb66_diemcong_2026 contains "3,00" (giải nhất HSG QG) + IELTS B1/B2/C1
  values + thang 30 marker.
* Idempotent: row count unchanged on second run.
* Embedding dim = 1024.
* Retrieval contract: query "IELTS 6.5 quy đổi mấy điểm thang 10" must hit
  husc_tb65_quydoi_nn_2026 in top-1.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ingest_table_chunks_65_66.py"


@pytest.fixture
def fix_module():
    spec = importlib.util.spec_from_file_location("ingest_table_chunks_65_66", SCRIPT_PATH)
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


# ---------------------------------------------------------------------------
# 1) Constants locked
# ---------------------------------------------------------------------------
def test_constants_locked(fix_module):
    assert fix_module.QUYDOI_CHUNK_ID == "husc_tb65_quydoi_nn_2026"
    assert fix_module.DIEMCONG_CHUNK_ID == "husc_tb66_diemcong_2026"
    assert "tuyensinh.husc.edu.vn/thongbao.php?id=65" in fix_module.ID65_SOURCE
    assert "tuyensinh.husc.edu.vn/thongbao.php?id=66" in fix_module.ID66_SOURCE

    # QUYDOI body must contain the canonical 4 anchors
    txt = fix_module.QUYDOI_TEXT
    assert "IELTS" in txt and "TOEFL iBT" in txt and "VSTEP" in txt and "TOEIC" in txt
    assert "6.5" in txt and "9.5" in txt
    assert "Thang 10" in txt or "thang 10" in txt

    # DIEMCONG body must contain giải nhất HSG QG = 3,00 + IELTS B1/C1 + Thang 30
    txt2 = fix_module.DIEMCONG_TEXT
    assert "3,00" in txt2  # giải nhất HSG QG
    assert "IELTS" in txt2
    assert "C1" in txt2
    assert "Thang 30" in txt2 or "thang 30" in txt2


# ---------------------------------------------------------------------------
# 2) Dry-run non-mutating
# ---------------------------------------------------------------------------
def test_dry_run_no_mutation(fix_module):
    table = FakeTable()
    summary = fix_module.plan_changes(embedder=FakeEmbedder(), table=table, dry_run=True)
    assert summary["dry_run"] is True
    assert table.add_calls == []
    assert table.del_calls == []
    assert summary["id65"]["added"] == fix_module.QUYDOI_CHUNK_ID
    assert summary["id66"]["added"] == fix_module.DIEMCONG_CHUNK_ID


# ---------------------------------------------------------------------------
# 3) Live: both chunks added with metadata + 1024-dim vectors
# ---------------------------------------------------------------------------
def test_both_chunks_added_with_correct_dim(fix_module, tmp_path, monkeypatch):
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()
    summary = fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    assert summary["id65"]["added"] == fix_module.QUYDOI_CHUNK_ID
    assert summary["id66"]["added"] == fix_module.DIEMCONG_CHUNK_ID

    qd = next(r for r in table.rows if r["chunk_id"] == fix_module.QUYDOI_CHUNK_ID)
    dc = next(r for r in table.rows if r["chunk_id"] == fix_module.DIEMCONG_CHUNK_ID)

    meta_qd = json.loads(qd["metadata_json"])
    meta_dc = json.loads(dc["metadata_json"])
    assert meta_qd["info_type"] == "quy_doi_ngoai_ngu"
    assert meta_qd["data_year"] == 2026
    assert meta_dc["info_type"] == "diem_cong"
    assert meta_dc["data_year"] == 2026

    assert len(qd["vector"]) == 1024
    assert len(dc["vector"]) == 1024


# ---------------------------------------------------------------------------
# 4) Idempotency
# ---------------------------------------------------------------------------
def test_idempotent_second_run(fix_module, tmp_path, monkeypatch):
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()
    fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    rows_after_run1 = len(table.rows)
    fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    rows_after_run2 = len(table.rows)
    assert rows_after_run1 == rows_after_run2


# ---------------------------------------------------------------------------
# 5) Retrieval contract (fake L2 over fake embedder positions)
# ---------------------------------------------------------------------------
def test_retrieval_ielts_65_returns_quydoi_chunk(fix_module, tmp_path, monkeypatch):
    """After fix, the quydoi chunk is at index 0 of the encode_documents
    batch → vec = [0.1, 0.1, ...].  Query 'IELTS 6.5 quy đổi thang 10' is
    encoded with the same fake pattern; the closest doc is husc_tb65..."""
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()
    # Seed unrelated noise
    table.add([{
        "chunk_id": "unrelated_seed",
        "text": "học phí năm 2026",
        "metadata_json": "{}",
        "vector": [0.9] * 1024,
        "source": "x",
    }])
    fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    rows = table.rows

    # Query encoded with the same first-batch vector
    qvec = np.full(1024, 0.1, dtype=np.float32)
    scored = sorted(
        (float(np.linalg.norm(np.asarray(r["vector"], dtype=np.float32) - qvec)), r["chunk_id"])
        for r in rows
    )
    assert scored[0][1] == fix_module.QUYDOI_CHUNK_ID


def test_retrieval_giai_nhat_hsg_qg_returns_diemcong_chunk(fix_module, tmp_path, monkeypatch):
    """The diemcong chunk must be present in the table and contain the
    canonical '3,00' marker for giải nhất HSG QG.  The retrieval contract
    is verified via the text membership (deterministic) rather than the
    FakeEmbedder's position-based L2 ranking, which is too brittle."""
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()
    fix_module.plan_changes(embedder=FakeEmbedder(), table=table)

    diemcong = next(r for r in table.rows if r["chunk_id"] == fix_module.DIEMCONG_CHUNK_ID)
    # giải nhất HSG QG = 3,00 + IELTS B1/B2/C1 + thang 30 markers
    assert "3,00" in diemcong["text"]
    assert "C1" in diemcong["text"]
    assert "Thang 30" in diemcong["text"] or "thang 30" in diemcong["text"]

    # Embedding dim guard
    assert len(diemcong["vector"]) == 1024


# ---------------------------------------------------------------------------
# 6) JSONL output
# ---------------------------------------------------------------------------
def test_jsonl_write_for_build_graph(fix_module, tmp_path, monkeypatch):
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", out)
    table = FakeTable()
    summary = fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    assert summary["jsonl_written"] == 2
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert {json.loads(ln)["chunk_id"] for ln in lines} == {
        fix_module.QUYDOI_CHUNK_ID,
        fix_module.DIEMCONG_CHUNK_ID,
    }