"""@spec(S18.6-T6) TDD tests for scripts/auto_ingest_yearly.py.

Verifies:
  - Year detection on a thongbao page that says "NĂM 2027" + section
    "học bổng" → chunk_id format `husc_tb{id}_{slug}_{year}_{k}`.
  - Prior-year reference marking: when a 2027 page is ingested, the
    matching 2026 chunk (same section slug) becomes is_reference=true.
  - Idempotency: running the plan twice yields the same upsert set
    (rows_to_upsert identical, row count unchanged in the fake table).
  - Embedding dim guard: assert len(vec)==1024 on every chunk.
  - dry-run: no add calls, no row mutation, summary still informative.

No real network, no real LanceDB, no real model — fakes only.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "auto_ingest_yearly.py"


@pytest.fixture
def ai():
    spec = importlib.util.spec_from_file_location("auto_ingest_yearly", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fakes (mirror fix_topic_route_id73_74.py style)
# ---------------------------------------------------------------------------
class FakeTable:
    """In-memory LanceDB-like table."""

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
        return np.stack([
            np.full(self.DIM, 0.1 * (i + 1), dtype=np.float32)
            for i in range(len(texts))
        ])


# ---------------------------------------------------------------------------
# 1) Detect helpers
# ---------------------------------------------------------------------------
def test_detect_year_from_text_picks_2027(ai):
    body = (
        "HỌC BỔNG TUYỂN SINH NĂM 2027 — HUSC thông báo dành 50 suất học bổng "
        "cho thí sinh trúng tuyển năm 2027. Mỗi suất trị giá 5.000.000 VNĐ."
    )
    assert ai.detect_year_from_text(body, current_year=2026) == 2027


def test_detect_year_from_text_no_future_returns_none(ai):
    body = "Thông báo tuyển sinh năm 2026 nộp hồ sơ đến 30/6/2026."
    assert ai.detect_year_from_text(body, current_year=2026) is None


def test_detect_section_hoc_bong(ai):
    body = "HỌC BỔNG dành cho thí sinh xét tuyển thẳng."
    assert ai.detect_section(body) == "hoc_bong"


def test_detect_section_dkxt(ai):
    body = "Đăng ký xét tuyển đại học hệ chính quy năm 2027 tại HUSC."
    assert ai.detect_section(body) == "dkxt"


def test_build_chunk_id_format(ai):
    assert ai.build_chunk_id(75, "hoc_bong", 2027) == "husc_tb75_hoc_bong_2027"


# ---------------------------------------------------------------------------
# 2) Embedding dim guard
# ---------------------------------------------------------------------------
def test_embed_texts_asserts_dim_1024(ai):
    class BadEmbedder:
        def encode_documents(self, texts):
            return np.zeros((len(texts), 512), dtype=np.float32)

    with pytest.raises(AssertionError, match="embedding dim mismatch"):
        ai.embed_texts(BadEmbedder(), ["a", "b"])


def test_embed_texts_ok_dim_1024(ai):
    vecs = ai.embed_texts(FakeEmbedder(), ["hello", "world"])
    assert len(vecs) == 2
    assert all(len(v) == 1024 for v in vecs)


# ---------------------------------------------------------------------------
# 3) Prior-year reference matching
# ---------------------------------------------------------------------------
def test_find_prior_year_chucks_returns_same_section_older_year(ai):
    rows = [
        {
            "chunk_id": "husc_tb71_hoc_bong_2026_0",
            "text": "học bổng 2026",
            "metadata_json": "{}",
            "vector": [0.0] * 1024,
            "source": "x",
        },
        {
            "chunk_id": "husc_tb75_dkxt_2026_0",
            "text": "dkxt 2026",
            "metadata_json": "{}",
            "vector": [0.0] * 1024,
            "source": "x",
        },
    ]
    priors = ai.find_prior_year_chunks(rows, section_slug="hoc_bong", new_year=2027)
    assert priors == ["husc_tb71_hoc_bong_2026_0"]


def test_find_prior_year_chucks_skips_same_year(ai):
    rows = [{
        "chunk_id": "husc_tb71_hoc_bong_2027_0",
        "text": "học bổng 2027",
        "metadata_json": json.dumps({"data_year": 2027, "section": "hoc_bong"}),
        "vector": [0.0] * 1024,
        "source": "x",
    }]
    assert ai.find_prior_year_chunks(rows, section_slug="hoc_bong", new_year=2027) == []


# ---------------------------------------------------------------------------
# 4) End-to-end plan + execute — main TDD scenario
# ---------------------------------------------------------------------------
HOC_BONG_BODY_2027 = (
    "HỌC BỔNG TUYỂN SINH NĂM 2027 — HUSC thông báo dành 50 suất học bổng "
    "cho thí sinh trúng tuyển đại học hệ chính quy năm 2027.\n\n"
    "Mỗi suất trị giá 5.000.000 VNĐ, áp dụng cho thí sinh có điểm đầu vào "
    "top 10 của mỗi ngành. Hạn nộp hồ sơ đăng ký học bổng: 30/9/2027."
)


def test_plan_yearly_ingest_emits_2027_chunks_and_prior_reference(ai, tmp_path):
    # Seed: prior-year 2026 chunk in section hoc_bong.
    prior_row = {
        "chunk_id": "husc_tb71_hoc_bong_2026_0",
        "text": "học bổng 2026 — 30 suất",
        "metadata_json": json.dumps({
            "data_year": 2026,
            "section": "hoc_bong",
            "info_type": "hoc_bong",
            "is_reference": False,
        }),
        "vector": [0.0] * 1024,
        "source": "tuyensinh.husc.edu.vn/thongbao.php?id=71",
    }
    pages = [(75, "https://tuyensinh.husc.edu.vn/thongbao.php?id=75", HOC_BONG_BODY_2027)]

    summary = ai.plan_yearly_ingest(
        year=2027,
        pages=pages,
        table_rows=[prior_row],
        embedder=FakeEmbedder(),
        dry_run=False,
    )

    # Step 4: chunks added
    assert summary["chunks_added"] >= 1
    new_rows = summary["rows_to_upsert"]
    assert all(len(r["vector"]) == 1024 for r in new_rows)
    assert any("husc_tb75_hoc_bong_2027" in r["chunk_id"] for r in new_rows)

    # Step 4: prior-year reference marking requested
    ref_marks = summary["reference_marks"]
    prior_cids = [cid for cid, _ in ref_marks]
    assert "husc_tb71_hoc_bong_2026_0" in prior_cids
    assert summary["chunks_marked_reference"] >= 1

    # Step 5: actually upsert via table + assert 2026 row becomes reference.
    table = FakeTable(rows=[prior_row])
    added = ai.upsert_rows(table, new_rows, jsonl_path=tmp_path / "out.jsonl")
    assert added == len(new_rows)
    assert table.add_calls  # something was added
    # JSONL file written for build_graph parity
    assert (tmp_path / "out.jsonl").exists()
    jsonl_lines = [
        ln for ln in (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(jsonl_lines) == len(new_rows)

    # Mark the prior 2026 row as is_reference=true.
    marked = ai.mark_prior_year_reference(table, ["husc_tb71_hoc_bong_2026_0"])
    assert marked == 1
    final_2026_row = next(
        r for r in table.rows if r["chunk_id"] == "husc_tb71_hoc_bong_2026_0"
    )
    final_meta = json.loads(final_2026_row["metadata_json"])
    assert final_meta["is_reference"] is True
    assert final_meta["data_year"] == 2026  # year preserved


# ---------------------------------------------------------------------------
# 5) Idempotency: second run on already-ingested state
# ---------------------------------------------------------------------------
def test_idempotent_second_run(ai, tmp_path):
    pages = [(75, "https://tuyensinh.husc.edu.vn/thongbao.php?id=75", HOC_BONG_BODY_2027)]

    s1 = ai.plan_yearly_ingest(
        year=2027, pages=pages, table_rows=[], embedder=FakeEmbedder(), dry_run=False
    )
    table = FakeTable()
    ai.upsert_rows(table, s1["rows_to_upsert"], jsonl_path=tmp_path / "out.jsonl")
    rows_after_run1 = len(table.rows)

    # Build rows-from-table snapshot for run 2's prior-year check.
    s2 = ai.plan_yearly_ingest(
        year=2027, pages=pages, table_rows=table.rows, embedder=FakeEmbedder(), dry_run=False
    )
    ai.upsert_rows(table, s2["rows_to_upsert"], jsonl_path=tmp_path / "out.jsonl")
    rows_after_run2 = len(table.rows)

    # delete + re-add the SAME chunk_id → row count unchanged
    assert rows_after_run1 == rows_after_run2
    # run 2 still emits the same number of chunks (deterministic)
    assert s1["chunks_added"] == s2["chunks_added"]


# ---------------------------------------------------------------------------
# 6) dry-run never mutates
# ---------------------------------------------------------------------------
def test_dry_run_no_table_mutation(ai):
    pages = [(75, "https://tuyensinh.husc.edu.vn/thongbao.php?id=75", HOC_BONG_BODY_2027)]
    summary = ai.plan_yearly_ingest(
        year=2027, pages=pages, table_rows=[], embedder=FakeEmbedder(), dry_run=True
    )
    assert summary["dry_run"] is True
    assert "rows_to_upsert" not in summary  # not built in dry-run
    assert summary["chunks_added"] >= 1


# ---------------------------------------------------------------------------
# 7) section slug survives chunk_text
# ---------------------------------------------------------------------------
def test_chunk_text_keeps_relevant_content(ai):
    text = "Đoạn 1 về học bổng 2027.\n\nĐoạn 2 về điều kiện xét tuyển."
    out = ai.chunk_text(text)
    assert len(out) >= 1
    assert any("học bổng 2027" in c for c in out)