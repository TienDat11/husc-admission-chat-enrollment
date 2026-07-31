"""@spec(S18.2-T4b) TDD test for scripts/fix_canonical_64_73.py.

Strategy: fake LanceDB + fake embedder. Verifies:

* Constants locked: 29 ngành, 2.079 chỉ tiêu, 8 đối tượng XTT listed in
  CANONICAL_TEXT.
* Dry-run is non-mutating (no add/delete).
* Live: every BAD_CHUNK_FIXES chunk_id gets deleted-by-id then re-added
  with corrected text; metadata carries data_year=2026, nganh_count=29,
  tong_chi_tieu=2079, is_superseded=false.
* Live: husc_canonical_2079_29nganh_2026 chunk added with 8 đối tượng
  XTT in metadata.doi_tuong_xtt list AND in text body.
* GUARDED_CHUNK_IDS (historical chunks) are NEVER mutated.
* Idempotent re-run: row count unchanged.
* Embedding dim = 1024 for every added chunk.
* JSONL output: written for every re-added chunk + canonical aggregate.
* After plan_changes: "25 ngành" / "1.969" / "1.939" / "26 ngành" /
  "28 ngành" no longer appear in any rewritten chunk's text (only in
  the historical guard rows).
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "fix_canonical_64_73.py"
)


@pytest.fixture
def fix_module():
    spec = importlib.util.spec_from_file_location(
        "fix_canonical_64_73", SCRIPT_PATH
    )
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
        import numpy as np
        return np.stack(
            [np.full(self.DIM, 0.1 * (i + 1), dtype=np.float32)
             for i in range(len(texts))]
        )

    def encode_single(self, text: str):
        import numpy as np
        return np.full(self.DIM, 0.5, dtype=np.float32)


def _seed_bad_and_guarded(table: FakeTable) -> None:
    """Seed the live table with all known bad chunks + 3 historical guards."""
    seeds = [
        ("thay_doi_tuyensinh_2026",
         "Tuyển sinh HUSC năm 2026 … Tổng chỉ tiêu 1.939 cho 25 ngành + 1 chương "
         "trình song song CNTT Việt-Nhật.", 2026, False),
        ("liet_ke_26_nganh_2026",
         "Danh sách đầy đủ 25 ngành + 1 chương trình song song CNTT Việt-Nhật tại HUSC "
         "tuyển sinh năm 2026 …", 2026, False),
        ("bang_xep_hang_chi_tieu_2026",
         "Bảng xếp hạng chỉ tiêu 25 ngành + 1 chương trình song song tại HUSC năm "
         "2026 (cao xuống thấp):", 2026, True),
        ("qa_chi_tieu_khoi_xet_2026_overview",
         "Năm 2026, … tuyển sinh 28 ngành đại học chính quy …", 2026, False),
        ("tuyensinh_overview_2026_v2",
         "Trường Đại học Khoa học, Đại học Huế (HUSC) - Mã trường: DHT. Năm 2026: "
         "tuyển sinh 28 ngành đào tạo đại học chính quy … Tổng chỉ tiêu dự kiến: "
         "2.079.", 2026, False),
        ("bang_xep_hang_chi_tieu_2026_v2",
         "Bảng xếp hạng 28 ngành HUSC năm 2026 theo chỉ tiêu …", 2026, False),
        ("liet_ke_28_nganh_2026_v2",
         "Danh sách đầy đủ 28 ngành đào tạo đại học chính quy của Trường Đại học "
         "Khoa học …", 2026, False),
        ("nganh_ngung_tuyen_2026",
         "DANH SÁCH NGÀNH HUSC ĐÃ NGỮNG TUYỂN SINH TỪ NĂM 2026 … 25 ngành + 1 chương "
         "trình song song …", 2026, False),
        ("qa_xet_hoc_ba_2026_v2",
         "Câu hỏi: HUSC 2026 — Xét học bạ … 26 ngành × 4 phương thức.", 2026, False),
        ("husc_vision_tb40_img00_2026",
         "[Trích xuất từ hình ảnh] … TỔNG CHỈ TIÊU: 2.079. SỐ NGÀNH/CTĐT: 28 …",
         2026, False),
        ("tuyensinh_overview_2026",
         "Trường Đại học Khoa học, Đại học Huế … tuyển sinh 25 ngành + 1 chương "
         "trình song song CNTT Việt-Nhật, tổng chỉ tiêu 1969.", 2026, True),
        # Historical guards — must NOT be mutated.
        ("historical_chi_tieu_2025_husc",
         "Chỉ tiêu HUSC 2025: Hán Nôm 32 … tổng 1.844 sinh viên.", 2025, False),
        ("diem_chuan_2025_full",
         "Điểm chuẩn HUSC năm 2025 — ĐẦY ĐỦ 26 ngành × 4 phương thức.", 2025, False),
        ("husc_summary_general_2025",
         "Năm 2025, Trường Đại học Khoa học - Đại học Huế tuyển sinh tổng cộng 25 "
         "ngành + 1 chương trình song song …", 2025, False),
    ]
    for cid, txt, dy, sup in seeds:
        meta = {"data_year": dy, "is_superseded": sup, "info_type": "seed"}
        table._rows.append({
            "chunk_id": cid,
            "text": txt,
            "metadata_json": json.dumps(meta, ensure_ascii=False),
            "vector": [0.0] * 1024,
            "source": "tuyensinh.husc.edu.vn/thongbao.php?id=64",
        })


# ---------------------------------------------------------------------------
# 1) Constants locked
# ---------------------------------------------------------------------------
def test_constants_locked(fix_module):
    assert fix_module.CANONICAL_NGANH == 29
    assert fix_module.CANONICAL_CHI_TIEU == 2079
    assert fix_module.CANONICAL_CHUNK_ID == "husc_canonical_2079_29nganh_2026"
    # 8 đối tượng XTT in canonical text
    txt = fix_module.CANONICAL_TEXT
    for needle in [
        "Anh hùng Lao động",
        "Chiến sĩ thi đua toàn quốc",
        "học sinh giỏi",
        "Khoa học kỹ thuật",
        "nước ngoài",
        "khuyết tật nặng",
        "dân tộc thiểu số",
    ]:
        assert needle in txt, f"missing in canonical text: {needle}"
    # 8 entries in metadata
    assert len(fix_module.CANONICAL_META["doi_tuong_xtt"]) == 8
    # 29 + 2.079 present
    assert "29 ngành/CTĐT" in txt or "29 ngành" in txt
    assert "2.079" in txt
    # Guards
    assert "historical_chi_tieu_2025_husc" in fix_module.GUARDED_CHUNK_IDS
    assert "diem_chuan_2025_full" in fix_module.GUARDED_CHUNK_IDS
    assert "husc_summary_general_2025" in fix_module.GUARDED_CHUNK_IDS
    # Bad chunk list covers ≥10 ids (matches plan §T4b.1)
    assert len(fix_module.BAD_CHUNK_FIXES) >= 10


# ---------------------------------------------------------------------------
# 2) Dry-run is non-mutating
# ---------------------------------------------------------------------------
def test_dry_run_no_mutation(fix_module):
    table = FakeTable()
    _seed_bad_and_guarded(table)
    summary = fix_module.plan_changes(embedder=FakeEmbedder(), table=table, dry_run=True)
    assert summary["dry_run"] is True
    assert table.add_calls == []
    assert table.del_calls == []
    assert summary["canonical_added"] == "husc_canonical_2079_29nganh_2026"
    assert len(summary["fixed"]) == len(fix_module.BAD_CHUNK_FIXES)


# ---------------------------------------------------------------------------
# 3) Live: bad chunks rewritten with canonical numbers
# ---------------------------------------------------------------------------
def test_bad_chunks_rewritten_with_2079_29(fix_module, tmp_path, monkeypatch):
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()
    _seed_bad_and_guarded(table)
    rows_before = len(table.rows)

    summary = fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    # 11 bad chunks rewritten + 1 canonical added = 12 adds
    assert summary["rows_added"] == len(fix_module.BAD_CHUNK_FIXES) + 1
    # At least 11 deletes (historical guards also delete-attempted but
    # they're skipped → summary["skipped"] contains them; the delete-by-id
    # still runs but is a no-op since those ids don't exist as bad fixes
    # in BAD_CHUNK_FIXES).
    assert summary["rows_deleted"] == len(fix_module.BAD_CHUNK_FIXES)

    # Verify each rewritten chunk carries the corrected numbers + data_year 2026
    for fix in fix_module.BAD_CHUNK_FIXES:
        cid = fix["chunk_id"]
        row = next(r for r in table.rows if r["chunk_id"] == cid)
        meta = json.loads(row["metadata_json"])
        assert meta["data_year"] == 2026
        assert meta.get("is_superseded") is False
        # meta carries canonical figures (either nganh_count or tong_chi_tieu)
        assert meta.get("nganh_count") == 29 or meta.get("tong_chi_tieu") == 2079, \
            f"{cid} missing canonical meta"
        # text contains the corrected number "29 ngành/CTĐT" or "2.079"
        assert "2.079" in row["text"] or "29 ngành" in row["text"], \
            f"{cid} text missing canonical numbers"
        # dim 1024
        assert len(row["vector"]) == 1024


# ---------------------------------------------------------------------------
# 4) Historical guards untouched
# ---------------------------------------------------------------------------
def test_historical_guards_not_mutated(fix_module, tmp_path, monkeypatch):
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()
    _seed_bad_and_guarded(table)

    fix_module.plan_changes(embedder=FakeEmbedder(), table=table)

    for cid in fix_module.GUARDED_CHUNK_IDS:
        rows = [r for r in table.rows if r["chunk_id"] == cid]
        assert rows, f"{cid} disappeared from table"
        row = rows[0]
        meta = json.loads(row["metadata_json"])
        # data_year stays 2025
        assert meta["data_year"] == 2025, f"{cid} data_year mutated"
        # Bad numbers remain (they are historical facts)
        assert ("25 ngành" in row["text"] or "26 ngành" in row["text"]
                or "1.844" in row["text"]), f"{cid} text mutated"


# ---------------------------------------------------------------------------
# 5) Canonical aggregate chunk added with 8 đối tượng XTT
# ---------------------------------------------------------------------------
def test_canonical_chunk_added_with_8_doituong(fix_module, tmp_path, monkeypatch):
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()
    _seed_bad_and_guarded(table)

    fix_module.plan_changes(embedder=FakeEmbedder(), table=table)

    row = next(r for r in table.rows if r["chunk_id"] == "husc_canonical_2079_29nganh_2026")
    meta = json.loads(row["metadata_json"])
    # 8 đối tượng
    assert len(meta["doi_tuong_xtt"]) == 8
    for needle in ["Anh hùng", "Chiến sĩ thi đua", "học sinh giỏi",
                   "Khoa học kỹ thuật", "khuyết tật", "dân tộc thiểu số"]:
        assert needle in row["text"], f"canonical chunk missing: {needle}"
    # Numbers
    assert "2.079" in row["text"]
    assert "29 ngành" in row["text"]
    assert len(row["vector"]) == 1024


# ---------------------------------------------------------------------------
# 6) Idempotent re-run: row count unchanged
# ---------------------------------------------------------------------------
def test_idempotent_second_run(fix_module, tmp_path, monkeypatch):
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()
    _seed_bad_and_guarded(table)

    fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    rows_run1 = len(table.rows)
    fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    rows_run2 = len(table.rows)
    assert rows_run1 == rows_run2


# ---------------------------------------------------------------------------
# 7) JSONL output: written for T7 build_graph parity
# ---------------------------------------------------------------------------
def test_jsonl_write_for_build_graph(fix_module, tmp_path, monkeypatch):
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", out)
    table = FakeTable()
    _seed_bad_and_guarded(table)

    summary = fix_module.plan_changes(embedder=FakeEmbedder(), table=table)
    # 11 fixes + 1 canonical = 12 rows
    assert summary["jsonl_written"] == len(fix_module.BAD_CHUNK_FIXES) + 1
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == len(fix_module.BAD_CHUNK_FIXES) + 1
    ids = {json.loads(ln)["chunk_id"] for ln in lines}
    assert "husc_canonical_2079_29nganh_2026" in ids


# ---------------------------------------------------------------------------
# 8) After plan_changes: rewritten 2026 chunks no longer carry 25/26/28 ngành
#    or 1.969/1.939 (historical guards are exempt by design).
# ---------------------------------------------------------------------------
def test_no_bad_numbers_in_rewritten_chunks(fix_module, tmp_path, monkeypatch):
    monkeypatch.setattr(fix_module, "CHUNKED_JSONL_OUT", tmp_path / "out.jsonl")
    table = FakeTable()
    _seed_bad_and_guarded(table)

    fix_module.plan_changes(embedder=FakeEmbedder(), table=table)

    bad_id_set = {fix["chunk_id"] for fix in fix_module.BAD_CHUNK_FIXES}
    # Word-boundary patterns: "25 ngành" must not appear (but "26 mã ngành"
    # which mentions a count + "ngành" is fine; "25 ngành" specifically
    # means the total ngành count is being asserted).
    forbidden_patterns = [
        re.compile(r"\b25\s*ngành"),
        re.compile(r"\b28\s*ngành"),
        re.compile(r"\b1\.969\b"),
        re.compile(r"\b1\.939\b"),
        re.compile(r"\b1\.844\b"),
    ]
    for row in table.rows:
        cid = row["chunk_id"]
        if cid in bad_id_set or cid == "husc_canonical_2079_29nganh_2026":
            for pat in forbidden_patterns:
                assert not pat.search(row["text"]), \
                    f"{cid} still contains forbidden pattern: {pat.pattern!r}"
