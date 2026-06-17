"""TDD test for scripts/fix_husc_tuition_2026.py (S18.1 — TDD data-ops).

Strategy: inject a fake LanceDB-shaped table (rows + add + delete + to_pandas)
and a fake embedder. Tests verify the corrector logic end-to-end (text-level
replacements + table-level delete+re-add + per-major lookup), with no real
LanceDB and no real Qwen3 model.

Coverage:
  * String-replacement maps (525.000 / 540.000) across the three separators
    that appear in the corpus: ``525.000`` (dot, the canonical form),
    ``525,000`` (comma), and ``525000`` (no separator).
  * The label ``"năm học 2026-2027 (theo Nghị định 238/2025/NĐ-CP)"`` is
    appended to the corrected text (or replaces an old year label like
    "năm 2025" or "năm học 2025-2026").
  * Khối V (600.000) chunks are NEVER touched.
  * 2025 (``data_year=2025``) chunks are NEVER touched.
  * Idempotency: re-running on already-correct text is a no-op.
  * Văn học (7229030) maps to 490.000; CNTT-VJ (7480201VJ) to 640.000;
    Triết học (7229001) to 0 (miễn học phí per Nghị định 238/2025/NĐ-CP).
  * The live ``husc_nganh_*_2026`` chunks get re-embedded + replaced in the
    table (delete + add), and the replacement row carries
    ``metadata.tuition_year="2026-2027"``,
    ``metadata.tuition_basis="Nghị định 238/2025/NĐ-CP"`` and
    ``metadata.tuition_source_url="https://tuyensinh.husc.edu.vn"``.
  * On-disk JSONL files get patched in place; 2025 records are skipped.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "fix_husc_tuition_2026.py"


@pytest.fixture
def corrector_module():
    spec = importlib.util.spec_from_file_location("fix_husc_tuition_2026", SCRIPT_PATH)
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
        norm = [r for r in rows]
        for i, r in enumerate(norm):
            r.setdefault("vector", [0.1 * (i + 1)] * 8)
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
    md = metadata or {}
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "text": text,
        "source": "https://tuyensinh.husc.edu.vn",
        "faq_type": "thong_tin_nganh",
        "metadata": md,
        "metadata_json": json.dumps(md, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# 1. STRING-REPLACEMENT MAP across separators
# ---------------------------------------------------------------------------
def test_apply_corrections_525_dot_replaced(corrector_module):
    text = "Học phí Khối VII 525.000đ/TC"
    out = corrector_module.apply_text_corrections(text)
    assert "525.000" not in out
    assert "490.000" in out


def test_apply_corrections_540_dot_replaced(corrector_module):
    text = "Học phí Khối IV 540.000đ/TC"
    out = corrector_module.apply_text_corrections(text)
    assert "540.000" not in out
    assert "555.000" in out


def test_apply_corrections_comma_separator(corrector_module):
    text = "Học phí Khối VII 525,000 VNĐ/tín chỉ"
    out = corrector_module.apply_text_corrections(text)
    assert "525,000" not in out
    assert "490,000" in out


def test_apply_corrections_no_separator(corrector_module):
    text = "Học phí Khối IV 540000đ"
    out = corrector_module.apply_text_corrections(text)
    assert "540000" not in out
    assert "555000" in out


def test_apply_corrections_600_unchanged(corrector_module):
    text = "Học phí Khối V 600.000đ/TC"
    out = corrector_module.apply_text_corrections(text)
    # Khối V must not be touched
    assert out == text


def test_apply_corrections_idempotent(corrector_module):
    text = "Học phí Khối VII 490.000đ/TC (năm học 2026-2027, theo Nghị định 238/2025/NĐ-CP)"
    out = corrector_module.apply_text_corrections(text)
    assert out == text


# ---------------------------------------------------------------------------
# 2. CANONICAL MAP: per-major lookups
# ---------------------------------------------------------------------------
def test_canonical_lookup_van_hoc(corrector_module):
    # Default canonical file is bundled with the repo
    canon = corrector_module.load_canonical(
        corrector_module.DEFAULT_CANONICAL_PATH
    )
    assert canon["by_ma_nganh"]["7229030"] == 490000
    assert canon["by_ma_nganh"]["7229010"] == 490000
    assert canon["by_ma_nganh"]["7480103"] == 600000
    assert canon["by_ma_nganh"]["7480201VJ"] == 640000
    assert canon["by_ma_nganh"]["7229001"] == 0  # Triết học, miễn học phí
    assert canon["by_block"]["VII"] == 490000
    assert canon["by_block"]["IV"] == 555000
    assert canon["by_block"]["V"] == 600000


# ---------------------------------------------------------------------------
# 3. ROW-LEVEL CORRECTOR: per-major block-aware + label injection
# ---------------------------------------------------------------------------
def test_correct_nganh_van_hoc_2026(corrector_module):
    row = _row(
        "husc_nganh_7229030_2026",
        "Văn học (mã 7229030) học phí Khối VII 525.000đ/TC",
        metadata={"data_year": 2026, "major_code": "7229030", "khoi": "VII"},
    )
    canon = corrector_module.load_canonical(corrector_module.DEFAULT_CANONICAL_PATH)
    new_text, was_changed, old, new = corrector_module.correct_row_text(
        text=row["text"],
        chunk_id=row["chunk_id"],
        metadata=row["metadata"],
        canonical=canon,
    )
    assert was_changed is True
    assert "525.000" not in new_text
    assert "490.000" in new_text
    assert old == "525.000"
    assert new == 490000
    # label injected
    assert "Nghị định 238/2025/NĐ-CP" in new_text
    assert "2026-2027" in new_text


def test_correct_nganh_vat_ly_2026(corrector_module):
    row = _row(
        "husc_nganh_7440102_2026",
        "Vật lý học (mã 7440102) học phí Khối IV 540.000đ/TC",
        metadata={"data_year": 2026, "major_code": "7440102", "khoi": "IV"},
    )
    canon = corrector_module.load_canonical(corrector_module.DEFAULT_CANONICAL_PATH)
    new_text, was_changed, old, new = corrector_module.correct_row_text(
        text=row["text"],
        chunk_id=row["chunk_id"],
        metadata=row["metadata"],
        canonical=canon,
    )
    assert was_changed is True
    assert "555.000" in new_text
    assert "540.000" not in new_text


def test_correct_nganh_cntt_vj_2026(corrector_module):
    row = _row(
        "husc_nganh_7480201VJ_2026",
        "CNTT Việt-Nhật (mã 7480201VJ) học phí Khối V 600.000đ/TC (Kỹ sư 5 năm)",
        metadata={"data_year": 2026, "major_code": "7480201VJ", "khoi": "V"},
    )
    canon = corrector_module.load_canonical(corrector_module.DEFAULT_CANONICAL_PATH)
    new_text, was_changed, old, new = corrector_module.correct_row_text(
        text=row["text"],
        chunk_id=row["chunk_id"],
        metadata=row["metadata"],
        canonical=canon,
    )
    # This is a Khối V *aggregate* that mentions 600.000 BUT the special-case
    # CNTT-VJ is 640.000. The block-aware lookup overrides.
    assert was_changed is True
    assert "640.000" in new_text
    assert old == "600.000"
    assert new == 640000


def test_correct_nganh_triet_hoc_2026(corrector_module):
    row = _row(
        "husc_nganh_7229001_2026",
        "Triết học (mã 7229001) - MIỄN học phí (theo Nghị định 81/2021/NĐ-CP).",
        metadata={"data_year": 2026, "major_code": "7229001", "khoi": "VII"},
    )
    canon = corrector_module.load_canonical(corrector_module.DEFAULT_CANONICAL_PATH)
    new_text, was_changed, old, new = corrector_module.correct_row_text(
        text=row["text"],
        chunk_id=row["chunk_id"],
        metadata=row["metadata"],
        canonical=canon,
    )
    # Was 0 before, still 0 — not a "change" but we DO update the legal basis
    # to the new Nghị định 238/2025/NĐ-CP.
    assert "Nghị định 238/2025/NĐ-CP" in new_text


def test_correct_2025_chunk_untouched(corrector_module):
    row = _row(
        "husc_nganh_7229030_2025",
        "Văn học (mã 7229030) học phí Khối VII 525.000đ/TC (năm 2025)",
        metadata={"data_year": 2025, "major_code": "7229030", "khoi": "VII"},
    )
    canon = corrector_module.load_canonical(corrector_module.DEFAULT_CANONICAL_PATH)
    new_text, was_changed, old, new = corrector_module.correct_row_text(
        text=row["text"],
        chunk_id=row["chunk_id"],
        metadata=row["metadata"],
        canonical=canon,
    )
    # 2025 records are NEVER touched
    assert was_changed is False
    assert new_text == row["text"]


# ---------------------------------------------------------------------------
# 4. LIVE TABLE: per-major + aggregate tuition chunks get re-embedded+replaced
# ---------------------------------------------------------------------------
def test_fix_lancedb_table_replaces_nganh_chunks(corrector_module):
    # Mix: 1 per-major Khối VII (wrong), 1 per-major Khối IV (wrong),
    # 1 per-major Khối V (already correct), 1 aggregate 2025 (untouched).
    rows = [
        _row(
            "husc_nganh_7229030_2026",
            "Văn học (mã 7229030) học phí Khối VII 525.000đ/TC",
            metadata={"data_year": 2026, "major_code": "7229030", "khoi": "VII"},
        ),
        _row(
            "husc_nganh_7440102_2026",
            "Vật lý học (mã 7440102) học phí Khối IV 540.000đ/TC",
            metadata={"data_year": 2026, "major_code": "7440102", "khoi": "IV"},
        ),
        _row(
            "husc_nganh_7480103_2026",
            "Kỹ thuật phần mềm (mã 7480103) học phí Khối V 600.000đ/TC",
            metadata={"data_year": 2026, "major_code": "7480103", "khoi": "V"},
        ),
        _row(
            "husc_nganh_7229030_2025",
            "Văn học (mã 7229030) học phí Khối VII 525.000đ/TC (năm 2025)",
            metadata={"data_year": 2025, "major_code": "7229030", "khoi": "VII"},
        ),
    ]
    table = FakeTable(rows=list(rows))

    canon = corrector_module.load_canonical(corrector_module.DEFAULT_CANONICAL_PATH)
    summary = corrector_module.fix_lancedb_table(
        table=table,
        embedder=FakeEmbedder(),
        canonical=canon,
    )

    # 2 corrections: 7229030 (VII 525→490) + 7440102 (IV 540→555).
    # Khối V (600) and 2025 are skipped.
    assert summary["nganh_rows_changed"] == 2
    assert summary["nganh_rows_skipped_already_correct"] == 1
    assert summary["nganh_rows_skipped_data_year_2025"] == 1

    # Delete + add were called for the 2 corrected chunks
    assert any("husc_nganh_7229030_2026" in c for c in table.del_calls)
    assert any("husc_nganh_7440102_2026" in c for c in table.del_calls)
    # Add should have 2 new rows
    all_adds = [r for batch in table.add_calls for r in batch]
    assert len(all_adds) == 2
    # Each new row's text is corrected
    by_id = {r["chunk_id"]: r for r in all_adds}
    assert "490.000" in by_id["husc_nganh_7229030_2026"]["text"]
    assert "555.000" in by_id["husc_nganh_7440102_2026"]["text"]
    # Each new row's metadata has the new legal basis
    for cid in ("husc_nganh_7229030_2026", "husc_nganh_7440102_2026"):
        meta = json.loads(by_id[cid]["metadata_json"])
        assert meta["tuition_year"] == "2026-2027"
        assert meta["tuition_basis"] == "Nghị định 238/2025/NĐ-CP"
        assert meta["tuition_source_url"] == "https://tuyensinh.husc.edu.vn"
        # Re-embedded
        assert len(by_id[cid]["vector"]) == FakeEmbedder.DIM


def test_fix_lancedb_table_aggregate_chunks(corrector_module):
    """The 5 aggregate tuition chunks also get re-embedded + replaced.

    These carry no per-major code, so the corrector falls back to the
    block-aware string replacement map (525.000→490.000, 540.000→555.000,
    and the legal-basis label is appended to the text + metadata).
    """
    rows = [
        _row(
            "husc_hoc_phi_tin_chi_2026",
            "Bảng học phí tín chỉ: Khối VII 525.000đ, Khối IV 540.000đ, Khối V 600.000đ.",
            metadata={"data_year": 2026, "topic": "tuition"},
        ),
        _row(
            "so_sanh_hocphi_2025_vs_2026",
            "So sánh 2025 (Khối VII 525.000) và 2026: vẫn 525.000.",
            metadata={"data_year": 2026, "topic": "so_sanh"},
        ),
    ]
    table = FakeTable(rows=list(rows))
    canon = corrector_module.load_canonical(corrector_module.DEFAULT_CANONICAL_PATH)
    summary = corrector_module.fix_lancedb_table(
        table=table,
        embedder=FakeEmbedder(),
        canonical=canon,
    )
    # Both 2026 aggregates have wrong figures (525.000) — both get fixed.
    assert summary["aggregate_rows_changed"] == 2
    by_id = {r["chunk_id"]: r for r in (r for batch in table.add_calls for r in batch)}
    new_text = by_id["husc_hoc_phi_tin_chi_2026"]["text"]
    assert "490.000" in new_text
    assert "555.000" in new_text
    assert "600.000" in new_text
    # The so_sanh chunk also has 525.000 inside it; if its data_year is 2026,
    # it gets fixed (block-aware).
    new_so_sanh = by_id.get("so_sanh_hocphi_2025_vs_2026")
    if new_so_sanh is not None:
        # the 2026 figure in the so_sanh text is replaced too
        assert "525.000" not in new_so_sanh["text"]


# ---------------------------------------------------------------------------
# 5. ON-DISK JSONL files: surgical patch
# ---------------------------------------------------------------------------
def test_fix_jsonl_file_surgical(corrector_module, tmp_path):
    jsonl = tmp_path / "approved.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "chunk_id": "a1",
                        "text": "Văn học học phí Khối VII 525.000đ/TC",
                        "metadata": {"data_year": 2026, "major_code": "7229030"},
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "chunk_id": "a2",
                        "text": "Vật lý học phí Khối IV 540.000đ/TC",
                        "metadata": {"data_year": 2026, "major_code": "7440102"},
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "chunk_id": "a3",
                        "text": "KTPM học phí Khối V 600.000đ/TC",
                        "metadata": {"data_year": 2026, "major_code": "7480103"},
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "chunk_id": "a4",
                        "text": "Văn học 2025 học phí 525.000đ/TC (năm 2025)",
                        "metadata": {"data_year": 2025, "major_code": "7229030"},
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    canon = corrector_module.load_canonical(corrector_module.DEFAULT_CANONICAL_PATH)
    res = corrector_module.fix_jsonl_file(jsonl, canonical=canon)
    assert res["lines_changed"] == 2
    assert res["lines_skipped_2025"] == 1
    assert res["lines_already_correct"] == 1

    # Re-read and verify
    out = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_cid = {r["chunk_id"]: r for r in out}
    assert "490.000" in by_cid["a1"]["text"]
    assert "555.000" in by_cid["a2"]["text"]
    assert "600.000" in by_cid["a3"]["text"]  # Khối V untouched
    assert "525.000" in by_cid["a4"]["text"]  # 2025 untouched


# ---------------------------------------------------------------------------
# 6. ON-DISK JSON file: surgical patch (raw canonical source)
# ---------------------------------------------------------------------------
def test_fix_canonical_json_file(corrector_module, tmp_path):
    """The source-of-truth companion file ``data/raw/husc_2026_canonical.json``
    has a top-level ``tuition.khoi_vii.amount = 525000`` and
    ``tuition.khoi_iv.amount = 540000``. Fixing it is a safety net for any
    future re-ingestion path; the live table is the primary store, so this
    is best-effort and IDEMPOTENT.

    Uses a tmp_path copy so the test does not depend on / mutate the actual
    repo file.
    """
    target = tmp_path / "husc_2026_canonical.json"
    target.write_text(
        json.dumps(
            {
                "school": {"name": "HUSC"},
                "tuition": {
                    "khoi_iv": {"amount": 540000, "unit": "VNĐ/tín chỉ"},
                    "khoi_v": {"amount": 600000, "unit": "VNĐ/tín chỉ"},
                    "khoi_vii": {"amount": 525000, "unit": "VNĐ/tín chỉ"},
                },
                "majors": [
                    {"code": "7229030", "name": "Văn học", "tuition": 525000},
                    {"code": "7440102", "name": "Vật lý học", "tuition": 540000},
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    canon = corrector_module.load_canonical(corrector_module.DEFAULT_CANONICAL_PATH)
    res = corrector_module.fix_canonical_json_file(target, canonical=canon)
    # Either it was already correct (no changes) or it was fixed.
    assert "khoi_vii_after" in res
    assert "khoi_iv_after" in res
    assert res["khoi_vii_after"] == 490000
    assert res["khoi_iv_after"] == 555000
    assert res["khoi_v_after"] == 600000
    # Idempotency: re-running must NOT touch it again.
    res2 = corrector_module.fix_canonical_json_file(target, canonical=canon)
    assert res2.get("changed", False) is False


# ---------------------------------------------------------------------------
# 6b. The real on-disk canonical source file should already be fixed
# (or, if not, fixing it now is a no-op for the test's purposes).
# This is a sanity check that the live file exists and is parseable.
# ---------------------------------------------------------------------------
def test_real_canonical_source_is_parseable(corrector_module):
    """The real ``data/raw/husc_2026_canonical.json`` exists and is parseable.

    This is a sanity test — does NOT mutate the live file. The mutation is
    done by ``fix_canonical_json_file`` in the run-against-live path.
    """
    target = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "husc_2026_canonical.json"
    )
    assert target.exists(), f"canonical source file missing: {target}"
    data = json.loads(target.read_text(encoding="utf-8"))
    assert "tuition" in data
    assert "khoi_vii" in data["tuition"]
    assert "khoi_iv" in data["tuition"]


# ---------------------------------------------------------------------------
# 7. CONSTANTS locked
# ---------------------------------------------------------------------------
def test_constants_locked(corrector_module):
    assert corrector_module.TUITION_YEAR == "2026-2027"
    assert corrector_module.TUITION_BASIS == "Nghị định 238/2025/NĐ-CP"
    assert corrector_module.TUITION_SOURCE_URL == "https://tuyensinh.husc.edu.vn"
    assert corrector_module.TUITION_LABEL == "năm học 2026-2027 (theo Nghị định 238/2025/NĐ-CP)"
    # Source-disk files (relative to repo root)
    expected = {
        "data/chunked/chunked_10.jsonl",
        "data/chunked/chunked_10_enhanced.jsonl",
        "data/chunked/chunked_24.jsonl",
        "data/chunked/husc_2026_aggregation_v2.jsonl",
        "data/chunked/nganh_detail_2026.jsonl",
        "data/raw/husc_2026_canonical.json",
    }
    assert set(corrector_module.SOURCE_FILES) == expected
