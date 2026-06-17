"""TDD test for scripts/ingest_missing_2026.py (S17.1 — TDD data-ops).

Strategy: inject a fake `table` (records list + add() + existing-ids set) and
a fake `EmbeddingService` (encode_documents returns predetermined vectors) so
the test NEVER touches the real LanceDB or the Qwen3 embedding model.

Coverage:
  * new chunk_id → row added (chunk_id, text, source, faq_type, metadata_json
    with diacritics, vector of correct length)
  * already-present chunk_id → SKIPPED (idempotent re-run)
  * metadata_json is valid JSON with Vietnamese diacritics preserved
  * missing `text` is skipped
  * `id` and `chunk_id` aliases both resolve to the same canonical id
  * text is embedded via the injected service (not bypassed)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ingest_missing_2026.py"


@pytest.fixture
def ingest_module():
    spec = importlib.util.spec_from_file_location("ingest_missing_2026", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fake table that mimics the LanceDB table interface used by the script.
# ---------------------------------------------------------------------------
class FakeTable:
    def __init__(self, existing_chunk_ids: List[str]) -> None:
        # existing rows (id, chunk_id, text, ...)
        self._records: List[Dict[str, Any]] = [
            {"chunk_id": cid, "id": cid, "text": f"old-{cid}", "metadata_json": "{}"}
            for cid in existing_chunk_ids
        ]
        self.add_calls: List[List[Dict[str, Any]]] = []
        self.del_calls: List[str] = []

    def add(self, rows: List[Dict[str, Any]]) -> None:
        self.add_calls.append(list(rows))
        self._records.extend(rows)

    def delete(self, where_clause: str) -> None:
        self.del_calls.append(where_clause)

    def to_pandas(self, columns: List[str] = None):  # noqa: D401
        """Return a tiny stand-in DataFrame with the existing chunk_ids.

        Tests use this to verify the script correctly skips duplicates by
        reading the table's chunk_id column.
        """
        import pandas as pd

        if columns is None:
            data = {col: [r.get(col) for r in self._records] for col in ("chunk_id", "id")}
            return pd.DataFrame(data)
        data = {col: [r.get(col) for r in self._records] for col in columns if col in ("chunk_id", "id")}
        # backfill missing requested columns with None
        for col in columns:
            data.setdefault(col, [None] * len(self._records))
        return pd.DataFrame(data)

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)


class FakeEmbedder:
    """Returns deterministic vectors: each text → [0.1, 0.2, ...] of dim=DIM."""

    DIM = 8

    def encode_documents(self, texts: List[str]):
        rows = []
        for i, _t in enumerate(texts):
            vec = np.full(self.DIM, 0.1 * (i + 1), dtype=np.float32)
            rows.append(vec)
        return np.stack(rows)


# ---------------------------------------------------------------------------
# 1. New chunk_id is added with correct row shape.
# ---------------------------------------------------------------------------
def test_new_chunk_id_is_added(ingest_module, tmp_path):
    jsonl = tmp_path / "approved.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "id": "qa_dang_ky_xet_tuyen_2026_v2",
                "text": "HUSC 2026 — đăng ký xét tuyển 02/7/2026 đến 14/7/2026",
                "metadata": {
                    "data_year": 2026,
                    "source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=74",
                    "notification_id": 74,
                    "faq_type": "operational_qa",
                    "topic": "dang_ky_xet_tuyen",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    table = FakeTable(existing_chunk_ids=[])
    summary = ingest_module.ingest_files(
        files=[jsonl],
        table=table,
        embedder=FakeEmbedder(),
    )

    assert summary["approved.jsonl"] == {"added": 1, "skipped": 0}
    assert len(table.add_calls) == 1
    added = table.add_calls[0][0]
    assert added["chunk_id"] == "qa_dang_ky_xet_tuyen_2026_v2"
    assert added["id"] == "qa_dang_ky_xet_tuyen_2026_v2"
    assert "02/7/2026" in added["text"]
    assert added["source"] == "https://tuyensinh.husc.edu.vn/thongbao.php?id=74"
    assert added["faq_type"] == "operational_qa"
    # vector preserved
    assert len(added["vector"]) == FakeEmbedder.DIM
    assert added["vector"][0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 2. Already-present chunk_id is SKIPPED (idempotent re-run).
# ---------------------------------------------------------------------------
def test_already_present_chunk_id_is_skipped(ingest_module, tmp_path):
    jsonl = tmp_path / "approved.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "id": "qa_dang_ky_xet_tuyen_2026_v2",
                "text": "duplicate",
                "metadata": {"data_year": 2026},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    table = FakeTable(existing_chunk_ids=["qa_dang_ky_xet_tuyen_2026_v2"])
    summary = ingest_module.ingest_files(
        files=[jsonl],
        table=table,
        embedder=FakeEmbedder(),
    )

    assert summary["approved.jsonl"] == {"added": 0, "skipped": 1}
    # nothing was added
    assert table.add_calls == [] or all(len(c) == 0 for c in table.add_calls)


# ---------------------------------------------------------------------------
# 3. metadata_json is valid JSON, diacritics preserved, metadata roundtrip.
# ---------------------------------------------------------------------------
def test_metadata_json_preserves_diacritics(ingest_module, tmp_path):
    jsonl = tmp_path / "approved.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "id": "phuongthuc_xettuyen_2026_v2",
                "text": "5 phương thức xét tuyển HUSC 2026",
                "metadata": {
                    "topic": "phương_thức_xét_tuyển",   # Vietnamese diacritics
                    "data_year": 2026,
                    "source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=74",
                    "faq_type": "phuong_thuc_2026",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    table = FakeTable(existing_chunk_ids=[])
    summary = ingest_module.ingest_files(
        files=[jsonl],
        table=table,
        embedder=FakeEmbedder(),
    )

    assert summary["approved.jsonl"] == {"added": 1, "skipped": 0}
    added = table.add_calls[0][0]
    parsed = json.loads(added["metadata_json"])
    assert parsed["topic"] == "phương_thức_xét_tuyển"
    assert parsed["data_year"] == 2026
    # ensure_ascii=False → Vietnamese chars stay literal
    assert "phương" in added["metadata_json"]
    assert "xét_tuyển" in added["metadata_json"]


# ---------------------------------------------------------------------------
# 4. Records with no `text` are skipped (not added, no error).
# ---------------------------------------------------------------------------
def test_record_without_text_is_skipped(ingest_module, tmp_path):
    jsonl = tmp_path / "approved.jsonl"
    jsonl.write_text(
        json.dumps({"id": "empty_chunk_2026", "metadata": {}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    table = FakeTable(existing_chunk_ids=[])
    summary = ingest_module.ingest_files(
        files=[jsonl],
        table=table,
        embedder=FakeEmbedder(),
    )

    assert summary["approved.jsonl"] == {"added": 0, "skipped": 1}
    assert all(len(c) == 0 for c in table.add_calls)


# ---------------------------------------------------------------------------
# 5. `chunk_id` alias (no `id` field) is also supported.
# ---------------------------------------------------------------------------
def test_chunk_id_alias_resolved(ingest_module, tmp_path):
    jsonl = tmp_path / "approved.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "chunk_id": "nganh_detail_7220104_2026",
                "text": "Hán - Nôm mã 7220104",
                "metadata": {"data_year": 2026, "major_code": "7220104"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    table = FakeTable(existing_chunk_ids=[])
    summary = ingest_module.ingest_files(
        files=[jsonl],
        table=table,
        embedder=FakeEmbedder(),
    )

    assert summary["approved.jsonl"] == {"added": 1, "skipped": 0}
    added = table.add_calls[0][0]
    assert added["chunk_id"] == "nganh_detail_7220104_2026"
    assert added["id"] == "nganh_detail_7220104_2026"


# ---------------------------------------------------------------------------
# 6. Default APPROVED_FILES list matches the plan exactly (5 files).
# ---------------------------------------------------------------------------
def test_default_approved_files(ingest_module):
    expected = {
        "husc_2026_canonical_v2.jsonl",
        "husc_2026_canonical_v3.jsonl",
        "husc_2026_aggregation_v2.jsonl",
        "nganh_detail_2026.jsonl",
        "husc_2026_vision_ocr.jsonl",
    }
    assert set(ingest_module.APPROVED_FILES) == expected
    # explicit skip set
    for skip in ("baiviet_2026.jsonl", "thongbao_2026_v2.jsonl", "chunked_10_enhanced.jsonl"):
        assert skip in ingest_module.SKIP_FILES


# ---------------------------------------------------------------------------
# 7. REGRESSION (bug 2026-06-10): real LanceDB `to_pandas()` does NOT accept
#    a `columns=` kwarg. If the script blindly calls `to_pandas(columns=...)`
#    the TypeError is swallowed, `existing` stays empty, and an
#    already-present chunk gets RE-ADDED as a duplicate. This table mimics
#    the real API (no columns kwarg) and asserts dedup still works.
# ---------------------------------------------------------------------------
class FakeTableNoColumnsKwarg:
    """Mimics the REAL LanceDB table: to_pandas() takes NO columns kwarg."""

    def __init__(self, existing_chunk_ids: List[str]) -> None:
        self._records: List[Dict[str, Any]] = [
            {"chunk_id": cid, "id": cid, "text": f"old-{cid}", "metadata_json": "{}"}
            for cid in existing_chunk_ids
        ]
        self.add_calls: List[List[Dict[str, Any]]] = []

    def add(self, rows: List[Dict[str, Any]]) -> None:
        self.add_calls.append(list(rows))
        self._records.extend(rows)

    def to_pandas(self):  # NO columns kwarg — matches real LanceTable
        import pandas as pd

        return pd.DataFrame(
            {
                "chunk_id": [r.get("chunk_id") for r in self._records],
                "id": [r.get("id") for r in self._records],
            }
        )


def test_dedup_works_when_to_pandas_has_no_columns_kwarg(ingest_module, tmp_path):
    jsonl = tmp_path / "approved.jsonl"
    jsonl.write_text(
        json.dumps(
            {"id": "bang_xep_hang_chi_tieu_2026_v2", "text": "dup", "metadata": {"data_year": 2026}},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    table = FakeTableNoColumnsKwarg(existing_chunk_ids=["bang_xep_hang_chi_tieu_2026_v2"])
    summary = ingest_module.ingest_files(
        files=[jsonl],
        table=table,
        embedder=FakeEmbedder(),
    )

    # The chunk already exists → MUST be skipped, NOT re-added as a duplicate.
    assert summary["approved.jsonl"] == {"added": 0, "skipped": 1}
    assert table.add_calls == [] or all(len(c) == 0 for c in table.add_calls)
