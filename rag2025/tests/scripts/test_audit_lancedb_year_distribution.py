"""Tests for audit_lancedb_year_distribution — TDD V5-R030."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Patch lancedb before importing the module under test
_mock_lancedb = MagicMock()
sys_modules_patch = patch.dict("sys.modules", {"lancedb": _mock_lancedb})

SYNTHETIC_ROWS = [
    {
        "id": "c1",
        "metadata": {
            "data_year": 2025,
            "chunk_method": "rule_v2",
            "info_type": "hoc_phi",
            "source_url": "https://x",
        },
    },
    {
        "id": "c2",
        "metadata": {
            "data_year": 2026,
            "chunk_method": "rule_v2",
            "info_type": "diem_chuan",
            "source_url": "https://y",
        },
    },
    {
        "id": "c3",
        "metadata": {
            "data_year": 2025,
            "chunk_method": "haiku_v1",
            "info_type": "qa",
            "source_url": "https://z",
        },
    },
    {
        "id": "c4",
        "metadata": {"info_type": "qa"},
    },
]


@pytest.fixture(autouse=True)
def _patch_lancedb():
    """Patch lancedb.connect for every test in this module."""
    mock_tbl = MagicMock()
    mock_tbl.to_arrow.return_value.to_pylist.return_value = SYNTHETIC_ROWS
    mock_db = MagicMock()
    mock_db.open_table.return_value = mock_tbl
    _mock_lancedb.connect.return_value = mock_db
    with sys_modules_patch:
        # Force re-import so the patched lancedb is picked up
        import importlib

        import rag2025.scripts.audit_lancedb_year_distribution as mod

        importlib.reload(mod)
        yield mod


def _audit(mod):
    return mod.audit_table("mock://uri", "husc")


def test_year_distribution_counts(_patch_lancedb):
    mod = _patch_lancedb
    result = _audit(mod)
    assert result["year_distribution"] == {"2025": 2, "2026": 1, "null": 1}


def test_method_distribution_counts(_patch_lancedb):
    mod = _patch_lancedb
    result = _audit(mod)
    assert result["method_distribution"] == {"rule_v2": 2, "haiku_v1": 1, "null": 1}


def test_missing_field_counts(_patch_lancedb):
    mod = _patch_lancedb
    result = _audit(mod)
    assert result["missing_data_year_count"] == 1
    assert result["missing_source_url_count"] == 1


def test_save_audit_writes_valid_json(_patch_lancedb, tmp_path):
    mod = _patch_lancedb
    audit = _audit(mod)
    out_path = tmp_path / "out.json"
    mod.save_audit(audit, out_path)
    assert out_path.exists()
    with open(out_path, encoding="utf-8") as f:
        loaded = json.load(f)
    required_keys = {
        "audit_date",
        "table",
        "total_rows",
        "year_distribution",
        "method_distribution",
        "info_type_distribution",
        "missing_data_year_count",
        "missing_source_url_count",
    }
    assert required_keys.issubset(loaded.keys())


def test_total_rows_equals_input_length(_patch_lancedb):
    mod = _patch_lancedb
    result = _audit(mod)
    assert result["total_rows"] == 4
