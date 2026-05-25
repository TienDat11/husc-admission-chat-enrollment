"""Tests for bootstrap_lancedb_blue (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_lancedb_blue.py"


@pytest.fixture
def boot_module():
    spec = importlib.util.spec_from_file_location("bootstrap_lancedb_blue", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_connector(existing_tables: list[str]):
    db = MagicMock()
    db.table_names.return_value = existing_tables
    db.create_table = MagicMock()
    db.drop_table = MagicMock()
    connector = MagicMock()
    connector.connect.return_value = db
    return connector, db


def test_create_when_table_missing(boot_module):
    connector, db = _fake_connector(existing_tables=[])
    res = boot_module.bootstrap_blue_table(
        db_uri="ignored", table_name="husc_v2026_blue", connector=connector,
    )
    assert res["status"] == "created"
    db.create_table.assert_called_once()
    db.drop_table.assert_not_called()


def test_skip_when_table_exists_without_force(boot_module):
    connector, db = _fake_connector(existing_tables=["husc_v2026_blue"])
    res = boot_module.bootstrap_blue_table(
        db_uri="x", table_name="husc_v2026_blue", connector=connector,
    )
    assert res["status"] == "exists"
    db.create_table.assert_not_called()
    db.drop_table.assert_not_called()


def test_force_recreates_existing(boot_module):
    connector, db = _fake_connector(existing_tables=["husc_v2026_blue"])
    res = boot_module.bootstrap_blue_table(
        db_uri="x", table_name="husc_v2026_blue", force=True, connector=connector,
    )
    assert res["status"] == "recreated"
    db.drop_table.assert_called_once_with("husc_v2026_blue")
    db.create_table.assert_called_once()


def test_schema_includes_required_fields(boot_module):
    connector, _db = _fake_connector(existing_tables=[])
    res = boot_module.bootstrap_blue_table(
        db_uri="x", table_name="husc_v2026_blue", connector=connector,
    )
    fields = res["fields"]
    for required in ("id", "text", "summary", "embedding", "metadata"):
        assert required in fields, f"Missing schema field: {required}"


def test_embedding_dim_is_1024_for_qwen3(boot_module):
    connector, _db = _fake_connector(existing_tables=[])
    res = boot_module.bootstrap_blue_table(
        db_uri="x", table_name="husc_v2026_blue", connector=connector,
    )
    assert res["schema_dim"] == 1024
