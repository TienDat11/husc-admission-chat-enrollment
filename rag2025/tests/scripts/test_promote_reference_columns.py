"""TDD: promote_reference_columns.py — add flat is_reference/crawl_date + BITMAP index (M3-N4a).

REAL LanceDB table under D:/tmp — no mocks.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import lancedb
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "promote_reference_columns.py"


@pytest.fixture()
def temp_db():
    d = tempfile.mkdtemp(dir="D:/tmp", prefix="test_promote_")
    db = lancedb.connect(d)
    yield db
    try:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass


def _create_legacy_table(db, table_name="husc_chunks"):
    """Create a 15-column table (pre-N2a shape) with metadata_json containing is_reference."""
    db.create_table(table_name, data=[
        {
            "id": "r1",
            "doc_id": "r1",
            "release_id": "legacy",
            "text": "Reference doc",
            "summary": None,
            "vector": [0.1] * 1024,
            "data_year": 2025,
            "scope": "admissions",
            "doc_type": "chunk",
            "status": "current",
            "effective_from": 0,
            "effective_to": 99991231,
            "is_tombstone": False,
            "superseded_by": None,
            "metadata_json": json.dumps({"is_reference": True, "crawl_date": "2026-07-01"}),
        },
        {
            "id": "r2",
            "doc_id": "r2",
            "release_id": "legacy",
            "text": "Not reference",
            "summary": None,
            "vector": [0.2] * 1024,
            "data_year": 2026,
            "scope": "admissions",
            "doc_type": "chunk",
            "status": "current",
            "effective_from": 0,
            "effective_to": 99991231,
            "is_tombstone": False,
            "superseded_by": None,
            "metadata_json": json.dumps({"is_reference": False}),
        },
        {
            "id": "r3",
            "doc_id": "r3",
            "release_id": "legacy",
            "text": "No metadata",
            "summary": None,
            "vector": [0.3] * 1024,
            "data_year": 2026,
            "scope": "admissions",
            "doc_type": "chunk",
            "status": "current",
            "effective_from": 0,
            "effective_to": 99991231,
            "is_tombstone": False,
            "superseded_by": None,
            "metadata_json": "{}",
        },
    ])


# ---------------------------------------------------------------------------
# Test 1 — adds flat columns and backfills
# ---------------------------------------------------------------------------


def test_promote_adds_flat_columns_and_backfills(temp_db: Any):
    """After --execute: is_reference column exists, r1=True, r2=False, r3=False."""
    _create_legacy_table(temp_db)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--uri", temp_db.uri, "--table", "husc_chunks", "--execute"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    table = temp_db.open_table("husc_chunks")
    assert "is_reference" in table.schema.names
    assert "crawl_date" in table.schema.names

    rows = {r["id"]: r for r in table.to_arrow().to_pylist()}
    assert rows["r1"]["is_reference"] is True
    assert rows["r2"]["is_reference"] is False
    assert rows["r3"]["is_reference"] is False  # None → False
    assert rows["r1"]["crawl_date"] == "2026-07-01"


# ---------------------------------------------------------------------------
# Test 2 — creates BITMAP index with pushdown
# ---------------------------------------------------------------------------


def test_promote_creates_bitmap_index_pushdown(temp_db: Any):
    """After promote: explain_plan of where('is_reference = true') contains ScalarIndexQuery."""
    _create_legacy_table(temp_db)

    subprocess.run(
        [sys.executable, str(SCRIPT), "--uri", temp_db.uri, "--table", "husc_chunks", "--execute"],
        capture_output=True, text=True, cwd=str(ROOT),
    )

    table = temp_db.open_table("husc_chunks")
    plan = table.search().where("is_reference = true").explain_plan(True)
    assert "ScalarIndexQuery" in plan, f"No pushdown detected: {plan}"


# ---------------------------------------------------------------------------
# Test 3 — idempotent (run twice, no error, same shape)
# ---------------------------------------------------------------------------


def test_promote_idempotent(temp_db: Any):
    """Running script twice: no error, column count and row count unchanged."""
    _create_legacy_table(temp_db)

    for _ in range(2):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--uri", temp_db.uri, "--table", "husc_chunks", "--execute"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    table = temp_db.open_table("husc_chunks")
    assert table.count_rows() == 3
    assert "is_reference" in table.schema.names


# ---------------------------------------------------------------------------
# Test 4 — dry-run does NOT write
# ---------------------------------------------------------------------------


def test_promote_dry_run_no_write(temp_db: Any):
    """--dry-run: columns NOT added."""
    _create_legacy_table(temp_db)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--uri", temp_db.uri, "--table", "husc_chunks", "--dry-run"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    table = temp_db.open_table("husc_chunks")
    assert "is_reference" not in table.schema.names
