"""Tests for parse_diem_chuan_history.py (PHASE-A1 TDD 1:1)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


# Repo-root / script-path resolution (mirror pattern used by other tests).
RAG_ROOT = Path(__file__).resolve().parents[2]
SRC = RAG_ROOT / "src"
SCRIPT_PATH = RAG_ROOT / "scripts" / "parse_diem_chuan_history.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))


@pytest.fixture
def pdh():
    spec = importlib.util.spec_from_file_location("parse_diem_chuan_history", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---- 1) ≥1 score per available major ----------------------------------------

def test_at_least_one_score_per_available_major(pdh):
    doc = pdh.parse_all(log_skipped=True)
    by_code: dict[str, list[dict]] = {}
    for e in doc["entries"]:
        by_code.setdefault(e["major_code"], []).append(e)
    assert by_code, "expected at least one parsed major"
    for code, rows in by_code.items():
        # Each major that appears in 2025 should have at least 1 entry
        # (THPT or HB). 2024 was unparseable so we don't require 2024.
        assert len(rows) >= 1, f"major {code} has no rows"
        # Every row has a numeric diem_chuan in 0..30 (thang-30 scale)
        for r in rows:
            assert isinstance(r["diem_chuan"], float)
            assert 0 <= r["diem_chuan"] <= 30, f"out-of-scale score: {r}"


# ---- 2) Parsed numbers match the HTML source --------------------------------

def test_parsed_numbers_match_html_source(pdh):
    """Spot-check: row 17 of diem_chuan_2025_full.html is CNTT 7480201
    THPT=17.50 / HB=19.69 / DGNL TPHCM=700 / DGNL HN=88.
    """
    doc = pdh.parse_all(log_skipped=True)
    cntt_thi = next(
        (e for e in doc["entries"]
         if e["major_code"] == "7480201" and e["year"] == 2025 and e["method"] == "thi_thpt"),
        None,
    )
    cntt_hb = next(
        (e for e in doc["entries"]
         if e["major_code"] == "7480201" and e["year"] == 2025 and e["method"] == "hoc_ba"),
        None,
    )
    assert cntt_thi is not None and cntt_hb is not None
    assert cntt_thi["diem_chuan"] == 17.5
    assert cntt_hb["diem_chuan"] == 19.69

    # And Hán - Nôm 7220104 THPT 16.00 (row 1)
    nom = next(
        (e for e in doc["entries"]
         if e["major_code"] == "7220104" and e["year"] == 2025 and e["method"] == "thi_thpt"),
        None,
    )
    # 7220104 is NOT in 2026 whitelist, so it's skipped; verify skip log.
    if nom is None:
        skipped = doc["_meta"].get("skipped", [])
        assert any(
            s.get("reason") == "not_in_2026_whitelist" and s.get("code") == "7220104"
            for s in skipped
        ), "expected 7220104 to be in skip log"

    # KTPM 7480103 THPT 17.50 (row 15)
    ktpm = next(
        (e for e in doc["entries"]
         if e["major_code"] == "7480103" and e["year"] == 2025 and e["method"] == "thi_thpt"),
        None,
    )
    assert ktpm is not None
    assert ktpm["diem_chuan"] == 17.5


# ---- 3) method field is one of the two allowed values -----------------------

def test_method_field_is_valid(pdh):
    doc = pdh.parse_all()
    allowed = {"thi_thpt", "hoc_ba"}
    for e in doc["entries"]:
        assert e["method"] in allowed, f"bad method: {e}"


# ---- 4) No NaN / no None in required numeric fields -------------------------

def test_no_nan_in_required_fields(pdh):
    doc = pdh.parse_all()
    import math
    for e in doc["entries"]:
        for k in ("major_code", "major_name", "year", "method", "diem_chuan"):
            assert e.get(k) is not None, f"missing {k}: {e}"
            if isinstance(e[k], float):
                assert not math.isnan(e[k]), f"NaN in {k}: {e}"


# ---- 5) The 2024 file's no-tabular-data skip is logged ---------------------

def test_2024_files_skip_is_logged(pdh):
    doc = pdh.parse_all()
    skipped = doc["_meta"].get("skipped", [])
    reasons = [s.get("reason") for s in skipped]
    # 2024 thi and 2024 hb both produce a "no_tabular_data" skip
    assert reasons.count("no_tabular_data") >= 2, (
        f"expected 2 no_tabular_data skips (one per 2024 file); got: {reasons}"
    )


# ---- 6) write_history produces a stable on-disk artifact -------------------

def test_write_history_creates_file(pdh, tmp_path):
    out = tmp_path / "out.json"
    pdh.write_history(out)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "entries" in data
    assert isinstance(data["entries"], list)
    assert data["_meta"]["schema"] == "phase_a_v1"
