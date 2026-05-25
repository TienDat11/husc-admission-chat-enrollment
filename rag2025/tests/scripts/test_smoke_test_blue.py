"""Tests for smoke_test_blue (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "smoke_test_blue.py"


@pytest.fixture
def smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_test_blue", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_gold_returns_synthetic_when_file_missing(smoke_module, tmp_path):
    missing = tmp_path / "no_such.json"
    gold = smoke_module._load_gold(missing)
    assert isinstance(gold, list) and len(gold) >= 2
    for q in gold:
        assert "question" in q


def test_load_gold_reads_file_when_present(smoke_module, tmp_path):
    p = tmp_path / "gold.json"
    p.write_text(json.dumps([{"id": "g1", "question": "Q?"}]), encoding="utf-8")
    gold = smoke_module._load_gold(p)
    assert gold == [{"id": "g1", "question": "Q?"}]


def test_score_one_year_match_via_metadata(smoke_module):
    result = {"metadata": {"data_year": 2026}, "answer": ""}
    s = smoke_module._score_one(result, expected_year=2026, expected_scope=True)
    assert s["year_match"] is True


def test_score_one_year_match_via_answer_text(smoke_module):
    result = {"metadata": {}, "answer": "Trong năm 2025 ngành ..."}
    s = smoke_module._score_one(result, expected_year=2025, expected_scope=True)
    assert s["year_match"] is True


def test_run_smoke_writes_json_report(smoke_module, tmp_path):
    async def stub_query(_q: str):
        return {"answer": "ans 2026", "metadata": {"data_year": 2026}, "in_scope": True, "grounded": True}

    out = tmp_path / "regression_blue.json"
    report = asyncio.run(smoke_module.run_smoke(
        table_name="husc_v2026_blue",
        gold_path=tmp_path / "missing.json",
        output_path=out,
        query_fn=stub_query,
    ))
    assert out.exists()
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["table_name"] == "husc_v2026_blue"
    assert "year_match_pct" in parsed
    assert "scope_match_pct" in parsed
    assert "grounded_pct" in parsed
    assert "per_question" in parsed
    assert report == parsed


def test_run_smoke_handles_query_failure(smoke_module, tmp_path):
    async def failing_query(_q: str):
        raise RuntimeError("provider down")

    out = tmp_path / "regression_blue.json"
    report = asyncio.run(smoke_module.run_smoke(
        table_name="husc_v2026_blue",
        gold_path=tmp_path / "missing.json",
        output_path=out,
        query_fn=failing_query,
    ))
    # Failures should not raise; counters reflect zero matches.
    assert report["total_questions"] >= 2
    assert report["year_match_pct"] == 0.0


def test_run_smoke_counts_correctly(smoke_module, tmp_path):
    async def half_match(q: str):
        # Only "2026" gold returns matching year; "2025" gold returns wrong year.
        if "2026" in q:
            return {"answer": "ok 2026", "metadata": {"data_year": 2026}, "in_scope": True, "grounded": True}
        return {"answer": "ok 2024", "metadata": {"data_year": 2024}, "in_scope": False, "grounded": False}

    out = tmp_path / "regression_blue.json"
    report = asyncio.run(smoke_module.run_smoke(
        table_name="husc_v2026_blue",
        gold_path=tmp_path / "missing.json",
        output_path=out,
        query_fn=half_match,
    ))
    assert report["total_questions"] >= 3
    assert report["counters"]["year_match"] >= 1
    assert report["counters"]["year_match"] < report["total_questions"]
