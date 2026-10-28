"""Tests for scripts/tombstone_legacy_dry_run.py (T15 — legacy tombstone dry-run).

Fakes-only, no live LanceDB:
  * FakeTable mimics the LanceDB-shaped table (to_pandas + update).
  * Pure functions under test: classify_rows (eligibility),
    check_recall_delta (2pp rule), apply_tombstones (idempotency),
    and the main() gate (--apply without the ack flag → exit 2, zero writes).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "tombstone_legacy_dry_run.py"


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("tombstone_legacy_dry_run", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tombstone_legacy_dry_run"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fake table
# ---------------------------------------------------------------------------
class FakeTable:
    """LanceDB-shaped: to_pandas + update, with call recording."""

    COLS = ["id", "doc_id", "release_id", "doc_type", "data_year", "status", "is_tombstone"]

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows: List[Dict[str, Any]] = [dict(r) for r in rows]
        self.update_calls: List[Any] = []

    def to_pandas(self):
        return pd.DataFrame({c: [r.get(c) for r in self._rows] for c in self.COLS})

    def update(self, where: str, values: Dict[str, Any]) -> None:
        self.update_calls.append((where, values))
        # apply to our in-memory rows so a second pass sees is_tombstone=true
        import re

        m = re.search(r'id = "([^"]+)"', where)
        if m:
            for r in self._rows:
                if str(r.get("id")) == m.group(1):
                    r.update(values)


def _row(rid: str, doc_type: str, year: int, release: str = "legacy", tomb: bool = False) -> Dict[str, Any]:
    return {
        "id": rid, "doc_id": rid, "release_id": release, "doc_type": doc_type,
        "data_year": year, "status": "published", "is_tombstone": tomb,
    }


ROWS = [
    # eligible group: CHI_TIEU/2026 has canonical coverage
    _row("leg-1", "CHI_TIEU", 2026),
    _row("leg-2", "CHI_TIEU", 2026),
    _row("can-1", "CHI_TIEU", 2026, release="rel-A"),
    # NOT eligible: HOC_PHI/2026 has legacy rows but zero canonical
    _row("leg-3", "HOC_PHI", 2026),
    # eligible: only canonical, no legacy → nothing to tombstone
    _row("can-2", "TRUONG", 2026, release="rel-B"),
]
def test_classify_rows_eligibility(mod):
    report = mod.classify_rows(ROWS)
    by_key = {(g["doc_type"], g["data_year"]): g for g in report["groups"]}

    ct = by_key[("CHI_TIEU", 2026)]
    assert ct["legacy_count"] == 2
    assert ct["canonical_count"] == 1
    assert ct["eligible"] is True
    assert sorted(ct["would_tombstone_ids"]) == ["leg-1", "leg-2"]

    hp = by_key[("HOC_PHI", 2026)]
    assert hp["legacy_count"] == 1
    assert hp["canonical_count"] == 0
    assert hp["eligible"] is False
    assert hp["would_tombstone_ids"] == []

    tr = by_key[("TRUONG", 2026)]
    assert tr["eligible"] is True  # canonical coverage exists
    assert tr["would_tombstone_ids"] == []  # but no legacy rows to tombstone

    assert report["total_would_tombstone"] == 2


def test_classify_rows_already_tombstoned_rows_still_counted_as_legacy(mod):
    rows = [
        _row("l-1", "X", 2026, tomb=True),
        _row("c-1", "X", 2026, release="rel-A"),
    ]
    report = mod.classify_rows(rows)
    g = report["groups"][0]
    # The report is a snapshot: a legacy row already tombstoned is still a
    # legacy row in the classification; apply_tombstones is what skips it.
    assert g["legacy_count"] == 1
    assert g["would_tombstone_ids"] == ["l-1"]


# ---------------------------------------------------------------------------
# (b) dry-run makes zero write calls
# ---------------------------------------------------------------------------
def test_dry_run_makes_zero_write_calls(mod):
    table = FakeTable(ROWS)
    report = mod.classify_rows(mod.load_all_rows(table))
    mod.print_report(report)  # pure print, no I/O
    assert table.update_calls == []
    assert report["total_would_tombstone"] == 2


# ---------------------------------------------------------------------------
# (c) --apply without the acknowledgement flag → non-zero, zero writes
# ---------------------------------------------------------------------------
def test_apply_without_ack_flag_exits_nonzero_zero_writes(mod):
    table = FakeTable(ROWS)
    with patch.dict(sys.modules, {"lancedb": _fake_lancedb(table)}):
        with patch.object(mod, "RAGSettings", return_value=_fake_settings()):
            rc = mod.main(["--apply"])
    assert rc == 2
    assert table.update_calls == []


def test_apply_with_ack_flag_writes_only_eligible_rows(mod):
    table = FakeTable(ROWS)
    with patch.dict(sys.modules, {"lancedb": _fake_lancedb(table)}):
        with patch.object(mod, "RAGSettings", return_value=_fake_settings()):
            rc = mod.main(["--apply", "--i-ran-the-3-route-http-check"])
    assert rc == 0
    # only leg-1, leg-2 written (leg-3's group not eligible; can-2 has no legacy)
    written_ids = [w[0] for w in table.update_calls]
    assert any('leg-1' in s for s in written_ids)
    assert any('leg-2' in s for s in written_ids)
    assert not any('leg-3' in s for s in written_ids)
    assert not any('can-2' in s for s in written_ids)


# ---------------------------------------------------------------------------
# (d) idempotence — second apply reports 0 changes
# ---------------------------------------------------------------------------
def test_apply_is_idempotent(mod):
    table = FakeTable(ROWS)
    with patch.dict(sys.modules, {"lancedb": _fake_lancedb(table)}):
        with patch.object(mod, "RAGSettings", return_value=_fake_settings()):
            mod.main(["--apply", "--i-ran-the-3-route-http-check"])
            first = len(table.update_calls)
            assert first == 2
            mod.main(["--apply", "--i-ran-the-3-route-http-check"])
            assert len(table.update_calls) == first  # second run: 0 new writes


# ---------------------------------------------------------------------------
# (e) recall-delta rule: 2.0pp passes, 2.1pp fails
# ---------------------------------------------------------------------------
def test_recall_delta_2pp_passes(mod):
    result = mod.check_recall_delta(t00_recall_pp=50.0, after_recall_pp=48.0)
    assert result["delta_pp"] == pytest.approx(2.0)
    assert result["passed"] is True


def test_recall_delta_2_1pp_fails(mod):
    result = mod.check_recall_delta(t00_recall_pp=50.0, after_recall_pp=47.9)
    assert result["delta_pp"] == pytest.approx(2.1)
    assert result["passed"] is False


def test_recall_delta_improvement_passes(mod):
    result = mod.check_recall_delta(t00_recall_pp=40.0, after_recall_pp=45.0)
    assert result["delta_pp"] == pytest.approx(-5.0)
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class _FakeSettings:
    LANCEDB_URI = "/tmp/fake-lancedb"
    LANCEDB_TABLE = "husc_chunks"


def _fake_settings() -> _FakeSettings:
    return _FakeSettings()


def _fake_lancedb(table: FakeTable) -> types.ModuleType:
    db = types.SimpleNamespace(table_names=lambda: ["husc_chunks"], open_table=lambda n: table)
    m = types.ModuleType("lancedb")
    m.connect = lambda uri: db
    return m


def test_eval_protocol_commands_mention_runner_and_t00(mod):
    cmds = mod.eval_protocol_commands()
    joined = " ".join(cmds)
    assert "eval_run_instrumented" in joined
    assert "baseline_T00" in joined
    assert "86q_records" in joined
    assert len(cmds) >= 3
