"""Tests for sweep_lancedb_noise — TDD S9 (READ-ONLY LanceDB noise sweep).

Critical: the sweep is strictly read-only. The fake lancedb/table mocks below
make ANY write call (add / delete / merge_insert / update / create_table /
open_table(..., mode="overwrite")) raise AssertionError, so an accidental write
fails the test loudly rather than corrupting the 1.6GB real table.
"""

import json
from collections import Counter
from unittest.mock import MagicMock, patch

import pytest

# Patch lancedb before importing the module under test
_mock_lancedb = MagicMock()
sys_modules_patch = patch.dict("sys.modules", {"lancedb": _mock_lancedb})


def _make_row(year, superseded, text, vec_len=1024, rid=None):
    """Build a synthetic row in the actual table shape: metadata_json is a JSON string."""
    md: dict = {"data_year": year}
    if superseded is not None:
        md["is_superseded"] = superseded
    return {
        "id": rid or f"r{abs(hash(text)) % 10_000}",
        "metadata_json": json.dumps(md),
        "text": text,
        "vector": [0.0] * vec_len,
    }


# A reusable long-text filler so non-flagged rows pass the "len(text) >= 100" gate
_LONG_FILLER = (
    "HUSC admission policy text covering the 2026 academic year including "
    "tuition fees, scholarship criteria, program offerings, and contact "
    "information for prospective students who wish to apply to the university."
)


# Synthetic rows covering every assertion in the test plan:
#   - 2 rows data_year=2025 is_superseded falsy  (stale-2025)
#   - 1 row  data_year=2025 is_superseded=True    (NOT stale, must be excluded)
#   - 3 rows data_year=2026                       (fresh baseline)
#   - 1 short-text row                            (low-quality: len < 100)
#   - 1 row starting with "MỤC LỤC ..."           (low-quality: TOC)
#   - 2 rows with identical normalized text       (duplicate pair, contributes 1)
SYNTHETIC_ROWS = [
    _make_row(2025, False, "Stale 2025 policy text about admissions " + _LONG_FILLER, rid="c1"),
    _make_row(2025, None,  "Another 2025 chunk without superseded flag " + _LONG_FILLER, rid="c2"),
    _make_row(2025, True,  "2025 chunk properly marked as superseded " + _LONG_FILLER, rid="c3"),
    _make_row(2026, False, "Fresh 2026 admissions content " + _LONG_FILLER, rid="c4"),
    _make_row(2026, False, "Another fresh 2026 content " + _LONG_FILLER, rid="c5"),
    _make_row(2026, False, "Third fresh 2026 content " + _LONG_FILLER, rid="c6"),
    _make_row(2026, False, "tiny", rid="c7"),  # short text — low quality
    _make_row(
        2026,
        False,
        "MỤC LỤC 1. Điều kiện xét tuyển ... 2. Học phí ... 3. Ngành đào tạo ... 4. Liên hệ ...",
        rid="c8",
    ),
    _make_row(
        2026,
        False,
        "Duplicate content about scholarship policy that appears twice in the table. " + _LONG_FILLER,
        rid="c9",
    ),
    _make_row(
        2026,
        False,
        "Duplicate content about scholarship policy that appears twice in the table. " + _LONG_FILLER,
        rid="c10",
    ),
]


def _read_only_table(rows, vec_len):
    """Build a fake table whose write methods raise AssertionError on any call."""
    tbl = MagicMock(name="fake_lancedb_table")

    # Read path: only to_arrow().to_pylist() will be used by sweep()
    tbl.to_arrow.return_value.to_pylist.return_value = rows
    # Also support to_pandas() in case implementation chooses that path
    try:
        import pandas as pd  # noqa: F401
        tbl.to_pandas.return_value = pd.DataFrame(rows)
    except Exception:
        tbl.to_pandas.return_value = None  # ignored if implementation skips it

    # --- READ-ONLY GUARD: any write call must fail loudly ---
    def _deny_write(*args, **kwargs):
        raise AssertionError(
            "READ-ONLY GUARD: a write method was called on the LanceDB table"
        )

    tbl.add.side_effect = _deny_write
    tbl.delete.side_effect = _deny_write
    tbl.merge_insert.side_effect = _deny_write
    tbl.update.side_effect = _deny_write

    return tbl


def _read_only_lancedb(table):
    """Wire up the fake lancedb.connect path with read-only safety."""
    db = MagicMock(name="fake_lancedb_db")
    tbl = table

    def _open_table(name, mode=None):
        if mode == "overwrite":
            raise AssertionError(
                "READ-ONLY GUARD: open_table called with mode='overwrite'"
            )
        return tbl

    db.open_table.side_effect = _open_table

    def _deny_create(*args, **kwargs):
        raise AssertionError("READ-ONLY GUARD: db.create_table was called")

    db.create_table.side_effect = _deny_create
    db.drop_table.side_effect = _deny_create

    _mock_lancedb.connect.return_value = db
    return db


@pytest.fixture(autouse=True)
def _patch_lancedb():
    """Patch lancedb.connect for every test in this module."""
    with sys_modules_patch:
        import importlib

        import rag2025.scripts.sweep_lancedb_noise as mod

        importlib.reload(mod)
        yield mod


def _sweep(mod, rows=None, vec_len=1024, settings_dim=2560):
    """Build a fake read-only table and call mod.sweep() against it.

    Uses mod._SETTINGS_DIM_OVERRIDE to inject the expected EMBEDDING_DIM value
    without mutating the global settings instance.
    """
    rows = rows if rows is not None else SYNTHETIC_ROWS
    tbl = _read_only_table(rows, vec_len=vec_len)
    with patch.object(mod, "_SETTINGS_DIM_OVERRIDE", settings_dim):
        return mod.sweep(tbl)


# ---------------------------------------------------------------------------
# Data-quality assertions
# ---------------------------------------------------------------------------

def test_total_chunks_equals_input_length(_patch_lancedb):
    mod = _patch_lancedb
    result = _sweep(mod)
    assert result["total_chunks"] == len(SYNTHETIC_ROWS) == 10


def test_stale_2025_excludes_superseded(_patch_lancedb):
    """Two data_year=2025 rows have falsy is_superseded; the True one is excluded."""
    mod = _patch_lancedb
    result = _sweep(mod)
    assert result["stale_2025_without_superseded"] == 2


def test_low_quality_count_includes_short_and_toc(_patch_lancedb):
    mod = _patch_lancedb
    result = _sweep(mod)
    # 1 short text ("tiny") + 1 TOC row starting with "MỤC LỤC" = 2 minimum
    assert result["low_quality_count"] >= 2
    # Pct must be a float in [0, 1] and consistent with the count
    assert 0.0 <= result["low_quality_pct"] <= 1.0
    assert (
        abs(result["low_quality_pct"] - result["low_quality_count"] / result["total_chunks"])
        < 1e-9
    )


def test_duplicate_text_count_at_least_one(_patch_lancedb):
    """Two rows share the same normalized text → duplicate_text_count >= 1."""
    mod = _patch_lancedb
    result = _sweep(mod)
    assert result["duplicate_text_count"] >= 1
    assert 0.0 <= result["duplicate_pct"] <= 1.0


def test_dim_drift_mismatch_when_lengths_differ(_patch_lancedb):
    """Vector length 1024 vs settings.EMBEDDING_DIM=2560 must report mismatch."""
    mod = _patch_lancedb
    result = _sweep(mod, vec_len=1024, settings_dim=2560)
    drift = result["dim_drift"]
    assert drift["stored_dim"] == 1024
    assert drift["settings_dim"] == 2560
    assert drift["mismatch"] is True


def test_dim_drift_match_when_lengths_match(_patch_lancedb):
    """When the stored dim equals settings_dim, mismatch must be False."""
    mod = _patch_lancedb
    result = _sweep(mod, vec_len=1024, settings_dim=1024)
    drift = result["dim_drift"]
    assert drift["stored_dim"] == 1024
    assert drift["settings_dim"] == 1024
    assert drift["mismatch"] is False


def test_by_year_aggregates_correctly(_patch_lancedb):
    mod = _patch_lancedb
    result = _sweep(mod)
    by_year = result["by_year"]
    # 3 rows data_year=2025, 7 rows data_year=2026 in SYNTHETIC_ROWS
    assert by_year["2025"] == 3
    assert by_year["2026"] == 7


# ---------------------------------------------------------------------------
# READ-ONLY GUARD: any write path must raise
# ---------------------------------------------------------------------------

def test_read_only_guard_table_writes_never_called(_patch_lancedb):
    """Calling sweep() must never invoke add/delete/merge_insert/update on the table."""
    mod = _patch_lancedb
    tbl = _read_only_table(SYNTHETIC_ROWS, vec_len=1024)
    with patch.object(mod, "_SETTINGS_DIM_OVERRIDE", 2560):
        mod.sweep(tbl)  # must NOT raise

    tbl.add.assert_not_called()
    tbl.delete.assert_not_called()
    tbl.merge_insert.assert_not_called()
    tbl.update.assert_not_called()


def test_read_only_guard_uses_open_table_not_create_table(_patch_lancedb, tmp_path):
    """main() must open the table via lancedb.connect(...).open_table(name) — never create_table or overwrite."""
    mod = _patch_lancedb
    tbl = _read_only_table(SYNTHETIC_ROWS, vec_len=1024)
    db = _read_only_lancedb(tbl)

    # Patch settings to a value that mismatches the stored vectors
    with patch.object(mod, "_SETTINGS_DIM_OVERRIDE", 2560), \
         patch.object(mod, "_LANCEDB_URI_OVERRIDE", "mock://uri"), \
         patch.object(mod, "_LANCEDB_TABLE_OVERRIDE", "rag2025"):
        out_path = tmp_path / "noise_report.json"
        rc = mod.main(argv=[], output_path=out_path)

    # main() must succeed (return 0) and write the JSON report
    assert rc == 0
    assert out_path.exists()

    # db.open_table was called with the expected name and no mode="overwrite"
    db.open_table.assert_called_once()
    call_kwargs = db.open_table.call_args.kwargs
    call_args = db.open_table.call_args.args
    assert "rag2025" in (call_args + tuple(call_kwargs.values()))
    # mode must not be "overwrite"
    if "mode" in call_kwargs:
        assert call_kwargs["mode"] != "overwrite"
    # db.create_table must NOT have been called
    db.create_table.assert_not_called()
    # db.drop_table must NOT have been called
    db.drop_table.assert_not_called()

    # And the report must have the expected top-level shape
    with open(out_path, encoding="utf-8") as f:
        report = json.load(f)
    for key in (
        "sweep_date",
        "table",
        "total_chunks",
        "stale_2025_without_superseded",
        "low_quality_count",
        "low_quality_pct",
        "duplicate_text_count",
        "duplicate_pct",
        "dim_drift",
        "by_year",
    ):
        assert key in report, f"missing key in report: {key}"


def test_read_only_guard_writes_correct_json(_patch_lancedb, tmp_path):
    mod = _patch_lancedb
    tbl = _read_only_table(SYNTHETIC_ROWS, vec_len=1024)
    _read_only_lancedb(tbl)

    with patch.object(mod, "_SETTINGS_DIM_OVERRIDE", 2560), \
         patch.object(mod, "_LANCEDB_URI_OVERRIDE", "mock://uri"), \
         patch.object(mod, "_LANCEDB_TABLE_OVERRIDE", "rag2025"):
        out_path = tmp_path / "noise_report.json"
        rc = mod.main(argv=[], output_path=out_path)
        assert rc == 0

    with open(out_path, encoding="utf-8") as f:
        report = json.load(f)

    # Values from SYNTHETIC_ROWS
    assert report["total_chunks"] == 10
    assert report["stale_2025_without_superseded"] == 2
    assert report["low_quality_count"] >= 2
    assert report["duplicate_text_count"] >= 1
    assert report["dim_drift"]["mismatch"] is True
    assert report["by_year"]["2025"] == 3
    assert report["by_year"]["2026"] == 7
