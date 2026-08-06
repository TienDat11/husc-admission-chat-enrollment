"""T1.5-bl / G4: `_list_chunk_files` must include ALL canonical JSONL sources.

Regression: the legacy glob `chunked_*.jsonl` only matched `chunked_N.jsonl`
(~1844 chunks), silently DROPPING rows from every other source in the corpus
(canonical v2/v3, aggregation, vision_ocr, nganh_detail, fix files,
thongbao_2026_v2, baiviet_2026). The widened glob must include every `*.jsonl`
in data/chunked except stubs/noise.
"""
from __future__ import annotations

import rag2025.scripts.ingest_lancedb as mod


def _mk(d, names):
    for n in names:
        (d / n).write_text("{}\n", encoding="utf-8")


def test_list_chunk_files_includes_all_sources(tmp_path):
    """Canonical corpus sources must ALL be matched by the widened glob."""
    d = tmp_path / "chunked"
    d.mkdir()
    names = [
        # legacy chunked_N
        "chunked_1.jsonl",
        "chunked_25.jsonl",
        # manual canonical / curated
        "husc_2026_canonical_v2.jsonl",
        "husc_2026_canonical_v3.jsonl",
        "husc_2026_aggregation_v2.jsonl",
        "husc_2026_vision_ocr.jsonl",
        "husc_2026_timeline_hocba_v1.jsonl",
        # fix files
        "husc_canonical_fix_2026.jsonl",
        "husc_thongbao_fix_2026.jsonl",
        "husc_lietke29nganh_fix.jsonl",
        "husc_nganh_moi_fix.jsonl",
        # other canonical sources
        "nganh_detail_2026.jsonl",
        "thongbao_2026_v2.jsonl",
        "husc_diem_ranking_2025.jsonl",
        "baiviet_2026.jsonl",
    ]
    _mk(d, names)

    files = {f.name for f in mod._list_chunk_files(d)}
    assert files == set(names), f"missing canonical sources: {set(names) - files}"


def test_list_chunk_files_excludes_stubs_and_noise(tmp_path):
    """Crawl stubs, fixtures, and templates stay out; *_enhanced stays IN.

    T1.5-bl: `*_enhanced.jsonl` carries real 2026 subject data and must be
    included (see test_ingest_incremental_no_mass_delete.py). Only true crawl
    stubs (chunked_thongbao_/batch_thongbao_/crawl_thongbao_), the test
    fixture, and the yearly template are excluded.
    """
    d = tmp_path / "chunked"
    d.mkdir()
    names = [
        "chunked_1.jsonl",
        "chunked_thongbao_78_2026-07-30T05.03.57.325376+00.00.jsonl",
        "batch_thongbao_76_2026-07-30T04.34.34.893400+00.00.jsonl",
        "crawl_thongbao_76_2026-07-30T04.03.14.002542+00.00.jsonl",
        "chunked_10_enhanced.jsonl",
        "chunked_test.jsonl",
        "husc_yearly_{year}.jsonl",
    ]
    _mk(d, names)

    files = [f.name for f in mod._list_chunk_files(d)]
    assert files == ["chunked_1.jsonl", "chunked_10_enhanced.jsonl"], (
        f"expected canonical + enhanced, got {files}"
    )


def test_list_chunk_files_hides_dotfiles(tmp_path):
    """Manifest / hidden dotfiles must never enter the batch."""
    d = tmp_path / "chunked"
    d.mkdir()
    _mk(d, ["chunked_1.jsonl", ".ingest_manifest.json", ".ingest_manifest.jsonl"])

    files = [f.name for f in mod._list_chunk_files(d)]
    assert files == ["chunked_1.jsonl"], f"dotfiles leaked into batch: {files}"
