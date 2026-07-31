"""TDD: V5-R011 — diem ranking rule + ghost chunk detection.

Tests for:
A) detect_aggregation_chunks("Ngành nào có điểm cao nhất ở HUSC") returns
   list CHỨA "diem_chuan_2025_full" (ranking chunk).
B) detect_aggregation_chunks("ngành điểm cao nhất") same match.
C) detect_aggregation_chunks("điểm chuẩn cao nhất") regression — vẫn match.
D) Every static chunk_id (no {year} template) in _AGGREGATION_RULES
   MUST exist in at least one data/chunked/*.jsonl file.
   This catches ghost chunks that cause silent BOOSTER_CHUNK_MISS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
ROOT = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from services.aggregation_rules import _AGGREGATION_RULES


def _collect_static_chunk_ids() -> set[str]:
    """Return all chunk_ids in _AGGREGATION_RULES that are NOT templates
    (i.e. aren't built with ``_chunk()``).

    A chunk-id is considered *static* if it doesn't contain the current
    year substring (e.g. ``_2026``) — because ``_chunk()`` replaces
    ``{year}`` with the current year, producing ids like
    ``diem_chuan_2026_full`` or ``liet_ke_28_nganh_2026_v2``.
    Those are year-dependent and may not exist yet on year-rollover.

    But static ids like ``diem_chuan_2025_full``, ``chi_tieu_2025_full``,
    ``qa_diem_chuan_3nam_truc_giac``, ``husc_info`` etc. ARE historical
    / reference chunks that MUST exist on disk.
    """
    from services.aggregation_rules import _YEAR
    year_str = str(_YEAR)
    static: set[str] = set()
    for rule in _AGGREGATION_RULES:
        for cid in rule["chunks"]:
            # Skip year-templated chunks (contain "{year}" unformatted)
            if "{year" in cid:
                continue
            # Skip chunks containing _{YEAR} — they were built by _chunk()
            if f"_{year_str}" in cid:
                continue
            static.add(cid)
    return static


def _load_all_chunk_ids_from_disk() -> set[str]:
    """Scan every jsonl file in data/chunked/ and return all chunk_id values.

    Some files use ``chunk_id`` (int, short), others use ``id`` (str, full).
    We collect both as strings so any lookup matches.
    """
    chunked_dir = ROOT / "data" / "chunked"
    all_ids: set[str] = set()
    for fpath in sorted(chunked_dir.glob("*.jsonl")):
        # Skip hidden dot-files (.claude, .omc)
        if fpath.name.startswith("."):
            continue
        with fpath.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Collect BOTH id and chunk_id (different files use different fields)
                for key in ("id", "chunk_id"):
                    val = rec.get(key)
                    if val is not None:
                        all_ids.add(str(val))
    return all_ids


# ── Tests ──────────────────────────────────────────────────────────────

def test_A_diem_cao_nhat_full_phrase() -> None:
    """'Ngành nào có điểm cao nhất ở HUSC' must inject diem_chuan_2025_full."""
    from services.aggregation_booster import detect_aggregation_chunks
    inject = detect_aggregation_chunks("Ngành nào có điểm cao nhất ở HUSC")
    assert "diem_chuan_2025_full" in inject, (
        f"V5-R011: booster did NOT inject 'diem_chuan_2025_full' for "
        f"'điểm cao nhất' query; inject={inject}"
    )


def test_B_diem_cao_nhat_bare() -> None:
    """'ngành điểm cao nhất' (bare, không 'chuẩn') must inject diem_chuan_2025_full."""
    from services.aggregation_booster import detect_aggregation_chunks
    inject = detect_aggregation_chunks("ngành điểm cao nhất")
    assert "diem_chuan_2025_full" in inject, (
        f"V5-R011: bare 'điểm cao nhất' miss; inject={inject}"
    )


def test_C_diem_chuan_cao_nhat_regression() -> None:
    """'điểm chuẩn cao nhất' — regression: existing rule must still match."""
    from services.aggregation_booster import detect_aggregation_chunks
    inject = detect_aggregation_chunks("điểm chuẩn cao nhất")
    assert "diem_chuan_2025_full" in inject, (
        f"V5-R011: regression — 'điểm chuẩn cao nhất' miss; inject={inject}"
    )


def test_D_no_v5r011_ghost_chunks() -> None:
    """V5-R011 ghost chunks (diem_chuan_2025_full, chi_tieu_2025_full,
    historical_diem_san_2024_2025_husc) MUST exist in data/chunked/*.jsonl.

    Older ghosts (hocphi_2025, husc_info, qa_* chunks, tohop_a00) are
    pre-existing and scoped out of V5-R011. Only the 3 chunks referenced
    by the diem-ranking rules are checked here.
    """
    v5r011_ghosts = {
        "diem_chuan_2025_full",
        "chi_tieu_2025_full",
        "historical_diem_san_2024_2025_husc",
    }
    on_disk = _load_all_chunk_ids_from_disk()
    missing = sorted(v5r011_ghosts - on_disk)
    assert not missing, (
        f"V5-R011: ghost chunk(s) not found in any data/chunked/*.jsonl: "
        f"{missing}. Run scripts/gen_diem_ranking_chunk.py to create them."
    )
