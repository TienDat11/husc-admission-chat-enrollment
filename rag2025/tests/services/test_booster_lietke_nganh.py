"""TDD FIX-RETLISTKE (msg07x): program-list / liệt-kê-ngành queries must
inject CẢ HAI chunk `liet_ke_28_nganh_<year>_v2` (full 29-major list with
tên + mã + chỉ tiêu) AND `bang_xep_hang_chi_tieu_<year>_v2` (chỉ tiêu
sorted-rank with the same 29 majors).

Live root cause:
  - Query A "Trường có những ngành nào? danh sách đầy đủ" → booster trước
    fix cũng inject đủ 2 chunk (overshot top_k=5; lý do: query đầu vào
    nhận đủ overlap với rule cũ); output đủ 29 ngành.
  - Query B "Liệt kê chi tiết tất cả các ngành đào tạo của HUSC 2026" →
    trước fix, retrieval CHỈ lấy `liet_ke_28_nganh_2026_v2` (chunk có 16
    mã đầy đủ + 13 mã còn lại gộp ở dòng note); model thấy "thiếu" → tự
    bịa "tài liệu chỉ liệt kê 19/29 ngành".

Fix contract verified by this test:
  1. `detect_aggregation_chunks` returns a list whose FIRST TWO entries
     are exactly (`liet_ke_28_nganh_<year>_v2`, `bang_xep_hang_chi_tieu_<year>_v2`)
     for both Query A and Query B.
  2. `boost_with_aggregation(..., max_inject=2)` (the production cap from
     `src/main.py:1259`) injects BOTH liệt-kê chunks into the returned
     docs — proving the wired-in path works, not just the detector.
  3. Idempotent: re-running detect on the same query yields the same
     (order-stable) chunk-id list — no random oscillation between calls.

Year parameterization: the test reads `get_current_admission_year()` from
`temporal_authority` so it survives year-rollover (mirrors the convention
in `test_booster_ngoaingu.py`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Two real candidate queries that exhibit the live root-cause split.
QUERY_A = "Trường có những ngành nào? danh sách đầy đủ"  # the one that "worked"
QUERY_B = "Liệt kê chi tiết tất cả các ngành đào tạo của HUSC 2026"  # the one that failed


def _year() -> int:
    from src.services.temporal_authority import get_current_admission_year
    return int(get_current_admission_year())


def _expected_lietke_cid() -> str:
    return f"liet_ke_28_nganh_{_year()}_v2"


def _expected_bang_cid() -> str:
    return f"bang_xep_hang_chi_tieu_{_year()}_v2"


# ── Detector-level assertions ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "query",
    [QUERY_A, QUERY_B],
)
def test_detect_injects_both_lietke_chunks_first(query: str) -> None:
    """Both liệt-kê chunks MUST appear in the detector output, and they
    MUST come ahead of the (optional) tuyensinh_overview chunk so the
    production `max_inject=2` cap lets them through."""
    from services.aggregation_booster import detect_aggregation_chunks

    inject = detect_aggregation_chunks(query)
    expected_lk = _expected_lietke_cid()
    expected_bx = _expected_bang_cid()

    assert expected_lk in inject, (
        f"FAIL: liet_ke chunk {expected_lk!r} missing for query={query!r}; "
        f"inject={inject!r}"
    )
    assert expected_bx in inject, (
        f"FAIL: bang_xep_hang chunk {expected_bx!r} missing for "
        f"query={query!r}; inject={inject!r}"
    )

    # Order: the two liệt-kê chunks must come FIRST (within the first 2)
    # so the production `max_inject=2` cap doesn't drop either in favor
    # of the optional tuyensinh_overview chunk.
    first_two = inject[:2]
    assert expected_lk in first_two, (
        f"FAIL: liet_ke chunk must be in first 2 of inject for "
        f"query={query!r}; first_two={first_two!r}; inject={inject!r}"
    )
    assert expected_bx in first_two, (
        f"FAIL: bang_xep_hang chunk must be in first 2 of inject for "
        f"query={query!r}; first_two={first_two!r}; inject={inject!r}"
    )


@pytest.mark.parametrize(
    "query",
    [QUERY_A, QUERY_B],
)
def test_detect_idempotent_same_query(query: str) -> None:
    """Idempotent: same query → same ordered chunk-id list every call."""
    from services.aggregation_booster import detect_aggregation_chunks

    out1 = detect_aggregation_chunks(query)
    out2 = detect_aggregation_chunks(query)
    assert out1 == out2, (
        f"FAIL: detect_aggregation_chunks is non-idempotent for "
        f"query={query!r}; first={out1!r}; second={out2!r}"
    )


# ── Booster-level assertion (wired-in path) ───────────────────────────────

@pytest.mark.parametrize(
    "query",
    [QUERY_A, QUERY_B],
)
def test_boost_with_aggregation_max_inject_2_injects_both_chunks(
    query: str, monkeypatch
) -> None:
    """End-to-end: `boost_with_aggregation(query, baseline=[], top_k=5,
    max_inject=2)` MUST return docs containing BOTH liệt-kê chunks.
    This proves the wired-in production path (called from `main.py:1254`
    with `max_inject=2`) injects both, not just the detector."""
    from unittest.mock import MagicMock

    from services.aggregation_booster import boost_with_aggregation
    from services.lancedb_retrieval import RetrievedDocument

    expected_lk = _expected_lietke_cid()
    expected_bx = _expected_bang_cid()
    expected_to = f"tuyensinh_overview_{_year()}_v2"

    def fetch_by_id(cid: str):
        # Return a fake doc for any of the expected liệt-kê chunk ids;
        # None for anything else (defensive — booster should never ask).
        if cid in {expected_lk, expected_bx, expected_to}:
            return RetrievedDocument(
                text=f"FAKE_TEXT for {cid}",
                source="fake_source",
                chunk_id=cid,
                metadata={"source_kind": "fake"},
                score=1.0,
                point_id=cid,
            )
        return None

    fake = MagicMock()
    fake.fetch_by_id.side_effect = fetch_by_id

    out = boost_with_aggregation(
        query=query,
        baseline_docs=[],
        lancedb_retriever=fake,
        top_k=5,
        max_inject=2,  # production cap from src/main.py:1259
    )
    injected_ids = [d.chunk_id for d in out]

    assert expected_lk in injected_ids, (
        f"FAIL: boost_with_aggregation did not include "
        f"{expected_lk!r} in returned docs for query={query!r}; "
        f"injected_ids={injected_ids!r}"
    )
    assert expected_bx in injected_ids, (
        f"FAIL: boost_with_aggregation did not include "
        f"{expected_bx!r} in returned docs for query={query!r}; "
        f"injected_ids={injected_ids!r}"
    )

    # The 2 injected docs must carry booster_injected=True so the
    # precision-cutoff / fusion exempt-block keeps them.
    for cid in (expected_lk, expected_bx):
        doc = next(d for d in out if d.chunk_id == cid)
        assert (doc.metadata or {}).get("booster_injected") is True, (
            f"FAIL: injected chunk {cid!r} missing booster_injected tag; "
            f"metadata={doc.metadata!r}"
        )
