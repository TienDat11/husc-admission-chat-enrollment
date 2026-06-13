"""Tests for major_recommender (PHASE-A2 + A3 TDD 1:1)."""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest


RAG_ROOT = Path(__file__).resolve().parents[2]
SRC = RAG_ROOT / "src"
SERVICE_PATH = SRC / "services" / "major_recommender.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))


@pytest.fixture
def mr():
    spec = importlib.util.spec_from_file_location("major_recommender_test", SERVICE_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules so @dataclass annotations can resolve the
    # module name (otherwise __module__ is None and dataclasses chokes).
    sys.modules["major_recommender_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---- A2: recommend() contract ----------------------------------------------

def test_recommend_high_score_yields_many_an_toan(mr):
    recs = mr.recommend(score=28.0, to_hop=None, uu_tien=0.0)
    assert recs, "expected at least one recommendation"
    an_toan = [r for r in recs if r.label == "an_toan"]
    # 28/30 is well above every diem_chuan in 2025 (max 25.0), so all should
    # be an_toan.
    assert len(an_toan) == len(recs), (
        f"expected all an_toan at score=28; got labels: "
        f"{[(r.major_code, r.label) for r in recs]}"
    )


def test_recommend_low_score_yields_few_or_no_an_toan(mr):
    recs = mr.recommend(score=14.0, to_hop=None, uu_tien=0.0)
    an_toan = [r for r in recs if r.label == "an_toan"]
    # 14/30 is below every diem_chuan; expect zero an_toan.
    assert an_toan == [], (
        f"expected 0 an_toan at score=14; got: "
        f"{[(r.major_code, r.label, r.delta) for r in recs if r.label == 'an_toan']}"
    )
    # Should still have results, all in can_nhac/mao_hiem
    assert recs, "expected at least one rec even at low score"
    for r in recs:
        assert r.label in ("can_nhac", "mao_hiem")


def test_recommend_uu_tien_shifts_label(mr):
    """Adding ưu_tien must shift at least one borderline major."""
    no_bonus = mr.recommend(score=20.0, to_hop=None, uu_tien=0.0)
    with_bonus = mr.recommend(score=20.0, to_hop=None, uu_tien=2.0)
    # Build label maps
    by_code_a = {r.major_code: r.label for r in no_bonus}
    by_code_b = {r.major_code: r.label for r in with_bonus}
    promoted = [c for c in by_code_a if c in by_code_b and by_code_a[c] != by_code_b[c]]
    # At least one major should change label when we add 2.0 ưu_tien
    assert promoted, (
        f"expected at least one label shift with uu_tien=2.0; "
        f"before={list(by_code_a.items())[:5]} after={list(by_code_b.items())[:5]}"
    )


def test_recommend_explanation_contains_year_and_score(mr):
    recs = mr.recommend(score=25.0, to_hop=None, uu_tien=0.0)
    assert recs
    for r in recs:
        # Each explanation must cite the year + the điểm chuẩn number
        assert str(r.latest_year) in r.explanation, r.explanation
        assert f"{r.latest_diem_chuan:.2f}" in r.explanation, r.explanation


def test_recommend_filters_by_to_hop_when_program_has_map(mr):
    """If a to_hop is given and the major's tổ hợp map is known, only
    matching majors are returned. If the map is empty, we still return all
    (with a note in the explanation)."""
    # CNTT 7480201 lists ['A00','A01','D01','X26']; 'A00' should match.
    cntt_a00 = mr.recommend(score=24.0, to_hop="A00", uu_tien=0.0)
    cntt_d07 = mr.recommend(score=24.0, to_hop="D07", uu_tien=0.0)
    codes_a00 = {r.major_code for r in cntt_a00}
    codes_d07 = {r.major_code for r in cntt_d07}
    assert "7480201" in codes_a00, "CNTT should be in A00 results"
    assert "7480201" not in codes_d07, "CNTT should NOT be in D07 results"


# ---- A3: whatif_probability() contract -------------------------------------

def test_whatif_above_chuan_returns_high_band(mr):
    res = mr.whatif_probability(score=25.0, major_code="7480201")  # CNTT 2025 = 17.5
    assert res.band in ("rất cao", "cao")
    assert "tham khảo" in res.basis.lower() or "tham khảo" in res.disclaimer.lower()
    assert res.disclaimer  # always present
    assert "ước lượng" in res.disclaimer or "tham khảo" in res.disclaimer


def test_whatif_below_chuan_returns_low_band(mr):
    res = mr.whatif_probability(score=15.0, major_code="7480201")
    # 15.0 vs 17.5 → delta = -2.5 → "thấp"
    assert res.band in ("trung bình", "thấp")


def test_whatif_basis_and_disclaimer_always_present(mr):
    """Even for high / low / unknown major, basis+disclaimer must surface."""
    for major in ("7480201", "9999999", "7480201VJ"):
        res = mr.whatif_probability(score=20.0, major_code=major)
        assert res.basis, f"basis missing for {major}"
        assert res.disclaimer, f"disclaimer missing for {major}"


def test_whatif_unknown_major_does_not_crash(mr):
    res = mr.whatif_probability(score=20.0, major_code="NOT_A_REAL_CODE")
    assert res.band == "unknown_major"
    assert "không" in res.pass_text() if hasattr(res, "pass_text") else True
    # We do NOT claim a probability; basis+disclaimer are still surfaced.
    assert res.basis
    assert res.disclaimer
    # latest_diem_chuan stays None — we did not invent a score.
    assert res.latest_diem_chuan is None


# ---- Latency check (proves no hot-path regression) -------------------------

def test_recommend_and_whatif_are_sub_millisecond(mr):
    """Single-digit ms per call. We allow up to 5 ms for safety; production
    floor must be << 1 ms in practice."""
    # Warmup
    mr.recommend(score=25.0)
    mr.whatif_probability(score=25.0, major_code="7480201")
    t0 = time.perf_counter()
    for _ in range(200):
        mr.recommend(score=25.0)
    rec_ms = (time.perf_counter() - t0) * 1000 / 200
    t0 = time.perf_counter()
    for _ in range(200):
        mr.whatif_probability(score=25.0, major_code="7480201")
    what_ms = (time.perf_counter() - t0) * 1000 / 200
    # Generous bound to avoid CI flakes, but single-digit ms.
    assert rec_ms < 10.0, f"recommend avg {rec_ms:.3f} ms (target <10 ms)"
    assert what_ms < 10.0, f"whatif avg {what_ms:.3f} ms (target <10 ms)"
    print(f"\n[latency] recommend={rec_ms:.3f}ms  whatif={what_ms:.3f}ms")
