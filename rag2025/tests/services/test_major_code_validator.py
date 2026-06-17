"""Tests for major_code_validator (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import importlib.util
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src" / "services" / "major_code_validator.py"


@pytest.fixture
def mv():
    spec = importlib.util.spec_from_file_location("major_code_validator", SRC)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_well_formed_7_digit(mv):
    assert mv.is_well_formed_major_code("7480201") is True


def test_well_formed_with_variant_suffix(mv):
    assert mv.is_well_formed_major_code("7480201VJ") is True
    assert mv.is_well_formed_major_code("7510302IC") is True


def test_not_well_formed_wrong_prefix(mv):
    assert mv.is_well_formed_major_code("8480201") is False


def test_not_well_formed_wrong_length(mv):
    assert mv.is_well_formed_major_code("748020") is False  # 6 digits
    assert mv.is_well_formed_major_code("748020111") is False  # 9 digits


def test_2026_new_majors_in_whitelist(mv):
    """C11: KHDL, Vi mạch, Bán dẫn must be in the 2026 whitelist."""
    assert mv.is_valid_2026_major("7460108") is True   # Khoa học dữ liệu
    assert mv.is_valid_2026_major("7510302IC") is True # Vi mạch tích hợp
    assert mv.is_valid_2026_major("7440102SC") is True # Bán dẫn


def test_existing_2026_major_in_whitelist(mv):
    assert mv.is_valid_2026_major("7480201") is True   # CNTT base
    assert mv.is_valid_2026_major("7460101") is True   # Toán học


def test_2025_only_variant_now_in_2026_whitelist(mv):
    """C11 update 2026-05-26: 7480201VJ (CNTT Việt-Nhật) IS in 2026 whitelist
    per live nganhdaotao-dh.php page (28 ngành) and TB74. Earlier ruling that
    excluded VJ has been superseded by cô tuyển sinh's verified live data."""
    assert mv.is_valid_2026_major("7480201VJ") is True


def test_quan_tri_phan_tich_du_lieu_NOT_in_2026_whitelist(mv):
    """C11 — 7340405 (Quản trị và phân tích dữ liệu) was REMOVED for 2026."""
    assert mv.is_valid_2026_major("7340405") is False
    assert "7340405" in mv.REMOVED_FROM_2026
    assert "7480107TD" in mv.REMOVED_FROM_2026


def test_unknown_code_not_in_whitelist(mv):
    assert mv.is_valid_2026_major("7999999") is False


def test_malformed_code_returns_false(mv):
    assert mv.is_valid_2026_major("not-a-code") is False
    assert mv.is_valid_2026_major("") is False


def test_get_2026_whitelist_is_frozen(mv):
    wl = mv.get_2026_whitelist()
    assert isinstance(wl, frozenset)
    # Whitelist should have at least the 28 entries we documented.
    assert len(wl) >= 28


def test_validate_answer_flags_unknown_codes(mv):
    answer = "Ngành 7999999 và 7480201 đều có ở HUSC."
    flagged = mv.validate_answer_majors(answer)
    assert "7999999" in flagged
    assert "7480201" not in flagged


def test_validate_answer_flags_removed_2026_code(mv):
    """C11 update 2026-05-26: 7340405 (Quản trị PT dữ liệu) is in REMOVED_FROM_2026
    so any answer mentioning it must be flagged as unknown for 2026."""
    answer = "Ngành 7340405 tuyển 50 sinh viên."
    flagged = mv.validate_answer_majors(answer)
    assert "7340405" in flagged


def test_validate_answer_does_not_flag_VJ_anymore(mv):
    """C11 update 2026-05-26: 7480201VJ is now valid for 2026, no longer flagged."""
    answer = "Ngành 7480201VJ tuyển 30 sinh viên."
    flagged = mv.validate_answer_majors(answer)
    assert "7480201VJ" not in flagged


def test_validate_answer_no_codes_returns_empty(mv):
    assert mv.validate_answer_majors("HUSC ở Huế.") == []


def test_validate_answer_dedupes_repeats(mv):
    answer = "Ngành 7999999 ở khoa A. Ngành 7999999 ở khoa B."
    assert mv.validate_answer_majors(answer) == ["7999999"]


def test_filter_chunks_keeps_no_major_code(mv):
    chunks = [{"metadata": {"info_type": "general"}}]
    kept, rejected = mv.filter_chunks_by_major_whitelist(chunks)
    assert len(kept) == 1
    assert rejected == []


def test_filter_chunks_rejects_unknown_major(mv):
    chunks = [
        {"metadata": {"major_code": "7480201"}},  # valid
        {"metadata": {"major_code": "7999999"}},  # invalid
    ]
    kept, rejected = mv.filter_chunks_by_major_whitelist(chunks)
    assert len(kept) == 1
    assert len(rejected) == 1
    assert rejected[0]["metadata"]["major_code"] == "7999999"


# =============================================================================
# S14.7 — year-loadable whitelist (de-snapshot C11 registry)
# =============================================================================
#
# These tests assert the NEW contract: a per-year data/major_codes/<year>.json
# file is preferred over the legacy WHITELIST_2026 frozenset; the legacy set is
# used as a fail-soft fallback for any year that has no JSON file. The 5 codes
# below were wrongly flagged as Type-4 hallucinations in baseline_2026 (msg
# 007/018/068/079/081) because they were missing from the legacy frozenset —
# they are real majors in husc_2026_canonical.json and must be valid for 2026.

NEW_2026_REAL_CODES = [
    "7510302",   # CNKT điện tử-viễn thông (Khoa V, khoi_v)
    "7520503",   # Kỹ thuật trắc địa-bản đồ (Khoa V)
    "7580211",   # Địa kỹ thuật xây dựng (Khoa V)
    "7229030",   # Văn học (Khoa VII)
    "7320111",   # Truyền thông số (Khoa VII)
    "7229010",   # Lịch sử (Khoa VII)
]


def test_get_whitelist_2026_includes_false_positive_real_codes(mv):
    """S14.7 — data/major_codes/2026.json is seeded with the 5 real codes
    that baseline_2026 eval falsely flagged as Type-4 hallucinations.
    After seeding, get_whitelist(2026) MUST return a set that includes
    every one of them."""
    wl = mv.get_whitelist(2026)
    assert isinstance(wl, frozenset)
    for code in NEW_2026_REAL_CODES:
        assert code in wl, (
            f"get_whitelist(2026) missing real code {code!r} — "
            "data/major_codes/2026.json is not seeded correctly"
        )


def test_is_valid_2026_major_no_arg_accepts_real_codes(mv):
    """S14.7 — no-arg ``is_valid_2026_major`` (backward-compat shim) must
    resolve to the current-year whitelist. With the 2026 seed file present,
    every previously-false-positive real code is now valid."""
    for code in NEW_2026_REAL_CODES:
        assert mv.is_valid_2026_major(code) is True, (
            f"is_valid_2026_major({code!r}) must be True under the seeded 2026 file"
        )


def test_is_valid_2026_major_no_arg_keeps_existing_valid(mv):
    """Backward-compat: legacy valid codes remain valid via the no-arg shim."""
    for code in ("7480101", "7480201", "7460101", "7510302IC", "7440102SC", "7480201VJ"):
        assert mv.is_valid_2026_major(code) is True


def test_is_valid_2026_major_no_arg_still_rejects_unknown(mv):
    """Backward-compat: fabricated codes must still be rejected."""
    for code in ("7999999", "7340405", "7480107TD"):
        assert mv.is_valid_2026_major(code) is False


def test_get_whitelist_9999_falls_back_to_legacy(mv):
    """S14.7 — absent-file fallback: get_whitelist(9999) returns the legacy
    WHITELIST_2026 frozenset without raising. This is the year-rollover
    safety guarantee."""
    wl = mv.get_whitelist(9999)
    assert isinstance(wl, frozenset)
    # Legacy snapshot still includes the original 28 codes.
    assert "7480201" in wl
    assert "7460108" in wl
    assert "7510302IC" in wl
    assert "7440102SC" in wl


def test_get_whitelist_2026_is_superset_of_legacy(mv):
    """S14.7 — the seeded 2026 file is the UNION of WHITELIST_2026 + the
    5 newly-validated real codes. Concretely, no code in WHITELIST_2026
    may be missing from get_whitelist(2026)."""
    wl_2026 = mv.get_whitelist(2026)
    legacy = mv.get_2026_whitelist()  # current-year → 2026 (file present)
    # The current-year getter resolves to the file path; the legacy constant
    # is the no-file fallback. Compare get_whitelist(2026) against the
    # legacy fallback shape (get_whitelist(9999)) for a stable invariant.
    legacy_fallback = mv.get_whitelist(9999)
    for code in legacy_fallback:
        assert code in wl_2026, (
            f"get_whitelist(2026) dropped legacy code {code!r} — "
            "2026.json must be a SUPERSET of WHITELIST_2026"
        )


def test_validate_answer_majors_no_longer_flags_real_2026_code(mv):
    """S14.7 — the 5 real codes that were falsely flagged under the legacy
    frozen set must now produce an empty invalid-list when seen in answer text.
    The no-arg signature (backward compat) still works."""
    for code in NEW_2026_REAL_CODES:
        text = f"Trường đào tạo ngành mã {code} trong năm 2026."
        assert mv.validate_answer_majors(text) == [], (
            f"validate_answer_majors must NOT flag real 2026 code {code!r}; "
            f"got {mv.validate_answer_majors(text)!r}"
        )


def test_validate_answer_majors_still_flags_fabricated(mv):
    """Backward-compat: fabricated / unknown codes remain flagged."""
    assert "7999999" in mv.validate_answer_majors("Ngành 7999999 không tồn tại.")
    # Removed-for-2026 codes must still be flagged.
    assert "7340405" in mv.validate_answer_majors("Ngành 7340405 tuyển sinh viên.")


def test_validate_answer_majors_explicit_year_kwarg(mv):
    """S14.7 — new optional ``year`` kwarg works; passing 9999 forces the
    legacy fallback (which does NOT include the 5 new real codes)."""
    # 2026 path (default + explicit) — real code is accepted
    assert mv.validate_answer_majors("Ngành 7520503", year=2026) == []
    # 9999 path — no file, falls back to legacy set, 7520503 NOT in legacy
    flagged_9999 = mv.validate_answer_majors("Ngành 7520503", year=9999)
    assert "7520503" in flagged_9999, (
        f"year=9999 must fall back to legacy (missing 7520503); got {flagged_9999!r}"
    )


def test_filter_chunks_by_major_whitelist_explicit_year(mv):
    """S14.7 — new optional ``year`` kwarg works on the chunk filter."""
    chunks_2026 = [{"metadata": {"major_code": "7520503"}}]  # valid in 2026
    kept, rejected = mv.filter_chunks_by_major_whitelist(chunks_2026, year=2026)
    assert kept == chunks_2026 and rejected == []

    # Year 9999 → legacy fallback → 7520503 NOT in legacy → rejected
    kept2, rejected2 = mv.filter_chunks_by_major_whitelist(chunks_2026, year=9999)
    assert kept2 == [] and rejected2 == chunks_2026


def test_get_whitelist_is_memoized(mv):
    """S14.7 — repeated calls return the same frozenset object (cheap, no I/O)."""
    a = mv.get_whitelist(2026)
    b = mv.get_whitelist(2026)
    assert a is b, "get_whitelist(2026) should return the cached frozenset object"


def test_get_whitelist_never_raises(mv):
    """S14.7 — the contract is 'NEVER raises' across weird inputs."""
    for y in (None, 0, -1, 9999, 10**9):
        wl = mv.get_whitelist(y)
        assert isinstance(wl, frozenset) and wl, (
            f"get_whitelist({y!r}) must return a non-empty frozenset"
        )
