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


def test_2025_only_variant_not_in_2026_whitelist(mv):
    """C11: codes that exist in 2025 dossier (id=59) but NOT id=74 must be rejected."""
    # Per user feedback, only id=74 dossier counts for 2026.
    assert mv.is_valid_2026_major("7480201VJ") is False  # CNTT Việt-Nhật — 2025 only


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


def test_validate_answer_flags_2025_only_variant(mv):
    answer = "Ngành 7480201VJ tuyển 50 sinh viên."
    flagged = mv.validate_answer_majors(answer)
    assert "7480201VJ" in flagged


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
