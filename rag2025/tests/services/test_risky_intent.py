"""Tests for risky_intent (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import importlib.util
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src" / "services" / "risky_intent.py"


@pytest.fixture
def ri():
    spec = importlib.util.spec_from_file_location("risky_intent", SRC)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_diem_chuan_detected(ri):
    assert ri.infer_intent_from_query("Điểm chuẩn ngành CNTT?") == "diem_chuan"


def test_diem_chuan_synonym_diem_trung_tuyen(ri):
    assert ri.infer_intent_from_query("Điểm trúng tuyển ngành Toán?") == "diem_chuan"


def test_hoc_phi_detected(ri):
    assert ri.infer_intent_from_query("Học phí 2026 là bao nhiêu?") == "hoc_phi"


def test_hoc_phi_with_nganh(ri):
    assert ri.infer_intent_from_query("Học phí ngành CNTT là bao nhiêu?") == "hoc_phi"


def test_chi_tieu_detected(ri):
    assert ri.infer_intent_from_query("Chỉ tiêu tuyển sinh năm nay?") == "chi_tieu"


def test_chi_tieu_synonym_so_luong_tuyen(ri):
    assert ri.infer_intent_from_query("Số lượng tuyển ngành Lý?") == "chi_tieu"


def test_da_hop_detected(ri):
    assert ri.infer_intent_from_query("Tổ hợp xét tuyển ngành CNTT") == "da_hop"


def test_non_risky_returns_none(ri):
    assert ri.infer_intent_from_query("HUSC ở đâu?") is None


def test_general_admin_question_returns_none(ri):
    assert ri.infer_intent_from_query("Quy trình xét tuyển ra sao?") is None


def test_empty_returns_none(ri):
    assert ri.infer_intent_from_query("") is None


def test_non_string_returns_none(ri):
    assert ri.infer_intent_from_query(None) is None  # type: ignore[arg-type]


def test_is_risky_intent_true(ri):
    assert ri.is_risky_intent("Học phí ngành CNTT?") is True


def test_is_risky_intent_false(ri):
    assert ri.is_risky_intent("HUSC ở đâu?") is False


def test_RISKY_INTENTS_set_complete(ri):
    assert ri.RISKY_INTENTS == frozenset({"diem_chuan", "hoc_phi", "chi_tieu", "da_hop"})


def test_to_hop_hop_le_NOT_classified_as_da_hop(ri):
    """HIGH-2 regression: 'Tổ hợp hợp lệ là gì?' must NOT match da_hop pattern."""
    assert ri.infer_intent_from_query("Tổ hợp hợp lệ là gì?") is None
    assert ri.infer_intent_from_query("Có hợp lệ không?") is None


def test_to_hop_xet_tuyen_still_detected(ri):
    """Sanity: legitimate da_hop queries still classified."""
    assert ri.infer_intent_from_query("Tổ hợp xét tuyển ngành CNTT") == "da_hop"
    assert ri.infer_intent_from_query("Tổ hợp môn ngành Toán?") == "da_hop"
