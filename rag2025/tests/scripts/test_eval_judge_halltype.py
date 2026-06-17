"""TDD tests for Slice S6 — deterministic hallucination-type classifier.

Four detectors under test (NO LLM in the tested path — all regex + validator):

  * detect_type4 — fabricated major code (delegates to validate_answer_majors)
  * detect_type5 — fabricated numerics (number in answer not in any context / GT)
  * detect_type1 — year-mix (answer presents 2025 fact as if current-year)
  * detect_type6 — anti-fallback fabrication (specifics when no current-year ctx)

Plus aggregate helpers:
  * classify_record(record, gt_entry, current_year) -> dict
  * build_confusion_matrix(classifications, gt_by_id) -> dict

Contract verified here:
  1. Type-4 deterministic: invalid code flagged, valid code clean
  2. Type-4 clean: only-valid-codes answer returns empty / type4 False
  3. Type-5 fired: unsupported number with no GT fact
  4. Type-5 supported: numeric-normalized match suppresses detection
  5. Type-1 fires: 2025 number offered as current; safe disambiguation suppresses
  6. Type-6 fires: empty chunks + specifics; fallback wording suppresses
  7. classify_record + confusion matrix: hand-built records yield correct cells
  8. Clean answer all-zero: grounded + valid codes + supported nums → all False
"""
from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
from typing import Any, Dict, List

import pytest


# ---------- Paths ----------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # rag2025/
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "eval_judge_halltype.py"


# ---------- Module loader ----------

@pytest.fixture(scope="module")
def jh_module():
    """Import eval_judge_halltype.py as a module (no side effects)."""
    assert SCRIPT_PATH.is_file(), f"missing script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("eval_judge_halltype", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# 1. Type-4 deterministic
# =============================================================================

def test_type4_detects_invalid_major_code(jh_module):
    """An answer mentioning 7999999 (NOT in WHITELIST_2026) must be flagged."""
    invalid_codes = jh_module.detect_type4("Ngành mã 7999999 không tồn tại trong HUSC.")
    assert "7999999" in invalid_codes, (
        f"expected 7999999 in invalid codes, got {invalid_codes!r}"
    )


def test_type4_valid_code_is_clean(jh_module):
    """An answer mentioning 7480201 (in WHITELIST_2026) must NOT be flagged."""
    clean = jh_module.detect_type4("Ngành CNTT mã 7480201 thuộc khối Kỹ thuật.")
    assert clean == [], f"expected empty invalid list, got {clean!r}"


# =============================================================================
# 2. Type-4 clean (cross-check vs is_valid)
# =============================================================================

def test_type4_clean_answer_with_only_valid_codes(jh_module):
    """An answer mentioning only valid codes → type4 False in classification."""
    answer = (
        "Trường đào tạo các ngành: Khoa học máy tính 7480101, "
        "Công nghệ thông tin 7480201, Toán-Tin 7460112."
    )
    gt = {
        "id": "q-clean-codes",
        "expected_behavior": "answer",
        "critical_facts": [
            {"value": "7480201", "fact_kind": "ma_nganh"},
        ],
    }
    record = {
        "id": "q-clean-codes",
        "question": "Trường có đào tạo ngành nào?",
        "answer": answer,
        "retrieved_chunks": [
            {
                "text": "Khoa học máy tính 7480101, CNTT 7480201, Toán-Tin 7460112.",
                "data_year": 2026,
            }
        ],
    }
    cls = jh_module.classify_record(record, gt, current_year=2026)
    assert cls["types"]["type4"] is False, (
        f"valid codes should not fire Type-4; got {cls}"
    )
    assert cls["type4_codes"] == []


# =============================================================================
# 3. Type-5 fires — unsupported number, no GT fact
# =============================================================================

def test_type5_fires_on_unsupported_money(jh_module):
    """Answer asserts 999.999 đồng/tín chỉ with no context/GT support → Type-5."""
    answer = "Học phí là 999.999 đồng/tín chỉ."
    contexts = ["Học phí khoá 2025 ngành CNTT là 600.000 VNĐ/tín chỉ."]
    crit = []
    unsupported = jh_module.detect_type5(answer, contexts, crit)
    # 999.999 normalizes to 999999; 600.000 normalizes to 600000.
    # So 999.999 must be in the unsupported set.
    joined = " | ".join(unsupported)
    assert "999999" in joined, f"expected 999999 in unsupported, got {unsupported!r}"
    assert any("999" in u for u in unsupported), (
        f"expected a 999-form value in unsupported, got {unsupported!r}"
    )


# =============================================================================
# 4. Type-5 supported — numeric-normalized match suppresses detection
# =============================================================================

def test_type5_supported_number_does_not_fire(jh_module):
    """Answer 600.000đ with context mentioning 600.000 → Type-5 must NOT fire."""
    answer = "Học phí ngành CNTT là 600.000đ/tín chỉ theo TB74."
    contexts = ["Học phí năm 2026 ngành CNTT: 600.000 VNĐ/tín chỉ."]
    crit = []
    unsupported = jh_module.detect_type5(answer, contexts, crit)
    # Both "600.000" tokens should normalize identically → supported.
    assert unsupported == [], f"expected no unsupported, got {unsupported!r}"


def test_type5_supported_via_gt_critical_fact(jh_module):
    """A number only in GT critical_facts (not in context) is also supported."""
    answer = "Học phí ngành CNTT là 750.000đ/tín chỉ."
    contexts = ["Học phí dao động theo ngành."]
    crit = [
        {"value": "750.000", "fact_kind": "hoc_phi"},
    ]
    unsupported = jh_module.detect_type5(answer, contexts, crit)
    assert unsupported == [], (
        f"GT-supported numbers should not fire; got {unsupported!r}"
    )


# =============================================================================
# 5. Type-1 year-mix
# =============================================================================

def test_type1_fires_when_2025_number_offered_as_current(jh_module):
    """Answer "điểm chuẩn 2025 là 18.50" to a current-year question → Type-1."""
    answer = "Điểm chuẩn ngành CNTT là 18.50."
    contexts_meta = [
        {"text": "Điểm chuẩn năm 2025 ngành CNTT: 18.50.", "data_year": 2025},
    ]
    question = "Điểm chuẩn năm nay của ngành CNTT là bao nhiêu?"
    assert jh_module.detect_type1(
        answer, contexts_meta, question, current_year=2026
    ) is True


def test_type1_does_not_fire_with_safe_disambiguation(jh_module):
    """Answer that explicitly tags 2025 as historical does NOT fire Type-1."""
    answer = "Năm 2025 điểm chuẩn ngành CNTT là 18.50 (năm nay chưa công bố)."
    contexts_meta = [
        {"text": "Điểm chuẩn năm 2025 ngành CNTT: 18.50.", "data_year": 2025},
    ]
    question = "Điểm chuẩn năm nay của ngành CNTT là bao nhiêu?"
    assert jh_module.detect_type1(
        answer, contexts_meta, question, current_year=2026
    ) is False


def test_type1_does_not_fire_when_no_2025_context(jh_module):
    """If NO 2025 context exists, Type-1 cannot fire (no year-mix)."""
    answer = "Điểm chuẩn ngành CNTT là 18.50."
    contexts_meta = [
        {"text": "Điểm chuẩn năm 2026 ngành CNTT: 19.00.", "data_year": 2026},
    ]
    question = "Điểm chuẩn năm nay của ngành CNTT là bao nhiêu?"
    assert jh_module.detect_type1(
        answer, contexts_meta, question, current_year=2026
    ) is False


# =============================================================================
# 6. Type-6 anti-fallback fabrication
# =============================================================================

def test_type6_fires_when_empty_chunks_plus_specifics(jh_module):
    """Empty retrieved_chunks + assertive numbers → Type-6 fired."""
    answer = "Học phí ngành CNTT năm 2026 là 18.000.000 VNĐ/năm."
    chunks: List[Dict[str, Any]] = []
    assert jh_module.detect_type6(answer, chunks, current_year=2026) is True


def test_type6_does_not_fire_on_proper_fallback(jh_module):
    """Empty chunks + a safe fallback wording → Type-6 NOT fired."""
    answer = "Tôi không tìm thấy thông tin về học phí ngành này."
    chunks: List[Dict[str, Any]] = []
    assert jh_module.detect_type6(answer, chunks, current_year=2026) is False


def test_type6_does_not_fire_when_only_2025_chunks_but_answer_safe(jh_module):
    """If chunks only have 2025 data, but answer is a safe fallback → no Type-6."""
    answer = (
        "Chưa có thông tin chính thức năm 2026. "
        "Năm 2025 học phí ngành CNTT là 14.000.000 VNĐ."
    )
    chunks = [
        {
            "text": "Học phí năm 2025 ngành CNTT: 14.000.000 VNĐ/năm.",
            "data_year": 2025,
        }
    ]
    assert jh_module.detect_type6(answer, chunks, current_year=2026) is False


# =============================================================================
# 7. classify_record + build_confusion_matrix
# =============================================================================

def test_classify_record_and_confusion_matrix(jh_module):
    """Hand-built records: 1 clean, 1 with type4, 1 with type5, 1 abstain."""
    gt_by_id = {
        "q1-clean": {
            "id": "q1-clean",
            "expected_behavior": "answer",
            "critical_facts": [{"value": "7480201", "fact_kind": "ma_nganh"}],
        },
        "q2-type4": {
            "id": "q2-type4",
            "expected_behavior": "answer",
            "critical_facts": [],
        },
        "q3-type5": {
            "id": "q3-type5",
            "expected_behavior": "answer",
            "critical_facts": [],
        },
        "q4-abstain": {
            "id": "q4-abstain",
            "expected_behavior": "abstain",
            "critical_facts": [],
        },
    }

    records = [
        # Clean answer — valid code, supported number, current-year context.
        {
            "id": "q1-clean",
            "question": "Ngành CNTT mã gì?",
            "answer": "Ngành CNTT mã 7480201 thuộc khối Kỹ thuật. "
                      "Học phí 600.000 VNĐ/tín chỉ.",
            "retrieved_chunks": [
                {
                    "text": "Ngành CNTT mã 7480201, học phí 600.000 VNĐ/tín chỉ.",
                    "data_year": 2026,
                }
            ],
        },
        # Type-4 fire: invalid code 7999999.
        {
            "id": "q2-type4",
            "question": "Trường có ngành 7999999 không?",
            "answer": "Có ngành 7999999 đào tạo tại trường.",
            "retrieved_chunks": [
                {
                    "text": "Các ngành đào tạo: 7480201, 7480101.",
                    "data_year": 2026,
                }
            ],
        },
        # Type-5 fire: fabricated money, no support.
        {
            "id": "q3-type5",
            "question": "Học phí năm 2026?",
            "answer": "Học phí năm 2026 là 999.999 VNĐ/tín chỉ.",
            "retrieved_chunks": [
                {
                    "text": "Học phí năm 2026 ngành CNTT: 600.000 VNĐ/tín chỉ.",
                    "data_year": 2026,
                }
            ],
        },
        # Abstain: empty answer, expected_behavior=abstain.
        {
            "id": "q4-abstain",
            "question": "Câu hỏi ngoài phạm vi?",
            "answer": "Tôi không tìm thấy thông tin.",
            "retrieved_chunks": [],
        },
    ]

    classifications = [
        jh_module.classify_record(r, gt_by_id[r["id"]], current_year=2026)
        for r in records
    ]

    # Sanity-check classifications.
    assert classifications[0]["types"]["type4"] is False
    assert classifications[0]["types"]["type5"] is False
    assert classifications[1]["types"]["type4"] is True
    assert "7999999" in classifications[1]["type4_codes"]
    assert classifications[2]["types"]["type5"] is True
    assert classifications[2]["type5_values"], "type5_values should be non-empty"
    # Abstain record: short fallback → observed = abstained.
    assert classifications[3]["types"]["type4"] is False
    assert classifications[3]["types"]["type5"] is False

    matrix = jh_module.build_confusion_matrix(classifications, gt_by_id)

    # Grid structure: rows=expected (answer/abstain/refuse), cols=observed
    assert "rows" in matrix and "cols" in matrix and "grid" in matrix
    rows = matrix["rows"]
    cols = matrix["cols"]
    assert rows == ["answer", "abstain", "refuse"]
    expected_cols = {
        "answered_clean",
        "answered_with_hallucination",
        "abstained",
        "refused",
    }
    assert expected_cols.issubset(set(cols)), f"missing cols; got {cols}"

    grid = matrix["grid"]

    def cell(row, col):
        for r in grid:
            if r["row"] == row:
                return r["counts"].get(col, 0)
        return 0

    # q1-clean: expected=answer, observed=answered_clean → answer×clean=1
    assert cell("answer", "answered_clean") == 1, matrix
    # q2-type4 + q3-type5 both expected=answer → 2 hallucinated.
    assert cell("answer", "answered_with_hallucination") == 2, matrix
    # q4-abstain: expected=abstain, observed=abstained → abstain×abstained=1
    assert cell("abstain", "abstained") == 1, matrix
    # Total per-row sanity.
    assert matrix["totals"]["rows_total"] == 4
    assert matrix["totals"]["cols_total"] == 4

    # Per-type counts (sum across classifications).
    pc = matrix["per_type_counts"]
    assert pc["type1"] == 0
    assert pc["type4"] == 1
    assert pc["type5"] == 1
    assert pc["type6"] == 0


# =============================================================================
# 8. Clean answer all-zero
# =============================================================================

def test_clean_answer_all_four_types_false(jh_module):
    """A fully grounded answer → all four types False."""
    answer = (
        "Ngành CNTT mã 7480201 thuộc khối Kỹ thuật. "
        "Học phí 600.000 VNĐ/tín chỉ. "
        "Điểm chuẩn 2026 ngành CNTT: 19.00."
    )
    record = {
        "id": "q-clean",
        "question": "Thông tin ngành CNTT?",
        "answer": answer,
        "retrieved_chunks": [
            {
                "text": (
                    "Ngành CNTT mã 7480201. "
                    "Học phí 600.000 VNĐ/tín chỉ. "
                    "Điểm chuẩn 2026 ngành CNTT: 19.00."
                ),
                "data_year": 2026,
            }
        ],
    }
    gt = {
        "id": "q-clean",
        "expected_behavior": "answer",
        "critical_facts": [
            {"value": "7480201", "fact_kind": "ma_nganh"},
            {"value": "600.000", "fact_kind": "hoc_phi"},
        ],
    }
    cls = jh_module.classify_record(record, gt, current_year=2026)
    assert cls["types"] == {
        "type1": False,
        "type4": False,
        "type5": False,
        "type6": False,
    }, f"expected all-false; got {cls['types']}"
    assert cls["type4_codes"] == []
    assert cls["type5_values"] == []


# =============================================================================
# Helper exposure (imported validators reachable from the module)
# =============================================================================

def test_module_exposes_numeric_normalize(jh_module):
    """The module must expose a numeric-normalize helper (spec requirement)."""
    norm = jh_module.numeric_normalize
    assert norm("600.000") == "600000"
    assert norm("600,000") == "600000"
    assert norm("600 000") == "600000"
    assert norm("999.999") == "999999"


def test_module_exposes_is_valid_alias(jh_module):
    """The module must re-export is_valid_2026_major for downstream use."""
    assert hasattr(jh_module, "is_valid_2026_major")
    assert jh_module.is_valid_2026_major("7480201") is True
    assert jh_module.is_valid_2026_major("7999999") is False
