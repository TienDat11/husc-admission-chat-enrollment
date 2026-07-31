"""TDD: V5-R013 / FIX 2 — router/intent mapping for 'liệt kê ngành' queries.

Root cause (2 parts):
 (a) main.py INTENT_TO_FAQ_TYPES was missing key "liet_ke" → fell through
     to {"faq_type": "thong_tin_nganh"} (chỉ detail chunks, không lấy
     listing chunks như liet_ke_28_nganh_2026_v2 /
     bang_xep_hang_chi_tieu_2026_v2 / tuyensinh_overview_2026_v2).
 (b) query_router.py _ENUMERATION_PATTERNS did not match
     "Chi tiết các ngành năm nay tuyển sinh của HUSC" → fell through
     to hybrid/general path → no graph_rag intent=liet_ke.

Fix:
 - main.py: add "liet_ke" key with the union of aggregate + detail
   faq_type values, and route is_program_list through it.
 - query_router.py: add "chi tiết các ngành", "thông tin các ngành",
   "tìm hiểu về các ngành", "các ngành.*tuyển sinh", "ngành.*đào tạo"
   to BOTH _ENUMERATION_PATTERNS and _MAJOR_QUESTION_PATTERNS so the
   route is graph_rag and intent is liet_ke.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Patterns the FIX must now match (enumeration-style listing)
LISTING_QUERIES = [
    "Em muốn tìm hiểu về các ngành đào tạo",
    "Chi tiết các ngành năm nay tuyển sinh HUSC",
    "Chi tiết các ngành năm nay tuyển sinh của HUSC",
    "thông tin các ngành đào tạo",
    "tìm hiểu các ngành đào tạo",
    "các ngành tuyển sinh 2026",
    "ngành nào đào tạo",
]


def test_router_enumeration_patterns_match_listing():
    """V5-R013 FIX 2b: _ENUMERATION_PATTERNS must match enumerate-style queries."""
    from services.query_router import _ENUMERATION_PATTERNS
    for q in LISTING_QUERIES:
        assert _ENUMERATION_PATTERNS.search(q), (
            f"_ENUMERATION_PATTERNS missing query: {q!r}"
        )


def test_router_major_question_patterns_match_listing():
    """V5-R013 FIX 2b: _MAJOR_QUESTION_PATTERNS must also match — so
    the contact-block auto-answer does NOT trigger for listing queries."""
    from services.query_router import _MAJOR_QUESTION_PATTERNS
    for q in LISTING_QUERIES:
        assert _MAJOR_QUESTION_PATTERNS.search(q), (
            f"_MAJOR_QUESTION_PATTERNS missing query: {q!r}"
        )


def test_main_intent_to_faq_types_has_liet_ke():
    """V5-R013 FIX 2a: INTENT_TO_FAQ_TYPES must have 'liet_ke' key.

    We verify the main module is importable + has the constants.
    """
    import importlib
    main_mod = importlib.import_module("src.main")
    src = Path(main_mod.__file__).read_text(encoding="utf-8")
    # The literal dict literal in main.py must contain "liet_ke" key.
    assert '"liet_ke"' in src or "'liet_ke'" in src, (
        "INTENT_TO_FAQ_TYPES missing liet_ke key"
    )
    # And it must include tong_hop_tuyen_sinh (aggregate listing chunks).
    assert "tong_hop_tuyen_sinh" in src, (
        "INTENT_TO_FAQ_TYPES liet_ke missing tong_hop_tuyen_sinh"
    )


def test_main_get_faq_type_filter_liet_ke_includes_aggregate():
    """V5-R013 FIX 2a: get_faq_type_filter must produce or_conditions
    that include tong_hop_tuyen_sinh for listing queries."""
    # Read main.py source and check that get_faq_type_filter handles
    # is_program_list via the new liet_ke mapping. We can't easily
    # import the inner function (it's nested) so we use a static check.
    import importlib
    main_mod = importlib.import_module("src.main")
    src = Path(main_mod.__file__).read_text(encoding="utf-8")
    # The new code path: when is_program_list, return or_conditions
    # using INTENT_TO_FAQ_TYPES["liet_ke"].
    assert "INTENT_TO_FAQ_TYPES[\"liet_ke\"]" in src, (
        "get_faq_type_filter should route is_program_list through liet_ke"
    )
    # And it must NOT be the old single-faqqtype {"faq_type": "thong_tin_nganh"}
    # anymore in the is_program_list branch.
    # Find the is_program_list block.
    m = re.search(r"is_program_list.*?return None", src, re.DOTALL)
    assert m, "could not locate is_program_list block in main.py"
    block = m.group(0)
    assert '"thong_tin_nganh"' not in block, (
        "is_program_list must no longer return bare thong_tin_nganh"
    )
    assert "or_conditions" in block, (
        "is_program_list must use or_conditions to allow liet_ke aggregates"
    )