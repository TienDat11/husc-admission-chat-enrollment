"""Deterministic test: enumerate-equivalent queries MUST all match _ENUMERATION_PATTERNS.

10 groups x 5 queries = 50 total. Every query in a group is a semantic-equivalent
variant of the same "list / count majors" intent. All must hit the regex fast-path
so the router deterministically returns graph_rag without falling through to LLM classify.

Regression guard: any new variant that should match but doesn't is a P0 bug.
"""

import pytest

from src.services.query_router import _ENUMERATION_PATTERNS


# ── Helpers ────────────────────────────────────────────────────────────────

def _match(query: str) -> bool:
    """Simulate the fast-path check in QueryRouter.route()."""
    return bool(_ENUMERATION_PATTERNS.search(query.strip().lower()))


# ── Group 1: "bao nhiêu ngành" variants (count-majors) ───────────────────
G1_BAO_NHIEU_NGANH = [
    "Tuyển sinh năm 2026 HUSC có bao nhiêu ngành?",
    "HUSC năm 2026 tuyển sinh tổng cộng mấy ngành?",
    "Tổng số ngành HUSC tuyển sinh 2026 là bao nhiêu?",
    "Năm 2026 HUSC đào tạo những ngành nào?",
    "bao nhiêu ngành tuyển sinh 2026",
]

# ── Group 2: "liệt kê / danh sách" variants (list-majors) ────────────────
G2_LIET_KE = [
    "Liệt kê các ngành đào tạo của HUSC",
    "Danh sách ngành tuyển sinh 2026",
    "Cho xem danh sách các ngành học",
    "Tất cả các ngành đào tạo tại HUSC",
    "Toàn bộ ngành học tại HUSC năm 2026",
]

# ── Group 3: "có mấy ngành" variants (bare count) ────────────────────────
G3_CO_MAY_NGANH = [
    "HUSC có mấy ngành tuyển sinh?",
    "HUSC có bao nhiêu ngành học?",
    "Có mấy ngành đào tạo tại HUSC?",
    "Có bao nhiêu ngành tuyển sinh 2026?",
    "HUSC năm nay có mấy ngành?",
]

# ── Group 4: "ngành nào" variants (which majors) ────────────────────────
G4_NGANH_NAO = [
    "HUSC có những ngành nào?",
    "Ngành nào đang tuyển sinh tại HUSC?",
    "Tuyển sinh những ngành nào năm 2026?",
    "Tuyển sinh ngành nào?",
    "Các ngành nào thuộc HUSC?",
]

# ── Group 5: "đào tạo những ngành nào" variants ────────────────────────
G5_DAO_TAO_NGANH = [
    "Đào tạo những ngành nào?",
    "Đào tạo các ngành nào tại HUSC?",
    "HUSC đào tạo những ngành nào?",
    "Đào tạo ngành nào ở HUSC?",
    "HUSC hiện đào tạo những ngành nào?",
]

# ── Group 6: "chi tiết / thông tin" variants ─────────────────────────────
G6_CHI_TIET = [
    "Chi tiết các ngành tuyển sinh 2026",
    "Thông tin các ngành đào tạo của HUSC",
    "Tìm hiểu về các ngành học tại HUSC",
    "Giới thiệu các ngành đào tạo HUSC",
    "Chi tiết ngành học tại HUSC",
]

# ── Group 7: "các ngành có / thuộc / tuyển sinh" variants ────────────────
G7_CAC_NGANH = [
    "Các ngành có tuyển sinh năm 2026",
    "Các ngành thuộc khoa CNTT",
    "Các ngành tuyển sinh năm nay",
    "Các ngành đào tạo hệ đại học",
    "Ngành nào đào tạo ngành CNTT?",
]

# ── Group 8: "mấy ngành / bao nhiêu ngành" bare (no leading "có") ────────
G8_BARE_MAY = [
    "mấy ngành tuyển sinh",
    "bao nhiêu ngành",
    "bao nhiêu ngành tuyển sinh",
    "mấy ngành học",
    "mấy ngành đào tạo",
]

# ── Group 9: "tổng cộng / tổng số ngành" variants ────────────────────────
G9_TONG_SO = [
    "Tổng cộng bao nhiêu ngành?",
    "Tổng số ngành là bao nhiêu?",
    "Tổng cộng mấy ngành tuyển sinh?",
    "HUSC tổng cộng có mấy ngành?",
    "Tổng số ngành đào tạo tại HUSC",
]

# ── Group 10: mixed context (year + location + modifier) ─────────────────
G10_MIXED_CONTEXT = [
    "HUSC 2026 tuyển sinh những ngành nào?",
    "Năm 2026 có bao nhiêu ngành học tại HUSC?",
    "Tuyển sinh đại học HUSC có mấy ngành?",
    "Trường HUSC đào tạo bao nhiêu ngành?",
    "Năm nay HUSC có mấy ngành tuyển sinh?",
]

ALL_GROUPS = [
    ("G1-bao-nhieu-nganh", G1_BAO_NHIEU_NGANH),
    ("G2-liet-ke", G2_LIET_KE),
    ("G3-co-may-nganh", G3_CO_MAY_NGANH),
    ("G4-nganh-nao", G4_NGANH_NAO),
    ("G5-dao-tao-nganh", G5_DAO_TAO_NGANH),
    ("G6-chi-tiet", G6_CHI_TIET),
    ("G7-cac-nganh", G7_CAC_NGANH),
    ("G8-bare-may", G8_BARE_MAY),
    ("G9-tong-so", G9_TONG_SO),
    ("G10-mixed-context", G10_MIXED_CONTEXT),
]


# ── Parametrized test ──────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "group_name,query",
    [(gname, q) for gname, queries in ALL_GROUPS for q in queries],
    ids=[f"{gname}::q{i}" for gname, queries in ALL_GROUPS for i, _ in enumerate(queries)],
)
def test_enumeration_pattern_matches(group_name: str, query: str):
    """Each enumeration-equivalent query MUST match _ENUMERATION_PATTERNS."""
    assert _match(query), (
        f"Regex fast-path MISS for {group_name!r}: {query!r}\n"
        "This query is semantically equivalent to other enumeration queries "
        "and must be caught deterministically without LLM classify."
    )


# ── Sanity: non-enumeration queries must NOT match ────────────────────────
FALSE_POSITIVE_CASES = [
    "điểm chuẩn ngành CNTT bao nhiêu?",
    "học phí ngành QTKD là bao nhiêu?",
    "HUSC ở đâu?",
    "thời gian tuyển sinh khi nào?",
    "ngành nào dễ xin việc?",
    "điểm chuẩn 2026",
    "học phí bao nhiêu?",
    "tổ hợp xét tuyển A00",
    "Xem điểm chuẩn ngành CNTT",
    "HUSC ở quận mấy?",
]


@pytest.mark.parametrize(
    "query",
    FALSE_POSITIVE_CASES,
    ids=[f"FP::{i}" for i, _ in enumerate(FALSE_POSITIVE_CASES)],
)
def test_enumeration_pattern_no_false_positive(query: str):
    """Non-enumeration queries must NOT match the pattern."""
    assert not _match(query), (
        f"False positive: {query!r} matched _ENUMERATION_PATTERNS but should NOT."
    )
