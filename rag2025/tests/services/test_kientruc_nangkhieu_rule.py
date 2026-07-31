"""Test: BUG-RETRIEVAL-MISS — Kien truc nang khieu / ve my thuat queries
must surface chunked_12_chunk_37 (6-school list) via aggregation booster.

Pins down the retrieval miss where queries about Kien truc's drawing exam
score sources were answered "tai lieu bi khuyet" because the booster did
not inject the chunk containing the full list of 6 accepted schools.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from src.services.aggregation_booster import detect_aggregation_chunks


# --- Query texts from live c30 / c32 that trigger the bug -------------------

QUERY_C30 = (
    "Ngành Kiến trúc của trường xét những tổ hợp nào ạ? Môn thi năng khiếu "
    "em ở xa, có thể thi ở các trường gần địa phương em được không hay phải về "
    "thi trực tiếp tại trường Đại học Khoa học?"
)

QUERY_C32 = (
    "Dạ em muốn hỏi ngành kiến trúc ở trường mình sẽ lấy điểm vẽ từ đâu vậy ạ"
)


# --- Negative: should NOT trigger for generic kien truc queries -------------

QUERY_HOC_PHI = "Học phí ngành Kiến trúc là bao nhiêu?"


class TestKienTrucNangKhieuBooster:
    """Aggregation rule must inject chunk_37 (6-school list) for drawing-exam
    queries about Kien truc nang khieu / ve my thuat, and must NOT overmatch
    for generic Kien truc queries like 'hoc phi nganh kien truc'."""

    def test_c30_query_injects_chunk_37(self):
        """c30 query: nang khieu + o xa + truong gan dia phuong -> chunk_37."""
        result = detect_aggregation_chunks(QUERY_C30)
        assert "chunked_12_chunk_37" in result, (
            f"c30 query must inject chunk_37 (6-school list); got: {result}"
        )

    def test_c32_query_injects_chunk_37(self):
        """c32 query: lay diem ve tu dau -> chunk_37."""
        result = detect_aggregation_chunks(QUERY_C32)
        assert "chunked_12_chunk_37" in result, (
            f"c32 query must inject chunk_37 (6-school list); got: {result}"
        )

    def test_c30_query_also_injects_chunk_36(self):
        """c30 query should also inject chunk_36 for HUSC-organizes-exam context."""
        result = detect_aggregation_chunks(QUERY_C30)
        assert "chunked_12_chunk_36" in result, (
            f"c30 query should inject chunk_36; got: {result}"
        )

    def test_c32_query_also_injects_chunk_36(self):
        """c32 query should also inject chunk_36."""
        result = detect_aggregation_chunks(QUERY_C32)
        assert "chunked_12_chunk_36" in result, (
            f"c32 query should inject chunk_36; got: {result}"
        )

    def test_hoc_phi_kien_truc_does_NOT_trigger(self):
        """Generic 'hoc phi nganh kien truc' must NOT match the nang khieu rule."""
        result = detect_aggregation_chunks(QUERY_HOC_PHI)
        assert "chunked_12_chunk_37" not in result, (
            f"Generic hoc phi query must NOT trigger chunk_37 injection; got: {result}"
        )
        assert "chunked_12_chunk_36" not in result, (
            f"Generic hoc phi query must NOT trigger chunk_36 injection; got: {result}"
        )

    def test_draw_score_other_schools_injects_chunk_37(self):
        """Alt phrasing: 'dung ket qua thi ve tu truong khac'."""
        q = "dùng kết quả thi vẽ mỹ thuật từ trường khác được không"
        result = detect_aggregation_chunks(q)
        assert "chunked_12_chunk_37" in result, (
            f"Alt phrasing must inject chunk_37; got: {result}"
        )

    def test_nang_khieu_o_dau_injects_chunk_37(self):
        """Alt phrasing: 'nang khieu ve o dau'."""
        q = "môn thi năng khiếu vẽ mỹ thuật lấy điểm ở đâu"
        result = detect_aggregation_chunks(q)
        assert "chunked_12_chunk_37" in result, (
            f"Nang khieu o dau must inject chunk_37; got: {result}"
        )

    def test_thi_ve_gan_dia_phuong_injects_chunk_37(self):
        """Alt phrasing: 'thi ve o truong gan dia phuong'."""
        q = "em ở xa có thể thi vẽ ở các trường gần địa phương không"
        result = detect_aggregation_chunks(q)
        assert "chunked_12_chunk_37" in result, (
            f"Thi ve gan dia phuong must inject chunk_37; got: {result}"
        )
