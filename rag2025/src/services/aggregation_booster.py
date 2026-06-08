"""Aggregation Chunk Booster — inject high-value summary chunks into retrieval results.

When the query is enumeration / listing / counting / superlative / comparison,
prepend relevant aggregation chunks to baseline retrieval so the generator has a
summary view alongside per-ngành chunks.

Phase E — Slice E.2.

G1-T2 / G1-T3 (durability, S14.x): the current admission year is now derived
at module init from `temporal_authority.get_current_admission_year()`. The
year-suffixed chunk-ids (`_2026` style) and year-literal regex patterns are
built at runtime from this authority. Explicit historical rules (e.g. 2025)
are kept as intentional literal rules. The chunk-id set at year=2026 is
byte-identical to the golden fixture `tests/services/fixtures/
booster_chunkids_2026.golden.json` (45 ids).

A `BOOSTER_CHUNK_MISS` counter and a visible WARN log are emitted whenever
`fetch_by_id` returns None for an aggregation chunk — so silent de-boost
failures are observable in production.
"""
from __future__ import annotations

import re
from dataclasses import replace
from threading import Lock
from typing import List, Optional

from loguru import logger

from src.services.lancedb_retrieval import RetrievedDocument


# ─── G1-T3: booster chunk-miss counter ──────────────────────────────────────
# Lightweight in-process counter (no Prometheus dep) so the WARN-vs-DEBUG
# regression in G1-T3 is unit-testable. Production metric export is wired
# elsewhere (see observability.request_tracker). Read with `.get()`.

class _BoosterChunkMiss:
    """Tiny thread-safe counter used by the booster to surface None-fetches."""

    def __init__(self) -> None:
        self._n = 0
        self._lock = Lock()

    def inc(self, n: int = 1) -> None:
        with self._lock:
            self._n += n

    def get(self) -> int:
        with self._lock:
            return self._n


BOOSTER_CHUNK_MISS = _BoosterChunkMiss()


# ─── G1-T2: derive the current year at module init ─────────────────────────
# Lazy import is unnecessary: temporal_authority itself is dependency-free.

def _current_admission_year() -> int:
    """Return the current admission year from the temporal authority.

    Reads the env or the dev/test fallback calendar via
    `temporal_authority.get_current_admission_year()`. Never returns a
    hardcoded literal; fail-loud in prod is the authority's contract.
    """
    from src.services.temporal_authority import get_current_admission_year
    return int(get_current_admission_year())


# Snapshot the year ONCE at module import. If the env is patched later
# (e.g. by tests), the booster's at-import snapshot still drives the
# chunk-id suffixes. Tests that need a different year should reload the
# module (importlib.reload) AFTER patching the env. This matches the
# authority's own "frozen-at-init" semantics in lancedb_retrieval.
_YEAR: int = _current_admission_year()


# Historical year constant — kept as an EXPLICIT literal because the rule
# set intentionally references 2025 admissions data. (The spec calls for
# "keep explicit historical (2025) rules and year-agnostic anchors".)
_HISTORICAL_YEAR_1: int = 2025  # previous-cycle data; per-year 2025
                                # rules remain static on purpose.

# Year-agnostic anchor for queries that don't name a year (e.g. "học phí
# hiện tại"). Treated as the year suffix in chunk-ids so the booster still
# resolves to the current-year chunk.
_YEAR_AGNOSTIC_SUFFIX = f"{_YEAR}"


# ─── Rule builder helpers ──────────────────────────────────────────────────

def _chunk(template: str) -> str:
    """Build a current-year chunk-id by templating `{year}` into a template.

    Templates like `"hocphi_{year}"` or `"liet_ke_28_nganh_{year}_v2"` are
    resolved against the current admission year. Pure helper — no I/O.
    """
    return template.format(year=_YEAR)


def _pat(template: str) -> str:
    """Build a current-year regex pattern by templating `{year}` into a
    raw-string template. Mirrors the `_chunk` helper for patterns."""
    return template.format(year=_YEAR)


# Map query pattern → list of priority chunk_ids to inject.
# Each rule has EITHER literal patterns (for year-agnostic anchors /
# historical-year rules) OR runtime-built `{year}` patterns. Mixing is
# allowed; both forms end up as compiled regex objects at import.
_AGGREGATION_RULES = [
    {
        "patterns": [
            r"bao\s+nhiêu\s+ngành",
            r"tất\s+cả\s+ngành",
            r"danh\s+sách\s+ngành",
            r"liệt\s+kê.*ngành",
            r"toàn\s+bộ\s+ngành",
            r"có\s+mấy\s+ngành",
            r"các\s+ngành",
            r"ngành\s+đào\s+tạo",
            r"ngành\s+học",
            r"chương\s+trình\s+đào\s+tạo",
            r"tìm\s+hiểu.*ngành",
            r"ngành.*trường",
            r"ngành\s+gì",
            r"có\s+ngành",
            r"những\s+ngành",
            r"trường\s+có.*ngành.*gì",
        ],
        "chunks": [
            _chunk("liet_ke_28_nganh_{year}_v2"),
            _chunk("tuyensinh_overview_{year}_v2"),
            _chunk("bang_xep_hang_chi_tieu_{year}_v2"),
        ],
    },
    {
        "patterns": [
            r"chỉ\s+tiêu\s+cao\s+nhất",
            r"top\s+\d+\s+ngành",
            r"xếp\s+hạng",
            r"chỉ\s+tiêu\s+thấp\s+nhất",
            r"ngành\s+nào.*nhiều\s+chỉ\s+tiêu",
        ],
        "chunks": [
            _chunk("bang_xep_hang_chi_tieu_{year}_v2"),
            _chunk("liet_ke_28_nganh_{year}_v2"),
        ],
    },
    {
        "patterns": [
            r"so\s+sánh.*học\s+phí",
            # Historical-1 vs current comparison (both 2025/2026 today; expands
            # automatically to 2026/2027 next year via the {year} template).
            _pat(r"học\s+phí.*\b{year}\b.*\b2025\b"),
            _pat(r"học\s+phí.*\b2025\b.*\b{year}\b"),
        ],
        "chunks": [
            # The comparison chunk id itself is historical-y (literal 2025 vs
            # 2026). When the year changes the cross-year comparison id will
            # need a re-ingest; we keep the same id format so retrieval keeps
            # working through the 2025->{year} boundary.
            "so_sanh_hocphi_2025_vs_2026",
            _chunk("hocphi_{year}"),
            "hocphi_2025",
        ],
    },
    # Historical-year-only rules (2025). Keep them LITERAL — the plan
    # requires explicit historical anchors.
    {
        "patterns": [r"học\s+phí.*2025", r"học\s+phí.*năm\s+2025"],
        "chunks": ["hocphi_2025"],
    },
    {
        "patterns": [
            _pat(r"học\s+phí.*\b{year}\b"),
            _pat(r"học\s+phí.*năm\s+{year}"),
            # Year-agnostic anchor (kept literal: "hiện tại" = "current").
            r"học\s+phí.*hiện\s+tại",
        ],
        "chunks": [_chunk("hocphi_{year}")],
    },
    {
        "patterns": [r"điểm\s+chuẩn.*2025", r"điểm\s+chuẩn.*năm\s+ngoái"],
        "chunks": ["diem_chuan_2025_full"],
    },
    {
        "patterns": [
            r"phương\s+thức.*xét\s+tuyển",
            r"5\s+phương\s+thức",
            r"các\s+cách\s+xét\s+tuyển",
            r"phương\s+thức.*tuyển\s+sinh",
            r"có\s+những\s+phương\s+thức",
        ],
        "chunks": [_chunk("phuongthuc_xettuyen_{year}_v2"), _chunk("phuongthuc_xettuyen_{year}")],
    },
    {
        "patterns": [
            r"điểm\s+cộng",
            r"ưu\s+tiên",
            r"chứng\s+chỉ\s+ngoại\s+ngữ",
            r"học\s+sinh\s+giỏi",
            r"giỏi.*điểm",
        ],
        "chunks": [_chunk("chinhsach_{year}")],
    },
    {
        "patterns": [r"tổ\s+hợp\s+a00", r"ngành.*a00", r"a00.*ngành", r"tổ\s+hợp.*xét", r"xét.*tổ\s+hợp"],
        "chunks": [_chunk("to_hop_a00_full_{year}"), "tohop_a00"],
    },
    {
        "patterns": [
            _pat(r"điểm\s+mới.*\b{year}\b"),
            _pat(r"thay\s+đổi.*\b{year}\b"),
            _pat(r"khác\s+biệt.*\b2025\b.*\b{year}\b"),
        ],
        "chunks": [_chunk("thay_doi_tuyensinh_{year}")],
    },
    {
        "patterns": [
            r"không\s+tuyển",
            r"toán\s+học",
            r"sinh\s+học\s+thuần",
            r"khoa\s+học\s+máy\s+tính",
            r"\btin\s+học\b",
        ],
        "chunks": [_chunk("nganh_khong_tuyen_{year}")],
    },
    {
        "patterns": [
            r"địa\s+chỉ",
            r"hotline",
            r"liên\s+hệ",
            r"số\s+điện\s+thoại",
            r"email.*tuyển\s+sinh",
            r"mã\s+trường",
        ],
        "chunks": ["husc_info"],
    },
    # ── Phase F additions (21 rules) ──────────────────────────────────
    {
        "patterns": [r"điểm\s+sàn", r"điểm.*năm\s+trước", r"điểm.*năm\s+ngoái"],
        "chunks": ["diem_chuan_2025_full"],
    },
    {
        "patterns": [r"điểm.*3\s+năm", r"điểm.*chênh\s+lệch", r"điểm.*biến\s+động"],
        "chunks": ["diem_chuan_2025_full"],
    },
    {
        "patterns": [r"chỉ\s+tiêu.*2025", r"chỉ\s+tiêu.*năm\s+trước"],
        "chunks": ["chi_tieu_2025_full"],
    },
    {
        "patterns": [r"khối\s+c01", r"tổ\s+hợp\s+c01"],
        "chunks": [_chunk("tohop_c01_full_{year}")],
    },
    {
        "patterns": [r"khối\s+d01", r"tổ\s+hợp\s+d01"],
        "chunks": [_chunk("tohop_d01_full_{year}")],
    },
    {
        "patterns": [r"cntt.*việt[- ]nhật", r"cntt.*vj", r"việt[- ]nhật.*cntt", r"so\s+sánh.*cntt"],
        "chunks": [_chunk("husc_nganh_7480201_{year}"), _chunk("husc_nganh_7480201VJ_{year}")],
    },
    {
        "patterns": [r"khi\s+nào.*mở.*xét\s+tuyển", r"khi\s+nào.*đăng\s+ký", r"lịch.*tuyển\s+sinh", r"hạn.*nộp"],
        "chunks": [_chunk("lich_tuyen_sinh_{year}")],
    },
    {
        "patterns": [r"ktx", r"ký\s+túc\s+xá"],
        "chunks": ["qa_ktx_husc"],
    },
    {
        "patterns": [r"đi\s+dạy", r"nghiệp\s+vụ\s+sư\s+phạm", r"sư\s+phạm.*văn", r"sư\s+phạm.*sử"],
        "chunks": ["qa_nghe_nghiep_van_hoc_lich_su"],
    },
    {
        "patterns": [r"năng\s+khiếu.*xa", r"thi.*năng\s+khiếu.*địa\s+phương", r"điểm\s+vẽ.*chuyển"],
        "chunks": ["qa_nang_khieu_kien_truc_thi_xa"],
    },
    {
        "patterns": [r"song\s+ngành", r"học.*2\s+trường"],
        "chunks": ["qa_song_nganh_2_truong"],
    },
    {
        "patterns": [r"link.*xét\s+tuyển", r"cổng.*riêng.*husc", r"link.*học\s+bạ"],
        "chunks": ["qa_link_cong_xet_tuyen_husc"],
    },
    {
        "patterns": [r"zalo", r"group.*tư\s+vấn", r"hotline"],
        "chunks": ["qa_lien_he_tu_van_husc"],
    },
    {
        "patterns": [r"thpt.*gdtx", r"gdtx.*thpt", r"hệ\s+gdtx"],
        "chunks": ["qa_thpt_vs_gdtx"],
    },
    {
        "patterns": [r"chỉ\s+tiêu.*thay\s+đổi", r"chỉ\s+tiêu.*tăng", r"chỉ\s+tiêu.*điều\s+chỉnh"],
        "chunks": ["qa_chi_tieu_co_thay_doi"],
    },
    {
        "patterns": [r"đgnl.*học\s+bạ", r"đánh\s+giá\s+năng\s+lực.*học\s+bạ", r"dgnl.*scan"],
        "chunks": [_chunk("qa_dgnl_minh_chung_{year}")],
    },
    {
        "patterns": [r"hệ\s+thống.*bộ", r"không.*hệ\s+thống.*đại\s+học\s+huế"],
        "chunks": [_chunk("qa_dang_ky_bo_vs_truong_{year}")],
    },
    {
        "patterns": [r"nv\d", r"nguyện\s+vọng\s+\d", r"nv1.*nv2", r"nv2.*nv3", r"đăng\s+ký.*2\s+ngành", r"2\s+ngành.*cùng\s+trường", r"nguyện\s+vọng.*ngành"],
        "chunks": ["qa_NV1_NV2_NV3_quy_che"],
    },
    {
        "patterns": [r"xét\s+tuyển\s+thẳng", r"đăng\s+ký.*xét\s+tuyển\s+thẳng", r"xét\s+tuyển\s+thẳng.*đăng\s+ký"],
        "chunks": [_chunk("qa_xet_tuyen_thang_{year}")],
    },
    {
        "patterns": [r"trực\s+tuyến.*hồ\s+sơ", r"hồ\s+sơ.*gửi.*trường", r"nộp\s+bản\s+giấy",
                     r"đăng\s+ký\s+xét\s+tuyển", r"đăng\s+kí.*xét\s+tuyển",
                     r"phải\s+làm\s+sao", r"tư\s+vấn.*đăng\s+ký", r"tư\s+vấn.*đăng\s+kí",
                     r"hạn\s+cuối", r"khi\s+nào.*đăng\s+ký", r"thời\s+gian.*đăng\s+ký",
                     r"đăng\s+ký.*như\s+thế\s+nào", r"đăng\s+kí.*như\s+thế\s+nào",
                     r"trực\s+tuyến\s+hay", r"online\s+hay\s+giấy", r"online\s+hay\s+offline"],
        "chunks": [_chunk("qa_dang_ky_xet_tuyen_{year}_v2"), _chunk("qa_xet_truc_tuyen_vs_giay_{year}")],
    },
    {
        "patterns": [r"học\s+sinh\s+giỏi.*3\s+năm", r"hsg.*3\s+năm", r"giỏi.*cộng\s+điểm", r"thí\s+sinh\s+tự\s+do", r"tự\s+do.*xét\s+tuyển", r"tự\s+do.*học\s+bạ", r"tự\s+do.*tốt\s+nghiệp"],
        "chunks": [_chunk("chinhsach_{year}"), _chunk("qa_xet_hoc_ba_quy_trinh_{year}")],
    },
    # ── Hotfix Nhóm B: canonical_v2 + thông báo hard-boost ──────────────
    {
        "patterns": [
            r"cách\s+tính\s+điểm\s+học\s+bạ",
            r"điểm\s+học\s+bạ.*tính", r"tính\s+điểm.*học\s+bạ",
            r"học\s+bạ.*lớp\s+mấy", r"học\s+bạ.*lớp\s+1[012]",
            r"điều\s+kiện.*học\s+bạ", r"học\s+bạ.*điểm.*bao\s+nhiêu",
            r"xét\s+học\s+bạ.*điểm", r"điểm.*xét\s+học\s+bạ",
            r"xét\s+hb", r"học\s+bạ.*ngưỡng", r"ngưỡng.*học\s+bạ",
        ],
        "chunks": [_chunk("qa_xet_hoc_ba_{year}_v2")],
    },
    {
        "patterns": [r"xét\s+tuyển\s+thẳng", r"xtt", r"ưu\s+tiên\s+xét\s+tuyển"],
        "chunks": [_chunk("husc_thongbao_id73_{year}"), _chunk("qa_xet_tuyen_thang_{year}")],
    },
    {
        "patterns": [r"thí\s+sinh\s+tự\s+do", r"tự\s+do.*xét\s+tuyển", r"tự\s+do.*đăng\s+ký",
                     r"tốt\s+nghiệp.*năm.*trước", r"tự\s+do.*tài\s+khoản",
                     r"tự\s+do.*không.*thi\s+lại", r"tự\s+do.*thpt"],
        "chunks": [_chunk("husc_thongbao_id71_{year}"), _chunk("qa_xet_hoc_ba_quy_trinh_{year}")],
    },
    {
        "patterns": [r"học\s+bổng", r"chính\s+sách\s+học\s+bổng"],
        "chunks": [_chunk("husc_thongbao_id67_{year}")],
    },
    {
        "patterns": [r"điểm\s+cộng", r"điểm\s+thưởng"],
        "chunks": [_chunk("husc_thongbao_id66_{year}")],
    },
    {
        "patterns": [r"chứng\s+chỉ\s+ngoại\s+ngữ", r"ielts", r"toefl", r"toeic",
                     r"quy\s+đổi.*ngoại\s+ngữ", r"quy\s+đổi.*tiếng\s+anh"],
        "chunks": [_chunk("husc_thongbao_id65_{year}")],
    },
    {
        "patterns": [r"chính\s+sách\s+ưu\s+tiên", r"đối\s+tượng\s+ưu\s+tiên", r"khu\s+vực\s+ưu\s+tiên"],
        "chunks": [_chunk("husc_thongbao_id68_{year}")],
    },
    {
        "patterns": [r"vẽ\s+mỹ\s+thuật", r"đgnl.*vẽ", r"môn\s+vẽ", r"năng\s+khiếu\s+vẽ", r"kiến\s+trúc.*vẽ"],
        "chunks": [_chunk("husc_thongbao_id69_{year}"), _chunk("husc_thongbao_id70_{year}")],
    },
    {
        "patterns": [r"ktx", r"ký\s+túc\s+xá", r"chỗ\s+ở.*sinh\s+viên", r"đăng\s+ký.*ktx"],
        "chunks": [_chunk("husc_thongbao_id53_{year}"), "qa_ktx_husc"],
    },
    {
        "patterns": [r"minh\s+chứng", r"duyệt.*minh\s+chứng", r"điều\s+chỉnh\s+thời\s+gian"],
        "chunks": [_chunk("husc_thongbao_id50_{year}")],
    },
    {
        "patterns": [r"điểm\s+sàn", r"ngưỡng\s+đầu\s+vào", r"ngưỡng\s+điểm"],
        "chunks": [_chunk("husc_thongbao_id49_{year}"), "diem_chuan_2025_full"],
    },
    {
        "patterns": [r"chỉ\s+tiêu", r"số\s+lượng\s+tuyển", r"khối\s+nào", r"tuyển.*khối"],
        "chunks": [_chunk("liet_ke_28_nganh_{year}_v2"), _chunk("bang_xep_hang_chi_tieu_{year}_v2"), _chunk("tuyensinh_overview_{year}_v2")],
    },
    {
        "patterns": [r"khối\s+nào", r"tuyển.*khối", r"những\s+khối", r"các\s+khối", r"khối\s+xét\s+tuyển", r"khối\s+thi", r"trường.*những\s+khối"],
        "chunks": [_chunk("liet_ke_28_nganh_{year}_v2"), _chunk("tuyensinh_overview_{year}_v2"), _chunk("bang_xep_hang_chi_tieu_{year}_v2")],
    },
    {
        "patterns": [r"điểm\s+chuẩn.*3\s+năm", r"điểm\s+chuẩn.*gần\s+đây", r"điểm\s+chuẩn.*biến\s+động",
                     r"điểm\s+chuẩn.*năm\s+trước", r"điểm\s+chuẩn.*năm\s+ngoái"],
        "chunks": ["qa_diem_chuan_3nam_truc_giac", _chunk("qa_diem_chuan_{year}_pending_thpt"), _chunk("husc_thongbao_id49_{year}")],
    },
    {
        "patterns": [r"kiến\s+trúc.*tổ\s+hợp", r"kiến\s+trúc.*xét", r"kiến\s+trúc.*ngành",
                     r"kiến\s+trúc.*vẽ", r"vẽ.*xa.*thi", r"môn\s+vẽ.*ở\s+đâu", r"chuyển\s+điểm.*vẽ"],
        "chunks": ["qa_kien_truc_to_hop_ve_thi_xa", _chunk("husc_thongbao_id69_{year}"), _chunk("husc_thongbao_id70_{year}"), _chunk("nganh_detail_7580101_{year}")],
    },
    {
        "patterns": [r"cntt.*việt[- ]nhật", r"việt[- ]nhật.*cntt", r"so\s+sánh.*cntt", r"khác\s+gì\s+nhau.*cntt",
                     r"cntt.*cnnt", r"cnnt.*cntt"],
        "chunks": [_chunk("nganh_detail_7480201_{year}"), _chunk("nganh_detail_7480201VJ_{year}")],
    },
    {
        "patterns": [r"học\s+sinh\s+giỏi.*cộng\s+điểm", r"hsg.*cộng", r"giỏi.*3\s+năm.*điểm"],
        "chunks": [_chunk("husc_thongbao_id66_{year}"), _chunk("husc_thongbao_id68_{year}")],
    },
    {
        "patterns": [r"song\s+ngành", r"học.*2\s+trường", r"liên\s+trường"],
        "chunks": [_chunk("husc_thongbao_id72_{year}")],
    },
    {
        "patterns": [r"zalo", r"group.*tư\s+vấn", r"liên\s+hệ.*tư\s+vấn"],
        "chunks": ["husc_info"],
    },
    {
        "patterns": [r"học\s+phí.*chia", r"học\s+phí.*nhiều\s+lần", r"học\s+phí.*đóng"],
        "chunks": [_chunk("hocphi_{year}")],
    },
    {
        "patterns": [r"nv1.*husc", r"bắt\s+buộc.*nv1", r"nguyện\s+vọng\s+1.*trường"],
        "chunks": ["qa_NV1_NV2_NV3_quy_che"],
    },
    {
        "patterns": [r"khó\s+khăn.*hỗ\s+trợ", r"hỗ\s+trợ.*sinh\s+viên", r"chính\s+sách.*sinh\s+viên"],
        "chunks": [_chunk("husc_thongbao_id67_{year}")],
    },
    {
        "patterns": [r"tự\s+do.*đăng\s+ký.*bộ", r"tự\s+do.*hệ\s+thống.*bộ", r"tự\s+do.*thpt.*bộ"],
        "chunks": [_chunk("husc_thongbao_id71_{year}"), _chunk("husc_thongbao_id72_{year}")],
    },
    {
        "patterns": [r"văn\s+học.*việc\s+làm", r"văn\s+học.*nghề\s+nghiệp", r"văn\s+học.*sư\s+phạm"],
        "chunks": [_chunk("nganh_detail_7229030_{year}")],
    },
    {
        "patterns": [r"cổng.*xét\s+tuyển", r"cổng.*riêng", r"cổng.*trường", r"xét\s+tuyển.*riêng.*trường", r"hệ\s+thống.*trường"],
        "chunks": ["qa_link_cong_xet_tuyen_husc"],
    },
    {
        "patterns": [r"điểm\s+cộng.*như\s+thế\s+nào", r"điểm\s+ưu\s+tiên.*tính", r"điểm\s+thưởng", r"ưu\s+tiên.*điểm", r"cộng\s+điểm.*như\s+thế"],
        "chunks": [_chunk("chinhsach_{year}"), _chunk("qa_xet_hoc_ba_quy_trinh_{year}")],
    },
    {
        "patterns": [r"đăng\s+ký.*nguyện\s+vọng", r"nguyện\s+vọng.*đăng\s+ký", r"1\s+trường.*2\s+ngành", r"xét.*tổ\s+hợp.*khác",
                     r"đăng\s+ký.*nhiều.*phương\s+thức", r"1\s+ngành.*nhiều.*phương\s+thức", r"nhiều.*phương\s+thức.*xét",
                     r"phương\s+thức.*khác\s+nhau"],
        "chunks": ["qa_NV1_NV2_NV3_quy_che", _chunk("phuongthuc_xettuyen_{year}_v2")],
    },
    {
        "patterns": [r"thí\s+sinh\s+tự\s+do.*chuẩn\s+bị", r"tự\s+do.*cần.*chuẩn\s+bị", r"tự\s+do.*xét\s+tuyển.*bằng", r"tự\s+do.*phương\s+thức"],
        "chunks": [_chunk("qa_xet_hoc_ba_quy_trinh_{year}"), _chunk("chinhsach_{year}"), _chunk("phuongthuc_xettuyen_{year}")],
    },
    # ── Khôn khéo year-mismatch fallback (canonical_v3) ──────────────────
    {
        "patterns": [_pat(r"điểm\s+chuẩn.*\b{year}\b"), r"điểm\s+chuẩn.*husc.*bao\s+nhiêu", r"điểm\s+chuẩn.*trường",
                     r"điểm\s+chuẩn.*cao\s+nhất", r"điểm\s+chuẩn.*thấp\s+nhất"],
        "chunks": [_chunk("qa_diem_chuan_{year}_pending_thpt"), "qa_diem_chuan_3nam_truc_giac", _chunk("husc_thongbao_id49_{year}")],
    },
    {
        "patterns": [r"trường.*tuyển.*khối\s+nào", r"husc.*tuyển.*khối", r"các\s+khối.*husc", r"tổ\s+hợp.*husc",
                     r"khối\s+nào.*xét\s+tuyển", r"các\s+tổ\s+hợp.*xét\s+tuyển"],
        "chunks": [_chunk("qa_chi_tieu_khoi_xet_{year}_overview"), _chunk("husc_thongbao_id40_{year}")],
    },
    {
        "patterns": [r"khi\s+nào.*có\s+thông\s+tin", r"đợi.*thông\s+tin\s+mới", r"đợi.*chính\s+thức",
                     r"theo\s+dõi.*tuyển\s+sinh", r"cập\s+nhật.*tuyển\s+sinh", r"ở\s+đâu.*đợi.*thông\s+tin",
                     r"web.*tuyển\s+sinh", r"trang.*tuyển\s+sinh", r"website.*tuyển\s+sinh"],
        "chunks": [_chunk("qa_de_an_{year}_xem_o_dau"), "qa_thong_tin_chua_co_template_tong_quat"],
    },
    {
        "patterns": [_pat(r"lịch.*tuyển\s+sinh.*\b{year}\b"), r"khi\s+nào.*đăng\s+ký", r"hạn.*đăng\s+ký", r"hạn\s+cuối",
                     r"thời\s+gian.*xét\s+tuyển", r"đợt.*xét\s+tuyển"],
        "chunks": [_chunk("qa_lich_tuyensinh_{year}_pending"), _chunk("qa_de_an_{year}_xem_o_dau")],
    },
]


def detect_aggregation_chunks(query: str) -> List[str]:
    """Return list of priority chunk_ids that should be injected for this query."""
    query_lower = query.lower()
    chunks_to_inject: List[str] = []
    seen = set()
    for rule in _AGGREGATION_RULES:
        for pattern in rule["patterns"]:
            if re.search(pattern, query_lower):
                for cid in rule["chunks"]:
                    if cid not in seen:
                        seen.add(cid)
                        chunks_to_inject.append(cid)
                break
    return chunks_to_inject


def boost_with_aggregation(
    query: str,
    baseline_docs: List[RetrievedDocument],
    lancedb_retriever,
    top_k: int = 5,
    max_inject: int = 2,
) -> List[RetrievedDocument]:
    """Inject aggregation chunks ahead of baseline if query matches.

    Args:
        query: original user query
        baseline_docs: baseline retrieved docs (post-rerank)
        lancedb_retriever: LanceDBRetriever instance to fetch by chunk_id
        top_k: total final docs (auto-expanded by `max_inject` when injection happens)
        max_inject: hard cap on injected aggregation chunks per query

    Returns:
        Final docs with aggregation chunks prepended (deduped)

    Observability:
        On every `fetch_by_id` that returns `None`, the booster:
          - logs a WARNING with the chunk-id and query (G1-T3: visible, not debug)
          - increments the `BOOSTER_CHUNK_MISS` in-process counter
    """
    inject_ids = detect_aggregation_chunks(query)
    if not inject_ids:
        return baseline_docs[:top_k]

    seen_ids = {d.chunk_id for d in baseline_docs if d.chunk_id}
    injected_docs: List[RetrievedDocument] = []
    for cid in inject_ids:
        if len(injected_docs) >= max_inject:
            break
        if cid in seen_ids:
            continue  # already in baseline
        try:
            doc = lancedb_retriever.fetch_by_id(cid)
            if doc:
                # S15.5 / CF-2: tag injected docs so the precision-cutoff
                # (separate module, Lane A) can EXEMPT them from the
                # relative rerank-score drop. RetrievedDocument is a
                # frozen dataclass, so we reconstruct with merged
                # metadata and keep the score from `fetch_by_id`
                # (booster docs retain their injected priority by design;
                # only the GraphRAG expander (S15.1) zero-bases scores).
                tagged = replace(
                    doc,
                    metadata={**doc.metadata, "booster_injected": True},
                )
                injected_docs.append(tagged)
                seen_ids.add(cid)
            else:
                # G1-T3: visible failure. Year-rollover (e.g. 2026→2027)
                # produces a None here; the previous `logger.debug` was
                # invisible to operators. Emit WARN + bump counter.
                BOOSTER_CHUNK_MISS.inc()
                logger.warning(
                    f"booster_chunk_miss: fetch_by_id({cid!r}) returned None "
                    f"for query={query!r} (likely year-mismatch: chunk-id "
                    f"from year={_YEAR} but data not yet ingested)"
                )
        except Exception as e:
            BOOSTER_CHUNK_MISS.inc()
            logger.warning(
                f"booster_chunk_miss: fetch_by_id({cid!r}) raised "
                f"{type(e).__name__}: {e} for query={query!r}"
            )

    if not injected_docs:
        return baseline_docs[:top_k]

    logger.info(
        f"Aggregation booster: injected {len(injected_docs)} chunks "
        f"({[d.chunk_id for d in injected_docs]})"
    )

    # Expand top_k to keep room for original baseline relevance
    effective_top_k = top_k + len(injected_docs)
    final = injected_docs + [
        d for d in baseline_docs if d.chunk_id not in {x.chunk_id for x in injected_docs}
    ]
    return final[:effective_top_k]
