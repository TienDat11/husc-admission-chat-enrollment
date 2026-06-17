"""Aggregation rules — pure data used by `aggregation_booster`.

This module contains ONLY the `_AGGREGATION_RULES` data structure and the
small year-templating helpers needed to build it at import time. All
retrieval/injection logic remains in `aggregation_booster`.

The public constant `_AGGREGATION_RULES` is re-exported by
`aggregation_booster` so any existing imports of the form
`from services.aggregation_booster import _AGGREGATION_RULES` continue to work.
"""
from __future__ import annotations

from src.services.temporal_authority import get_current_admission_year


# ─── G1-T2: derive the current year at module init ─────────────────────────
# Snapshot the year ONCE at module import. If the env is patched later
# (e.g. by tests), the rule set's at-import snapshot still drives the
# chunk-id suffixes. Tests that need a different year should reload the
# module (importlib.reload) AFTER patching the env.


def _current_admission_year() -> int:
    """Return the current admission year from the temporal authority."""
    return int(get_current_admission_year())


_YEAR: int = _current_admission_year()


# Historical year constant — kept as an EXPLICIT literal because the rule
# set intentionally references 2025 admissions data.
_HISTORICAL_YEAR_1: int = 2025

# Year-agnostic suffix for queries that don't name a year.
_YEAR_AGNOSTIC_SUFFIX = f"{_YEAR}"


# ─── Rule builder helpers ──────────────────────────────────────────────────

def _chunk(template: str) -> str:
    """Build a current-year chunk-id by templating `{year}` into a template."""
    return template.format(year=_YEAR)


def _pat(template: str) -> str:
    """Build a current-year regex pattern by templating `{year}` into a
    raw-string template. Mirrors the `_chunk` helper for patterns."""
    return template.format(year=_YEAR)


# Map query pattern → list of priority chunk_ids to inject.
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
            _pat(r"học\s+phí.*\b{year}\b.*\b2025\b"),
            _pat(r"học\s+phí.*\b2025\b.*\b{year}\b"),
        ],
        "chunks": [
            "so_sanh_hocphi_2025_vs_2026",
            _chunk("hocphi_{year}"),
            "hocphi_2025",
        ],
    },
    {
        "patterns": [r"học\s+phí.*2025", r"học\s+phí.*năm\s+2025"],
        "chunks": ["hocphi_2025"],
    },
    {
        "patterns": [
            _pat(r"học\s+phí.*\b{year}\b"),
            _pat(r"học\s+phí.*năm\s+{year}"),
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

__all__ = [
    "_AGGREGATION_RULES",
    "_chunk",
    "_pat",
    "_YEAR",
    "_HISTORICAL_YEAR_1",
    "_YEAR_AGNOSTIC_SUFFIX",
]
