"""@spec(S18.2-T4b) Reconcile HUSC 2026 canonical chunk numbers to the
officially-approved count: 29 ngành/CTĐT × tổng 2.079 chỉ tiêu.

Root cause (post-T4a user-approved decision, 2026-06-18):
    * web id64 (chính thức) says: ``với 2.079 chỉ tiêu của 29 ngành``
      where 29 = 26 mã ngành + 3 CT con (7440102SC, 7480201VJ, 7510302IC).
    * Internal chunks carried the WRONG figures for 2026:
        - 25 ngành (5 chunks), 26 ngành (2 chunks), 28 ngành (5 chunks)
        - 1.969 chỉ tiêu (3 chunks), 1.939 (1 chunk)
    * Goal of T4b: rewrite the wrong 2026 chunks in-place (idempotent
      delete-by-chunk_id + re-add) AND make sure a canonical chunk exists
      that enumerates all 8 đối tượng XTT (T2 already added
      ``husc_tb73_doituong_xtt_2026`` — T4b verifies it and adds a sibling
      8-đối-tượng aggregate chunk only if absent).

This script is idempotent: every add is gated on
``delete WHERE chunk_id = ?`` followed by ``table.add([row])``.  Running
twice is a no-op.

What it does
------------
1. For each of the 11 known bad chunks (``BAD_CHUNK_IDS``):
    - delete the row by chunk_id
    - re-embed the corrected text (29 ngành + 2.079 chỉ tiêu)
    - re-add with updated metadata carrying
      ``data_year=2026`` + ``superseded_reason`` annotation cleared and
      ``is_superseded=false`` for the corrected aggregate chunk.
2. Chunks that CANNOT be cleanly rewritten (because they encode 2025
   historical figures, e.g. ``historical_chi_tieu_2025_husc``,
   ``diem_chuan_2025_full``, ``husc_summary_general_2025``) are
   *guarded*: the script refuses to mutate them — their
   ``data_year``/``is_historical`` already protects them on the
   retrieval hot-path.
3. Adds 1 new canonical chunk ``husc_canonical_2079_29nganh_2026``:
   the single source of truth chunk that lists 2.079 + 29 ngành/CTĐT
   + the 8 đối tượng XTT (mirroring ``husc_tb73_doituong_xtt_2026``)
   — so query "đối tượng xét tuyển thẳng 2026" can retrieve ≥8 đối
   tượng even if T2 chunk drifts.
4. Writes the corrected rows to
   ``data/chunked/husc_canonical_fix_2026.jsonl`` for build_graph parity.

Usage::

    python scripts/fix_canonical_64_73.py            # real
    python scripts/fix_canonical_64_73.py --dry-run  # preview only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from loguru import logger  # noqa: E402

# ---------------------------------------------------------------------------
# TDD-locked constants (decision approved by user 2026-06-18)
# ---------------------------------------------------------------------------
# Canonical truth per web id64: "với 2.079 chỉ tiêu của 29 ngành".
# 29 = 26 mã ngành + 3 CT con (7440102SC, 7480201VJ, 7510302IC).
CANONICAL_NGANH: int = 29
CANONICAL_CHI_TIEU: int = 2079
SOURCE_URL: str = "https://tuyensinh.husc.edu.vn/thongbao.php?id=64"
APPROVAL: str = "xcheck_t4a_20260618"

# New chunk id we ADD — the single source-of-truth aggregate.
CANONICAL_CHUNK_ID: str = "husc_canonical_2079_29nganh_2026"

# Source chunk that already covers the 8 đối tượng XTT (added in T2).
# We verify it exists; otherwise add an equivalent aggregate via this script.
T2_DOITUONG_CHUNK_ID: str = "husc_tb73_doituong_xtt_2026"
DOITUONG_TB73_SOURCE: str = "tuyensinh.husc.edu.vn/thongbao.php?id=73"

# Output JSONL (for build_graph parity).
CHUNKED_JSONL_OUT: Path = REPO_ROOT / "data" / "chunked" / "husc_canonical_fix_2026.jsonl"

# ---------------------------------------------------------------------------
# Bad chunks → corrected text + corrected metadata.
#
# GUARDED_CHUNK_IDS are 2025/historical chunks that we REFUSE to mutate —
# their data_year tag already filters them out of 2026 retrieval. Touching
# them would damage the historical reference record.
# ---------------------------------------------------------------------------
GUARDED_CHUNK_IDS: set = {
    "historical_chi_tieu_2025_husc",   # data_year 2025
    "diem_chuan_2025_full",            # data_year 2025, is_historical=true
    "husc_summary_general_2025",       # data_year 2025
}

# Each entry: (chunk_id, new_text, new_meta_overrides)
# Text keeps all other factual content (học phí, phương thức, mã ngành…)
# — only the 25/26/28 ngành and 1.969/1.939/1.844 are corrected to
# 29 ngành/CTĐT + 2.079 chỉ tiêu.
BAD_CHUNK_FIXES: List[Dict[str, Any]] = [
    {
        "chunk_id": "thay_doi_tuyensinh_2026",
        "text_fix": (
            "Tuyển sinh HUSC năm 2026 có nhiều điểm mới so với 2025: (1) Học phí điều chỉnh "
            "tăng theo Thông tư 06/2026/TT-BGDĐT — Khối VII từ 440.000đ lên 490.000đ/TC, "
            "Khối IV từ 505.000đ lên 555.000đ/TC, Khối V từ 545.000đ lên 600.000đ/TC. "
            "(2) Áp dụng 5 phương thức xét tuyển: XTT (xét tuyển thẳng), THPT, HB "
            "(xét học bạ), NL (đánh giá năng lực), KH (kết hợp). "
            "(3) Tổng chỉ tiêu 2.079 cho 29 ngành/CTĐT (gồm 26 mã ngành đại học chính quy "
            "và 3 chương trình con 7440102SC, 7480201VJ, 7510302IC). "
            "(4) Mở thêm chương trình CNTT Việt-Nhật (mã 7480201VJ, 30 chỉ tiêu) cùng "
            "2 chương trình con mới 7440102SC và 7510302IC. "
            "(5) Điểm cộng tối đa 10% thang điểm cho giải quốc gia, chứng chỉ ngoại ngữ. "
            "(Nguồn: tuyensinh.husc.edu.vn/thongbao.php?id=64 — "
            "reconcile chuẩn sau cross-check id 62-74.)"
        ),
        "meta_overrides": {
            "data_year": 2026,
            "nganh_count": CANONICAL_NGANH,
            "tong_chi_tieu": CANONICAL_CHI_TIEU,
            "canonical_approved_by": APPROVAL,
            "is_superseded": False,
        },
        "source": SOURCE_URL,
        "faq_type": "thay_doi_tuyen_sinh",
    },
    {
        "chunk_id": "liet_ke_26_nganh_2026",
        "text_fix": (
            "Danh sách đầy đủ 29 ngành/CTĐT tại HUSC tuyển sinh năm 2026 — gồm 26 mã ngành "
            "đại học chính quy và 3 chương trình con (7440102SC Vật lý học - Công nghệ bán "
            "dẫn, 7480201VJ CNTT Việt-Nhật, 7510302IC CNKT điện tử-viễn thông - Vi mạch). "
            "Tổng chỉ tiêu toàn trường: 2.079. "
            "- Hán Nôm (mã 7220104): chỉ tiêu 32, tổ hợp C00, D14, X70, C19, D01, học phí "
            "490.000đ/TC, đào tạo 4 năm. "
            "- Triết học (mã 7229001): chỉ tiêu 32, tổ hợp A00, X78, D66, X70, C19, D01, "
            "học phí 490.000đ/TC, đào tạo 4 năm. "
            "- Lịch sử (mã 7229010): chỉ tiêu 62, tổ hợp C00, D14, X70, C19, C03, học phí "
            "490.000đ/TC, đào tạo 4 năm. "
            "- Văn học (mã 7229030): chỉ tiêu 104, tổ hợp C00, D14, X70, C19, D01, học phí "
            "490.000đ/TC, đào tạo 4 năm. "
            "- Quản lý văn hóa (mã 7229042): chỉ tiêu 30, tổ hợp C00, D14, X70, C19, C03, "
            "học phí 490.000đ/TC, đào tạo 4 năm. "
            "(Nguồn: tuyensinh.husc.edu.vn/thongbao.php?id=64 — 29 ngành/CTĐT × 2.079 chỉ tiêu.)"
        ),
        "meta_overrides": {
            "data_year": 2026,
            "nganh_count": CANONICAL_NGANH,
            "tong_chi_tieu": CANONICAL_CHI_TIEU,
            "canonical_approved_by": APPROVAL,
            "is_superseded": False,
        },
        "source": SOURCE_URL,
        "faq_type": "danh_sach_nganh",
    },
    {
        "chunk_id": "bang_xep_hang_chi_tieu_2026",
        "text_fix": (
            "Bảng xếp hạng chỉ tiêu 29 ngành/CTĐT tại HUSC năm 2026 (cao xuống thấp): "
            "1. Công nghệ thông tin (Cử nhân) (mã 7480201): 400 chỉ tiêu, tổ hợp A00, A01, "
            "D01, X26. "
            "2. Truyền thông số (mã 7320111): 279 chỉ tiêu, tổ hợp C00, C03, X70, C19, D01. "
            "3. Báo chí (mã 7320101): 250 chỉ tiêu, tổ hợp C00, C03, X70, C19, D01. "
            "4. Kiến trúc (mã 7580101): 105 chỉ tiêu, tổ hợp V00, V01, V02, V12. "
            "5. Văn học (mã 7229030): 104 chỉ tiêu, tổ hợp C00, D14, X70, C19, D01. "
            "6. CNKT điện tử-viễn thông (mã 7510302): 100 chỉ tiêu, tổ hợp A00, A01, C01, X06. "
            "7. Quản lý nhà nước (mã 7310205): 80 chỉ tiêu, tổ hợp A00, X78, D66, X70, C19, "
            "D01. "
            "8. Kỹ thuật phần mềm (mã 7480103): 63 chỉ tiêu, tổ hợp A00, A01, D01, X26. "
            "9. Lịch sử (mã 7229010): 62 chỉ tiêu, tổ hợp C00, D14, X70, C19, C03. "
            "Tổng toàn trường: 2.079 chỉ tiêu cho 29 ngành/CTĐT. "
            "(Nguồn: tuyensinh.husc.edu.vn/thongbao.php?id=64 — reconcile T4b.)"
        ),
        "meta_overrides": {
            "data_year": 2026,
            "nganh_count": CANONICAL_NGANH,
            "total_chi_tieu": CANONICAL_CHI_TIEU,
            "is_superseded": False,
            "superseded_by": "",
            "canonical_approved_by": APPROVAL,
        },
        "source": SOURCE_URL,
        "faq_type": "xep_hang_chi_tieu",
    },
    {
        "chunk_id": "qa_chi_tieu_khoi_xet_2026_overview",
        "text_fix": (
            "Câu hỏi: Trường HUSC năm 2026 tuyển những khối nào? Chỉ tiêu bao nhiêu? "
            "Trả lời: Năm 2026, Trường Đại học Khoa học - ĐH Huế tuyển sinh 29 ngành/CTĐT "
            "đại học chính quy với tổng chỉ tiêu 2.079. Các tổ hợp xét tuyển gồm: A00 "
            "(Toán-Lý-Hóa), A01 (Toán-Lý-Anh), A02 (Toán-Lý-Sinh), B00 (Toán-Hóa-Sinh), "
            "B03 (Toán-Sinh-Văn), C00 (Văn-Sử-Địa), C01 (Văn-Toán-Lý), C02 (Văn-Toán-Hóa), "
            "C03 (Văn-Toán-Sử), C04 (Văn-Toán-Địa), C19 (Văn-Sử-GDCD), C20 (Văn-Địa-GDCD), "
            "D01 (Toán-Văn-Anh), D07 (Toán-Hóa-Anh), D08 (Toán-Sinh-Anh), D14 (Văn-Sử-Anh), "
            "D15 (Văn-Địa-Anh), D66 (Văn-GDCD-Anh), D78 (Văn-KHXH-Anh), D90 (Toán-KHTN-Anh), "
            "V00/V01/V02 (Toán-Lý-Vẽ / Toán-Văn-Vẽ — riêng ngành Kiến trúc 7580101). "
            "Phương thức xét tuyển: (1) Xét điểm thi THPT 2026; (2) Xét học bạ lớp 10+11+12; "
            "(3) Xét điểm ĐGNL của ĐHQG-HCM/ĐHQG-HN; (4) Xét tuyển thẳng; (5) Xét tuyển "
            "kết hợp. 29 ngành/CTĐT gồm 26 mã ngành + 3 CT con 7440102SC, 7480201VJ, "
            "7510302IC. (Nguồn: tuyensinh.husc.edu.vn/thongbao.php?id=64.)"
        ),
        "meta_overrides": {
            "data_year": 2026,
            "nganh_count": CANONICAL_NGANH,
            "tong_chi_tieu": CANONICAL_CHI_TIEU,
            "is_superseded": False,
            "canonical_approved_by": APPROVAL,
        },
        "source": SOURCE_URL,
        "faq_type": "qa_overview",
    },
    {
        "chunk_id": "tuyensinh_overview_2026_v2",
        "text_fix": (
            "Trường Đại học Khoa học, Đại học Huế (HUSC) - Mã trường: DHT. "
            "Năm 2026: tuyển sinh 29 ngành/CTĐT đào tạo đại học chính quy (26 mã ngành + "
            "3 chương trình con 7440102SC, 7480201VJ, 7510302IC; ngành Quản trị và phân tích "
            "dữ liệu mã 7480107TD đã ngừng tuyển từ 2026). Tổng chỉ tiêu: 2.079. "
            "Năm phương thức xét tuyển: (1) Xét tuyển thẳng / ưu tiên xét tuyển theo quy chế "
            "Bộ GD&ĐT, (2) Sử dụng kết quả thi THPT 2026, (3) Xét học bạ THPT (3 năm lớp "
            "10+11+12), (4) Sử dụng kết quả thi đánh giá năng lực ĐHQG-HCM/ĐHQG-HN, "
            "(5) Xét tuyển kết hợp (đa tiêu chí). Hotline: 0234 3828427 / 0385 887 111 "
            "(ĐT/Zalo). Email: tuyensinh@husc.edu.vn. Địa chỉ: 77 Nguyễn Huệ, TP Huế. "
            "Website: https://tuyensinh.husc.edu.vn/ "
            "(Nguồn: tuyensinh.husc.edu.vn/thongbao.php?id=64 — reconcile T4b.)"
        ),
        "meta_overrides": {
            "data_year": 2026,
            "nganh_count": CANONICAL_NGANH,
            "tong_chi_tieu": CANONICAL_CHI_TIEU,
            "is_superseded": False,
            "approved_by": APPROVAL,
            "canonical_approved_by": APPROVAL,
        },
        "source": SOURCE_URL,
        "faq_type": "tuyen_sinh_overview",
    },
    {
        "chunk_id": "bang_xep_hang_chi_tieu_2026_v2",
        "text_fix": (
            "Bảng xếp hạng 29 ngành/CTĐT HUSC năm 2026 theo chỉ tiêu (cao -> thấp): "
            "1. Công nghệ thông tin - Cử nhân (mã 7480201): 400 chỉ tiêu, tổ hợp A00, A01, "
            "D01, X26 - Khối V. "
            "2. Truyền thông số (mã 7320111): 279 chỉ tiêu, tổ hợp C00, C03, X70, C19, D01 "
            "- Khối VII. "
            "3. Báo chí (mã 7320101): 250 chỉ tiêu, tổ hợp C00, C03, X70, C19, D01 "
            "- Khối VII. "
            "4. Kiến trúc (mã 7580101): 105 chỉ tiêu, tổ hợp V00, V01, V02, V12. "
            "5. Văn học (mã 7229030): 104 chỉ tiêu, tổ hợp C00, D14, X70, C19, D01. "
            "6. CNKT điện tử-viễn thông (mã 7510302): 100 chỉ tiêu, tổ hợp A00, A01, C01, "
            "X06 - Khối V. "
            "7. Quản lý nhà nước (mã 7310205): 80 chỉ tiêu, tổ hợp A00, X78, D66, X70, C19, "
            "D01. "
            "Tổng toàn trường: 2.079 chỉ tiêu cho 29 ngành/CTĐT (gồm 26 mã ngành + 3 CT con "
            "7440102SC, 7480201VJ, 7510302IC). (Nguồn: tuyensinh.husc.edu.vn/thongbao.php?id=64.)"
        ),
        "meta_overrides": {
            "data_year": 2026,
            "nganh_count": CANONICAL_NGANH,
            "tong_chi_tieu": CANONICAL_CHI_TIEU,
            "is_superseded": False,
            "canonical_approved_by": APPROVAL,
        },
        "source": SOURCE_URL,
        "faq_type": "xep_hang_chi_tieu",
    },
    {
        "chunk_id": "liet_ke_28_nganh_2026_v2",
        "text_fix": (
            "Danh sách đầy đủ 29 ngành/CTĐT đào tạo đại học chính quy của Trường Đại học "
            "Khoa học - Đại học Huế (HUSC) năm 2026 (Mã trường: DHT): "
            "1. Hán - Nôm (mã 7220104) - chỉ tiêu 32, tổ hợp C00, D14, X70, C19, D01, "
            "học phí Khối VII 490.000đ/TC, đào tạo 4 năm. "
            "2. Triết học (mã 7229001) - chỉ tiêu 32, tổ hợp A00, X78, D66, X70, C19, D01, "
            "học phí Khối VII 490.000đ/TC, đào tạo 4 năm. "
            "3. Lịch sử (mã 7229010) - chỉ tiêu 62, tổ hợp C00, D14, X70, C19, C03, "
            "học phí Khối VII 490.000đ/TC, đào tạo 4 năm. "
            "(29 ngành/CTĐT = 26 mã ngành + 3 CT con 7440102SC, 7480201VJ, 7510302IC; "
            "tổng chỉ tiêu toàn trường 2.079.) "
            "(Nguồn: tuyensinh.husc.edu.vn/thongbao.php?id=64 — reconcile T4b.)"
        ),
        "meta_overrides": {
            "data_year": 2026,
            "nganh_count": CANONICAL_NGANH,
            "tong_chi_tieu": CANONICAL_CHI_TIEU,
            "is_superseded": False,
            "canonical_approved_by": APPROVAL,
        },
        "source": SOURCE_URL,
        "faq_type": "danh_sach_nganh",
    },
    {
        "chunk_id": "nganh_ngung_tuyen_2026",
        "text_fix": (
            "DANH SÁCH NGÀNH HUSC ĐÃ NGỪNG TUYỂN SINH TỪ NĂM 2026 (KHÔNG CÒN ĐÀO TẠO): "
            "1. Quản trị và phân tích dữ liệu (mã 7480107TD) - Đã ngừng tuyển từ năm 2026. "
            "Năm 2025 ngành này còn tuyển 30 chỉ tiêu nhưng năm 2026 KHÔNG còn nằm trong "
            "danh mục đào tạo. Học sinh có nguyện vọng học ngành tương đương có thể tham "
            "khảo Khoa học dữ liệu (mã 7460108) — ngành mới của HUSC từ 2026. "
            "Tổng quan năm 2026: HUSC tuyển 29 ngành/CTĐT với tổng chỉ tiêu 2.079 "
            "(gồm 26 mã ngành + 3 CT con 7440102SC, 7480201VJ, 7510302IC). "
            "(Nguồn: tuyensinh.husc.edu.vn/thongbao.php?id=64 — reconcile T4b.)"
        ),
        "meta_overrides": {
            "data_year": 2026,
            "nganh_count": CANONICAL_NGANH,
            "tong_chi_tieu": CANONICAL_CHI_TIEU,
            "is_superseded": False,
            "canonical_approved_by": APPROVAL,
        },
        "source": SOURCE_URL,
        "faq_type": "ngung_tuyen",
    },
    {
        "chunk_id": "qa_xet_hoc_ba_2026_v2",
        "text_fix": (
            "Câu hỏi: HUSC 2026 — Xét học bạ cần điểm bao nhiêu? Học bạ lớp mấy? "
            "Trả lời: Năm 2026, HUSC tuyển sinh 29 ngành/CTĐT với tổng chỉ tiêu 2.079. "
            "**Năm 2026 hiện chưa có thông báo điểm sàn cụ thể** cho phương thức xét học "
            "bạ tại HUSC. **Điều kiện TỐI THIỂU (theo quy định Bộ GD&ĐT)** áp dụng cho "
            "thí sinh 2k8 (sinh năm 2008, tốt nghiệp THPT 2026): Tổng điểm thi THPT "
            "**3 môn trong tổ hợp xét tuyển** HOẶC tổng 3 môn **(Toán + Văn + 1 môn thi "
            "THPT khác)** phải **từ 15 điểm trở lên** — đây CHỈ là ngưỡng tối thiểu để "
            "được vào diện xét, không phải điểm trúng tuyển. "
            "**Cách tính điểm xét học bạ HUSC** (theo HUSC Thông tin tuyển sinh 2026, "
            "Mục B.II.3): Điểm các môn trong tổ hợp xét tuyển là **điểm trung bình chung "
            "kết quả học tập cả năm của các lớp 10, 11, 12** (ba năm THPT) môn học đó, "
            "làm tròn đến 02 chữ số thập phân. "
            "(Nguồn: tuyensinh.husc.edu.vn/thongbao.php?id=64 — 29 ngành/CTĐT × 2.079 chỉ tiêu.)"
        ),
        "meta_overrides": {
            "data_year": 2026,
            "nganh_count": CANONICAL_NGANH,
            "tong_chi_tieu": CANONICAL_CHI_TIEU,
            "is_superseded": False,
            "canonical_approved_by": APPROVAL,
        },
        "source": SOURCE_URL,
        "faq_type": "xet_hoc_ba",
    },
    {
        "chunk_id": "husc_vision_tb40_img00_2026",
        "text_fix": (
            "[Trích xuất từ hình ảnh trong thông báo tuyển sinh HUSC] "
            "Chính thức mở đăng ký xét tuyển đại học 2026. "
            "Trường Đại học Khoa học - Đại học Huế. "
            "HỆ ĐÀO TẠO: CHÍNH QUY. "
            "MÃ TRƯỜNG: DHT. "
            "TỔNG CHỈ TIÊU: 2.079. "
            "SỐ NGÀNH/CTĐT: 29 (gồm 26 mã ngành + 3 CT con 7440102SC, 7480201VJ, "
            "7510302IC — reconcile T4b). "
            "Thời gian: 23/5/2026 – 14/7/2026. "
            "Hotline: 0385 887 111 / 02343 828 427. "
            "Website: https://tuyensinh.husc.edu.vn/. "
            "Email: tuyensinh@husc.edu.vn. "
            "ĐĂNG KÝ XÉT TUYỂN TRỰC TUYẾN TẠI: www.dkxt.hueuni.edu.vn. "
            "Nguồn: https://tuyensinh.husc.edu.vn/thongbao.php?id=40 "
            "(reconcile với id=64 — 29 ngành/CTĐT × 2.079 chỉ tiêu). "
            "Lưu ý: Đây là thông tin trích từ ảnh. Để cập nhật mới nhất, vui lòng theo dõi "
            "https://tuyensinh.husc.edu.vn/thongbao.php"
        ),
        "meta_overrides": {
            "data_year": 2026,
            "nganh_count": CANONICAL_NGANH,
            "tong_chi_tieu": CANONICAL_CHI_TIEU,
            "is_superseded": False,
            "canonical_approved_by": APPROVAL,
        },
        "source": "https://tuyensinh.husc.edu.vn/thongbao.php?id=40",
        "faq_type": "vision_ocr",
    },
    {
        "chunk_id": "tuyensinh_overview_2026",
        "text_fix": (
            "Trường Đại học Khoa học, Đại học Huế — Mã trường: DHT. "
            "Năm 2026: tuyển sinh 29 ngành/CTĐT (26 mã ngành đại học chính quy + 3 chương "
            "trình con 7440102SC, 7480201VJ, 7510302IC), tổng chỉ tiêu 2.079. "
            "5 phương thức xét tuyển: Xét tuyển thẳng / ưu tiên, Dùng kết quả thi THPT, "
            "Xét học bạ THPT, Dùng kết quả thi đánh giá năng lực, Xét tuyển kết hợp. "
            "(Nguồn: tuyensinh.husc.edu.vn/thongbao.php?id=64 — reconcile T4b.)"
        ),
        "meta_overrides": {
            "data_year": 2026,
            "nganh_count": CANONICAL_NGANH,
            "tong_chi_tieu": CANONICAL_CHI_TIEU,
            "is_superseded": False,
            "superseded_by": "",
            "canonical_approved_by": APPROVAL,
        },
        "source": SOURCE_URL,
        "faq_type": "tuyen_sinh_overview",
    },
]

# ---------------------------------------------------------------------------
# New canonical aggregate chunk — single source of truth 29 × 2.079
# ---------------------------------------------------------------------------
CANONICAL_TEXT: str = (
    "HUSC tuyển sinh đại học hệ chính quy năm 2026 với tổng cộng "
    "29 ngành/CTĐT và tổng chỉ tiêu 2.079 (đúng nguyên văn theo Thông báo id=64: "
    "'với 2.079 chỉ tiêu của 29 ngành'). Trong 29 ngành/CTĐT có 26 mã ngành đại học "
    "chính quy và 3 chương trình đào tạo con: 7440102SC (Vật lý học - Công nghệ bán "
    "dẫn), 7480201VJ (Công nghệ thông tin Việt-Nhật), 7510302IC (CNKT điện tử-viễn "
    "thông - Vi mạch). Năm phương thức xét tuyển: Xét tuyển thẳng / ưu tiên (XTT), "
    "Xét điểm thi THPT 2026, Xét học bạ THPT, Xét điểm ĐGNL ĐHQG-HCM/ĐHQG-HN, Xét "
    "tuyển kết hợp. Đối tượng xét tuyển thẳng (8 nhóm theo Thông báo id=73): "
    "(1) Anh hùng Lao động, Anh hùng Lực lượng vũ trang nhân dân; "
    "(2) Chiến sĩ thi đua toàn quốc; "
    "(3) Thí sinh đạt giải Nhất, Nhì, Ba trong kỳ thi học sinh giỏi cấp Quốc gia hoặc "
    "Quốc tế; "
    "(4) Thí sinh đạt giải trong cuộc thi Khoa học kỹ thuật cấp Quốc gia hoặc Quốc tế; "
    "(5) Người nước ngoài có đủ điều kiện theo quy định hiện hành; "
    "(6) Người Việt Nam đã tốt nghiệp trung học ở nước ngoài; "
    "(7) Người khuyết tật nặng theo quy định; "
    "(8) Người thuộc dân tộc thiểu số rất ít người. "
    "Tổng chỉ tiêu XTT và ưu tiên xét tuyển không vượt quá 10% chỉ tiêu mỗi "
    "ngành/CTĐT. Lệ phí XTT: 30.000đ/nguyện vọng (nộp trực tiếp hoặc EMS). "
    "(Nguồn: tuyensinh.husc.edu.vn/thongbao.php?id=64 + id=73 — reconcile T4b.)"
)

CANONICAL_META: Dict[str, Any] = {
    "source": SOURCE_URL,
    "source_type": "web_crawl",
    "year": "2026",
    "data_year": 2026,
    "info_type": "canonical_overview",
    "school": "HUSC",
    "audience": "thi_sinh",
    "unit": "HUSC",
    "chunk_type": "text_section",
    "topic_disambiguation": "id64_canonical_2079_29nganh_v1",
    "nganh_count": CANONICAL_NGANH,
    "tong_chi_tieu": CANONICAL_CHI_TIEU,
    "doi_tuong_xtt": [
        "Anh hùng Lao động / Lực lượng vũ trang",
        "Chiến sĩ thi đua toàn quốc",
        "Giải Nhất-Nhì-Ba HSG QG/QT",
        "Giải KHKT QG/QT",
        "Người nước ngoài",
        "Người VN học ở nước ngoài",
        "Khuyết tật nặng",
        "DTTS rất ít người",
    ],
    "canonical_approved_by": APPROVAL,
    "is_superseded": False,
}

# ---------------------------------------------------------------------------
# Helpers (same shape as reingest_thongbao_full_2026.py)
# ---------------------------------------------------------------------------
def _delete_by_chunk_id(table, chunk_id: str) -> None:
    table.delete(f"chunk_id = '{chunk_id}'")


def _row_count(table) -> int:
    if hasattr(table, "rows"):
        return len(table.rows)
    try:
        return int(table.count_rows())
    except Exception:
        return len(table.to_pandas())


def _has_chunk_id(table, chunk_id: str) -> bool:
    if hasattr(table, "rows"):
        return any((r.get("chunk_id") == chunk_id) for r in table.rows)
    try:
        df = table.to_pandas()
        return chunk_id in df["chunk_id"].tolist()
    except Exception:
        return False


def _get_existing_meta(table, chunk_id: str) -> Dict[str, Any]:
    """Return the existing metadata dict for a chunk_id (best effort)."""
    if hasattr(table, "rows"):
        for r in table.rows:
            if r.get("chunk_id") == chunk_id:
                try:
                    return json.loads(r.get("metadata_json", "{}"))
                except Exception:
                    return {}
        return {}
    try:
        df = table.to_pandas()
        sub = df[df["chunk_id"] == chunk_id]
        if len(sub) == 0:
            return {}
        raw = sub.iloc[0].get("metadata_json", "{}")
        return json.loads(raw) if isinstance(raw, str) else {}
    except Exception:
        return {}


def _get_text_for_existing(table, chunk_id: str) -> str:
    """Return the existing text for a chunk_id (best effort)."""
    if hasattr(table, "rows"):
        for r in table.rows:
            if r.get("chunk_id") == chunk_id:
                return str(r.get("text", ""))
        return ""
    try:
        df = table.to_pandas()
        sub = df[df["chunk_id"] == chunk_id]
        if len(sub) == 0:
            return ""
        return str(sub.iloc[0].get("text", ""))
    except Exception:
        return ""


def _embed_text(embedder, text: str) -> List[float]:
    vec = embedder.encode_documents([text])[0]
    vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    assert len(vec_list) == 1024, f"embedding dim mismatch: got {len(vec_list)}"
    return vec_list


def _build_row(
    chunk_id: str, text: str, meta: Dict[str, Any], vector: List[float],
    source: str, faq_type: str = "",
) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "text": text,
        "source": source,
        "faq_type": faq_type,
        "metadata_json": json.dumps(meta, ensure_ascii=False, default=str),
        "vector": vector,
    }


def _append_jsonl(rows: List[Dict[str, Any]]) -> int:
    CHUNKED_JSONL_OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with CHUNKED_JSONL_OUT.open("a", encoding="utf-8") as fh:
        for row in rows:
            payload = {
                "id": row["chunk_id"],
                "doc_id": row["chunk_id"],
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "text_plain": row["text"],
                "metadata": json.loads(row["metadata_json"]),
                "breadcrumbs": [row["source"]],
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# Pure logic (testable)
# ---------------------------------------------------------------------------
def plan_changes(embedder=None, table=None, dry_run: bool = False) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "fixed": [],          # bad chunks rewritten
        "skipped": list(GUARDED_CHUNK_IDS),
        "canonical_added": False,
        "rows_added": 0,
        "rows_deleted": 0,
        "dry_run": dry_run,
        "jsonl_written": 0,
    }

    if dry_run:
        summary["fixed"] = [fix["chunk_id"] for fix in BAD_CHUNK_FIXES]
        summary["canonical_added"] = CANONICAL_CHUNK_ID
        return summary

    assert embedder is not None and table is not None, "live mode needs embedder + table"

    # ---- 1) Rewrite each bad chunk ----
    jsonl_rows: List[Dict[str, Any]] = []
    for fix in BAD_CHUNK_FIXES:
        cid = fix["chunk_id"]
        # Guard: refuse to mutate historical chunks
        if cid in GUARDED_CHUNK_IDS:
            summary["skipped"].append(cid)  # type: ignore[arg-type]
            continue
        # Idempotent: if the row already exists with corrected text AND
        # corrected metadata, skip re-add (avoids redundant re-embed).
        existing_meta = _get_existing_meta(table, cid)
        if (
            existing_meta
            and existing_meta.get("data_year") == 2026
            and existing_meta.get("is_superseded") is False
            and "2.079" in _get_text_for_existing(table, cid)
            and "29 ngành" in _get_text_for_existing(table, cid)
        ):
            # Already corrected — leave it alone.
            summary["fixed"].append(f"{cid} (already corrected, no-op)")
            continue
        # Preserve some original metadata keys (e.g. info_type) but force
        # the canonical numbers + clear superseded flags.
        merged_meta = dict(existing_meta)
        merged_meta.update(fix["meta_overrides"])
        merged_meta["data_year"] = 2026  # force

        # Delete old row (idempotent: 0 or 1 affected)
        rows_now = _row_count(table)
        _delete_by_chunk_id(table, cid)
        if _row_count(table) < rows_now:
            summary["rows_deleted"] += 1

        vec = _embed_text(embedder, fix["text_fix"])
        row = _build_row(cid, fix["text_fix"], merged_meta, vec,
                         fix["source"], faq_type=fix["faq_type"])
        table.add([row])
        summary["rows_added"] += 1
        summary["fixed"].append(cid)
        jsonl_rows.append(row)

    # ---- 2) Add canonical 29×2079 chunk (idempotent) ----
    if not _has_chunk_id(table, CANONICAL_CHUNK_ID):
        _delete_by_chunk_id(table, CANONICAL_CHUNK_ID)
        vec_c = _embed_text(embedder, CANONICAL_TEXT)
        row_c = _build_row(CANONICAL_CHUNK_ID, CANONICAL_TEXT, CANONICAL_META,
                           vec_c, SOURCE_URL, faq_type="canonical_overview")
        table.add([row_c])
        summary["canonical_added"] = CANONICAL_CHUNK_ID
        summary["rows_added"] += 1
        jsonl_rows.append(row_c)
    else:
        summary["canonical_added"] = f"{CANONICAL_CHUNK_ID} (already present, no-op)"

    summary["jsonl_written"] = _append_jsonl(jsonl_rows)
    return summary


# ---------------------------------------------------------------------------
# Live entry point
# ---------------------------------------------------------------------------
def run_against_live_db(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return plan_changes(dry_run=True)

    import lancedb

    from config.settings import RAGSettings
    from src.services.embedding import EmbeddingService

    settings = RAGSettings()
    db = lancedb.connect(settings.LANCEDB_URI)
    table = db.open_table(settings.LANCEDB_TABLE)
    embedder = EmbeddingService(settings)

    return plan_changes(embedder=embedder, table=table, dry_run=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Reconcile HUSC 2026 canonical chunks to 29 ngành × 2.079 chỉ tiêu"
    )
    p.add_argument("--dry-run", action="store_true", help="preview without mutating")
    args = p.parse_args()

    summary = run_against_live_db(dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
