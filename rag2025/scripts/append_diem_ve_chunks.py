"""Append 6 chunks về điểm vẽ năng khiếu + công thức điểm 5 phương thức vào husc_2026_operational_qa.jsonl."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
FP = ROOT / "data" / "chunked" / "husc_2026_operational_qa.jsonl"

NEW_CHUNKS = [
    {
        "chunk_id": "qa_cong_thuc_diem_pt2_thpt_2026",
        "text": (
            "Câu hỏi: Công thức tính điểm xét tuyển bằng kết quả thi THPT (Phương thức 2) năm 2026 tại HUSC?\n"
            "Trả lời (theo thông báo tuyển sinh HUSC 2026, thang 30):\n"
            "ĐXT = M1×H1 + M2×H2 + M3×H3 + Điểm cộng + Điểm ưu tiên\n"
            "- M1, M2, M3: điểm 3 môn của tổ hợp xét tuyển từ kỳ thi tốt nghiệp THPT năm 2026.\n"
            "- H1, H2, H3: hệ số môn (mặc định = 1, ngành Kiến trúc dùng công thức riêng KH-5b).\n"
            "- Điểm ưu tiên (KV/ĐT): KV1 +0,75 ; KV2-NT +0,5 ; KV2 +0,25 ; đối tượng 1-4 +1,0 đến +2,0 ; đối tượng 5-7 +0,5 đến +1,0 (thang 30).\n"
            "- Điểm cộng (HSG, IELTS/TOEFL/VSTEP, HSG 3 năm trường chuyên) — tổng điểm cộng tối đa = 10% × thang điểm = 3,0 điểm (thang 30)."
        ),
        "source": "tuyensinh.husc.edu.vn/thongbao.php?id=63 (canonical 2026)",
        "faq_type": "operational_qa",
        "metadata": {
            "data_year": 2026,
            "info_type": "operational_qa",
            "topic": "cong_thuc_diem_pt2",
            "summary": "Công thức PT2 (THPT 2026): ĐXT = M1×H1 + M2×H2 + M3×H3 + Điểm cộng + Điểm ưu tiên (thang 30).",
        },
    },
    {
        "chunk_id": "qa_cong_thuc_diem_pt3_hocba_2026",
        "text": (
            "Câu hỏi: Công thức tính điểm xét tuyển học bạ THPT (Phương thức 3) năm 2026 tại HUSC?\n"
            "Trả lời (theo thông báo tuyển sinh HUSC 2026, thang 30):\n"
            "ĐXT = M1×H1 + M2×H2 + M3×H3 + Điểm cộng + Điểm ưu tiên\n"
            "- M1, M2, M3: điểm trung bình chung kết quả học tập CẢ NĂM của 3 môn trong tổ hợp xét tuyển ở 3 năm THPT, làm tròn 02 chữ số thập phân. "
            "Cụ thể với mỗi môn: M = (điểm TB cả năm lớp 10 + điểm TB cả năm lớp 11 + điểm TB cả năm lớp 12) / 3.\n"
            "- H1, H2, H3: hệ số môn (mặc định = 1).\n"
            "- Điểm cộng và điểm ưu tiên: như Phương thức 2 (KV/ĐT/HSG/CCNN).\n"
            "Ví dụ: tổ hợp A00, môn Toán có TB cả năm 10/11/12 lần lượt 8,0/8,5/9,0 → M Toán = (8,0+8,5+9,0)/3 = 8,50."
        ),
        "source": "tuyensinh.husc.edu.vn/thongbao.php?id=63 (canonical 2026)",
        "faq_type": "operational_qa",
        "metadata": {
            "data_year": 2026,
            "info_type": "operational_qa",
            "topic": "cong_thuc_diem_pt3_hocba",
            "summary": "PT3 học bạ: M = TB cả năm lớp 10 + 11 + 12 / 3 (làm tròn 2 chữ số), ĐXT = M1×H1 + M2×H2 + M3×H3 + cộng + ưu tiên.",
        },
    },
    {
        "chunk_id": "qa_cong_thuc_diem_pt4_dgnl_2026",
        "text": (
            "Câu hỏi: Công thức tính điểm xét tuyển bằng kết quả thi đánh giá năng lực (Phương thức 4) năm 2026 tại HUSC?\n"
            "Trả lời (theo thông báo tuyển sinh HUSC 2026):\n"
            "- Phương thức 4a (ĐGNL Đại học Quốc gia Hà Nội): ĐXT = M + Điểm cộng + Điểm ưu tiên (thang 150).\n"
            "- Phương thức 4b (ĐGNL Đại học Quốc gia TP.HCM): cùng công thức 4a, dùng kết quả đợt thi có điểm cao nhất "
            "(thang 150 hoặc thang theo quy định ĐGNL TP.HCM, được quy về thang chung khi xét).\n"
            "- M: điểm bài thi đánh giá năng lực 2026.\n"
            "- Điểm cộng + Điểm ưu tiên được quy đổi tương ứng về thang điểm 150 (KV1, KV2-NT, KV2, đối tượng 1-7, HSG, CCNN).\n"
            "Thí sinh có thể thi nhiều đợt và sử dụng kết quả cao nhất."
        ),
        "source": "tuyensinh.husc.edu.vn/thongbao.php?id=63 (canonical 2026)",
        "faq_type": "operational_qa",
        "metadata": {
            "data_year": 2026,
            "info_type": "operational_qa",
            "topic": "cong_thuc_diem_pt4_dgnl",
            "summary": "PT4a/4b ĐGNL: ĐXT = M + cộng + ưu tiên (thang 150); 4b lấy điểm đợt cao nhất.",
        },
    },
    {
        "chunk_id": "qa_cong_thuc_diem_pt5a_ccnn_2026",
        "text": (
            "Câu hỏi: Công thức tính điểm xét tuyển kết hợp chứng chỉ ngoại ngữ (Phương thức 5a) năm 2026 tại HUSC?\n"
            "Trả lời (theo thông báo tuyển sinh HUSC 2026, thang 30):\n"
            "ĐXT = M1×H1 + M2×H2 + M3×H3 + Điểm cộng + Điểm ưu tiên\n"
            "- M1, M2: điểm 2 môn văn hóa từ kỳ thi THPT 2026 hoặc điểm trung bình chung cả năm 3 năm THPT (theo Phương thức 3).\n"
            "- M3: điểm môn ngoại ngữ — quy đổi từ chứng chỉ ngoại ngữ về thang 10. "
            "Quy đổi tham khảo: IELTS 5.5 ≈ 8,0 ; IELTS 6.0-6.5 ≈ 9,0 ; IELTS 7.0+ ≈ 10,0 ; "
            "TOEFL iBT 64-78 ≈ 8,0 ; TOEFL iBT 79-93 ≈ 9,0 ; TOEFL iBT 94+ ≈ 10,0 ; "
            "B1 VSTEP ≈ 8,0 ; B2 VSTEP ≈ 9,0 ; C1-C2 VSTEP ≈ 10,0.\n"
            "- Hạn chứng chỉ ngoại ngữ: đến 03/8/2026, KHÔNG chấp nhận Home Edition."
        ),
        "source": "tuyensinh.husc.edu.vn/thongbao.php?id=63 (canonical 2026)",
        "faq_type": "operational_qa",
        "metadata": {
            "data_year": 2026,
            "info_type": "operational_qa",
            "topic": "cong_thuc_diem_pt5a_ccnn",
            "summary": "PT5a kết hợp CCNN: M3 = điểm CCNN quy về thang 10; ĐXT = M1×H1 + M2×H2 + M3×H3 + cộng + ưu tiên (thang 30).",
        },
    },
    {
        "chunk_id": "qa_cong_thuc_diem_pt5b_kientruc_2026",
        "text": (
            "Câu hỏi: Công thức tính điểm xét tuyển ngành Kiến trúc (Phương thức 5b - kết hợp năng khiếu) năm 2026 tại HUSC?\n"
            "Trả lời (theo thông báo tuyển sinh HUSC 2026, áp dụng riêng cho ngành Kiến trúc mã 7580101, thang 40):\n"
            "ĐXT = M1×H1 + M2×H2 + M_NK×2 + Điểm cộng + Điểm ưu tiên\n"
            "- M1, M2: điểm 2 môn văn hóa của tổ hợp V00/V01/V02/V12 (Toán + Vật lý/Tiếng Anh/Ngữ văn) — lấy từ thi THPT 2026 hoặc TB chung cả năm lớp 10-11-12.\n"
            "- M_NK: điểm môn thi năng khiếu Vẽ mỹ thuật năm 2026 (thang 10), HỆ SỐ 2.\n"
            "- Điều kiện sàn: M_NK ≥ 5,00 điểm (thang 10). Nếu không đạt, bị loại khỏi xét tuyển ngành Kiến trúc.\n"
            "- HUSC chấp nhận M_NK từ 6 nguồn: HUSC, ĐH Kiến trúc Hà Nội, ĐH Xây dựng Hà Nội, ĐHBK Đà Nẵng, ĐH Kiến trúc TPHCM, ĐHBK ĐHQG TPHCM. "
            "Trường khác có nhiều thành phần điểm: M_NK = trung bình cộng (đã quy thang 10).\n"
            "Ví dụ: M1=Toán 8,0; M2=Vật lý 7,5; M_NK=6,5; KV2-NT (+0,5) → ĐXT = 8,0×1 + 7,5×1 + 6,5×2 + 0 + 0,5 = 29,0 điểm."
        ),
        "source": "tuyensinh.husc.edu.vn/thongbao.php?id=63 (canonical 2026)",
        "faq_type": "operational_qa",
        "metadata": {
            "data_year": 2026,
            "info_type": "operational_qa",
            "topic": "cong_thuc_diem_pt5b_kientruc",
            "summary": "PT5b Kiến trúc: ĐXT = M1×H1 + M2×H2 + M_NK×2 + cộng + ưu tiên; M_NK ≥ 5,00 (thang 10); chấp nhận điểm vẽ từ 6 trường.",
        },
    },
    {
        "chunk_id": "qa_diem_ve_chap_nhan_6_truong_2026",
        "text": (
            "Câu hỏi: HUSC chấp nhận điểm thi năng khiếu Vẽ mỹ thuật của những trường Đại học nào năm 2026?\n"
            "Trả lời (theo thông báo tuyển sinh HUSC 2026 — id=63):\n"
            "HUSC chấp nhận điểm thi năng khiếu môn Vẽ mỹ thuật năm 2026 từ 6 nguồn:\n"
            "1. Trường Đại học Khoa học, Đại học Huế (HUSC) — đơn vị chủ quản, tổ chức tại TP Huế.\n"
            "2. Trường Đại học Kiến trúc Hà Nội.\n"
            "3. Trường Đại học Xây dựng Hà Nội.\n"
            "4. Trường Đại học Bách khoa, Đại học Đà Nẵng.\n"
            "5. Trường Đại học Kiến trúc Thành phố Hồ Chí Minh.\n"
            "6. Trường Đại học Bách Khoa, Đại học Quốc gia TP.HCM.\n"
            "Ngưỡng tối thiểu: ≥ 5,00 điểm (thang 10). Nếu kỳ thi của trường khác có nhiều thành phần điểm (đã quy thang 10), "
            "điểm môn Vẽ mỹ thuật của thí sinh = trung bình cộng các điểm thành phần. "
            "Thí sinh nộp giấy chứng nhận kết quả thi năng khiếu hợp lệ trong hồ sơ xét tuyển."
        ),
        "source": "tuyensinh.husc.edu.vn/thongbao.php?id=63 (canonical 2026)",
        "faq_type": "operational_qa",
        "metadata": {
            "data_year": 2026,
            "info_type": "operational_qa",
            "topic": "diem_ve_chap_nhan_6_truong",
            "summary": "6 trường được HUSC công nhận điểm thi năng khiếu Vẽ mỹ thuật 2026: HUSC, ĐHKT HN, ĐHXD HN, ĐHBK Đà Nẵng, ĐHKT TPHCM, ĐHBK ĐHQG TPHCM.",
        },
    },
]


def main() -> int:
    existing_ids = set()
    if FP.exists():
        with FP.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj.get("chunk_id") or obj.get("id")
                if cid:
                    existing_ids.add(cid)

    appended = 0
    skipped = 0
    with FP.open("a", encoding="utf-8") as fh:
        for chunk in NEW_CHUNKS:
            if chunk["chunk_id"] in existing_ids:
                skipped += 1
                continue
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            appended += 1

    print(f"Appended: {appended} | Skipped (already exist): {skipped}")

    with FP.open(encoding="utf-8") as fh:
        total = sum(1 for line in fh if line.strip())
    print(f"Total chunks now: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
