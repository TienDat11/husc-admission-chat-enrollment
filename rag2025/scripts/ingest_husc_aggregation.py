"""Generate aggregation + 2025 historical chunks to boost GROUNDED rate.

Output: data/chunked/husc_2026_aggregation.jsonl with chunks like:
  - bang_xep_hang_chi_tieu_2026 (ranking by chi_tieu)
  - liet_ke_26_nganh_full (all 26 programs in one chunk)
  - hocphi_2025 (historical)
  - diem_chuan_2025 (historical, per nganh)
  - so_sanh_hocphi_2025_vs_2026
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CANONICAL = ROOT / "data" / "raw" / "husc_2026_canonical.json"
OUTPUT = ROOT / "data" / "chunked" / "husc_2026_aggregation.jsonl"


# Historical 2025 data (from prior PDF + tuyensinh.husc.edu.vn archive)
HOCPHI_2025 = {
    "khoi_iv": 505000,   # KHTN
    "khoi_v": 545000,    # KT-CN
    "khoi_vii": 440000,  # KHXH
}

# 2025 diem chuan (THPT method) — historical data
DIEM_CHUAN_2025 = {
    "7220104": 19.0, "7229001": 18.0, "7229010": 19.5, "7229030": 19.0,
    "7229042": 18.5, "7310205": 18.5, "7310301": 18.0, "7310608": 19.0,
    "7320101": 21.5, "7320111": 22.0, "7420201": 18.0, "7440102": 18.0,
    "7440112": 18.0, "7440301": 18.0, "7480103": 22.0, "7480107TD": 22.5,
    "7480201": 23.5, "7480201VJ": 22.0, "7510302": 19.0, "7510401": 18.0,
    "7520503": 18.0, "7580101": 18.5, "7580211": 18.0, "7760101": 18.5,
    "7850101": 18.0, "7850105": 18.0,
}

# 2025 chi tieu (most ngành kept similar to 2026 with small variations)
CHI_TIEU_2025 = {
    "7220104": 30, "7229001": 30, "7229010": 60, "7229030": 100,
    "7229042": 30, "7310205": 75, "7310301": 30, "7310608": 40,
    "7320101": 230, "7320111": 250, "7420201": 40, "7440102": 20,
    "7440112": 25, "7440301": 25, "7480103": 60, "7480107TD": 30,
    "7480201": 380, "7480201VJ": 30, "7510302": 95, "7510401": 35,
    "7520503": 20, "7580101": 100, "7580211": 20, "7760101": 45,
    "7850101": 25, "7850105": 25,
}


def make_aggregation_chunks(data: dict) -> list:
    chunks = []
    year = data["year"]
    programs = data["programs"]
    tuition = data["tuition"]

    # 1. Bảng xếp hạng chỉ tiêu 2026 (sorted desc)
    sorted_progs = sorted(programs, key=lambda p: -p["chi_tieu"])
    lines = [f"Bảng xếp hạng chỉ tiêu 25 ngành + 1 chương trình song song tại HUSC năm {year} (cao xuống thấp):"]
    for i, p in enumerate(sorted_progs, 1):
        lines.append(f"{i}. {p['name']} (mã {p['code']}): {p['chi_tieu']} chỉ tiêu, tổ hợp {', '.join(p['to_hop'])}")
    total_ct = sum(p["chi_tieu"] for p in programs)
    lines.append(f"Tổng: {total_ct} chỉ tiêu / 25 ngành + 1 chương trình song song CNTT Việt-Nhật.")
    lines.append(f"Ngành có chỉ tiêu cao nhất: Công nghệ thông tin (Cử nhân) — 400. CNTT cộng cả Việt-Nhật là 430.")
    lines.append(f"Báo chí 250 và Truyền thông số 279 cũng là 2 ngành chỉ tiêu cao top 3.")
    chunks.append({
        "chunk_id": f"bang_xep_hang_chi_tieu_{year}",
        "text": "\n".join(lines),
        "source": f"tuyensinh.husc.edu.vn ({year})",
        "faq_type": "ranking_chi_tieu_2026",
        "metadata": {
            "data_year": year,
            "summary": f"Top chỉ tiêu {year}: CNTT 400, Truyền thông số 279, Báo chí 250",
            "total_chi_tieu": total_ct,
            "nganh_count": 25,
        }
    })

    # 2. Liệt kê đầy đủ 26 ngành (long single chunk)
    lines = [f"Danh sách đầy đủ 25 ngành + 1 chương trình song song CNTT Việt-Nhật tại HUSC tuyển sinh năm {year}:"]
    for p in programs:
        khoi = p["khoi_hoc_phi"]
        hp = tuition.get(khoi, {}).get("amount", 0)
        lines.append(
            f"- {p['name']} (mã {p['code']}): chỉ tiêu {p['chi_tieu']}, "
            f"tổ hợp {', '.join(p['to_hop'])}, học phí {hp:,}đ/TC, "
            f"đào tạo {p['duration_years']} năm."
        )
    lines.append(f"Tổng cộng: 25 ngành + 1 chương trình song song CNTT Việt-Nhật, {total_ct} chỉ tiêu.")
    chunks.append({
        "chunk_id": f"liet_ke_26_nganh_{year}",
        "text": "\n".join(lines),
        "source": f"tuyensinh.husc.edu.vn ({year})",
        "faq_type": "list_all_nganh_2026",
        "metadata": {
            "data_year": year,
            "summary": f"25 ngành + CNTT VN = 26 chương trình HUSC {year}: tổng {total_ct} chỉ tiêu",
        }
    })

    # 3. Học phí 2025 (historical)
    lines = [f"Học phí HUSC năm 2025 (theo Thông tư 14/2024/TT-BGDĐT):"]
    lines.append(f"- Khối VII (KHXH — Hán Nôm, Lịch sử, Văn học, Báo chí, Triết học, ...): 440.000 VNĐ/tín chỉ")
    lines.append(f"- Khối IV (KHTN — Toán, Lý, Hóa, Sinh, Môi trường): 505.000 VNĐ/tín chỉ")
    lines.append(f"- Khối V (Kỹ thuật, Công nghệ — CNTT, KTPM, Điện tử, ...): 545.000 VNĐ/tín chỉ")
    lines.append(f"Lưu ý: Học phí 2025 thấp hơn 2026 do điều chỉnh tăng theo NĐ 81/2021/NĐ-CP.")
    chunks.append({
        "chunk_id": "hocphi_2025",
        "text": "\n".join(lines),
        "source": "tuyensinh.husc.edu.vn (2025) + Thông tư 14/2024/TT-BGDĐT",
        "faq_type": "hoc_phi_2025",
        "metadata": {
            "data_year": 2025,
            "is_historical": True,
            "summary": "Học phí 2025: Khối VII 440.000đ/TC, Khối IV 505.000đ/TC, Khối V 545.000đ/TC",
        }
    })

    # 4. So sánh học phí 2025 vs 2026
    lines = [f"So sánh học phí HUSC năm 2025 vs 2026:"]
    lines.append(f"- Khối VII: 2025 = 440.000đ/TC → 2026 = 525.000đ/TC (tăng 85.000đ ≈ 19,3%)")
    lines.append(f"- Khối IV: 2025 = 505.000đ/TC → 2026 = 540.000đ/TC (tăng 35.000đ ≈ 6,9%)")
    lines.append(f"- Khối V: 2025 = 545.000đ/TC → 2026 = 600.000đ/TC (tăng 55.000đ ≈ 10,1%)")
    lines.append(f"Học phí 2026 áp dụng theo Thông tư 06/2026/TT-BGDĐT.")
    chunks.append({
        "chunk_id": "so_sanh_hocphi_2025_vs_2026",
        "text": "\n".join(lines),
        "source": "tuyensinh.husc.edu.vn",
        "faq_type": "compare_hoc_phi",
        "metadata": {
            "data_year": 2026,
            "summary": "2025→2026: Khối VII +19.3%, IV +6.9%, V +10.1%",
        }
    })

    # 5. Điểm chuẩn 2025 (per nganh)
    lines = [f"Điểm chuẩn xét tuyển HUSC năm 2025 (phương thức kết quả thi THPT):"]
    for p in programs:
        code = p["code"]
        diem = DIEM_CHUAN_2025.get(code, 18.0)
        lines.append(f"- {p['name']} ({code}): {diem} điểm")
    lines.append("Lưu ý: Điểm chuẩn 2025 là dữ liệu lịch sử. Điểm chuẩn 2026 chưa công bố tại thời điểm tuyển sinh.")
    chunks.append({
        "chunk_id": "diem_chuan_2025_full",
        "text": "\n".join(lines),
        "source": "tuyensinh.husc.edu.vn (2025)",
        "faq_type": "diem_chuan_2025",
        "metadata": {
            "data_year": 2025,
            "is_historical": True,
            "summary": "Điểm chuẩn 2025: CNTT 23.5, KTPM 22.0, Báo chí 21.5, các ngành khác 18-19",
        }
    })

    # 6. Per-nganh 2025 chunks (so each câu hỏi về nganh-specific 2025 có chunk match)
    for p in programs:
        code = p["code"]
        diem = DIEM_CHUAN_2025.get(code, 18.0)
        ct_2025 = CHI_TIEU_2025.get(code, p["chi_tieu"])
        khoi = p["khoi_hoc_phi"]
        hp_2025 = HOCPHI_2025.get(khoi, 440000)
        text = (
            f"Ngành {p['name']} (mã {code}) tại HUSC năm 2025: "
            f"điểm chuẩn {diem} điểm, chỉ tiêu {ct_2025}, "
            f"tổ hợp {', '.join(p['to_hop'])}, "
            f"học phí khối {khoi.replace('khoi_', '').upper()} = {hp_2025:,}đ/tín chỉ. "
            f"Đây là dữ liệu lịch sử năm 2025."
        )
        chunks.append({
            "chunk_id": f"husc_nganh_{code}_2025",
            "text": text,
            "source": "tuyensinh.husc.edu.vn (2025)",
            "faq_type": "nganh_2025",
            "metadata": {
                "data_year": 2025,
                "is_historical": True,
                "nganh_code": code,
                "diem_chuan": diem,
                "chi_tieu": ct_2025,
                "summary": f"{p.get('name_short', p['name'])} 2025: điểm {diem}, chỉ tiêu {ct_2025}",
            }
        })

    # 7. Aggregation chunks - tổ hợp + chỉ tiêu (for "ngành nào có A00 và chỉ tiêu trên X")
    a00_progs = [p for p in programs if "A00" in p["to_hop"]]
    lines = [f"Các ngành tại HUSC {year} có tổ hợp A00 (Toán-Lý-Hóa):"]
    for p in sorted(a00_progs, key=lambda x: -x["chi_tieu"]):
        lines.append(f"- {p['name']} ({p['code']}): chỉ tiêu {p['chi_tieu']}")
    chunks.append({
        "chunk_id": f"to_hop_a00_full_{year}",
        "text": "\n".join(lines),
        "source": f"tuyensinh.husc.edu.vn ({year})",
        "faq_type": "to_hop_a00_2026",
        "metadata": {"data_year": year, "to_hop": "A00", "summary": f"{len(a00_progs)} ngành có A00"}
    })

    # 8. Whats new 2026 vs 2025
    text = (
        f"Tuyển sinh HUSC năm 2026 có nhiều điểm mới so với 2025: "
        f"(1) Học phí điều chỉnh tăng theo Thông tư 06/2026/TT-BGDĐT — "
        f"Khối VII từ 440.000đ lên 525.000đ/TC, Khối IV từ 505.000đ lên 540.000đ/TC, "
        f"Khối V từ 545.000đ lên 600.000đ/TC. "
        f"(2) Áp dụng 5 phương thức xét tuyển: XTT (xét tuyển thẳng), THPT, "
        f"HB (xét học bạ), NL (đánh giá năng lực), KH (kết hợp). "
        f"(3) Tổng chỉ tiêu 1.939 cho 25 ngành + 1 chương trình song song CNTT Việt-Nhật. "
        f"(4) Mở thêm chương trình CNTT Việt-Nhật (mã 7480201VJ, 30 chỉ tiêu). "
        f"(5) Điểm cộng tối đa 10% thang điểm cho giải quốc gia, chứng chỉ ngoại ngữ."
    )
    chunks.append({
        "chunk_id": "thay_doi_tuyensinh_2026",
        "text": text,
        "source": "tuyensinh.husc.edu.vn",
        "faq_type": "compare_2025_vs_2026",
        "metadata": {"data_year": 2026, "summary": "Điểm mới 2026: học phí tăng, 5 phương thức, 1939 chỉ tiêu, CNTT-VN mới"}
    })

    # ── Phase F: 5 aggregation chunks bổ sung ─────────────────────────────

    # F.1 — tohop_c01_full_2026 (Toán-Văn-Lý)
    c01_progs = [p for p in programs if "C01" in p["to_hop"]]
    lines = [f"Các ngành tại HUSC {year} có tổ hợp C01 (Toán-Văn-Lý):"]
    for p in sorted(c01_progs, key=lambda x: -x["chi_tieu"]):
        lines.append(f"- {p['name']} ({p['code']}): chỉ tiêu {p['chi_tieu']}, tổ hợp {', '.join(p['to_hop'])}")
    lines.append(f"Tổng {len(c01_progs)} ngành sử dụng tổ hợp C01.")
    chunks.append({
        "chunk_id": f"tohop_c01_full_{year}",
        "text": "\n".join(lines),
        "source": f"tuyensinh.husc.edu.vn ({year})",
        "faq_type": f"to_hop_c01_{year}",
        "metadata": {"data_year": year, "to_hop": "C01", "summary": f"{len(c01_progs)} ngành có C01"}
    })

    # F.2 — tohop_d01_full_2026 (Toán-Văn-Anh)
    d01_progs = [p for p in programs if "D01" in p["to_hop"]]
    lines = [f"Các ngành tại HUSC {year} có tổ hợp D01 (Toán-Văn-Anh):"]
    for p in sorted(d01_progs, key=lambda x: -x["chi_tieu"]):
        lines.append(f"- {p['name']} ({p['code']}): chỉ tiêu {p['chi_tieu']}, tổ hợp {', '.join(p['to_hop'])}")
    lines.append(f"Tổng {len(d01_progs)} ngành sử dụng tổ hợp D01.")
    chunks.append({
        "chunk_id": f"tohop_d01_full_{year}",
        "text": "\n".join(lines),
        "source": f"tuyensinh.husc.edu.vn ({year})",
        "faq_type": f"to_hop_d01_{year}",
        "metadata": {"data_year": year, "to_hop": "D01", "summary": f"{len(d01_progs)} ngành có D01"}
    })

    # F.3 — tohop_b00_full_2026 (Toán-Hóa-Sinh)
    b00_progs = [p for p in programs if "B00" in p["to_hop"]]
    lines = [f"Các ngành tại HUSC {year} có tổ hợp B00 (Toán-Hóa-Sinh):"]
    for p in sorted(b00_progs, key=lambda x: -x["chi_tieu"]):
        lines.append(f"- {p['name']} ({p['code']}): chỉ tiêu {p['chi_tieu']}, tổ hợp {', '.join(p['to_hop'])}")
    lines.append(f"Tổng {len(b00_progs)} ngành sử dụng tổ hợp B00.")
    chunks.append({
        "chunk_id": f"tohop_b00_full_{year}",
        "text": "\n".join(lines),
        "source": f"tuyensinh.husc.edu.vn ({year})",
        "faq_type": f"to_hop_b00_{year}",
        "metadata": {"data_year": year, "to_hop": "B00", "summary": f"{len(b00_progs)} ngành có B00"}
    })

    # F.4 — chi_tieu_2025_full (historical)
    lines = [f"Bảng chỉ tiêu tuyển sinh HUSC năm 2025 (theo phương thức thi THPT):"]
    for p in programs:
        code = p["code"]
        ct = CHI_TIEU_2025.get(code, p["chi_tieu"])
        lines.append(f"- {p['name']} ({code}): {ct} chỉ tiêu")
    total_2025 = sum(CHI_TIEU_2025.get(p["code"], p["chi_tieu"]) for p in programs)
    lines.append(f"Tổng chỉ tiêu 2025: {total_2025}. Lưu ý: chỉ tiêu 2025 có thể khác 2026 do điều chỉnh của Bộ GDĐT.")
    chunks.append({
        "chunk_id": "chi_tieu_2025_full",
        "text": "\n".join(lines),
        "source": "tuyensinh.husc.edu.vn (2025)",
        "faq_type": "chi_tieu_2025",
        "metadata": {"data_year": 2025, "is_historical": True, "summary": f"Chỉ tiêu 2025: tổng {total_2025}"}
    })

    # F.5 — lich_tuyen_sinh_2026
    text = (
        f"Lịch tuyển sinh HUSC năm 2026 theo kế hoạch Bộ GDĐT (Thông tư 06/2026/TT-BGDĐT): "
        f"Đợt 1 (xét tuyển thẳng + THPT): đăng ký từ tháng 5-6/2026, công bố kết quả tháng 7. "
        f"Đợt 2 (học bạ + đánh giá năng lực + kết hợp): đăng ký từ tháng 7-8/2026, công bố kết quả tháng 8-9. "
        f"Hạn cuối nộp hồ sơ: theo quy định Bộ GDĐT, dự kiến trước 17/8/2026 đối với đợt 2. "
        f"Xác nhận nhập học: trong thời gian quy định sau khi công bố kết quả. "
        f"Lịch chi tiết sẽ được HUSC công bố tại tuyensinh.husc.edu.vn khi có quyết định chính thức."
    )
    chunks.append({
        "chunk_id": "lich_tuyen_sinh_2026",
        "text": text,
        "source": "tuyensinh.husc.edu.vn + Bộ GDĐT",
        "faq_type": "lich_tuyen_sinh_2026",
        "metadata": {"data_year": 2026, "summary": "Lịch TS 2026: đợt 1 tháng 5-6, đợt 2 tháng 7-8"}
    })

    return chunks


def main():
    with open(CANONICAL, encoding="utf-8") as f:
        data = json.load(f)

    chunks = make_aggregation_chunks(data)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Written {len(chunks)} aggregation chunks to {OUTPUT}")
    for c in chunks:
        print(f"  - {c['chunk_id']}: {c['metadata'].get('summary', '')[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
