"""Add 2026 tuition chunks to chunked JSONL files."""
import json
from pathlib import Path

BASE = Path("rag2025/data/chunked")

# --- chunked_10.jsonl chunks ---
chunks_10 = [
    {
        "id": "husc_hoc_phi_tin_chi_2026",
        "doc_id": "chunked_10",
        "chunk_id": 32,
        "text": "### husc hoc phi tin chi 2026\n\nMức thu học phí theo tín chỉ tại Trường Đại học Khoa học - Đại học Huế năm 2026 theo khối ngành (dự kiến theo Thông tư 06/2026/TT-BGDĐT): Khối VII (Hán-Nôm, Lịch sử, Văn học, Báo chí, Xã hội học...) 525.000 VNĐ/tín chỉ. Khối IV (Vật lý, Hóa học, Sinh học, Khoa học môi trường...) 540.000 VNĐ/tín chỉ. Khối V (Kỹ thuật, CNTT, KTPM, Kiến trúc, Điện tử...) 600.000 VNĐ/tín chỉ. Ngành Triết học miễn học phí (0 đồng/năm). Mức trung bình 30 tín chỉ/năm. Nguồn: Thông tin tuyển sinh HUSC 2026 + Thông tư 06/2026/TT-BGDĐT + tuyensinh.husc.edu.vn.",
        "text_plain": "Hoc phi 2026 theo khoi nganh: VII 525.000d/tc, IV 540.000d/tc, V 600.000d/tc. Triet hoc mien hoc phi. Nguon TT 06/2026 va tuyensinh.husc.edu.vn.",
        "summary": "Học phí 2026 theo khối ngành: VII 525.000đ/tín chỉ, IV 540.000đ/tín chỉ, V 600.000đ/tín chỉ. Triết học miễn học phí. Nguồn: TT06/2026 + HUSC website.",
        "metadata": {
            "topic": "hoc_phi_tin_chi",
            "current_price_khoi_VII": 525000,
            "current_price_khoi_IV": 540000,
            "current_price_khoi_V": 600000,
            "currency": "VND",
            "growth_rate_khoi_VII_percent": 19.3,
            "growth_rate_khoi_IV_percent": 6.9,
            "growth_rate_khoi_V_percent": 9.6,
            "unit": "Khoa_Hoc",
            "issuer": "Dai_Hoc_Hue",
            "year": "2026",
            "data_year": 2026,
            "source": "Thông tin tuyển sinh HUSC 2026 + Thông tư 06/2026/TT-BGDĐT + tuyensinh.husc.edu.vn",
            "source_type": "jsonl",
            "faq_type": "hoc_phi_va_chi_phi",
            "effective_date": "2026-06-01",
            "crawl_date": "2026-05-18T00:00:00Z",
            "expired": False,
            "is_superseded": False,
            "freshness_score": 1.0,
            "version": 2,
            "school": "HUSC",
            "supersedes": "husc_hoc_phi_tin_chi_2025",
        },
        "breadcrumbs": ["Thông tin tuyển sinh HUSC 2026", "Hoc Phi"],
        "prev_chunk_id": "husc_hoc_phi_tin_chi_2025",
        "next_chunk_id": "husc_uoc_tinh_hoc_phi_ky_2026",
        "sparse_terms": ["học", "phí", "tín", "chỉ", "khối", "VII", "IV", "V", "525", "540", "600", "000", "triết", "học", "miễn", "trường", "đại", "khoa", "huế", "2026", "thông", "tư"],
    },
    {
        "id": "husc_uoc_tinh_hoc_phi_ky_2026",
        "doc_id": "chunked_10",
        "chunk_id": 33,
        "text": "### husc uoc tinh hoc phi ky 2026\n\nƯớc tính học phí theo khối ngành tại Trường Đại học Khoa học - Đại học Huế năm 2026 (30 tín chỉ/năm): Khối VII (525.000đ/tín chỉ): khoảng 15.750.000đ/năm, mỗi kỳ 15 tín chỉ ≈ 7.875.000đ, 18 tín chỉ ≈ 9.450.000đ. Khối IV (540.000đ/tín chỉ): khoảng 16.200.000đ/năm, mỗi kỳ 15 tín chỉ ≈ 8.100.000đ, 18 tín chỉ ≈ 9.720.000đ. Khối V (600.000đ/tín chỉ): khoảng 18.000.000đ/năm, mỗi kỳ 15 tín chỉ ≈ 9.000.000đ, 18 tín chỉ ≈ 10.800.000đ. Ngành Triết học: 0đ (miễn học phí). Nguồn: Dự kiến theo TT06/2026/TT-BGDĐT + tuyensinh.husc.edu.vn.",
        "text_plain": "Uoc tinh hoc phi 2026 theo khoi nganh: VII 7.9-9.5 trieu/ky, IV 8.1-9.7 trieu/ky, V 9.0-10.8 trieu/ky. Triet hoc mien hoc phi.",
        "summary": "Ước tính học phí/kỳ 2026 theo khối: VII ~7.9-9.5 triệu, IV ~8.1-9.7 triệu, V ~9.0-10.8 triệu. Triết học miễn.",
        "metadata": {
            "topic": "uoc_tinh_hoc_phi",
            "base_price_khoi_VII": 525000,
            "base_price_khoi_IV": 540000,
            "base_price_khoi_V": 600000,
            "credits_min": 15,
            "credits_max": 18,
            "unit": "Khoa_Hoc",
            "issuer": "Dai_Hoc_Hue",
            "year": "2026",
            "data_year": 2026,
            "source": "Thông tin tuyển sinh HUSC 2026 + Thông tư 06/2026/TT-BGDĐT + tuyensinh.husc.edu.vn",
            "source_type": "jsonl",
            "faq_type": "hoc_phi_va_chi_phi",
            "effective_date": "2026-06-01",
            "crawl_date": "2026-05-18T00:00:00Z",
            "expired": False,
            "is_superseded": False,
            "freshness_score": 1.0,
            "version": 2,
            "school": "HUSC",
            "supersedes": "husc_uoc_tinh_hoc_phi_ky_2025",
        },
        "breadcrumbs": ["Thông tin tuyển sinh HUSC 2026", "Hoc Phi"],
        "prev_chunk_id": "husc_hoc_phi_tin_chi_2026",
        "next_chunk_id": "",
        "sparse_terms": ["ước", "tính", "học", "phí", "khối", "VII", "IV", "V", "525", "540", "600", "000", "tín", "chỉ", "kỳ", "triết", "học", "miễn", "đồng", "trường", "đại", "khoa", "huế", "2026"],
    },
]

# --- chunked_10_enhanced.jsonl chunks (3 chunks: hoc_phi + uoc_tinh + policy) ---
chunks_10_enhanced = [
    {
        "id": "husc_hoc_phi_tin_chi_2026",
        "doc_id": "husc_tuyen_sinh_2026",
        "chunk_id": 35,
        "text": "### husc hoc phi tin chi 2026\n\nMức thu học phí theo tín chỉ tại Trường Đại học Khoa học - Đại học Huế năm 2026 theo khối ngành (dự kiến theo Thông tư 06/2026/TT-BGDĐT): Khối VII (Hán-Nôm, Lịch sử, Văn học, Báo chí, Xã hội học...) 525.000 VNĐ/tín chỉ. Khối IV (Vật lý, Hóa học, Sinh học, Khoa học môi trường...) 540.000 VNĐ/tín chỉ. Khối V (Kỹ thuật, CNTT, KTPM, Kiến trúc, Điện tử...) 600.000 VNĐ/tín chỉ. Ngành Triết học miễn học phí (0 đồng/năm). Mức trung bình 30 tín chỉ/năm. Nguồn: Thông tin tuyển sinh HUSC 2026 + Thông tư 06/2026/TT-BGDĐT + tuyensinh.husc.edu.vn.",
        "text_plain": "Hoc phi 2026 theo khoi nganh: VII 525.000d/tc, IV 540.000d/tc, V 600.000d/tc. Triet hoc mien hoc phi. Nguon TT 06/2026 va tuyensinh.husc.edu.vn.",
        "summary": "Học phí 2026 theo khối ngành: VII 525.000đ/tín chỉ, IV 540.000đ/tín chỉ, V 600.000đ/tín chỉ. Triết học miễn học phí.",
        "metadata": {
            "topic": "hoc_phi_tin_chi",
            "current_price_khoi_VII": 525000,
            "current_price_khoi_IV": 540000,
            "current_price_khoi_V": 600000,
            "currency": "VND",
            "growth_rate_khoi_VII_percent": 19.3,
            "growth_rate_khoi_IV_percent": 6.9,
            "growth_rate_khoi_V_percent": 9.6,
            "unit": "Khoa_Hoc",
            "issuer": "Dai_Hoc_Hue",
            "audience": "thi_sinh",
            "info_type": "hoc_phi",
            "effective_date": "2026-06-01",
            "crawl_date": "2026-05-18T00:00:00Z",
            "year": "2026",
            "data_year": 2026,
            "expired": False,
            "is_superseded": False,
            "freshness_score": 1.0,
            "version": 2,
            "school": "HUSC",
            "supersedes": "husc_hoc_phi_tin_chi_2025",
            "source": "Thông tin tuyển sinh HUSC 2026 + Thông tư 06/2026/TT-BGDĐT + tuyensinh.husc.edu.vn",
            "source_type": "jsonl",
            "faq_type": "hoc_phi_va_chi_phi",
        },
        "breadcrumbs": ["Đại học Huế (HUSC) - Tuyển sinh 2026", "Hoc Phi"],
        "prev_chunk_id": "husc_policy_hoc_phi_2025",
        "next_chunk_id": "husc_uoc_tinh_hoc_phi_ky_2026",
        "sparse_terms": ["học", "phí", "tín", "chỉ", "khối", "VII", "IV", "V", "525", "540", "600", "000", "triết", "học", "miễn", "trường", "đại", "khoa", "huế", "2026", "thông", "tư"],
    },
    {
        "id": "husc_uoc_tinh_hoc_phi_ky_2026",
        "doc_id": "husc_tuyen_sinh_2026",
        "chunk_id": 36,
        "text": "### husc uoc tinh hoc phi ky 2026\n\nƯớc tính học phí theo khối ngành tại Trường Đại học Khoa học - Đại học Huế năm 2026 (30 tín chỉ/năm): Khối VII (525.000đ/tín chỉ): khoảng 15.750.000đ/năm, mỗi kỳ 15 tín chỉ ≈ 7.875.000đ, 18 tín chỉ ≈ 9.450.000đ. Khối IV (540.000đ/tín chỉ): khoảng 16.200.000đ/năm, mỗi kỳ 15 tín chỉ ≈ 8.100.000đ, 18 tín chỉ ≈ 9.720.000đ. Khối V (600.000đ/tín chỉ): khoảng 18.000.000đ/năm, mỗi kỳ 15 tín chỉ ≈ 9.000.000đ, 18 tín chỉ ≈ 10.800.000đ. Ngành Triết học: 0đ (miễn học phí). Nguồn: Dự kiến theo TT06/2026/TT-BGDĐT + tuyensinh.husc.edu.vn.",
        "text_plain": "Uoc tinh hoc phi 2026 theo khoi nganh: VII 7.9-9.5 trieu/ky, IV 8.1-9.7 trieu/ky, V 9.0-10.8 trieu/ky. Triet hoc mien hoc phi.",
        "summary": "Ước tính học phí/kỳ 2026 theo khối: VII ~7.9-9.5 triệu, IV ~8.1-9.7 triệu, V ~9.0-10.8 triệu. Triết học miễn.",
        "metadata": {
            "topic": "uoc_tinh_hoc_phi",
            "base_price_khoi_VII": 525000,
            "base_price_khoi_IV": 540000,
            "base_price_khoi_V": 600000,
            "credits_min": 15,
            "credits_max": 18,
            "unit": "Khoa_Hoc",
            "issuer": "Dai_Hoc_Hue",
            "audience": "thi_sinh",
            "info_type": "hoc_phi",
            "effective_date": "2026-06-01",
            "crawl_date": "2026-05-18T00:00:00Z",
            "year": "2026",
            "data_year": 2026,
            "expired": False,
            "is_superseded": False,
            "freshness_score": 1.0,
            "version": 2,
            "school": "HUSC",
            "supersedes": "husc_uoc_tinh_hoc_phi_ky_2025",
            "source": "Thông tin tuyển sinh HUSC 2026 + Thông tư 06/2026/TT-BGDĐT + tuyensinh.husc.edu.vn",
            "source_type": "jsonl",
            "faq_type": "hoc_phi_va_chi_phi",
        },
        "breadcrumbs": ["Đại học Huế (HUSC) - Tuyển sinh 2026", "Hoc Phi"],
        "prev_chunk_id": "husc_hoc_phi_tin_chi_2026",
        "next_chunk_id": "husc_policy_hoc_phi_2026",
        "sparse_terms": ["ước", "tính", "học", "phí", "khối", "VII", "IV", "V", "525", "540", "600", "000", "tín", "chỉ", "kỳ", "triết", "học", "miễn", "đồng", "trường", "đại", "khoa", "huế", "2026"],
    },
    {
        "id": "husc_policy_hoc_phi_2026",
        "doc_id": "husc_tuyen_sinh_2026",
        "chunk_id": 37,
        "text": "### husc policy hoc phi 2026\n\nThông tin Học phí năm 2026 tại Trường Đại học Khoa học - Đại học Huế (theo Thông tư 06/2026/TT-BGDĐT và dự kiến tuyển sinh):\n\n## 1. Đơn giá tín chỉ theo khối ngành (năm 2026)\n- **Khối VII** (Hán-Nôm, Lịch sử, Văn học, Báo chí, Xã hội học...): 525.000 VNĐ/tín chỉ.\n- **Khối IV** (Vật lý, Hóa học, Sinh học, Khoa học môi trường...): 540.000 VNĐ/tín chỉ.\n- **Khối V** (Kỹ thuật, CNTT, KTPM, Kiến trúc, Điện tử...): 600.000 VNĐ/tín chỉ.\n- **Triết học:** Miễn học phí (0 đồng/năm).\n\nSo với năm 2025, mức tăng theo khối: Khối VII tăng ~19.3% (440.000→525.000), Khối IV tăng ~6.9% (505.000→540.000), Khối V tăng ~9.6% (545.000-580.000→600.000).\n\n## 2. Ước tính học phí theo kỳ\nSinh viên thường đăng ký từ 15 đến 18 tín chỉ mỗi học kỳ:\n- Khối VII: 7.875.000đ - 9.450.000đ/kỳ\n- Khối IV: 8.100.000đ - 9.720.000đ/kỳ\n- Khối V: 9.000.000đ - 10.800.000đ/kỳ\n\n*Lưu ý: Sinh viên đóng tiền thực tế dựa trên số lượng tín chỉ đăng ký trên hệ thống.*\n\nNguồn: Thông tư 06/2026/TT-BGDĐT, Thông tin tuyển sinh HUSC 2026, tuyensinh.husc.edu.vn/thongbao.php?id=63-73.",
        "text_plain": "Hoc phi 2026 theo khoi nganh: VII 525.000d/tc, IV 540.000d/tc, V 600.000d/tc. Tang so voi 2025: VII +19.3%, IV +6.9%, V +9.6%. Triet hoc mien hoc phi. Nguon TT06/2026 va HUSC website.",
        "summary": "Học phí 2026 theo khối: VII 525.000đ/tín chỉ (+19.3%), IV 540.000đ/tín chỉ (+6.9%), V 600.000đ/tín chỉ (+9.6%). Triết học miễn. Nguồn: TT06/2026 + HUSC.",
        "metadata": {
            "category": "financial",
            "year": 2026,
            "data_year": 2026,
            "topic": "hoc_phi",
            "audience": "thi_sinh",
            "info_type": "thong_tin_chung",
            "effective_date": "2026-06-01",
            "crawl_date": "2026-05-18T00:00:00Z",
            "expired": False,
            "is_superseded": False,
            "freshness_score": 1.0,
            "version": 2,
            "school": "HUSC",
            "supersedes": "husc_policy_hoc_phi_2025",
            "price_khoi_VII": 525000,
            "price_khoi_IV": 540000,
            "price_khoi_V": 600000,
            "growth_rate_khoi_VII_percent": 19.3,
            "growth_rate_khoi_IV_percent": 6.9,
            "growth_rate_khoi_V_percent": 9.6,
            "unit": "Khoa_Hoc",
            "issuer": "Dai_Hoc_Hue",
            "source": "Thông tin tuyển sinh HUSC 2026 + Thông tư 06/2026/TT-BGDĐT + tuyensinh.husc.edu.vn",
            "source_type": "jsonl",
            "faq_type": "hoc_phi_va_chi_phi",
        },
        "breadcrumbs": ["Đại học Huế (HUSC) - Tuyển sinh 2026"],
        "prev_chunk_id": "husc_uoc_tinh_hoc_phi_ky_2026",
        "next_chunk_id": "",
        "sparse_terms": ["học", "phí", "khối", "VII", "IV", "V", "525", "540", "600", "000", "tín", "chỉ", "tăng", "triết", "học", "miễn", "trường", "đại", "khoa", "huế", "2026", "thông", "tư"],
    },
]


def main():
    # Append to chunked_10.jsonl
    with open(BASE / "chunked_10.jsonl", "a", encoding="utf-8") as f:
        for chunk in chunks_10:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Appended {len(chunks_10)} chunks to chunked_10.jsonl")

    # Append to chunked_10_enhanced.jsonl
    with open(BASE / "chunked_10_enhanced.jsonl", "a", encoding="utf-8") as f:
        for chunk in chunks_10_enhanced:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Appended {len(chunks_10_enhanced)} chunks to chunked_10_enhanced.jsonl")

    # Validate
    for fname in ["chunked_10.jsonl", "chunked_10_enhanced.jsonl"]:
        path = BASE / fname
        lines = path.read_text(encoding="utf-8").splitlines()
        valid = 0
        for line in lines:
            if not line.strip():
                continue
            try:
                json.loads(line)
                valid += 1
            except json.JSONDecodeError as e:
                print(f"INVALID JSON in {fname}: {e}")
        print(f"{fname}: {valid} valid lines")


if __name__ == "__main__":
    main()
