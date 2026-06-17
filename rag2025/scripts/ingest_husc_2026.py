"""Ingest HUSC 2026 canonical data → JSONL chunks for LanceDB.

Reads data/raw/husc_2026_canonical.json and produces:
  data/chunked/husc_2026_chunks.jsonl

Each program gets a chunk like husc_nganh_<code>_2026.
Plus supplementary chunks for overview, info, policies, etc.
"""
import json
import re
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent
CANONICAL = ROOT / "data" / "raw" / "husc_2026_canonical.json"
OUTPUT = ROOT / "data" / "chunked" / "husc_2026_chunks.jsonl"


def make_nganh_chunk(prog: dict, tuition: dict, year: int) -> dict:
    code = prog["code"]
    khoi = prog["khoi_hoc_phi"]
    hoc_phi = tuition.get(khoi, {}).get("amount", 0)
    to_hop_str = ", ".join(prog["to_hop"])
    text = (
        f"Ngành {prog['name']} — Mã ngành: {code}. "
        f"Chỉ tiêu năm {year}: {prog['chi_tieu']}. "
        f"Tổ hợp xét tuyển: {to_hop_str}. "
        f"Khối học phí: {khoi.replace('khoi_', 'Khối ').upper()} ({hoc_phi:,} VNĐ/tín chỉ). "
        f"Thời gian đào tạo: {prog['duration_years']} năm."
    )
    summary = (
        f"{prog.get('name_short', prog['name'])}, mã {code}, "
        f"chỉ tiêu {year}: {prog['chi_tieu']}, "
        f"tổ hợp {to_hop_str}, "
        f"khối {khoi.replace('khoi_', '').upper()} — {hoc_phi:,}đ/tín chỉ"
    )
    return {
        "chunk_id": f"husc_nganh_{code}_{year}",
        "text": text,
        "source": f"tuyensinh.husc.edu.vn ({year})",
        "faq_type": "nganh_2026",
        "metadata": {
            "data_year": year,
            "is_superseded": False,
            "nganh_code": code,
            "nganh_name": prog["name"],
            "chi_tieu": prog["chi_tieu"],
            "to_hop": prog["to_hop"],
            "khoi_hoc_phi": khoi,
            "hoc_phi_vnd_per_credit": hoc_phi,
            "summary": summary,
        }
    }


def make_overview_chunk(data: dict) -> dict:
    year = data["year"]
    total_chi_tieu = sum(p["chi_tieu"] for p in data["programs"])
    nganh_count = len(data["programs"])
    pt_names = ", ".join(pt["name"] for pt in data["phuong_thuc"])
    text = (
        f"Trường {data['school']['name']} — Mã trường: {data['school']['code']}. "
        f"Năm {year}: tuyển sinh {nganh_count} ngành/chương trình, "
        f"tổng chỉ tiêu {total_chi_tieu}. "
        f"5 phương thức xét tuyển: {pt_names}. "
        f"Hotline: {data['school']['hotline']}. Email: {data['school']['email']}. "
        f"Địa chỉ: {data['school']['address']}."
    )
    return {
        "chunk_id": f"tuyensinh_overview_{year}",
        "text": text,
        "source": f"tuyensinh.husc.edu.vn ({year})",
        "faq_type": "overview_2026",
        "metadata": {
            "data_year": year,
            "is_superseded": False,
            "total_chi_tieu": total_chi_tieu,
            "nganh_count": nganh_count,
            "summary": f"HUSC {year}: {nganh_count} ngành, {total_chi_tieu} chỉ tiêu, 5 phương thức",
        }
    }


def make_husc_info_chunk(data: dict) -> dict:
    s = data["school"]
    text = (
        f"Mã trường: {s['code']}. "
        f"Tên trường: {s['name']}. "
        f"Địa chỉ: {s['address']}. "
        f"Hotline: {s['hotline']}. "
        f"Email: {s['email']}."
    )
    return {
        "chunk_id": "husc_info",
        "text": text,
        "source": "tuyensinh.husc.edu.vn",
        "faq_type": "school_info",
        "metadata": {
            "data_year": 2026,
            "summary": "HUSC — mã DHT, 77 Nguyễn Huệ Huế",
        }
    }


def make_hocphi_chunk(data: dict) -> dict:
    year = data["year"]
    lines = [f"Học phí năm {year} theo Thông tư 06/2026/TT-BGDĐT:"]
    for khoi_key, khoi_data in data["tuition"].items():
        khoi_label = khoi_key.replace("khoi_", "Khối ").upper()
        amount = khoi_data["amount"]
        codes = ", ".join(khoi_data["nganh_codes"])
        lines.append(f"- {khoi_label}: {amount:,} VNĐ/tín chỉ (mã ngành: {codes})")
    text = "\n".join(lines)
    return {
        "chunk_id": f"hocphi_{year}",
        "text": text,
        "source": f"tuyensinh.husc.edu.vn ({year})",
        "faq_type": "hoc_phi_2026",
        "metadata": {
            "data_year": year,
            "summary": "Học phí 2026: Khối IV 540.000đ/TC, Khối V 600.000đ/TC, Khối VII 525.000đ/TC",
        }
    }


def make_phuongthuc_chunk(data: dict) -> dict:
    year = data["year"]
    lines = [f"5 phương thức xét tuyển năm {year}:"]
    for i, pt in enumerate(data["phuong_thuc"], 1):
        lines.append(f"{i}. {pt['name']}: {pt['mo_ta']}")
    text = "\n".join(lines)
    return {
        "chunk_id": f"phuongthuc_xettuyen_{year}",
        "text": text,
        "source": f"tuyensinh.husc.edu.vn ({year})",
        "faq_type": "phuong_thuc_2026",
        "metadata": {
            "data_year": year,
            "summary": "5 phương thức: XTT, THPT, HB, NL, KH",
        }
    }


def make_diem_cong_chunk(data: dict) -> dict:
    dc = data["diem_cong"]
    lines = [f"Điểm cộng ưu tiên năm {data['year']}: Tối đa {dc['max_pct']}."]
    for item in dc["items"]:
        lines.append(f"- {item['type']}: {item['value']}")
    lines.append(f"Lưu ý: {dc['note']}")
    text = "\n".join(lines)
    return {
        "chunk_id": f"chinhsach_{data['year']}",
        "text": text,
        "source": f"tuyensinh.husc.edu.vn ({data['year']})",
        "faq_type": "chinh_sach_2026",
        "metadata": {
            "data_year": data["year"],
            "summary": "Điểm cộng tối đa 10%, giải quốc gia 3đ/30, chứng chỉ NN 0.5-1.5đ/30",
        }
    }


def make_non_offered_chunk(data: dict) -> dict:
    year = data["year"]
    non_list = ", ".join(data["non_offered"])
    text = (
        f"Năm {year}, HUSC KHÔNG tuyển sinh các ngành sau: {non_list}. "
        f"Cụ thể: "
        f"Toán học chỉ có trong tổ hợp môn, không phải ngành đào tạo. "
        f"Sinh học thuần không tuyển (chỉ có Công nghệ sinh học 7420201). "
        f"Địa lý chỉ có trong tổ hợp môn, không phải ngành đào tạo. "
        f"Khoa học máy tính không tuyển (chỉ có CNTT 7480201, KTPM 7480103, QTPADL 7480107TD). "
        f"Tin học không tuyển (không có ngành Tin học thuần)."
    )
    return {
        "chunk_id": f"nganh_khong_tuyen_{year}",
        "text": text,
        "source": f"tuyensinh.husc.edu.vn ({year})",
        "faq_type": "non_offered_2026",
        "metadata": {
            "data_year": year,
            "non_offered": data["non_offered"],
            "summary": "2026 không tuyển: Toán, Sinh học, Địa lý, KHMT, Tin học",
        }
    }


def make_tohop_chunks(data: dict) -> list:
    tohop_map = {
        "A00": "Toán, Vật lý, Hóa học",
        "A01": "Toán, Vật lý, Tiếng Anh",
        "A02": "Toán, Hóa học, Sinh học",
        "A04": "Toán, Vật lý, Khoa học tự nhiên",
        "A09": "Toán, Hóa học, Giáo dục công dân",
        "B00": "Toán, Hóa học, Sinh học",
        "B03": "Toán, Sinh học, Tiếng Anh",
        "B08": "Toán, Hóa học, Tiếng Anh",
        "C00": "Ngữ văn, Lịch sử, Địa lý",
        "C01": "Ngữ văn, Toán, Vật lý",
        "C02": "Ngữ văn, Toán, Hóa học",
        "C03": "Ngữ văn, Lịch sử, Địa lý",
        "C04": "Ngữ văn, Toán, Khoa học xã hội",
        "C14": "Ngữ văn, Toán, Khoa học tự nhiên",
        "C19": "Ngữ văn, Lịch sử, Giáo dục công dân",
        "D01": "Ngữ văn, Toán, Tiếng Anh",
        "D06": "Ngữ văn, Toán, Khoa học xã hội",
        "D07": "Toán, Hóa học, Tiếng Anh",
        "D10": "Toán, Khoa học tự nhiên, Tiếng Anh",
        "D14": "Ngữ văn, Lịch sử, Tiếng Anh",
        "D66": "Toán, Vật lý, Giáo dục công dân",
        "D84": "Toán, Khoa học tự nhiên, Tiếng Anh",
        "X01": "Toán, Khoa học tự nhiên, Tiếng Anh",
        "X02": "Toán, Vật lý, Hóa học",
        "X06": "Toán, Vật lý, Khoa học tự nhiên",
        "X21": "Toán, Khoa học xã hội, Tiếng Anh",
        "X25": "Toán, Khoa học tự nhiên, Tiếng Anh",
        "X26": "Toán, Vật lý, Tiếng Anh",
        "X70": "Ngữ văn, Khoa học xã hội, Tiếng Anh",
        "X78": "Toán, Vật lý, Giáo dục công dân",
        "V00": "Toán, Vật lý, Vẽ mỹ thuật",
        "V01": "Toán, Tiếng Anh, Vẽ mỹ thuật",
        "V02": "Ngữ văn, Toán, Vẽ mỹ thuật",
        "V12": "Ngữ văn, Tiếng Anh, Vẽ mỹ thuật",
    }
    year = data["year"]
    chunks = []
    for code, desc in tohop_map.items():
        using = [p["name_short"] for p in data["programs"] if code in p["to_hop"]]
        using_str = ", ".join(using) if using else "—"
        text = f"Tổ hợp xét tuyển {code}: {desc}. Các ngành dùng tổ hợp này tại HUSC {year}: {using_str}."
        chunks.append({
            "chunk_id": f"tohop_{code.lower()}",
            "text": text,
            "source": f"tuyensinh.husc.edu.vn ({year})",
            "faq_type": "to_hop_2026",
            "metadata": {
                "data_year": year,
                "to_hop_code": code,
                "subjects": desc,
                "summary": f"{code}: {desc} — {len(using)} ngành",
            }
        })
    return chunks


def main():
    with open(CANONICAL, encoding="utf-8") as f:
        data = json.load(f)

    year = data["year"]
    tuition = data["tuition"]
    chunks = []

    for prog in data["programs"]:
        chunks.append(make_nganh_chunk(prog, tuition, year))

    chunks.append(make_overview_chunk(data))
    chunks.append(make_husc_info_chunk(data))
    chunks.append(make_hocphi_chunk(data))
    chunks.append(make_phuongthuc_chunk(data))
    chunks.append(make_diem_cong_chunk(data))
    chunks.append(make_non_offered_chunk(data))
    chunks.extend(make_tohop_chunks(data))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    nganh_count = sum(1 for c in chunks if c["chunk_id"].startswith("husc_nganh_"))
    print(f"Written {len(chunks)} chunks to {OUTPUT}")
    print(f"  - Ngành chunks: {nganh_count}")
    print(f"  - Supplementary chunks: {len(chunks) - nganh_count}")


if __name__ == "__main__":
    main()
