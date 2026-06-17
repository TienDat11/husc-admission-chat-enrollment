"""Build husc_2026_vision_ocr.jsonl from the OCR text extracted by vision_crawl_husc.py.

Reads vision_summary_*.jsonl files in data/vision_extracted/, deduplicates by image URL,
attaches each OCR text to a chunk keyed by source_html ID + image hash, and writes a
chunk-friendly JSONL.

Each chunk:
  - chunk_id: husc_vision_tb{ID}_img{N}_2026
  - text: '[Trích xuất từ hình ảnh trong thông báo HUSC tuyển sinh]\n\n{ocr_text}\n\nNguồn: {source_url}'
  - metadata: data_year, info_type=vision_ocr, source_url, image_url, parent_html
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).parent.parent
SUMMARY_DIR = ROOT / "data" / "vision_extracted"
OUT_PATH = ROOT / "data" / "chunked" / "husc_2026_vision_ocr.jsonl"

# Pages confirmed to be 2026 official thông báo
TB_2026_IDS = {65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 40}

# Pages that are 2025 reference (allow as fallback context)
TB_2025_REF_IDS = {49, 50, 53, 5, 6, 7, 47, 48, 51, 56, 57, 58, 59, 60, 61, 62, 63, 64,
                   1, 2, 9, 10, 12, 17, 18, 19, 26, 28, 33, 34, 52, 7}


def _parse_tb_id(source_html: str) -> int | None:
    m = re.search(r"thongbao_(\d+)\.html", source_html)
    return int(m.group(1)) if m else None


def main() -> int:
    if not SUMMARY_DIR.exists():
        print(f"No vision_extracted dir: {SUMMARY_DIR}")
        return 1

    seen_url: set[str] = set()
    chunks: list[dict] = []

    for fp in sorted(SUMMARY_DIR.glob("vision_summary_*.jsonl")):
        with fp.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                src_html = r.get("source_html", "")
                tb_id = _parse_tb_id(src_html)
                vision_items = r.get("vision_extracted", [])

                for img_idx, x in enumerate(vision_items):
                    ocr = (x.get("ocr") or "").strip()
                    if len(ocr) < 80:
                        continue
                    url = x.get("url", "")
                    if url in seen_url:
                        continue
                    seen_url.add(url)

                    # Skip if OCR is just the school name + contact (low value)
                    body = ocr.lower()
                    if "trường đại học khoa học" in body and len(ocr) < 250:
                        # short header-only blurb — skip unless it has formula keywords
                        if not any(k in body for k in [
                            "công thức", "điểm xét", "đxt", "quy đổi", "tổ hợp",
                            "chỉ tiêu", "điểm chuẩn", "điểm cộng", "ưu tiên",
                            "xét tuyển thẳng", "vẽ", "ngoại ngữ", "ielts", "toefl",
                            "đăng ký", "thời gian", "bảng",
                        ]):
                            continue

                    fname = urlparse(url).path.split("/")[-1] or "img"
                    chunk_id_base = f"husc_vision_tb{tb_id}_img{img_idx:02d}_2026" if tb_id else f"husc_vision_misc_{len(chunks)}_2026"

                    # Determine year tag
                    if tb_id in TB_2026_IDS:
                        data_year = 2026
                    elif tb_id in TB_2025_REF_IDS:
                        data_year = 2025
                    else:
                        data_year = 2026  # default

                    text = (
                        "[Trích xuất từ hình ảnh trong thông báo tuyển sinh HUSC]\n\n"
                        f"{ocr}\n\n"
                        f"Nguồn: https://tuyensinh.husc.edu.vn/thongbao.php?id={tb_id}"
                        if tb_id else
                        f"[Trích xuất từ hình ảnh]\n\n{ocr}"
                    )
                    text += "\n\nLưu ý: Đây là thông tin trích từ ảnh. Để cập nhật mới nhất, vui lòng theo dõi https://tuyensinh.husc.edu.vn/thongbao.php"

                    chunks.append({
                        "chunk_id": chunk_id_base,
                        "text": text,
                        "summary": ocr[:200].replace("\n", " ") + ("…" if len(ocr) > 200 else ""),
                        "metadata": {
                            "data_year": data_year,
                            "info_type": "vision_ocr",
                            "source_url": f"https://tuyensinh.husc.edu.vn/thongbao.php?id={tb_id}" if tb_id else "",
                            "image_url": url,
                            "parent_html": src_html,
                            "chunk_method": "vision_ocr_claude_sonnet_4_6",
                            "approved_by": "vision_pipeline_v1",
                            "school": "HUSC",
                            "audience": "thí_sinh_2026",
                            "thongbao_id": tb_id,
                        },
                    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} OCR chunks to {OUT_PATH}")
    print()
    print("Chunks by data_year:")
    from collections import Counter
    yr = Counter(c["metadata"]["data_year"] for c in chunks)
    for y, n in yr.most_common():
        print(f"  {y}: {n}")
    print()
    print("First 5 chunk_ids:")
    for c in chunks[:5]:
        print(f"  - {c['chunk_id']} | {len(c['text'])}c | year={c['metadata']['data_year']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
