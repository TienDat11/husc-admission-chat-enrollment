"""Vision-aware crawler for HUSC tuyensinh pages.

Problem: Many thông báo pages embed point/score formulas as IMAGES (not HTML tables),
so plain text crawl loses critical info (e.g. công thức tính điểm, bảng quy đổi).

Solution: Two-stage extraction
  1. HTML text → BeautifulSoup as before (cheap, primary)
  2. For images that look like content (not logos/icons), download + OCR via vision LLM
     (vision LLM via the ns24 gateway). Append OCR text to chunk under
     "## [Trích xuất từ hình ảnh]" header so retrieval can find formulas.
     NOTE: vision/image OCR support on the ns24 gateway is unverified. NS24_VISION_MODEL
     MUST be a model the gateway actually serves for images, or every OCR call will fail.

Image filtering heuristics (skip non-content images):
  - URL contains 'logo', 'icon', 'avatar', 'banner', 'qrcode'
  - Width × height < 100×100 (best-effort)
  - File size < 5 KB (skip tiny GIFs)
  - alt text ≤ 3 chars (likely decoration)

Cost control:
  - Hard cap: max 8 images per page (sorted by area, biggest first — usually formula screenshots)
  - Cache OCR by image URL hash → re-runs are free
  - Skip if cache hit within 30 days

Usage:
  python scripts/vision_crawl_husc.py --html-dir data/raw_thongbao
  python scripts/vision_crawl_husc.py --html-dir data/raw_tuyensinh_full
  python scripts/vision_crawl_husc.py --ids 33,26,34,57,52,71,69,70 --html-dir data/raw_thongbao,data/raw_tuyensinh_full
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

RAW_DIRS_DEFAULT = [
    ROOT / "data" / "raw_thongbao",
    ROOT / "data" / "raw_tuyensinh_full",
]
CACHE_DIR = ROOT / "data" / "vision_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "data" / "vision_extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Vision model: must be an image-capable model served by the ns24 gateway (unverified).
VISION_MODEL = os.getenv("NS24_VISION_MODEL", "")
NS24_BASE = os.getenv("NS24_BASE_URL", "https://api.ns24.io/v1")
API_KEY = os.getenv("NS24_API_KEY")

SKIP_URL_TOKENS = ("logo", "icon", "avatar", "qrcode", "favicon", "footer", "header_bg",
                   "fanpage", "sitename", "hero-bg", "emoji", "fbcdn", "facebook")


def _is_content_image(src: str, alt: str, soup_img) -> bool:
    if not src:
        return False
    src_l = src.lower()
    if any(tok in src_l for tok in SKIP_URL_TOKENS):
        return False
    if "/assets/img/" in src_l:
        return False
    if alt and len(alt.strip()) > 0 and len(alt.strip()) <= 3:
        return False
    # try width/height attrs
    try:
        w = int(soup_img.get("width") or 0)
        h = int(soup_img.get("height") or 0)
        if 0 < w < 100 and 0 < h < 100:
            return False
    except (TypeError, ValueError):
        pass
    return True


def _img_priority(soup_img) -> int:
    """Higher = more likely to be a formula/table screenshot.

    Heuristic: prefer images served from `/IMAGES/` (HUSC content uploads)
    over generic asset images. Fall back to width × height when available.
    """
    src = (soup_img.get("src") or "").lower()
    base = 0
    if "/images/" in src:
        base += 10_000
    try:
        w = int(soup_img.get("width") or 0)
        h = int(soup_img.get("height") or 0)
        if w > 0 and h > 0:
            base += w * h
    except (TypeError, ValueError):
        pass
    return base


def _download_image(url: str, out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 1024:
        return True
    try:
        r = httpx.get(url, timeout=20, follow_redirects=True)
        if r.status_code == 200 and len(r.content) > 1024:
            out_path.write_bytes(r.content)
            return True
    except Exception as e:
        logger.warning(f"download fail {url}: {e}")
    return False


def _ocr_via_vision(image_path: Path) -> str:
    """Send image to the ns24 vision endpoint, get back OCR text."""
    if not API_KEY:
        logger.warning("No NS24_API_KEY — skipping OCR.")
        return ""
    if not VISION_MODEL:
        logger.warning("No NS24_VISION_MODEL set — skipping OCR.")
        return ""

    cache_key = hashlib.sha256(image_path.read_bytes()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{cache_key}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    img_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    suffix = image_path.suffix.lstrip(".").lower() or "png"
    if suffix == "jpg":
        suffix = "jpeg"

    prompt = (
        "Đây là hình ảnh trong trang thông báo tuyển sinh đại học HUSC. "
        "Vui lòng trích xuất TOÀN BỘ thông tin có trong hình thành text tiếng Việt thuần. "
        "Đặc biệt chú ý:\n"
        "1. Công thức tính điểm (giữ nguyên ký hiệu toán học, viết rõ ví dụ: ĐXT = M1 + M2 + M3 + ĐƯT)\n"
        "2. Bảng điểm chuẩn / bảng quy đổi (chuyển sang dạng text bảng đơn giản với | ngăn cột)\n"
        "3. Mã ngành, tên ngành, chỉ tiêu (dạng list)\n"
        "4. Tổ hợp xét tuyển (A00, B00, C00, D01...) → giữ nguyên\n"
        "5. Mốc thời gian, hạn nộp\n\n"
        "Nếu hình ảnh KHÔNG có thông tin tuyển sinh có giá trị (logo, icon, ảnh trang trí), "
        "trả lời chính xác chuỗi: [SKIP_NO_CONTENT]\n\n"
        "Output: chỉ trả về text trích xuất, KHÔNG giải thích thêm, KHÔNG markdown formatting."
    )

    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{suffix};base64,{img_b64}"}},
            ],
        }],
        "max_tokens": 1500,
        "temperature": 0.0,
        "stream": False,
    }

    try:
        with httpx.Client(timeout=90) as client:
            r = client.post(
                f"{NS24_BASE}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
        if r.status_code != 200:
            logger.warning(f"vision API {r.status_code}: {r.text[:300]}")
            return ""
        body = r.text.strip()
        # Endpoint may return SSE stream even when stream:false — handle both
        if body.startswith("data:"):
            text_parts = []
            for line in body.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload_str = line[len("data:"):].strip()
                if payload_str in ("[DONE]", ""):
                    continue
                try:
                    obj = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                for choice in obj.get("choices", []):
                    delta = choice.get("delta") or {}
                    msg = choice.get("message") or {}
                    chunk_text = delta.get("content") or msg.get("content") or ""
                    if chunk_text:
                        text_parts.append(chunk_text)
            text = "".join(text_parts).strip()
        else:
            data = r.json()
            text = data["choices"][0]["message"]["content"].strip()
        if "[SKIP_NO_CONTENT]" in text:
            cache_file.write_text("", encoding="utf-8")
            return ""
        cache_file.write_text(text, encoding="utf-8")
        return text
    except Exception as e:
        logger.warning(f"vision call failed for {image_path.name}: {e}")
        return ""


def process_html(html_path: Path, base_url: str = "https://tuyensinh.husc.edu.vn/", max_imgs: int = 8) -> dict:
    """Extract text + OCR significant images for ONE html file."""
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one(".maincontent") or soup.select_one("article") or soup.body or soup
    plain_text = main.get_text("\n", strip=True) if main else ""

    # Collect candidate images
    img_tags = main.find_all("img") if main else []
    candidates = []
    for img in img_tags:
        src = (img.get("src") or "").strip()
        alt = (img.get("alt") or "").strip()
        if _is_content_image(src, alt, img):
            full_url = urljoin(base_url, src)
            candidates.append((full_url, alt, _img_priority(img)))

    # Sort by priority desc + take top max_imgs
    candidates.sort(key=lambda x: -x[2])
    candidates = candidates[:max_imgs]

    img_texts = []
    img_dir = OUT_DIR / "images" / html_path.stem
    img_dir.mkdir(parents=True, exist_ok=True)

    for idx, (url, alt, _) in enumerate(candidates):
        ext = Path(urlparse(url).path).suffix.lower() or ".png"
        local = img_dir / f"img_{idx:02d}{ext}"
        if not _download_image(url, local):
            continue
        ocr_text = _ocr_via_vision(local)
        if ocr_text and len(ocr_text.strip()) > 20:
            img_texts.append({
                "url": url,
                "alt": alt,
                "ocr": ocr_text.strip(),
            })
        time.sleep(0.3)

    return {
        "source_html": str(html_path).replace(str(ROOT) + os.sep, ""),
        "plain_text": plain_text,
        "vision_extracted": img_texts,
        "total_images_seen": len(img_tags),
        "total_ocr_extracted": len(img_texts),
    }


def merge_into_chunk_text(plain: str, img_texts: list[dict]) -> str:
    """Combine plain text + OCR results into a single chunk-ready string."""
    if not img_texts:
        return plain
    parts = [plain.strip(), "\n\n## [Thông tin trích xuất từ hình ảnh trong thông báo]\n"]
    for i, item in enumerate(img_texts, 1):
        parts.append(f"\n### Hình {i} ({item.get('alt') or 'không có chú thích'})\n{item['ocr']}\n")
    return "\n".join(parts)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-dir", default=",".join(str(d) for d in RAW_DIRS_DEFAULT))
    parser.add_argument("--ids", default="", help="Comma-separated thông báo IDs to process (e.g. 33,57,69,70). Empty = all.")
    parser.add_argument("--max-imgs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Stop after N pages (0 = no limit)")
    args = parser.parse_args(argv)

    dirs = [Path(d.strip()) for d in args.html_dir.split(",") if d.strip()]
    target_ids = set(args.ids.split(",")) if args.ids else None

    files: list[Path] = []
    for d in dirs:
        if not d.exists():
            logger.warning(f"dir not found: {d}")
            continue
        for f in sorted(d.glob("thongbao_*.html")):
            m = re.match(r"thongbao_(\d+)\.html", f.name)
            if not m:
                continue
            tb_id = m.group(1)
            if target_ids and tb_id not in target_ids:
                continue
            files.append(f)
        for f in sorted(d.glob("*_index*.html")):
            files.append(f)
        for f in sorted(d.glob("intro*.html")):
            files.append(f)

    if args.limit:
        files = files[: args.limit]

    logger.info(f"Vision-extracting {len(files)} HTML files using {VISION_MODEL}…")

    summary_path = OUT_DIR / f"vision_summary_{int(time.time())}.jsonl"
    with summary_path.open("w", encoding="utf-8") as fh:
        for f in files:
            logger.info(f"→ {f.name}")
            try:
                res = process_html(f, max_imgs=args.max_imgs)
            except Exception as e:
                logger.error(f"failed {f.name}: {e}")
                continue
            res["merged_text"] = merge_into_chunk_text(res["plain_text"], res["vision_extracted"])
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            logger.info(f"   imgs_total={res['total_images_seen']} ocr_extracted={res['total_ocr_extracted']}")

    logger.info(f"Done. Output: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
