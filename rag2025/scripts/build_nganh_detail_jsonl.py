"""Build rag2025/data/chunked/nganh_detail_2026.jsonl

One chunk per ngành from the 28 raw HTML files in rag2025/data/raw/nganh_*.html.
Uses BeautifulSoup to strip nav/footer/script/style and extract body text.

Chunk schema:
  chunk_id   : "nganh_detail_{code}_2026"
  text       : plain text extracted from body
  metadata   : {
      source_url, notification_id, data_year, info_type,
      chunk_method, school, audience, major_code,
      approved_by, approved_date
  }
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from loguru import logger

SCRIPTS_DIR = Path(__file__).resolve().parent  # rag2025/scripts/
RAG2025_DIR = SCRIPTS_DIR.parent               # rag2025/
RAW_DIR = RAG2025_DIR / "data" / "raw"
OUTPUT_FILE = RAG2025_DIR / "data" / "chunked" / "nganh_detail_2026.jsonl"
BASE_URL = "https://tuyensinh.husc.edu.vn/nganhdaotao-dh.php?manganh={code}"

STRIP_TAGS = {"nav", "footer", "header", "script", "style", "noscript", "aside"}


def _extract_text(html_bytes: bytes) -> str:
    """Strip nav/footer/script/style; return clean body text."""
    soup = BeautifulSoup(html_bytes, "html.parser")

    # Remove unwanted structural tags
    for tag in soup.find_all(STRIP_TAGS):
        tag.decompose()

    # Prefer <main> or <article> if present, else fall back to <body>
    body = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"content|main|body", re.I))
        or soup.find("body")
        or soup
    )

    text = body.get_text(separator="\n")
    # Collapse excessive blank lines
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def build_chunk(code: str, html_bytes: bytes) -> dict:
    text = _extract_text(html_bytes)
    return {
        "chunk_id": f"nganh_detail_{code}_2026",
        "text": text,
        "metadata": {
            "source_url": BASE_URL.format(code=code),
            "notification_id": 74,
            "data_year": 2026,
            "info_type": "nganh",
            "chunk_method": "web_crawl_canonical_v2",
            "school": "HUSC",
            "audience": "thi_sinh",
            "major_code": code,
            "approved_by": "co_tuyen_sinh",
            "approved_date": "2026-05-26",
        },
    }


def main() -> int:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    html_files = sorted(RAW_DIR.glob("nganh_*.html"))
    if not html_files:
        logger.error(f"No nganh_*.html files found in {RAW_DIR}")
        return 1

    chunks: list[dict] = []
    for html_path in html_files:
        # Extract code from filename: nganh_{code}.html
        code = html_path.stem.removeprefix("nganh_")
        html_bytes = html_path.read_bytes()
        chunk = build_chunk(code, html_bytes)
        text_len = len(chunk["text"])
        logger.info(f"code={code} text_len={text_len}")
        chunks.append(chunk)

    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(chunks)} chunks to {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
