#!/usr/bin/env python3
"""Crawl HUSC thongbao IDs 41-64 gap fill for RAG fallback knowledge base."""

import os
import time
import httpx
from bs4 import BeautifulSoup
from datetime import datetime

BASE_URL = "https://tuyensinh.husc.edu.vn/thongbao.php?id={}"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE_DIR, "data", "raw_tuyensinh_full")
os.makedirs(OUT_DIR, exist_ok=True)

IDS = [41,42,43,44,45,46,47,48,51,52,54,55,56,57,58,59,60,61,62,63,64]
SKIP_IDS = {49, 50, 53}  # already crawled

LOG_FILE = os.path.join(OUT_DIR, "_crawl_log_thongbao_41_64.txt")
SUMMARY_FILE = os.path.join(OUT_DIR, "_summary_thongbao_41_64.md")

HEADERS = {"User-Agent": "Mozilla/5.0 HUSC-RAG-Crawler/1.0"}
TIMEOUT = 15.0

def detect_year(text: str) -> str:
    """Detect year from text content."""
    for y in ["2025", "2024", "2023", "2022", "2021", "2020"]:
        if y in text:
            return y
    return "unknown"

def is_useful(title: str, text: str) -> bool:
    """Determine if content is useful for admissions fallback."""
    combined = (title + " " + text).lower()
    keywords = [
        "điểm chuẩn", "chỉ tiêu", "lịch tuyển sinh", "phương thức",
        "xét tuyển", "nhập học", "hướng dẫn", "thông tin chung",
        "tuyển sinh", "đại học", "cao đẳng", "điểm", "nguyện vọng",
        "hồ sơ", "đăng ký", "thí sinh", "trúng tuyển"
    ]
    return any(kw in combined for kw in keywords)

def extract_content(html: str) -> tuple:
    """Extract title and main content from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else "(no title)"

    # Try maincontent div first
    main = soup.find("div", class_="maincontent")
    if not main:
        main = soup.find("div", class_="main-content")
    if not main:
        main = soup.find("div", id="maincontent")
    if not main:
        # fallback: article or body text
        main = soup.find("article") or soup.find("body")

    text = main.get_text(separator="\n", strip=True) if main else ""
    return title, text

def main():
    results = []
    log_lines = [f"Crawl started: {datetime.now().isoformat()}", f"Target IDs: {IDS}", ""]

    client = httpx.Client(headers=HEADERS, timeout=TIMEOUT, follow_redirects=True)

    for i, tid in enumerate(IDS):
        out_path = os.path.join(OUT_DIR, f"thongbao_{tid}.html")

        # Skip if file already exists
        if os.path.exists(out_path):
            log_lines.append(f"[{tid}] SKIP - file already exists")
            # Still extract info from existing file for summary
            with open(out_path, "r", encoding="utf-8", errors="replace") as f:
                html = f.read()
            title, text = extract_content(html)
            year = detect_year(title + " " + text)
            useful = is_useful(title, text)
            results.append({"id": tid, "title": title, "year": year, "useful": useful, "status": "exists"})
            continue

        url = BASE_URL.format(tid)
        try:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text

            # Save raw HTML
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)

            title, text = extract_content(html)
            year = detect_year(title + " " + text)
            useful = is_useful(title, text)

            # Check for empty/404/not-found pages
            lower_text = text.lower()
            if (len(text.strip()) < 50
                or "không tồn tại" in lower_text
                or "không tìm thấy" in lower_text
                or "404" in title):
                log_lines.append(f"[{tid}] SKIP - empty or not found: {title}")
                results.append({"id": tid, "title": title, "year": year, "useful": False, "status": "empty/404"})
            else:
                log_lines.append(f"[{tid}] OK - {title} ({year}) [{len(text)} chars]")
                results.append({"id": tid, "title": title, "year": year, "useful": useful, "status": "ok"})

        except httpx.HTTPStatusError as e:
            log_lines.append(f"[{tid}] ERROR HTTP {e.response.status_code}: {e}")
            results.append({"id": tid, "title": f"HTTP {e.response.status_code}", "year": "-", "useful": False, "status": "error"})
        except Exception as e:
            log_lines.append(f"[{tid}] ERROR: {type(e).__name__}: {e}")
            results.append({"id": tid, "title": f"Error: {type(e).__name__}", "year": "-", "useful": False, "status": "error"})

        # Polite sleep between requests
        if i < len(IDS) - 1:
            time.sleep(0.5)

    client.close()

    # Write log
    log_lines.append("")
    log_lines.append(f"Crawl finished: {datetime.now().isoformat()}")
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # Generate markdown summary
    ok_count = sum(1 for r in results if r["status"] == "ok")
    useful_count = sum(1 for r in results if r["useful"])
    error_count = sum(1 for r in results if r["status"] == "error")
    empty_count = sum(1 for r in results if r["status"] == "empty/404")
    exists_count = sum(1 for r in results if r["status"] == "exists")

    md = [
        "# Summary: ThongBao IDs 41-64 Crawl",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Total IDs processed:** {len(results)}",
        f"**New content:** {ok_count} | **Already existed:** {exists_count} | **Empty/404:** {empty_count} | **Errors:** {error_count}",
        f"**Useful for fallback:** {useful_count}/{len(results)}",
        "",
        "| ID | Title | Year | Useful | Status |",
        "|:--:|:------|:----:|:------:|:------:|",
    ]
    for r in results:
        title_short = r["title"][:60] + ("..." if len(r["title"]) > 60 else "")
        useful_mark = "yes" if r["useful"] else "no"
        md.append(f"| {r['id']} | {title_short} | {r['year']} | {useful_mark} | {r['status']} |")

    md.append("")
    md.append("## Useful Content IDs")
    useful_ids = [str(r["id"]) for r in results if r["useful"]]
    if useful_ids:
        md.append(f"IDs with admissions-relevant content: {', '.join(useful_ids)}")
    else:
        md.append("No useful content found.")

    md.append("")
    md.append("## Skipped IDs (49, 50, 53)")
    md.append("Pre-existing files in `data/raw_thongbao/` — not re-crawled.")

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # Print summary to stdout
    print("\n".join(md))
    print(f"\nLog saved to: {LOG_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
