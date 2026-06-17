"""
Incremental crawl: fetch new thongbao from tuyensinh.husc.edu.vn
Starting from max_id+1, stop when HTTP 404.
Then run IngestPipeline to chunk -> embed -> LanceDB -> graph.

Usage:
    cd rag2025
    python scripts/crawl_incremental.py              # crawl new only
    python scripts/crawl_incremental.py --force-id 74  # force crawl specific ID
    python scripts/crawl_incremental.py --dry-run    # check for new IDs without ingesting
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
    from bs4 import BeautifulSoup
except ImportError:
    print("ERROR: Missing dependencies. Run: pip install httpx beautifulsoup4")
    sys.exit(1)

from loguru import logger
from src.infrastructure.auto_crawl import CrawlerState, IngestPipeline

BASE_URL = "https://tuyensinh.husc.edu.vn/thongbao.php?id={id}"
MAX_CONSECUTIVE_404 = 3  # Stop after 3 consecutive 404s


def crawl_fn() -> list[tuple[str, str]]:
    """Fetch new thongbao pages starting from max_id + 1."""
    state = CrawlerState.load()
    start_id = state.max_id + 1
    pages = []
    consecutive_404 = 0

    logger.info(f"Starting crawl from id={start_id} (max_id={state.max_id})")

    for tid in range(start_id, start_id + 50):  # Max 50 new pages per run
        url = BASE_URL.format(id=tid)
        try:
            resp = httpx.get(url, timeout=30, follow_redirects=True)
            if resp.status_code == 404 or "không tìm thấy" in resp.text.lower():
                consecutive_404 += 1
                logger.debug(f"404/not-found: {url} (consecutive: {consecutive_404})")
                if consecutive_404 >= MAX_CONSECUTIVE_404:
                    logger.info(f"Stopping after {MAX_CONSECUTIVE_404} consecutive 404s")
                    break
                continue
            consecutive_404 = 0
            soup = BeautifulSoup(resp.text, "html.parser")
            # Extract main content (adjust selector based on site structure)
            content_div = (
                soup.select_one(".content-detail")
                or soup.select_one("article")
                or soup.find("body")
            )
            text = content_div.get_text(separator="\n", strip=True) if content_div else resp.text
            pages.append((url, text))
            logger.info(f"Crawled: {url} ({len(text)} chars)")
        except Exception as e:
            logger.warning(f"Failed to crawl {url}: {e}")
            consecutive_404 += 1

    # Update max_id
    if pages:
        max_crawled = max(int(url.split("id=")[1]) for url, _ in pages)
        state.max_id = max_crawled
        state.last_crawl = datetime.now(timezone.utc).isoformat()
        state.save()
        logger.info(f"Updated max_id to {max_crawled}")
    else:
        logger.info("No new pages found")

    return pages


def dry_run():
    """Check for new IDs without ingesting."""
    state = CrawlerState.load()
    start_id = state.max_id + 1
    logger.info(f"DRY RUN: Checking for new thongbao starting from id={start_id}")

    found = []
    consecutive_404 = 0
    for tid in range(start_id, start_id + 10):
        url = BASE_URL.format(id=tid)
        try:
            resp = httpx.get(url, timeout=15, follow_redirects=True)
            if resp.status_code == 404 or "không tìm thấy" in resp.text.lower():
                consecutive_404 += 1
                if consecutive_404 >= MAX_CONSECUTIVE_404:
                    break
                continue
            consecutive_404 = 0
            found.append(tid)
            logger.info(f"Found: id={tid} ({url})")
        except Exception as e:
            logger.warning(f"Error checking {url}: {e}")

    if found:
        logger.info(f"DRY RUN: Found {len(found)} new pages: {found}")
    else:
        logger.info(f"DRY RUN: No new pages found (latest is id={state.max_id})")
    return found


def main():
    parser = argparse.ArgumentParser(description="Incremental crawl for HUSC thongbao")
    parser.add_argument("--dry-run", action="store_true", help="Check for new IDs without ingesting")
    parser.add_argument("--force-id", type=int, help="Force crawl a specific ID")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    if args.force_id:
        url = BASE_URL.format(id=args.force_id)
        logger.info(f"Force crawling id={args.force_id}")
        try:
            resp = httpx.get(url, timeout=30, follow_redirects=True)
            soup = BeautifulSoup(resp.text, "html.parser")
            content_div = soup.select_one(".content-detail") or soup.find("body")
            text = content_div.get_text(separator="\n", strip=True) if content_div else resp.text
            pages = [(url, text)]
            state = CrawlerState.load()
            pipeline = IngestPipeline(state)
            result = pipeline.run(crawl_fn=lambda: pages)
            logger.info(f"Force crawl result: {result}")
        except Exception as e:
            logger.error(f"Force crawl failed: {e}")
        return

    state = CrawlerState.load()
    pipeline = IngestPipeline(state)
    result = pipeline.run(crawl_fn=crawl_fn)
    logger.info(f"Pipeline result: {result}")


if __name__ == "__main__":
    main()
