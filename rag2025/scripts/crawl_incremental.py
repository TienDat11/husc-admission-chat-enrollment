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
import json
import os
import re
import hashlib
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

# The site returns HTTP 200 even for non-existent IDs, rendering a stub page
# that contains this sentinel instead of a real 404. Detect it by string.
NOT_FOUND_SENTINELS = ("nội dung không tồn tại", "không tìm thấy")
# Real thongbao pages are >2KB of body text; empty stubs are ~630 chars.
MIN_CONTENT_CHARS = 1500

# M3-N1a: derive max thongbao-id from LanceDB (self-healing crawl state)
_THONGBAO_ID_RE = re.compile(r"thongbao[_-]?id?(\d+)", re.IGNORECASE)
_URL_ID_RE = re.compile(r"[?&]id=(\d+)")


def derive_max_id(db_uri: str, table: str = "husc_chunks") -> int:
    """Read max thongbao-id from LanceDB. Returns 0 if table missing/empty."""
    try:
        import lancedb
        db = lancedb.connect(db_uri)
        if table not in db.table_names():
            return 0
        t = db.open_table(table)
        rows = t.to_arrow().select(["id", "metadata_json"]).to_pylist()
        if not rows:
            return 0
        max_id = 0
        for row in rows:
            # (a) parse from id field: husc_thongbao_id73_2026 → 73
            m = _THONGBAO_ID_RE.search(row.get("id") or "")
            if m:
                max_id = max(max_id, int(m.group(1)))
            # (b) parse from metadata_json: notification_id or source_url?id=NN
            meta_raw = row.get("metadata_json") or "{}"
            try:
                meta = json.loads(meta_raw)
            except (json.JSONDecodeError, TypeError):
                meta = {}
            nid = meta.get("notification_id")
            if isinstance(nid, int) and nid > 0:
                max_id = max(max_id, nid)
            src_url = meta.get("source_url") or ""
            um = _URL_ID_RE.search(src_url)
            if um:
                max_id = max(max_id, int(um.group(1)))
        return max_id
    except Exception as e:
        logger.warning(f"derive_max_id failed (fallback 0): {e}")
        return 0


def _is_empty_page(html: str, body_text: str) -> bool:
    """True when the page is a not-found stub despite a 200 status."""
    low = html.lower()
    if any(s in low for s in NOT_FOUND_SENTINELS):
        return True
    return len(body_text.strip()) < MIN_CONTENT_CHARS


def crawl_fn() -> list[tuple[str, str]]:
    """Fetch new thongbao pages starting from max(state.max_id, derive_max_id) + 1."""
    state = CrawlerState.load()
    # M3-N1a: self-healing — derive max from LanceDB, take max with file state
    try:
        from config.settings import RAGSettings
        db_uri = RAGSettings().LANCEDB_URI
    except Exception:
        db_uri = str(Path(__file__).parent.parent / "data" / "lancedb")
    derived = derive_max_id(db_uri)
    start_id = max(state.max_id, derived) + 1
    pages = []
    consecutive_404 = 0

    logger.info(f"Starting crawl from id={start_id} (state.max_id={state.max_id}, derived={derived})")

    for tid in range(start_id, start_id + 50):  # Max 50 new pages per run
        url = BASE_URL.format(id=tid)
        try:
            resp = httpx.get(url, timeout=30, follow_redirects=True)
            soup = BeautifulSoup(resp.text, "html.parser")
            content_div = (
                soup.select_one(".content-detail")
                or soup.select_one("article")
                or soup.find("body")
            )
            text = content_div.get_text(separator="\n", strip=True) if content_div else resp.text
            if resp.status_code == 404 or _is_empty_page(resp.text, text):
                consecutive_404 += 1
                logger.debug(f"empty/not-found: {url} (consecutive: {consecutive_404})")
                if consecutive_404 >= MAX_CONSECUTIVE_404:
                    logger.info(f"Stopping after {MAX_CONSECUTIVE_404} consecutive empty pages")
                    break
                continue
            consecutive_404 = 0
            pages.append((url, text))
            logger.info(f"Crawled: {url} ({len(text)} chars)")
        except Exception as e:
            logger.warning(f"Failed to crawl {url}: {e}")
            consecutive_404 += 1

    return pages


def dry_run():
    """Check for new IDs without ingesting."""
    state = CrawlerState.load()
    try:
        from config.settings import RAGSettings
        db_uri = RAGSettings().LANCEDB_URI
    except Exception:
        db_uri = str(Path(__file__).parent.parent / "data" / "lancedb")
    derived = derive_max_id(db_uri)
    start_id = max(state.max_id, derived) + 1
    logger.info(f"DRY RUN: Checking for new thongbao starting from id={start_id} (derived={derived})")

    found = []
    consecutive_404 = 0
    for tid in range(start_id, start_id + 10):
        url = BASE_URL.format(id=tid)
        try:
            resp = httpx.get(url, timeout=15, follow_redirects=True)
            soup = BeautifulSoup(resp.text, "html.parser")
            content_div = (
                soup.select_one(".content-detail")
                or soup.select_one("article")
                or soup.find("body")
            )
            text = content_div.get_text(separator="\n", strip=True) if content_div else resp.text
            if resp.status_code == 404 or _is_empty_page(resp.text, text):
                consecutive_404 += 1
                if consecutive_404 >= MAX_CONSECUTIVE_404:
                    break
                continue
            consecutive_404 = 0
            found.append(tid)
            logger.info(f"Found: id={tid} ({url}, {len(text)} chars)")
        except Exception as e:
            logger.warning(f"Error checking {url}: {e}")

    if found:
        logger.info(f"DRY RUN: Found {len(found)} new pages: {found}")
    else:
        logger.info(f"DRY RUN: No new pages found (latest is id={state.max_id})")
    return found


# --- Chunk adapter for IngestPipeline ---

_THONGBAO_ID_RE_CHUNK = re.compile(r"[?&]id=(\d+)", re.IGNORECASE)


def _parse_notification_id(url: str) -> int:
    """Extract thongbao id from URL. Returns 0 when absent."""
    m = _THONGBAO_ID_RE_CHUNK.search(url)
    return int(m.group(1)) if m else 0


def chunk_fn(text: str, metadata: dict, chunked_dir: Path | None = None) -> list[dict]:
    """Hybrid chunker adapter: split on single \\n, hard-cut at max_chars.
    Writes JSONL to data/chunked/ for ingest_lancedb.py --incremental.

    ``chunked_dir`` is injectable so tests can write to tmp_path instead of
    polluting the real data/chunked/ (regression: pytest run created
    chunked_thongbao_78_*.jsonl crawl stubs into production data)."""
    # Split on single newline (not blank-line — site returns \n-separated text)
    raw_parts = [p.strip() for p in text.split("\n") if p.strip()]
    if not raw_parts:
        return []

    max_chars = 1200
    groups: list[str] = []
    buf = ""
    for part in raw_parts:
        if not buf:
            buf = part
        elif len(buf) + 1 + len(part) <= max_chars:
            buf = buf + "\n" + part
        else:
            groups.append(buf)
            buf = part
    # Hard-cut: even a single part may exceed max_chars
    if buf:
        if len(buf) > max_chars:
            # Force-split long block on newline boundaries
            sub = buf.split("\n")
            sbuf = ""
            for sp in sub:
                if not sbuf:
                    sbuf = sp
                elif len(sbuf) + 1 + len(sp) <= max_chars:
                    sbuf = sbuf + "\n" + sp
                else:
                    groups.append(sbuf)
                    sbuf = sp
            if sbuf:
                groups.append(sbuf)
        else:
            groups.append(buf)

    nid = _parse_notification_id(metadata.get("source_url", ""))
    now = datetime.now(timezone.utc).isoformat()
    prefix = f"thongbao_{nid}" if nid else "thongbao_unknown"

    # Write JSONL to data/chunked/ for ingest_lancedb.py
    if chunked_dir is None:
        chunked_dir = Path(__file__).parent.parent / "data" / "chunked"
    chunked_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = chunked_dir / f"chunked_{prefix}_{now.replace(':','.')}.jsonl"

    chunks = []
    for i, g in enumerate(groups):
        cid = f"{prefix}_{i}"
        meta = {**metadata, "notification_id": nid, "crawl_date": now}
        chunk_obj = {"id": cid, "chunk_id": cid, "text": g, "metadata": meta}
        chunks.append({"id": cid, "chunk_id": cid, "text": g, "metadata": meta})
        jsonl_path.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks),
            encoding="utf-8",
        )

    return chunks


def embed_fn(chunks: list[dict]) -> int:
    """Embed only the new chunks and merge into husc_chunks (17-field).
    Reuses CHUNK_SCHEMA + content-hash from ingest_lancedb logic."""
    import lancedb as ldb_lib
    from config.settings import RAGSettings
    from src.services.embedding import EmbeddingService
    from src.infrastructure.lancedb_schema import CHUNK_SCHEMA

    if not chunks:
        return 0

    settings = RAGSettings()
    db = ldb_lib.connect(settings.LANCEDB_URI)
    target = os.environ.get("LANCEDB_UNIFIED_TABLE", "husc_chunks")

    # Embed only new chunks
    texts = [c["text"] for c in chunks]
    svc = EmbeddingService(settings)
    vectors = svc.encode_documents(texts)

    records = []
    for chunk, vec in zip(chunks, vectors.tolist()):
        raw_meta = chunk.get("metadata", {})
        data_year = 0
        needs_review = True
        for key in ("data_year", "year"):
            val = raw_meta.get(key)
            if isinstance(val, int) and val > 0:
                data_year, needs_review = val, False
                break
            if isinstance(val, str) and val.isdigit():
                data_year, needs_review = int(val), False
                break
        content_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
        meta_out = {**raw_meta, "needs_review": needs_review, "content_hash": content_hash}
        rec = {
            "id": chunk["id"],
            "doc_id": chunk.get("chunk_id") or chunk["id"],
            "release_id": "legacy",
            "text": chunk["text"],
            "summary": None,
            "vector": vec,
            "data_year": data_year,
            "scope": "admissions",
            "doc_type": raw_meta.get("faq_type", "chunk"),
            "status": "current",
            "effective_from": 0,
            "effective_to": 99991231,
            "is_tombstone": False,
            "superseded_by": None,
            "metadata_json": json.dumps(meta_out, ensure_ascii=False, default=str),
            "is_reference": bool(raw_meta.get("is_reference")),
            "crawl_date": raw_meta.get("crawl_date"),
        }
        records.append(rec)

    table_exists = target in db.table_names()
    if table_exists:
        table = db.open_table(target)
        job = (table.merge_insert("id")
                 .when_matched_update_all()
                 .when_not_matched_insert_all())
        job.execute(records)
        logger.info(f"embed_fn: merged {len(records)} chunks into {target}, total={table.count_rows()}")
    else:
        db.create_table(target, data=records)
        logger.info(f"embed_fn: created {target} with {len(records)} chunks")

    return len(records)


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
            result = pipeline.run(crawl_fn=lambda: pages, chunk_fn=chunk_fn, embed_fn=embed_fn)
            # Save max_id only after pipeline succeeds — prevents data loss
            if result.get("status") == "success" and args.force_id > state.max_id:
                state.max_id = args.force_id
                state.last_crawl = datetime.now(timezone.utc).isoformat()
                state.save()
                logger.info(f"Updated max_id to {args.force_id}")
            logger.info(f"Force crawl result: {result}")
        except Exception as e:
            logger.error(f"Force crawl failed: {e}")
        return

    state = CrawlerState.load()
    _capture = {"max_id": 0}

    def _crawl_with_capture():
        pages = crawl_fn()
        if pages:
            _capture["max_id"] = max(int(url.split("id=")[1]) for url, _ in pages)
        return pages

    pipeline = IngestPipeline(state)
    result = pipeline.run(crawl_fn=_crawl_with_capture, chunk_fn=chunk_fn, embed_fn=embed_fn)

    # Save max_id only after pipeline succeeds — prevents data loss
    if result.get("status") == "success" and _capture["max_id"]:
        state.max_id = _capture["max_id"]
        state.last_crawl = datetime.now(timezone.utc).isoformat()
        state.save()
        logger.info(f"Updated max_id to {_capture['max_id']}")
    elif result.get("status") == "success":
        logger.info("No new pages found")

    logger.info(f"Pipeline result: {result}")


if __name__ == "__main__":
    main()
