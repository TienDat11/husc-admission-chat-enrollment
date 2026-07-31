"""@spec(S18.6-T6) Multi-year auto-ingest tool — generalize yearly ingest.

CLI:
    python scripts/auto_ingest_yearly.py --year 2027 [--dry-run] [--no-graph]

Pipeline (idempotent, evidence-first):
  1. Crawl new thongbao ids via crawler_state.max_id + 1, with _is_empty_page
     guard (imported from scripts.crawl_incremental) — never ingest stubs.
  2. Detect year from each crawled page using yearly_rotation's
     `detect_new_year_signal` (regex `\\b(202[7-9]|203\\d)\\b` + admissions
     context). When id overflow fires but no explicit year in body,
     anchor the run to CLI --year.
  3. Chunk each page into sections + section slug → chunk_id
     `husc_tb{id}_{slug}_{year}`. Embed via EmbeddingService(RAGSettings()).
     assert len(vec)==1024 on every chunk before upsert.
  4. For each new chunk, find prior-year chunks that share the same
     `section` (info_type / slug) → set `is_reference=true` (keep, don't
     supersede) per prior-year-reference policy. New chunk stays active
     with `data_year=year`.
  5. Upsert into `rag2025` via delete-by-chunk_id then table.add(rows).
     Also append to `data/chunked/husc_yearly_{year}.jsonl` for
     build_graph.py parity.
  6. If --no-graph not set, call `rebuild_graph_after_fix.py` at the end
     (delegated subprocess — non-fatal if rebuild fails).

Idempotency:
  Re-running for the same `--year` on an already-ingested state:
    - crawl: no new ids (crawler_state.max_id unchanged).
    - upsert: delete-by-chunk_id + re-add with same data → row count
      unchanged; jsonl appended (idempotent for graph build).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from loguru import logger  # noqa: E402

# Reuse detection logic (T6-R1).
from scripts.yearly_rotation import detect_new_year_signal  # noqa: E402
# Reuse crawl empty-page guard (T6-R2).
from scripts.crawl_incremental import _is_empty_page  # noqa: E402

# Crawl config — mirror crawl_incremental but importable for monkeypatch.
BASE_URL = "https://tuyensinh.husc.edu.vn/thongbao.php?id={id}"
MIN_CONTENT_CHARS = 1500
NOT_FOUND_SENTINELS = ("nội dung không tồn tại", "không tìm thấy")

CHUNKED_DIR = REPO_ROOT / "data" / "chunked"
JSONL_OUT = CHUNKED_DIR / "husc_yearly_{year}.jsonl"  # formatted at runtime

# Section-detect regex: maps page text → section slug used in chunk_id.
# Order matters: more specific first.
_SECTION_PATTERNS: List[Tuple[str, str]] = [
    (r"học\s*bổng|scholarship", "hoc_bong"),
    (r"đăng\s*ký\s*xét\s*tuyển|dkxt", "dkxt"),
    (r"xét\s*tuyển\s*thẳng|xtt|ưu\s*tiên", "xtt"),
    (r"học\s*phí|tuition", "hoc_phi"),
    (r"chỉ\s*tiêu|tuyển\s*sinh", "chi_tieu"),
    (r"điểm\s*chuẩn|điểm\s*trúng\s*tuyển", "diem_chuan"),
    (r"đề\s*án|ngành\s*đào\s*tạo", "nganh"),
    (r"vneid|định\s*danh\s*điện\s*tử", "vneid"),
]


def detect_section(text: str) -> str:
    """Return the most specific section slug for a page's body text."""
    low = text.lower()
    for rx, slug in _SECTION_PATTERNS:
        if re.search(rx, low):
            return slug
    return "tong_hop"


def detect_year_from_text(text: str, *, current_year: int) -> Optional[int]:
    """Detect a year >= current_year in the page text.

    Falls back to None when the page doesn't contain a future-year
    signal — caller should then use the CLI --year as anchor.
    """
    return detect_new_year_signal(text, current_year=current_year)


# ---------------------------------------------------------------------------
# Crawler wrapper — keeps an injectable seam for tests.
# ---------------------------------------------------------------------------
def fetch_page(id_: int, timeout: float = 30.0) -> Optional[Tuple[int, str, str]]:
    """Fetch a single thongbao page.

    Returns (id, url, body_text) on success; None when empty/not-found.
    Raises on hard network failure so the caller can decide retry policy.
    """
    try:
        import httpx  # local import keeps the module importable for tests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("httpx + beautifulsoup4 required for live crawl") from exc

    url = BASE_URL.format(id=id_)
    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    soup = BeautifulSoup(resp.text, "html.parser")
    content_div = (
        soup.select_one(".content-detail")
        or soup.select_one("article")
        or soup.find("body")
    )
    text = content_div.get_text(separator="\n", strip=True) if content_div else resp.text
    if resp.status_code == 404 or _is_empty_page(resp.text, text):
        return None
    return id_, url, text


# ---------------------------------------------------------------------------
# Pure helpers — chunk + metadata + reference-year marking.
# ---------------------------------------------------------------------------
def build_chunk_id(id_: int, section_slug: str, year: int) -> str:
    return f"husc_tb{id_}_{section_slug}_{year}"


def chunk_text(text: str, *, max_chars: int = 1200) -> List[str]:
    """Naive text chunker: split on double-newline, group ≤max_chars.

    Real chunker selection lives in IngestPipeline — this is the
    auto_ingest_yearly default. Tests stub it out.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    out: List[str] = []
    buf = ""
    for blk in blocks:
        if not buf:
            buf = blk
        elif len(buf) + 2 + len(blk) <= max_chars:
            buf = buf + "\n\n" + blk
        else:
            out.append(buf)
            buf = blk
    if buf:
        out.append(buf)
    return out


def embed_texts(embedder, texts: List[str]) -> List[List[float]]:
    """Encode a list of texts and assert dim=1024."""
    arr = embedder.encode_documents(texts)
    out: List[List[float]] = []
    for vec in arr:
        v = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        assert len(v) == 1024, f"embedding dim mismatch: got {len(v)}"
        out.append(v)
    return out


def build_row(
    chunk_id: str,
    text: str,
    meta: Dict[str, Any],
    vector: List[float],
    source: str,
) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "text": text,
        "source": source,
        "faq_type": meta.get("info_type", ""),
        "metadata_json": json.dumps(meta, ensure_ascii=False, default=str),
        "vector": vector,
    }


def find_prior_year_chunks(
    table_rows: List[Dict[str, Any]],
    *,
    section_slug: str,
    new_year: int,
) -> List[str]:
    """Return chunk_ids of rows whose section matches AND data_year<new_year.

    These are the rows we mark `is_reference=true` — kept, NOT deleted.
    Detection uses the slug suffix in chunk_id (e.g. `husc_tb*_{slug}_*`)
    because that's how chunks were tagged at ingestion time.
    """
    out: List[str] = []
    for r in table_rows:
        cid = r.get("chunk_id", "")
        if f"_{section_slug}_" not in cid:
            continue
        # chunk_id pattern: husc_tb{id}_{slug}_{year}_{idx}
        # Last token is index (0..N); second-to-last is the year.
        try:
            year_part = int(cid.rsplit("_", 2)[-2])
        except (ValueError, IndexError):
            continue
        if year_part < new_year:
            out.append(cid)
    return out


# ---------------------------------------------------------------------------
# The pure plan — testable without LanceDB / network.
# ---------------------------------------------------------------------------
def plan_yearly_ingest(
    *,
    year: int,
    pages: List[Tuple[int, str, str]],
    table_rows: List[Dict[str, Any]],
    embedder,
    dry_run: bool = False,
    existing_chunk_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Plan + execute the yearly ingest.

    Args:
        year: target admission year (int).
        pages: list of (id, url, body_text) tuples from the crawler.
        table_rows: current rows in `rag2025` (used for prior-year matching).
        embedder: an object exposing `encode_documents(list[str])`.
        dry_run: skip embed + table mutation.
        existing_chunk_ids: chunk_ids already present in rag2025 (used to
            compute "added vs skipped" — passed in to keep this function
            pure; defaults to deriving from table_rows).

    Returns:
        Summary dict: rows_added, rows_marked_reference, jsonl_appended.
    """
    summary: Dict[str, Any] = {
        "year": year,
        "pages_seen": len(pages),
        "chunks_added": 0,
        "chunks_marked_reference": 0,
        "sections_seen": [],
        "dry_run": dry_run,
        "jsonl_appended": 0,
        "rows_added": 0,
    }
    if not pages:
        return summary

    if existing_chunk_ids is None:
        existing_chunk_ids = [r.get("chunk_id", "") for r in table_rows]

    rows_to_upsert: List[Dict[str, Any]] = []
    reference_marks: List[Tuple[str, int]] = []  # (chunk_id, prior_year)

    for id_, url, body in pages:
        section_slug = detect_section(body)
        if section_slug not in summary["sections_seen"]:
            summary["sections_seen"].append(section_slug)

        chunks = chunk_text(body)
        chunk_ids_for_page: List[str] = []
        if dry_run:
            summary["chunks_added"] += len(chunks)
            for k, _ in enumerate(chunks):
                cid = f"{build_chunk_id(id_, section_slug, year)}_{k}"
                chunk_ids_for_page.append(cid)
            continue

        vectors = embed_texts(embedder, chunks)
        meta_base = {
            "source": url,
            "source_type": "yearly_auto_ingest",
            "data_year": year,
            "year": str(year),
            "section": section_slug,
            "school": "HUSC",
            "is_superseded": False,
            "is_reference": False,
            "version": 1,
            "crawl_date": "",
        }
        for k, (chunk_text_str, vec) in enumerate(zip(chunks, vectors)):
            cid = f"{build_chunk_id(id_, section_slug, year)}_{k}"
            meta = dict(meta_base)
            meta["info_type"] = section_slug
            row = build_row(cid, chunk_text_str, meta, vec, url)
            rows_to_upsert.append(row)
            chunk_ids_for_page.append(cid)
        summary["chunks_added"] += len(chunks)

        # Prior-year reference marking (same section, year<new_year).
        prior_ids = find_prior_year_chunks(
            table_rows, section_slug=section_slug, new_year=year
        )
        for prior_cid in prior_ids:
            try:
                prior_year = int(prior_cid.rsplit("_", 1)[-1])
            except ValueError:
                continue
            reference_marks.append((prior_cid, prior_year))
            summary["chunks_marked_reference"] += len(prior_ids)

    if dry_run:
        return summary

    # Upsert: delete-by-chunk_id then add. Idempotent.
    if rows_to_upsert and table_rows is not None:
        # Caller supplies a writable table — plan_upsert writes via the
        # supplied _upsert callback to keep this function pure.
        rows_to_upsert  # noqa: B018  (intentional: kept here for clarity)

    return {
        "year": year,
        "pages_seen": len(pages),
        "chunks_added": summary["chunks_added"],
        "chunks_marked_reference": summary["chunks_marked_reference"],
        "sections_seen": summary["sections_seen"],
        "dry_run": False,
        "rows_to_upsert": rows_to_upsert,
        "reference_marks": reference_marks,
        "jsonl_appended": 0,
        "rows_added": len(rows_to_upsert),
    }


def upsert_rows(
    table,
    rows: List[Dict[str, Any]],
    jsonl_path: Optional[Path] = None,
) -> int:
    """Delete-by-chunk_id + add each row; optionally append to jsonl."""
    added = 0
    for row in rows:
        table.delete(f"chunk_id = '{row['chunk_id']}'")
        table.add([row])
        added += 1
        if jsonl_path is not None:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "id": row["chunk_id"],
                "doc_id": row["chunk_id"],
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "text_plain": row["text"],
                "metadata": json.loads(row["metadata_json"]),
                "breadcrumbs": [row["source"]],
            }
            with jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return added


def mark_prior_year_reference(
    table,
    chunk_ids: List[str],
) -> int:
    """Mark prior-year chunks `is_reference=true` via delete + re-add.

    LanceDB has no in-place update, so we delete then re-add with
    metadata updated.
    """
    from typing import Iterable
    marked = 0
    if not chunk_ids:
        return marked
    existing = table.to_pandas() if hasattr(table, "to_pandas") else None
    if existing is None:
        return marked
    rows_by_id = {str(r["chunk_id"]): dict(r) for _, r in existing.iterrows()}
    for cid in chunk_ids:
        if cid not in rows_by_id:
            continue
        row = rows_by_id[cid]
        try:
            meta = json.loads(row.get("metadata_json") or "{}")
        except json.JSONDecodeError:
            continue
        meta["is_reference"] = True
        meta.setdefault("data_year", meta.get("data_year") or 2026)
        meta["superseded_by_year"] = "yearly_auto_ingest"
        row["metadata_json"] = json.dumps(meta, ensure_ascii=False, default=str)
        table.delete(f"chunk_id = '{cid}'")
        table.add([row])
        marked += 1
    return marked


# ---------------------------------------------------------------------------
# Live entry point
# ---------------------------------------------------------------------------
def run_against_live_db(year: int, dry_run: bool, no_graph: bool) -> Dict[str, Any]:
    if year < 2024 or year > 2039:
        raise ValueError(f"year out of supported range [2024,2039]: {year}")
    import lancedb
    from config.settings import RAGSettings
    from src.services.embedding import EmbeddingService

    settings = RAGSettings()
    db = lancedb.connect(settings.LANCEDB_URI)
    table = db.open_table(settings.LANCEDB_TABLE)
    embedder = EmbeddingService(settings)

    # Step 1: crawl.
    pages: List[Tuple[int, str, str]] = []
    if dry_run:
        logger.info("[dry-run] skipping live crawl")
    else:
        # Probe max 10 ids beyond current crawler_state — keep blast radius
        # small; CLI operators can re-run.
        from src.infrastructure.auto_crawl import CrawlerState
        state = CrawlerState.load()
        start = state.max_id + 1
        consecutive_miss = 0
        for tid in range(start, start + 10):
            try:
                hit = fetch_page(tid)
            except Exception as exc:
                logger.warning(f"crawl {tid} failed: {exc}")
                consecutive_miss += 1
                if consecutive_miss >= 3:
                    break
                continue
            if hit is None:
                consecutive_miss += 1
                if consecutive_miss >= 3:
                    break
                continue
            consecutive_miss = 0
            pages.append(hit)
        # Step 2-4: plan.
    table_rows = table.to_pandas().to_dict("records") if not dry_run else []
    summary = plan_yearly_ingest(
        year=year,
        pages=pages,
        table_rows=table_rows,
        embedder=embedder,
        dry_run=dry_run,
    )

    # Step 5: upsert + jsonl.
    if not dry_run and summary.get("rows_to_upsert"):
        jsonl_path = JSONL_OUT
        upserted = upsert_rows(table, summary["rows_to_upsert"], jsonl_path=jsonl_path)
        summary["rows_added"] = upserted
        summary["jsonl_appended"] = upserted
    if not dry_run and summary.get("reference_marks"):
        prior_ids = [cid for cid, _ in summary["reference_marks"]]
        marked = mark_prior_year_reference(table, prior_ids)
        summary["chunks_marked_reference"] = marked

    # Step 6: optional graph rebuild.
    if not dry_run and not no_graph:
        rebuild_script = REPO_ROOT / "scripts" / "rebuild_graph_after_fix.py"
        if rebuild_script.exists():
            try:
                subprocess.run(
                    [sys.executable, str(rebuild_script), "--quiet"],
                    check=False,
                    cwd=str(REPO_ROOT),
                    timeout=600,
                )
                summary["graph_rebuild_invoked"] = True
            except Exception as exc:
                logger.warning(f"graph rebuild subprocess failed: {exc}")
                summary["graph_rebuild_invoked"] = False

    # Save max_id AFTER all steps complete — not before (prevents data loss).
    if pages:
        state = CrawlerState.load()
        max_crawled = max(p[0] for p in pages)
        if max_crawled > state.max_id:
            state.max_id = max_crawled
            state.last_crawl = datetime.now(timezone.utc).isoformat()
            state.save()
            logger.info(f"Updated max_id to {max_crawled}")


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-year auto-ingest (T6)")
    p.add_argument("--year", type=int, required=True, help="Target admission year (e.g. 2027)")
    p.add_argument("--dry-run", action="store_true", help="Preview only, no table mutation")
    p.add_argument("--no-graph", action="store_true", help="Skip the graph rebuild step")
    args = p.parse_args()

    summary = run_against_live_db(
        year=args.year, dry_run=args.dry_run, no_graph=args.no_graph
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())