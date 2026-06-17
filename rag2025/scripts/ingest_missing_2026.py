"""Ingest the approved, uningested 2026 JSONL files into LanceDB (S17.1).

TDD contract: see ``tests/scripts/test_ingest_missing_2026.py``.

Behaviors:
  * Only the 5 approved files are eligible; the 3 noisy files are explicitly
    skipped (NEVER touch the live table for them).
  * Dedup by ``chunk_id`` (idempotent re-run is safe).
  * ``id`` and ``chunk_id`` are interchangeable on the source record.
  * Records with empty ``text`` are skipped (no embedding call, no row).
  * The script NEVER drops the table — it only appends new rows.

Run (real, against the live ``rag2025`` table):

    python scripts/ingest_missing_2026.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol, Set

# Make ``config`` and ``src`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lancedb  # noqa: E402
from loguru import logger  # noqa: E402

from config.settings import RAGSettings  # noqa: E402
from src.services.embedding import EmbeddingService  # noqa: E402


# ---------------------------------------------------------------------------
# Approved / skipped allow-list (TDD-locked)
# ---------------------------------------------------------------------------
APPROVED_FILES: List[str] = [
    "husc_2026_canonical_v2.jsonl",
    "husc_2026_canonical_v3.jsonl",
    "husc_2026_aggregation_v2.jsonl",
    "nganh_detail_2026.jsonl",
    "husc_2026_vision_ocr.jsonl",
]

SKIP_FILES: Set[str] = {
    "baiviet_2026.jsonl",            # noisy HTML
    "thongbao_2026_v2.jsonl",        # data_year=2025 mislabel
    "chunked_10_enhanced.jsonl",     # dup of chunked_10
}


# ---------------------------------------------------------------------------
# Protocols — keep the script testable with fakes
# ---------------------------------------------------------------------------
class _TableLike(Protocol):
    def add(self, rows: List[Dict[str, Any]]) -> Any: ...


class _EmbedderLike(Protocol):
    def encode_documents(self, texts: List[str]): ...


# ---------------------------------------------------------------------------
# Core ingest routine (pure — testable)
# ---------------------------------------------------------------------------
def _resolve_chunk_id(obj: Dict[str, Any]) -> str:
    """Return the canonical chunk_id, accepting either ``id`` or ``chunk_id``."""
    raw = obj.get("id") or obj.get("chunk_id")
    if not raw:
        return ""
    return str(raw)


def _build_row(
    obj: Dict[str, Any],
    text: str,
    vector: List[float],
) -> Dict[str, Any]:
    """Build a single LanceDB row matching the live table schema."""
    chunk_id = _resolve_chunk_id(obj)
    metadata = obj.get("metadata", {}) or {}
    source = (
        metadata.get("source_url")
        or metadata.get("source")
        or obj.get("source", "")
        or ""
    )
    faq_type = (
        metadata.get("faq_type")
        or obj.get("faq_type", "")
        or ""
    )
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "text": text,
        "source": source,
        "faq_type": faq_type,
        "metadata_json": json.dumps(metadata, ensure_ascii=False, default=str),
        "vector": vector,
    }


def _load_records(file_path: Path) -> Iterable[Dict[str, Any]]:
    """Yield one parsed JSON object per non-empty line."""
    with file_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON line in {file_path.name}")
                continue
            if not isinstance(obj, dict):
                continue
            yield obj


def ingest_files(
    files: List[Path],
    table: _TableLike,
    embedder: _EmbedderLike,
) -> Dict[str, Dict[str, int]]:
    """Ingest the given files into the supplied (fake or real) LanceDB table.

    Returns a per-file summary ``{filename: {"added": N, "skipped": M}}``.
    Re-runs are safe: chunk_ids already present in the table are skipped.
    """
    # Read existing chunk_ids straight from the live table (idempotent re-run).
    # NB: the REAL LanceDB `to_pandas()` does NOT accept a `columns=` kwarg
    # (only the test fake does). Call it with no args and select columns from
    # the resulting DataFrame — otherwise the kwarg raises TypeError, the
    # except-swallow leaves `existing` empty, and every already-present chunk
    # gets re-added as a DUPLICATE (the bug seen on 2026-06-10).
    existing: Set[str] = set()
    try:
        if hasattr(table, "to_pandas"):
            try:
                df = table.to_pandas(columns=["chunk_id", "id"])  # fake table path
            except TypeError:
                df = table.to_pandas()  # real LanceDB path (no columns kwarg)
            for col in ("chunk_id", "id"):
                if col in df.columns:
                    existing.update(str(v) for v in df[col].tolist() if v is not None)
    except Exception as exc:
        logger.warning(f"could not read existing chunk_ids ({exc}); proceeding empty")

    summary: Dict[str, Dict[str, int]] = {}

    for file_path in files:
        file_name = file_path.name
        # Defense in depth: never ingest an explicitly-skipped file.
        if file_name in SKIP_FILES:
            logger.warning(f"Refusing to ingest skip-listed file: {file_name}")
            summary[file_name] = {"added": 0, "skipped": 0}
            continue

        pending_objs: List[Dict[str, Any]] = []
        pending_chunks: List[Dict[str, Any]] = []  # parallel list of valid records
        seen_in_file: Set[str] = set()
        skipped = 0

        for obj in _load_records(file_path):
            text = (obj.get("text") or "").strip()
            if not text:
                skipped += 1
                continue
            cid = _resolve_chunk_id(obj)
            if not cid:
                skipped += 1
                continue
            if cid in existing or cid in seen_in_file:
                skipped += 1
                continue
            seen_in_file.add(cid)
            pending_objs.append(obj)
            pending_chunks.append({"id": cid, "chunk_id": cid, "text": text, "metadata": obj.get("metadata", {})})

        if not pending_chunks:
            summary[file_name] = {"added": 0, "skipped": skipped}
            logger.info(f"{file_name}: 0 added, {skipped} skipped")
            continue

        # Embed only the pending texts (efficient: skips already-present + empty)
        vectors = embedder.encode_documents([c["text"] for c in pending_chunks])

        rows: List[Dict[str, Any]] = []
        for obj, vec in zip(pending_objs, vectors):
            vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            rows.append(_build_row(obj=obj, text=obj.get("text", ""), vector=vec_list))

        if rows:
            table.add(rows)

        summary[file_name] = {"added": len(rows), "skipped": skipped}
        logger.info(f"{file_name}: {len(rows)} added, {skipped} skipped")

    return summary


# ---------------------------------------------------------------------------
# Real-DB entry point
# ---------------------------------------------------------------------------
def run_against_live_db() -> Dict[str, Dict[str, int]]:
    settings = RAGSettings()
    chunks_dir = Path(__file__).resolve().parent.parent / "data" / "chunked"

    db = lancedb.connect(settings.LANCEDB_URI)
    if settings.LANCEDB_TABLE not in db.table_names():
        raise FileNotFoundError(
            f"LanceDB table '{settings.LANCEDB_TABLE}' not found at '{settings.LANCEDB_URI}'"
        )
    table = db.open_table(settings.LANCEDB_TABLE)

    files: List[Path] = []
    for name in APPROVED_FILES:
        path = chunks_dir / name
        if not path.exists():
            logger.warning(f"Approved file missing on disk: {name}")
            continue
        files.append(path)

    embedder = EmbeddingService(settings)
    summary = ingest_files(files=files, table=table, embedder=embedder)
    return summary


if __name__ == "__main__":
    summary = run_against_live_db()
    print("Ingest summary:")
    for fname, counts in summary.items():
        print(f"  {fname}: added={counts['added']} skipped={counts['skipped']}")
    total_added = sum(c["added"] for c in summary.values())
    total_skipped = sum(c["skipped"] for c in summary.values())
    print(f"TOTAL: added={total_added} skipped={total_skipped}")
