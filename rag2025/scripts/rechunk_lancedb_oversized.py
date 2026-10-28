"""T1.8: re-chunk oversized LanceDB chunks into <=512-token children.

For every active chunk whose text exceeds `--threshold` (default 800) tokens:
  1. split the text into <=HARD_MAX_TOKENS (512) children (row-aware for tables),
  2. supersede the old chunk IN-PLACE (metadata_json.is_superseded=true +
     superseded_by) — NO hard delete, old vector is kept (Agent B pattern),
  3. insert the split children with fresh embeddings (merge_insert).

Crash safety: children are only embedded+inserted after the split step; the
embedding phase writes a JSONL checkpoint so an interrupted run leaves the DB
untouched (old chunks stay active). A `--skip-embed`/`--dry-run` mode prepares
the child records without embedding.

Usage:
    cd rag2025
    python scripts/rechunk_lancedb_oversized.py \
        [--threshold 800] [--max-tokens 512] [--dry-run] [--skip-embed]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import lancedb
import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunk_table_aware_v2 import _split_oversized_preserving  # noqa: E402
from config.settings import RAGSettings  # noqa: E402
from src.services.embedding import EmbeddingService  # noqa: E402

HARD_MAX_TOKENS = 512
CHECKPOINT_DIR = Path(__file__).parent.parent / "data" / "t1.8_resplit"


def _unique_id(source: str, ingest_file: str, chunk_id: str) -> str:
    """T1.5/G9 globally-unique row id (same scheme as ingest_lancedb.py)."""
    raw = f"{source}|{ingest_file}|{chunk_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _parse_meta(raw: object) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _content_hash(text: str) -> str:
    norm = " ".join(text.split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _esc(raw: str) -> str:
    return raw.replace("'", "''")


def prepare_children(row: dict, enc, max_tokens: int) -> list[dict]:
    """Split one oversized row into child record dicts (no vector yet).

    Returns [] when the row is not oversized or cannot be split.
    """
    text = row.get("text") or ""
    tok = len(enc.encode(text))
    if tok <= max_tokens:
        return []
    pieces = _split_oversized_preserving(text, enc, max_tokens)
    if len(pieces) <= 1:
        return []

    old_id = str(row["id"])
    old_doc_id = row.get("doc_id") or old_id
    old_meta = _parse_meta(row.get("metadata_json"))
    child_type = old_meta.get("chunk_type") or "text_section"
    base_meta = {k: v for k, v in old_meta.items() if k != "token_count"}

    children = []
    for n, piece in enumerate(pieces):
        child_id = _unique_id(old_id, "t1.8_resplit", f"split{n}")
        meta = dict(base_meta)
        meta.update({
            "chunk_type": child_type,
            "parent_id": old_id,
            "parent_doc_id": old_doc_id,
            "split_index": n,
            "is_split_child": True,
            "ingest_file": "t1.8_resplit",
            "content_hash": _content_hash(piece),
            "needs_review": old_meta.get("needs_review", False),
        })
        if "token_count" in meta:
            meta["token_count"] = len(enc.encode(piece))
        children.append({
            "id": child_id,
            "doc_id": old_doc_id,
            "release_id": row.get("release_id") or "legacy",
            "text": piece,
            "summary": None,
            "vector": None,  # filled by embed phase
            "data_year": row.get("data_year", 0),
            "scope": row.get("scope") or "admissions",
            "doc_type": row.get("doc_type") or "chunk",
            "status": "current",
            "effective_from": row.get("effective_from", 0),
            "effective_to": row.get("effective_to", 99991231),
            "is_tombstone": False,
            "superseded_by": None,
            "metadata_json": json.dumps(meta, ensure_ascii=False, default=str),
            "is_reference": bool(row.get("is_reference")),
            "crawl_date": row.get("crawl_date"),
        })
    return children


def build_supersede_values(row: dict, children: list[dict]) -> dict:
    """In-place supersede values for an oversized row (Agent B pattern)."""
    old_meta = _parse_meta(row.get("metadata_json"))
    old_meta["is_superseded"] = True
    old_meta["superseded_by"] = children[0]["id"] if children else ""
    old_meta["superseded_reason"] = "T1.8 re-chunk >800 tokens into <=512 children"
    values = {"metadata_json": json.dumps(old_meta, ensure_ascii=False, default=str)}
    if "superseded_by" in row:
        values["superseded_by"] = old_meta["superseded_by"]
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="T1.8 re-chunk oversized LanceDB chunks")
    parser.add_argument("--threshold", type=int, default=800,
                        help="token count above which a chunk is re-chunked")
    parser.add_argument("--max-tokens", type=int, default=HARD_MAX_TOKENS)
    parser.add_argument("--dry-run", action="store_true",
                        help="prepare children and print counts, write nothing")
    parser.add_argument("--skip-embed", action="store_true",
                        help="prepare children, write checkpoint JSONL but do NOT embed/insert")
    args = parser.parse_args()

    settings = RAGSettings()
    db = lancedb.connect(settings.LANCEDB_URI)
    table = db.open_table(settings.LANCEDB_TABLE)
    enc = tiktoken.get_encoding("cl100k_base")

    rows = table.to_arrow().to_pylist()
    oversized = [r for r in rows if len(enc.encode(r.get("text") or "")) > args.threshold]
    # Skip rows already superseded (defensive).
    oversized = [r for r in oversized
                 if not _is_truthy(_parse_meta(r.get("metadata_json")).get("is_superseded"))]
    print(f"rows={len(rows)} oversized(>{args.threshold} tok, active)={len(oversized)}")

    all_children: dict[str, list[dict]] = {}
    for row in oversized:
        kids = prepare_children(row, enc, args.max_tokens)
        if kids:
            all_children[str(row["id"])] = kids
    total_children = sum(len(v) for v in all_children.values())
    print(f"old->children: {len(all_children)} rows -> {total_children} children")

    if args.dry_run:
        # Report sizes and a sample, write nothing.
        over = [(str(k), len(v), len(enc.encode(v[0]["text"]))) for k, v in all_children.items()]
        over.sort(key=lambda x: -x[2])
        print("sample children (old_id, n, child_tok):")
        for oid, n, c0 in over[:10]:
            print(f"  {oid[:40]:<42} n={n} child0_tok={c0}")
        print("DRY-RUN — no DB writes.")
        return 0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_DIR / "children.jsonl"
    with checkpoint.open("w", encoding="utf-8") as fh:
        for old_id, kids in all_children.items():
            for kid in kids:
                fh.write(json.dumps(kid, ensure_ascii=False, default=str) + "\n")
    print(f"checkpoint written: {checkpoint}")

    # Safety snapshot of the rows we are about to supersede (rollback reference).
    old_snapshot = CHECKPOINT_DIR / "old_rows.jsonl"
    with old_snapshot.open("w", encoding="utf-8") as fh:
        for row in oversized:
            row_out = {k: v for k, v in row.items()}
            fh.write(json.dumps(row_out, ensure_ascii=False, default=str) + "\n")
    print(f"old-row snapshot written: {old_snapshot} ({len(oversized)} rows)")

    if args.skip_embed:
        print("SKIP-EMBED — children prepared in checkpoint; no embedding/insert.")
        return 0

    # ── Embed phase (crash-safe: no DB writes until ALL embeddings done) ──
    print("loading embedding model ...", flush=True)
    t0 = time.time()
    svc = EmbeddingService(settings)
    all_texts = [kid["text"] for kids in all_children.values() for kid in kids]
    batch = int(settings.EMBEDDING_BATCH_SIZE)
    print(f"embedding {len(all_texts)} children (batch={batch}) ...", flush=True)
    vec_list = []
    n_batches = (len(all_texts) + batch - 1) // batch
    for i in range(0, len(all_texts), batch):
        chunk = all_texts[i : i + batch]
        t_b = time.time()
        vecs = svc.encode_batch(chunk)
        vec_list.append(vecs)
        done = min(i + batch, len(all_texts))
        print(
            f"  batch {i // batch + 1}/{n_batches} done ({done}/{len(all_texts)}) "
            f"in {time.time() - t_b:.1f}s",
            flush=True,
        )
    import numpy as np

    vectors = np.concatenate(vec_list, axis=0) if vec_list else np.empty((0, 1024), dtype=np.float32)
    print(f"embedding done in {time.time() - t0:.1f}s, shape={vectors.shape}", flush=True)

    # Attach vectors
    idx = 0
    for kids in all_children.values():
        for kid in kids:
            kid["vector"] = vectors[idx].tolist()
            idx += 1

    # ── Write phase: insert children + supersede old (fast, last) ──
    records = [kid for kids in all_children.values() for kid in kids]
    job = table.merge_insert("id").when_not_matched_insert_all()
    job.execute(records)
    print(f"inserted {len(records)} children")

    superseded = 0
    for old_id, kids in all_children.items():
        values = build_supersede_values(
            next(r for r in oversized if str(r["id"]) == old_id), kids)
        table.update(f"id = '{_esc(old_id)}'", values=values)
        superseded += 1
    print(f"superseded {superseded} old oversized chunks")

    # Idempotency guard: verify final state
    final = table.to_arrow().to_pylist()
    over512 = sum(1 for r in final if not _is_truthy(_parse_meta(r.get("metadata_json")).get("is_superseded"))
                  and len(enc.encode(r.get("text") or "")) > HARD_MAX_TOKENS)
    print(f"final rows={len(final)} active>512tok={over512}")
    return 0


def _is_truthy(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


if __name__ == "__main__":
    raise SystemExit(main())
