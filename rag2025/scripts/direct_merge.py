"""Direct incremental merge: embed 25 new chunks and add to husc_v2026_clean.

Bypasses the EmbeddingService lazy-load + progress_bar issues that cause hangs
in PowerShell terminal. Uses sentence_transformers directly with batch_size=8.
"""
import sys, json, os, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Step 1: load new chunks
from scripts.build_husc_v2026_clean import SOURCES, load_chunks, build_record
import lancedb

all_chunks = []
for src in SOURCES:
    if not src.exists():
        print(f"  SKIP (missing): {src.name}")
        continue
    kept, skipped = load_chunks(src)
    print(f"  {src.name}: kept={len(kept)} skipped={skipped}")
    all_chunks.extend(kept)
print(f"Total chunks from sources: {len(all_chunks)}")

db = lancedb.connect(str(ROOT / "data" / "lancedb"))
table = db.open_table("husc_v2026_clean")
existing_ids = set(table.to_pandas()["id"].tolist())
print(f"Existing rows: {len(existing_ids)}")

new_chunks = []
for c in all_chunks:
    cid = str(c.get("chunk_id") or c.get("id") or "")
    if cid and cid not in existing_ids:
        new_chunks.append(c)
print(f"NEW to embed: {len(new_chunks)}")

if not new_chunks:
    print("Nothing to add. Table is up to date.")
    sys.exit(0)

# Step 2: embed directly with sentence_transformers
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading embedding model...")
t0 = time.time()
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
print(f"Model loaded in {time.time()-t0:.1f}s")

texts = [c["text"] for c in new_chunks]
print(f"Encoding {len(texts)} texts (batch_size=8, no progress bar)...")
t0 = time.time()
embs = model.encode(
    texts,
    batch_size=8,
    normalize_embeddings=True,
    show_progress_bar=False,
    convert_to_numpy=True,
).astype(np.float32)
print(f"Encoded in {time.time()-t0:.1f}s, shape={embs.shape}")

# Step 3: build records — strip extra metadata keys to match existing table schema
EXISTING_META_KEYS = {
    "approved_by", "approved_date", "audience", "chunk_method", "chunk_type",
    "data_year", "info_type", "major_code", "notification_id", "references_pdf",
    "references_year_2025", "school", "source", "source_type", "source_url",
    "supersedes", "token_count", "topic", "unit", "year",
}


def build_record_normalized(chunk: dict, vector: list[float]) -> dict:
    """Like build_record but strips metadata keys not in existing table schema."""
    meta = chunk.get("metadata", {})
    normalized_meta = {k: v for k, v in meta.items() if k in EXISTING_META_KEYS}
    # Ensure all required keys exist with sensible defaults
    for k in EXISTING_META_KEYS:
        if k not in normalized_meta:
            if k == "supersedes":
                normalized_meta[k] = []
            elif k in ("data_year", "notification_id", "token_count"):
                normalized_meta[k] = None
            else:
                normalized_meta[k] = None
    chunk_id = chunk.get("chunk_id", chunk.get("id", ""))
    return {
        "id": str(chunk_id),
        "text": chunk.get("text", ""),
        "summary": chunk.get("summary", None),
        "embedding": vector,
        "metadata": normalized_meta,
    }


records = [build_record_normalized(new_chunks[i], embs[i].tolist()) for i in range(len(new_chunks))]
print(f"Adding {len(records)} records to table...")
table.add(records)
new_count = table.count_rows()
print(f"DONE. Table 'husc_v2026_clean' now has {new_count} rows.")
print()
print("New chunk IDs added:")
for r in records:
    print(f"  + {r['id']}")
