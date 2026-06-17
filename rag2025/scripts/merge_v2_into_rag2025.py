"""Embed and merge aggregation_v2 chunks into rag2025 table (the retriever's live table)."""
import json, sys, os, time
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import lancedb
from sentence_transformers import SentenceTransformer
import numpy as np

TABLE = "rag2025"
CHUNKS_PATH = ROOT / "data" / "chunked" / "husc_2026_aggregation_v2.jsonl"

# Load chunks
chunks = []
for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    chunks.append(json.loads(line))
print(f"Loaded {len(chunks)} aggregation_v2 chunks")

db = lancedb.connect(str(ROOT / "data" / "lancedb"))
table = db.open_table(TABLE)
df = table.to_pandas()
existing_ids = set(df["id"].astype(str).tolist())
print(f"rag2025 rows: {len(existing_ids)}")

# Filter new
new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
print(f"New to embed: {len(new_chunks)}")
if not new_chunks:
    print("Nothing new.")
    sys.exit(0)

print("Loading embedding model...")
t0 = time.time()
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
print(f"Model loaded in {time.time()-t0:.1f}s")

texts = [c["text"] for c in new_chunks]
t0 = time.time()
embs = model.encode(texts, batch_size=4, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
print(f"Encoded in {time.time()-t0:.1f}s shape={embs.shape}")

# Build records matching rag2025 schema: id, chunk_id, text, source, faq_type, metadata_json, vector
records = []
for i, c in enumerate(new_chunks):
    records.append({
        "id": c["chunk_id"],
        "chunk_id": c["chunk_id"],
        "text": c["text"],
        "source": c.get("metadata", {}).get("source_url", ""),
        "faq_type": c.get("metadata", {}).get("info_type", ""),
        "metadata_json": json.dumps(c.get("metadata", {}), ensure_ascii=False),
        "vector": embs[i].tolist(),
    })

table.add(records)
print(f"DONE. rag2025 now has {table.count_rows()} rows.")
for r in records:
    print(f"  + {r['id']}")
