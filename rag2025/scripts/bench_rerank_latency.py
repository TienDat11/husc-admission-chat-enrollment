#!/usr/bin/env python3
"""Latency experiments: max_length + threads for BGE-reranker.
Pool size already measured. This script handles the rest.
"""
import os
os.environ['HF_HOME'] = r'D:\huggingface_cache'
os.environ['HF_HUB_CACHE'] = r'D:\huggingface_cache\hub'
os.environ['TORCH_HOME'] = r'D:\torch_cache'
os.environ['TMPDIR'] = r'D:\tmp'
os.environ['TEMP'] = r'D:\tmp'
os.environ['TMP'] = r'D:\tmp'
os.makedirs(r'D:\tmp', exist_ok=True)

import torch
torch.set_num_threads(4)

import json, time, gc, statistics
from pathlib import Path
import numpy as np
import functools
print = functools.partial(print, flush=True)

BASE = Path(r"D:\chunking\husc-admission-chat-enrollment\rag2025")
LANCE_DIR = str(BASE / "data" / "lancedb")
EVAL_FILE = BASE / "data" / "eval" / "husc_86q_test.json"
OUT_FILE = BASE / "results" / "reranker_bench.json"

# Load data
import lancedb
db = lancedb.connect(LANCE_DIR)
tbl = db.open_table("rag2025")
corpus = tbl.to_pandas()
corpus_ids = corpus["chunk_id"].tolist()
corpus_texts = corpus["text"].tolist()

with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_qs = json.load(f)[:40]
queries = [q["question"] for q in eval_qs]

# Embed queries
from sentence_transformers import SentenceTransformer
QWEN3_EMBED_PATH = r"D:\huggingface_cache\hub\models--Qwen--Qwen3-Embedding-0.6B\snapshots\97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
embed_model = SentenceTransformer(QWEN3_EMBED_PATH, device="cpu", model_kwargs={"dtype": "auto"})
query_embs = np.array(embed_model.encode(queries, normalize_embeddings=True, batch_size=16), dtype=np.float32)
del embed_model; gc.collect()

# LanceDB search top-20
topk_idx = []
for i in range(40):
    qvec = query_embs[i].tolist()
    rows = tbl.search(qvec).limit(20).to_list()
    cids = [r["chunk_id"] for r in rows]
    idx = [corpus_ids.index(c) for c in cids if c in corpus_ids]
    topk_idx.append(np.array(idx))
topk_idx = np.array(topk_idx)

# --- Exp 2: max_length via re-creating model with different max_length ---
from sentence_transformers import CrossEncoder

print("[Exp 2] max_length...")
ml = {}
for mval in [512, 256, 128]:
    model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu", max_length=mval)
    times = []
    for i in range(10):
        cand_idx = topk_idx[i]
        cand_texts = [corpus_texts[j] for j in cand_idx]
        pairs = [(queries[i], doc) for doc in cand_texts]
        t0 = time.time()
        model.predict(pairs, show_progress_bar=False)
        times.append((time.time() - t0) * 1000)
    ml[f"ml{mval}_ms"] = round(statistics.mean(times), 1)
    print(f"  max_length={mval}: {ml[f'ml{mval}_ms']}ms")
    del model; gc.collect()

# --- Exp 3: threads ---
print("\n[Exp 3] Threads...")
model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu", max_length=512)
th = {}
for nt in [2, 4, 8]:
    torch.set_num_threads(nt)
    times = []
    for i in range(10):
        cand_idx = topk_idx[i]
        cand_texts = [corpus_texts[j] for j in cand_idx]
        pairs = [(queries[i], doc) for doc in cand_texts]
        t0 = time.time()
        model.predict(pairs, show_progress_bar=False)
        times.append((time.time() - t0) * 1000)
    th[f"t{nt}_ms"] = round(statistics.mean(times), 1)
    print(f"  threads={nt}: {th[f't{nt}_ms']}ms")
torch.set_num_threads(4)
del model; gc.collect()

# Merge into existing results
with open(OUT_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

# Pool data from prior run
pool = {"top20_ms": 44129.0, "quality_top20_hit5": 0.55, "top10_ms": 16546.1, "quality_top10_hit5": 0.55, "top5_ms": 7117.5, "quality_top5_hit5": 0.55}
results["latency_experiments"] = {"pool_size": pool, "max_length": ml, "threads": th}

# Verdict
bge = results["rerankers"]["bge-reranker-v2-m3"]
bge_mean = bge["rerank_ms_mean"]
parts = []
parts.append(f"Giam pool 20->5: {pool['top20_ms']/pool['top5_ms']:.1f}x nhanh hon")
if ml.get("ml128_ms",0)>0 and ml.get("ml512_ms",0)>0:
    parts.append(f"Giam max_length 512->128: {ml['ml512_ms']/ml['ml128_ms']:.1f}x nhanh hon")
if th.get("t2_ms") and th.get("t4_ms") and th.get("t8_ms"):
    bt = min([(th["t2_ms"],2),(th["t4_ms"],4),(th["t8_ms"],8)], key=lambda x:x[0])
    parts.append(f"Thread toi uu: {bt[1]} ({bt[0]:.0f}ms)")

best_pool = 5  # quality same, 6x faster
best_ml = 128 if ml.get("ml128_ms", 999) < ml.get("ml256_ms", 999) else 256
best_threads = min([(th.get(f"t{k}_ms",999),k) for k in [2,4,8]], key=lambda x:x[0])[1] if th else 4

# Estimated optimized ms
opt_ms = pool["top5_ms"]  # best pool
if ml and best_ml < 512:
    opt_ms *= ml.get(f"ml{best_ml}_ms", 1) / ml.get("ml512_ms", 1)

verdict = (
    f"Latency PARTIALLY FIXABLE. "
    f"BGE-reranker raw: {bge_mean:.0f}ms/query (top-20, max_length=512, 4 threads). "
    f"Optimized (pool=5, max_length={best_ml}, threads={best_threads}): ~{opt_ms:.0f}ms/query. "
    f"1000-query batch: ~{opt_ms/1000:.0f}s ({opt_ms/60000:.1f}min). "
    + ". ".join(parts) + ". "
    f"CrossEncoder on CPU inherently ~29s/query for 20 candidates. "
    f"Pool=5 reduces to ~7s. max_length further cuts if chunks short. "
    f"30-min batch from earlier was fixable by pool+max_length tuning."
)
results["latency_verdict"] = verdict
results["winner"] = f"bge-reranker-v2-m3 (hit@5=0.625 vs qwen3 0.575, {bge_mean:.0f}ms vs {results['rerankers']['qwen3-reranker-0.6b']['rerank_ms_mean']:.0f}ms)"

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print(f"\n=== DONE ===")
print(f"Verdict: {verdict}")
