#!/usr/bin/env python3
"""Verify safe reranker max_length for bge-reranker-v2-m3.

JOB 1 output: results/rerank_maxlen_verify.json

Steps:
  1. Corpus chunk length distribution (tokenized by bge-reranker tokenizer)
  2. Rerank quality at max_length in {128, 256, 512}
  3. Recommend safe max_length (smallest ml whose hit@5 is within 0.025 of ml=512)
"""
import os
os.environ['HF_HOME'] = r'D:\huggingface_cache'
os.environ['HF_HUB_CACHE'] = r'D:\huggingface_cache\hub'
os.environ['TORCH_HOME'] = r'D:\torch_cache'
os.environ['TMPDIR'] = r'D:\tmp'
os.environ['TEMP'] = r'D:\tmp'
os.environ['TMP'] = r'D:\tmp'
os.environ['MPLCONFIGDIR'] = r'D:\tmp\mpl'
os.makedirs(r'D:\tmp', exist_ok=True)
os.makedirs(r'D:\torch_cache', exist_ok=True)

import torch
torch.set_num_threads(8)

import json, time, gc, statistics
from pathlib import Path
import numpy as np

BASE = Path(r"D:\chunking\husc-admission-chat-enrollment\rag2025")
LANCE_DIR = str(BASE / "data" / "lancedb")
EVAL_FILE = BASE / "data" / "eval" / "husc_86q_test.json"
OUT_FILE = BASE / "results" / "rerank_maxlen_verify.json"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

N_EVAL = 40
TOP_RERANK = 20
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
EMBED_MODEL_PATH = r"D:\huggingface_cache\hub\models--Qwen--Qwen3-Embedding-0.6B\snapshots\97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"

print = __builtins__["print"] if isinstance(__builtins__, dict) else __builtins__.print
import functools
print = functools.partial(print, flush=True)

# ============================================================
# 1. Corpus chunk length distribution
# ============================================================
print("=== Step 1: Corpus chunk length distribution ===")
import lancedb
db = lancedb.connect(LANCE_DIR)
tbl = db.open_table("rag2025")
df = tbl.to_pandas()
corpus_ids = df["chunk_id"].tolist()
corpus_texts = df["text"].tolist()
print(f"Corpus: {len(corpus_texts)} chunks")

# Load bge-reranker tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    RERANKER_MODEL, cache_dir=r"D:\huggingface_cache\hub"
)

token_lengths = []
for t in corpus_texts:
    toks = tokenizer(t, truncation=False, return_tensors="pt")
    token_lengths.append(toks["input_ids"].shape[1])

token_lengths = np.array(token_lengths)
median = int(np.median(token_lengths))
p90 = int(np.percentile(token_lengths, 90))
p95 = int(np.percentile(token_lengths, 95))
p99 = int(np.percentile(token_lengths, 99))
mx = int(np.max(token_lengths))
total = len(token_lengths)

pct_128 = round(100 * np.sum(token_lengths > 128) / total, 2)
pct_256 = round(100 * np.sum(token_lengths > 256) / total, 2)
pct_512 = round(100 * np.sum(token_lengths > 512) / total, 2)

distro = {
    "total_chunks": total,
    "median_tokens": median,
    "p90_tokens": p90,
    "p95_tokens": p95,
    "p99_tokens": p99,
    "max_tokens": mx,
    "pct_exceeds_128": pct_128,
    "pct_exceeds_256": pct_256,
    "pct_exceeds_512": pct_512,
    "count_exceeds_128": int(np.sum(token_lengths > 128)),
    "count_exceeds_256": int(np.sum(token_lengths > 256)),
    "count_exceeds_512": int(np.sum(token_lengths > 512)),
}
print(f"  Median: {median} tokens, p90: {p90}, p95: {p95}, p99: {p99}, max: {mx}")
print(f"  Exceeds 128: {pct_128}% ({distro['count_exceeds_128']}/{total})")
print(f"  Exceeds 256: {pct_256}% ({distro['count_exceeds_256']}/{total})")
print(f"  Exceeds 512: {pct_512}% ({distro['count_exceeds_512']}/{total})")

del tokenizer
gc.collect()

# ============================================================
# 2. Load eval questions + embed queries + LanceDB search
# ============================================================
print("\n=== Step 2: Load eval + first-stage retrieval ===")
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_full = json.load(f)
eval_qs = eval_full[:N_EVAL]
print(f"  Eval: {len(eval_qs)} questions")

queries = [q["question"] for q in eval_qs]

print("  Loading Qwen3-Embedding-0.6B...")
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer(
    EMBED_MODEL_PATH,
    device="cpu",
    model_kwargs={"dtype": "auto"},
)

print("  Encoding 40 queries...")
t0 = time.time()
query_embs = embed_model.encode(
    queries, show_progress_bar=False, normalize_embeddings=True, batch_size=16,
)
query_embs = np.array(query_embs, dtype=np.float32)
print(f"  Query embeddings: {query_embs.shape} in {time.time()-t0:.1f}s")

del embed_model
gc.collect()

print(f"  LanceDB search top-{TOP_RERANK} per query...")
t0 = time.time()
topk_results = []
for i in range(N_EVAL):
    qvec = query_embs[i].tolist()
    rows = tbl.search(qvec).limit(TOP_RERANK).to_list()
    chunk_ids_found = [r["chunk_id"] for r in rows]
    indices = []
    for cid in chunk_ids_found:
        try:
            indices.append(corpus_ids.index(cid))
        except ValueError:
            pass
    topk_results.append(np.array(indices))
topk_idx = np.array(topk_results)
print(f"  Search done in {time.time()-t0:.1f}s")

# ============================================================
# 3. Metric helpers
# ============================================================
def must_include_check(topk_indices, questions, corpus_texts, K):
    hits = 0
    partials = []
    for i, q in enumerate(questions):
        topk_text = " ".join(corpus_texts[j] for j in topk_indices[i])
        must = q.get("must_include", [])
        if not must:
            hits += 1
            partials.append(1.0)
            continue
        found = sum(1 for m in must if m.lower() in topk_text.lower())
        partials.append(found / len(must))
        if found == len(must):
            hits += 1
    n = len(questions)
    return {
        f"hit@{K}": round(hits / n, 4),
        f"partial@{K}": round(sum(partials) / n, 4),
    }

def compute_all_metrics(topk_indices, questions, corpus_texts):
    m5 = must_include_check(topk_indices[:, :5], questions, corpus_texts, 5)
    m10 = must_include_check(topk_indices[:, :10], questions, corpus_texts, 10)
    return {**m5, **m10}

# ============================================================
# 4. Rerank at each max_length
# ============================================================
print("\n=== Step 3: Rerank quality per max_length ===")
from sentence_transformers import CrossEncoder

results_table = {}

for ml in [512, 256, 128]:
    print(f"\n  --- max_length={ml} ---")
    model = CrossEncoder(
        RERANKER_MODEL, device="cpu", max_length=ml,
    )

    all_sorted = []
    times = []
    for i, q in enumerate(queries):
        cand_idx = topk_idx[i]
        cand_texts = [corpus_texts[j] for j in cand_idx]
        pairs = [(q, doc) for doc in cand_texts]
        t0 = time.time()
        scores = model.predict(pairs, show_progress_bar=False)
        elapsed_ms = (time.time() - t0) * 1000
        times.append(elapsed_ms)
        order = np.argsort(-scores)
        sorted_idx = cand_idx[order]
        all_sorted.append(sorted_idx)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(queries)} done, last={elapsed_ms:.0f}ms")

    all_sorted = np.array(all_sorted)
    m = compute_all_metrics(all_sorted, eval_qs, corpus_texts)
    mean_ms = round(statistics.mean(times), 1)
    p50_ms = round(statistics.median(times), 1)

    results_table[f"ml{ml}"] = {
        "hit@5": m["hit@5"],
        "hit@10": m["hit@10"],
        "partial@5": m["partial@5"],
        "partial@10": m["partial@10"],
        "latency_ms_mean": mean_ms,
        "latency_ms_p50": p50_ms,
    }
    print(f"    hit@5={m['hit@5']}, hit@10={m['hit@10']}, partial@5={m['partial@5']}, latency={mean_ms:.0f}ms")

    del model
    gc.collect()

# ============================================================
# 5. Decide safe max_length
# ============================================================
print("\n=== Step 4: Recommend safe max_length ===")
ml512_hit5 = results_table["ml512"]["hit@5"]
THRESHOLD = 0.025  # within 1 point

recommended = "512"
reason = ""
for candidate_ml in ["256", "128"]:
    cand_hit5 = results_table[f"ml{candidate_ml}"]["hit@5"]
    delta = ml512_hit5 - cand_hit5
    if delta <= THRESHOLD:
        recommended = candidate_ml
        reason = (
            f"ml={candidate_ml} hit@5={cand_hit5} is within {delta:.4f} of ml=512 hit@5={ml512_hit5} "
            f"(threshold={THRESHOLD}). Safe to use."
        )
        break
    else:
        reason = (
            f"ml={candidate_ml} hit@5={cand_hit5} loses {delta:.4f} vs ml=512 hit@5={ml512_hit5} "
            f"(>{THRESHOLD}). Not safe."
        )

# If neither 256 nor 128 is safe, recommend 512
if recommended == "512":
    ml256_delta = ml512_hit5 - results_table["ml256"]["hit@5"]
    reason = (
        f"ml=256 hit@5={results_table['ml256']['hit@5']} loses {ml256_delta:.4f} vs ml=512 hit@5={ml512_hit5} "
        f"(>{THRESHOLD}). Recommend ml=512 for quality safety."
    )

rec_ml_int = int(recommended)
pct_truncated_at_rec = distro[f"pct_exceeds_{rec_ml_int}"]
count_truncated_at_rec = distro[f"count_exceeds_{rec_ml_int}"]

print(f"  Recommended max_length: {recommended}")
print(f"  Reasoning: {reason}")
print(f"  Chunks truncated at ml={recommended}: {pct_truncated_at_rec}% ({count_truncated_at_rec}/{distro['total_chunks']})")

# ============================================================
# 6. Save output
# ============================================================
output = {
    "chunk_length_distribution": distro,
    "quality_per_max_length": results_table,
    "recommended_max_length": int(recommended),
    "reasoning": reason,
    "pct_chunks_truncated_at_recommended": pct_truncated_at_rec,
    "count_chunks_truncated_at_recommended": count_truncated_at_rec,
}

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n=== Results saved to {OUT_FILE} ===")
print("=== DONE ===")
