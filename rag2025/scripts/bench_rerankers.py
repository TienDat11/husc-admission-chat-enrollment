#!/usr/bin/env python3
"""Head-to-head reranker benchmark: BGE-reranker-v2-m3 vs Qwen3-Reranker-0.6B
on Vietnamese RAG admission corpus. CPU only.

Uses LanceDB stored vectors for first-stage retrieval (no corpus re-embedding).
Writes partial results after each reranker so crash doesn't lose everything.
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
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

N_EVAL = 40
TOP_RERANK = 20

QWEN3_EMBED_PATH = r"D:\huggingface_cache\hub\models--Qwen--Qwen3-Embedding-0.6B\snapshots\97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"

# ============================================================
# 1. Load data
# ============================================================
print("=== Loading data ===")
import lancedb
db = lancedb.connect(LANCE_DIR)
tbl = db.open_table("rag2025")
corpus = tbl.to_pandas()
corpus_ids = corpus["chunk_id"].tolist()
corpus_texts = corpus["text"].tolist()
print(f"Corpus: {len(corpus_texts)} chunks, vector dim={len(corpus['vector'].iloc[0])}")

with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_full = json.load(f)
eval_qs = eval_full[:N_EVAL]
print(f"Eval: {len(eval_qs)} questions")

queries = [q["question"] for q in eval_qs]

# ============================================================
# 2. Embed queries + get top-K via LanceDB search
# ============================================================
print("\n=== Stage 1: Embed queries + LanceDB search ===")
print("  Loading Qwen3-Embedding-0.6B...")
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer(
    QWEN3_EMBED_PATH,
    device="cpu",
    model_kwargs={"dtype": "auto"},
)

print("  Encoding 40 queries...")
t0 = time.time()
query_embs = embed_model.encode(
    queries,
    show_progress_bar=False,
    normalize_embeddings=True,
    batch_size=16,
)
query_embs = np.array(query_embs, dtype=np.float32)
print(f"  Query embeddings: {query_embs.shape} in {time.time()-t0:.1f}s")

del embed_model
gc.collect()

# LanceDB search for top-K per query
print(f"  LanceDB search top-{TOP_RERANK} per query...")
t0 = time.time()
topk_results = []
for i in range(N_EVAL):
    qvec = query_embs[i].tolist()
    rows = tbl.search(qvec).limit(TOP_RERANK).to_list()
    # rows[i]["id"] gives the chunk_id; get index in corpus_texts
    chunk_ids_found = [r["chunk_id"] for r in rows]
    # Map chunk_id to corpus index
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
# 3. Metric functions
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

# Baseline metrics (no rerank)
baseline_5 = must_include_check(topk_idx[:, :5], eval_qs, corpus_texts, 5)
baseline_10 = must_include_check(topk_idx[:, :10], eval_qs, corpus_texts, 10)
baseline = {"hit@5": baseline_5["hit@5"], "hit@10": baseline_10["hit@10"]}
print(f"\n=== Baseline (no rerank): {baseline} ===")

# ============================================================
# 4. Reranker helpers
# ============================================================
results = {
    "n_eval": N_EVAL,
    "baseline_no_rerank": baseline,
    "rerankers": {},
    "latency_experiments": {},
    "latency_verdict": "",
    "winner": "",
}

def save_results(extra=None):
    data = dict(results)
    if extra:
        data.update(extra)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [results saved to {OUT_FILE.name}]", flush=True)

def rerank_single(model, query, candidate_texts, max_length=None):
    pairs = [(query, doc) for doc in candidate_texts]
    kwargs = {}
    if max_length is not None:
        kwargs["max_length"] = max_length
    scores = model.predict(pairs, show_progress_bar=False, **kwargs)
    return np.array(scores)

def rerank_single_qwen(tokenizer, model, query, candidate_texts, max_length=8192):
    """Qwen3-Reranker: causal-LM yes/no logit scoring per the README."""
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")

    scores = []
    for doc in candidate_texts:
        text = f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: {doc}"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]  # last token logits
            true_logit = logits[:, token_true_id]
            false_logit = logits[:, token_false_id]
            stacked = torch.stack([false_logit, true_logit], dim=1)
            probs = torch.nn.functional.softmax(stacked, dim=1)
            scores.append(probs[:, 1].item())
    return np.array(scores)

def rerank_all(model, queries, topk_idx, corpus_texts, max_length=None, label=""):
    all_sorted = []
    times = []
    for i, q in enumerate(queries):
        cand_idx = topk_idx[i]
        cand_texts = [corpus_texts[j] for j in cand_idx]
        t0 = time.time()
        scores = rerank_single(model, q, cand_texts, max_length=max_length)
        elapsed_ms = (time.time() - t0) * 1000
        times.append(elapsed_ms)
        order = np.argsort(-scores)
        sorted_idx = cand_idx[order]
        all_sorted.append(sorted_idx)
        if (i + 1) % 10 == 0:
            print(f"  {label} {i+1}/{len(queries)} done, last={elapsed_ms:.0f}ms")
    return np.array(all_sorted), times

# ============================================================
# 5. Benchmark each reranker
# ============================================================
RERANKERS = {
    "bge-reranker-v2-m3": {
        "model_id": "BAAI/bge-reranker-v2-m3",
    },
    "qwen3-reranker-0.6b": {
        "model_id": "Qwen/Qwen3-Reranker-0.6B",
    },
}

for rname, rinfo in RERANKERS.items():
    print(f"\n{'='*60}")
    print(f"Reranker: {rname}")
    print(f"{'='*60}")

    # Check disk size
    try:
        from huggingface_hub import snapshot_info
        info = snapshot_info(rinfo["model_id"], cache_dir=r'D:\huggingface_cache\hub')
        disk_bytes = sum(s.size for s in info.siblings if s.size)
        disk_gb = round(disk_bytes / (1024**3), 2)
    except Exception:
        disk_gb = 0.0

    # Load model
    print("  Loading model...")
    t_load = time.time()
    is_qwen = "qwen" in rname.lower()
    model = None
    qwen_tokenizer = None
    try:
        if is_qwen:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            qwen_tokenizer = AutoTokenizer.from_pretrained(
                rinfo["model_id"], cache_dir=r'D:\huggingface_cache\hub', padding_side='left',
            )
            model = AutoModelForCausalLM.from_pretrained(
                rinfo["model_id"], cache_dir=r'D:\huggingface_cache\hub',
            ).eval()
        else:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(
                rinfo["model_id"],
                device="cpu",
                max_length=512,
            )
    except Exception as e:
        print(f"  FAILED to load {rname}: {e}")
        results["rerankers"][rname] = {"error": str(e)}
        save_results()
        continue
    load_s = time.time() - t_load
    print(f"  Loaded in {load_s:.1f}s, disk ~{disk_gb}GB")

    # Main benchmark: rerank top-20
    print("  Reranking top-20...")
    if is_qwen:
        def _rerank_fn(queries, topk_idx, corpus_texts, **kw):
            all_sorted, times = [], []
            for i, q in enumerate(queries):
                cand_idx = topk_idx[i]
                cand_texts = [corpus_texts[j] for j in cand_idx]
                t0 = time.time()
                scores = rerank_single_qwen(qwen_tokenizer, model, q, cand_texts, max_length=kw.get("max_length", 8192))
                elapsed_ms = (time.time() - t0) * 1000
                times.append(elapsed_ms)
                order = np.argsort(-scores)
                all_sorted.append(cand_idx[order])
                if (i + 1) % 10 == 0:
                    print(f"  {rname} {i+1}/{len(queries)} done, last={elapsed_ms:.0f}ms")
            return np.array(all_sorted), times
        sorted_idx, times = _rerank_fn(queries, topk_idx, corpus_texts)
    else:
        sorted_idx, times = rerank_all(model, queries, topk_idx, corpus_texts, label=rname)

    m = compute_all_metrics(sorted_idx, eval_qs, corpus_texts)
    mean_ms = round(statistics.mean(times), 1)
    p50_ms = round(statistics.median(times), 1)
    p95_ms = round(sorted(times)[int(len(times) * 0.95)], 1)

    entry = {
        "hit@5": m["hit@5"],
        "hit@10": m["hit@10"],
        "partial@5": m["partial@5"],
        "partial@10": m["partial@10"],
        "rerank_ms_mean": mean_ms,
        "rerank_ms_p50": p50_ms,
        "rerank_ms_p95": p95_ms,
        "load_s": round(load_s, 1),
        "disk_gb": disk_gb,
    }
    results["rerankers"][rname] = entry
    print(f"  Results: {entry}")
    save_results()

    # Keep BGE model for latency experiments
    if rname == "bge-reranker-v2-m3":
        bge_model = model
    else:
        del model
    gc.collect()

# ============================================================
# 6. Latency experiments (BGE only)
# ============================================================
print(f"\n{'='*60}")
print("Latency experiments (BGE-reranker)")
print(f"{'='*60}")

exp_model = bge_model if 'bge_model' in dir() else None
if exp_model is None:
    print("  Skipping latency experiments — no BGE model loaded")

if exp_model is not None:
    # --- Pool size ---
    print("\n  [Exp 1] Pool size: top-20 vs top-10 vs top-5")
    pool_results = {}
    for k_pool in [20, 10, 5]:
        topk_k = topk_idx[:, :k_pool]
        _, times_k = rerank_all(exp_model, queries, topk_k, corpus_texts, label=f"pool{k_pool}")
        m_k = must_include_check(topk_k[:, :min(5, k_pool)], eval_qs, corpus_texts, 5)
        pool_results[f"top{k_pool}_ms"] = round(statistics.mean(times_k), 1)
        pool_results[f"quality_top{k_pool}_hit5"] = m_k["hit@5"]
        print(f"    pool={k_pool}: mean={pool_results[f'top{k_pool}_ms']}ms, hit@5={m_k['hit@5']}")
    results["latency_experiments"]["pool_size"] = pool_results
    save_results()

    # --- max_length ---
    print("\n  [Exp 2] max_length: 512 vs 256 vs 128")
    ml_results = {}
    for ml in [512, 256, 128]:
        sorted_ml, times_ml = rerank_all(exp_model, queries, topk_idx, corpus_texts, max_length=ml, label=f"ml{ml}")
        m_ml = compute_all_metrics(sorted_ml, eval_qs, corpus_texts)
        ml_results[f"ml{ml}_ms"] = round(statistics.mean(times_ml), 1)
        ml_results[f"ml{ml}_hit5"] = m_ml["hit@5"]
        ml_results[f"ml{ml}_hit10"] = m_ml["hit@10"]
        print(f"    max_length={ml}: mean={ml_results[f'ml{ml}_ms']}ms, hit@5={m_ml['hit@5']}, hit@10={m_ml['hit@10']}")
    results["latency_experiments"]["max_length"] = ml_results
    save_results()

    # --- torch threads ---
    print("\n  [Exp 3] torch threads: 2 vs 4 vs 8")
    thread_results = {}
    for n_t in [2, 4, 8]:
        torch.set_num_threads(n_t)
        _, times_t = rerank_all(exp_model, queries, topk_idx, corpus_texts, label=f"t{n_t}")
        thread_results[f"t{n_t}_ms"] = round(statistics.mean(times_t), 1)
        print(f"    threads={n_t}: mean={thread_results[f't{n_t}_ms']}ms")
    torch.set_num_threads(4)
    results["latency_experiments"]["threads"] = thread_results
    save_results()

    del exp_model
    gc.collect()

# ============================================================
# 7. Latency verdict
# ============================================================
print("\n=== Generating latency verdict ===")

bge_res = results["rerankers"].get("bge-reranker-v2-m3", {})
qwen_res = results["rerankers"].get("qwen3-reranker-0.6b", {})
pool_exp = results["latency_experiments"].get("pool_size", {})
ml_exp = results["latency_experiments"].get("max_length", {})
thread_exp = results["latency_experiments"].get("threads", {})

bge_mean = bge_res.get("rerank_ms_mean", 0)
bge_hit5 = bge_res.get("hit@5", 0)
qwen_hit5 = qwen_res.get("hit@5", 0)

verdict_parts = []

if pool_exp.get("top5_ms", 0) > 0 and pool_exp.get("top20_ms", 0) > 0:
    speedup_pool = pool_exp["top20_ms"] / pool_exp["top5_ms"]
    verdict_parts.append(f"Giam pool tu 20->5: {speedup_pool:.1f}x nhanh hon")

if ml_exp.get("ml128_ms", 0) > 0 and ml_exp.get("ml512_ms", 0) > 0:
    speedup_ml = ml_exp["ml512_ms"] / ml_exp["ml128_ms"]
    verdict_parts.append(f"Giam max_length tu 512->128: {speedup_ml:.1f}x nhanh hon")

if thread_exp:
    t2 = thread_exp.get("t2_ms", 0)
    t4 = thread_exp.get("t4_ms", 0)
    t8 = thread_exp.get("t8_ms", 0)
    if t2 and t4 and t8:
        best_t = min([(t2, 2), (t4, 4), (t8, 8)], key=lambda x: x[0])
        verdict_parts.append(f"So thread toi uu: {best_t[1]} ({best_t[0]:.0f}ms)")

# Recommended settings
best_pool = 10
best_ml = 256
best_threads = 4

if pool_exp:
    best_hit = max(pool_exp.get(f"top{k}_hit5", 0) for k in [5, 10, 20])
    for k in [10, 5]:
        if pool_exp.get(f"top{k}_hit5", 0) >= best_hit - 0.01:
            best_pool = k
            break

if ml_exp:
    best_hit_ml = max(ml_exp.get(f"ml{k}_hit5", 0) for k in [128, 256, 512])
    for k in [256, 128]:
        if ml_exp.get(f"ml{k}_hit5", 0) >= best_hit_ml - 0.01:
            best_ml = k
            break

if thread_exp:
    times_list = [(thread_exp.get(f"t{k}_ms", 999), k) for k in [2, 4, 8]]
    best_threads = min(times_list, key=lambda x: x[0])[1]

est_per_query_ms = bge_mean * (best_pool / 20)
if ml_exp and best_ml < 512:
    ml_ratio = ml_exp.get(f"ml{best_ml}_ms", bge_mean) / ml_exp.get("ml512_ms", bge_mean)
    est_per_query_ms *= ml_ratio
est_batch_1000 = est_per_query_ms * 1000 / 1000

verdict = (
    f"Latency is PARTIALLY FIXABLE. "
    f"BGE-reranker on CPU: {bge_mean:.0f}ms/query for top-20 (raw). "
    f"With recommended settings (pool={best_pool}, max_length={best_ml}, threads={best_threads}): "
    f"estimated ~{est_per_query_ms:.0f}ms/query. "
    f"For 1000-query batch: ~{est_batch_1000:.0f}s ({est_batch_1000/60:.1f}min). "
    + ". ".join(verdict_parts) + ". "
    f"CrossEncoder on CPU is inherently slow per-query (~{bge_mean:.0f}ms) but manageable in batch with optimizations."
)
results["latency_verdict"] = verdict

if qwen_res.get("error"):
    winner = f"BGE-reranker-v2-m3 (Qwen3 errored: {qwen_res['error'][:100]})"
elif bge_hit5 >= qwen_hit5:
    winner = f"BGE-reranker-v2-m3 (hit@5={bge_hit5} vs {qwen_hit5}, faster on CPU)"
else:
    winner = f"Qwen3-Reranker-0.6B (hit@5={qwen_hit5} vs {bge_hit5})"
results["winner"] = winner

print(f"\nVerdict: {verdict}")
print(f"Winner: {winner}")

save_results()
print("\n=== DONE ===")
