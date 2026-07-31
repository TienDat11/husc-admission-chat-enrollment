#!/usr/bin/env python3
"""LEAN embedding benchmark: Qwen3 baseline vs Vietnamese-finetuned models.
E2Rank dropped (too slow on CPU). 3 configs only.
All corpus/query embeddings pre-cached. Config 3 loads CrossEncoder only.
Key optimization: rerank scores computed ONCE per query, sliced for K=5 and K=10.
"""
import os
os.environ['HF_HOME'] = r'D:\huggingface_cache'
os.environ['HF_HUB_CACHE'] = r'D:\huggingface_cache\hub'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TORCH_HOME'] = r'D:\torch_cache'
os.environ['TMPDIR'] = r'D:\tmp'
os.environ['TEMP'] = r'D:\tmp'
os.environ['TMP'] = r'D:\tmp'
os.environ['MPLCONFIGDIR'] = r'D:\tmp\mpl'
os.makedirs(r'D:\tmp', exist_ok=True)
os.makedirs(r'D:\torch_cache', exist_ok=True)

import torch
torch.set_num_threads(4)

import sys, json, time, traceback, gc
from pathlib import Path
import numpy as np
import functools
print = functools.partial(print, flush=True)

BASE = Path(r"D:\chunking\husc-admission-chat-enrollment\rag2025")
LANCE_DIR = str(BASE / "data" / "lancedb")
EVAL_FILE = BASE / "data" / "eval" / "husc_86q_test.json"
HARD_FILE = BASE / "data" / "eval" / "husc_thi_sinh_thuc.json"
OUT_FILE = BASE / "results" / "embed_rerank_bench.json"
PARTIAL_FILE = BASE / "results" / "embed_rerank_bench.partial.json"
CACHE_DIR = BASE / "results" / "_bench_cache"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
N_EVAL = 40
K_VALUES = [5, 10]
TOP_RERANK = 20
QWEN3_CORPUS_NPY = CACHE_DIR / "qwen3_baseline_corpus.npy"
VN_CORPUS_NPY = CACHE_DIR / "vn_embed_v2_corpus.npy"
VN_QUERY_NPY = CACHE_DIR / "vn_embed_v2_query_40.npy"
VN_HARD_NPY = CACHE_DIR / "vn_embed_v2_hard_3.npy"
VN_RERANK_PATH = r"D:\huggingface_cache\hub\models--AITeamVN--Vietnamese_Reranker\snapshots\f536976248403314225d7fdfdbc87f0e9516a54e"
VN_EMBED_PATH = r"D:\huggingface_cache\hub\models--AITeamVN--Vietnamese_Embedding_v2\snapshots\18b44161e041bf1d3a333ab5144b5b7b93f914d2"
QWEN3_PATH = r"D:\huggingface_cache\hub\models--Qwen--Qwen3-Embedding-0.6B\snapshots\97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"

# -- Load data --
print("=== Loading data ===", flush=True)
import lancedb
db = lancedb.connect(LANCE_DIR)
tbl = db.open_table("rag2025")
corpus = tbl.to_pandas()
corpus_ids = corpus["chunk_id"].tolist()
corpus_texts = corpus["text"].tolist()
print(f"Corpus: {len(corpus_texts)} chunks", flush=True)

with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_full = json.load(f)
eval_qs = eval_full[:N_EVAL]
print(f"Eval: {len(eval_qs)} questions", flush=True)

with open(HARD_FILE, "r", encoding="utf-8") as f:
    hard_full = json.load(f)
hard_ids = ["msg029", "msg051", "msg080"]
hard_qs = [h for h in hard_full if h["id"] in hard_ids]
print(f"Hard questions: {len(hard_qs)}", flush=True)

def cosine_sim_matrix(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T

def compute_metrics(query_embs, corpus_embs, questions, k_vals):
    sims = cosine_sim_matrix(query_embs, corpus_embs)
    metrics = {}
    for K in k_vals:
        topk_idx = np.argsort(-sims, axis=1)[:, :K]
        hits = 0
        partials = []
        for i, q in enumerate(questions):
            topk_text = " ".join(corpus_texts[j] for j in topk_idx[i])
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
        metrics[f"hit@{K}"] = round(hits / n, 4)
        metrics[f"partial@{K}"] = round(sum(partials) / n, 4)
    return metrics

def embed_texts(model, texts, batch_size=16):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embs.append(np.array(embs))
    return np.vstack(all_embs)

def save_partial(results, extra=None):
    clean = [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in results]
    data = {"configs": clean, "status": "partial"}
    if extra:
        data.update(extra)
    with open(PARTIAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [partial saved]", flush=True)

results = []

def run_config(name, fn):
    print(f"\n{'='*60}", flush=True)
    print(f"Config: {name}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    try:
        res = fn()
        elapsed = round(time.time() - t0, 1)
        res["config"] = name
        res["time_s"] = elapsed
        results.append(res)
        print(f"  -> DONE in {elapsed}s | hit@5={res.get('hit@5','?')} hit@10={res.get('hit@10','?')}", flush=True)
        save_partial(results)
        return res
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        tb = traceback.format_exc()
        print(f"  -> FAILED in {elapsed}s: {e}", flush=True)
        res = {"config": name, "error": str(e), "time_s": elapsed}
        results.append(res)
        save_partial(results)
        return res

# -- Config 1: Qwen3 baseline --
def config_qwen3():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(QWEN3_PATH)
    corpus_embs = np.load(str(QWEN3_CORPUS_NPY))
    print(f"  Loaded cached corpus: {corpus_embs.shape}", flush=True)
    query_embs = embed_texts(model, [q["question"] for q in eval_qs])
    metrics = compute_metrics(query_embs, corpus_embs, eval_qs, K_VALUES)
    hard_embs = embed_texts(model, [h["question"] for h in hard_qs])
    sims_hard = cosine_sim_matrix(hard_embs, corpus_embs)
    hard_results = {}
    for i, h in enumerate(hard_qs):
        top3 = np.argsort(-sims_hard[i])[:3]
        hard_results[h["id"]] = [
            {"chunk_id": corpus_ids[j], "text_preview": corpus_texts[j][:100]}
            for j in top3
        ]
    del model; gc.collect()
    return {**metrics, "hard_questions": hard_results}

run_config("qwen3_baseline", config_qwen3)

# -- Config 2: Vietnamese Embedding v2 --
def config_vn_embed():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(VN_EMBED_PATH)
    if VN_CORPUS_NPY.exists():
        corpus_embs = np.load(str(VN_CORPUS_NPY))
        print(f"  Loaded cached VN corpus: {corpus_embs.shape}", flush=True)
    else:
        t0 = time.time()
        corpus_embs = embed_texts(model, corpus_texts)
        print(f"  Corpus embed: {round(time.time()-t0,1)}s", flush=True)
        np.save(str(VN_CORPUS_NPY), corpus_embs)
    query_embs = embed_texts(model, [q["question"] for q in eval_qs])
    metrics = compute_metrics(query_embs, corpus_embs, eval_qs, K_VALUES)
    np.save(str(VN_QUERY_NPY), query_embs)
    hard_embs = embed_texts(model, [h["question"] for h in hard_qs])
    np.save(str(VN_HARD_NPY), hard_embs)
    sims_hard = cosine_sim_matrix(hard_embs, corpus_embs)
    hard_results = {}
    for i, h in enumerate(hard_qs):
        top3 = np.argsort(-sims_hard[i])[:3]
        hard_results[h["id"]] = [
            {"chunk_id": corpus_ids[j], "text_preview": corpus_texts[j][:100]}
            for j in top3
        ]
    del model; gc.collect()
    return {**metrics, "hard_questions": hard_results}

run_config("vn_embed_v2", config_vn_embed)

# -- Config 3: VN Embed v2 + VN Reranker --
def config_vn_reranker():
    from sentence_transformers import CrossEncoder
    corpus_embs = np.load(str(VN_CORPUS_NPY))
    query_embs = np.load(str(VN_QUERY_NPY))
    hard_embs = np.load(str(VN_HARD_NPY))
    print(f"  Loaded cached: corpus={corpus_embs.shape}, query={query_embs.shape}, hard={hard_embs.shape}", flush=True)

    # Pre-compute top-TOP_RERANK per query
    sims = cosine_sim_matrix(query_embs, corpus_embs)
    topk_idx = np.argsort(-sims, axis=1)[:, :TOP_RERANK]

    # Load CrossEncoder
    ce = CrossEncoder(VN_RERANK_PATH, max_length=512)
    print(f"  CrossEncoder loaded", flush=True)

    # Rerank: compute scores ONCE per query, use for both K values
    # Pre-build all (query, candidate) pairs for batch prediction
    all_pairs = []
    pair_offsets = []  # (start_idx, n_candidates) per query
    for i, q in enumerate(eval_qs):
        start = len(all_pairs)
        candidates = [corpus_texts[j] for j in topk_idx[i]]
        for c in candidates:
            all_pairs.append((q["question"], c))
        pair_offsets.append((start, len(candidates)))
    print(f"  Total rerank pairs: {len(all_pairs)} (40 queries x {TOP_RERANK} cands)", flush=True)

    # Batch predict ALL pairs at once
    t0 = time.time()
    all_scores = ce.predict(all_pairs, show_progress_bar=True, batch_size=32)
    all_scores = np.array(all_scores)
    print(f"  Batch rerank done in {round(time.time()-t0,1)}s", flush=True)

    # Compute metrics for each K
    metrics = {}
    for K in K_VALUES:
        hits = 0
        partials = []
        for i, q in enumerate(eval_qs):
            start, n_cands = pair_offsets[i]
            q_scores = all_scores[start:start+n_cands]
            reranked_local = np.argsort(-q_scores)[:K]
            candidates = [corpus_texts[j] for j in topk_idx[i]]
            topk_text = " ".join(candidates[j] for j in reranked_local)
            must = q.get("must_include", [])
            if not must:
                hits += 1
                partials.append(1.0)
            else:
                found = sum(1 for m in must if m.lower() in topk_text.lower())
                partials.append(found / len(must))
                if found == len(must):
                    hits += 1
        n = len(eval_qs)
        metrics[f"hit@{K}"] = round(hits / n, 4)
        metrics[f"partial@{K}"] = round(sum(partials) / n, 4)
        print(f"  K={K}: hit@{K}={metrics[f'hit@{K}']}, partial@{K}={metrics[f'partial@{K}']}", flush=True)

    # Hard questions - also batch rerank
    sims_hard = cosine_sim_matrix(hard_embs, corpus_embs)
    hard_pairs = []
    hard_pair_offsets = []
    for i, h in enumerate(hard_qs):
        start = len(hard_pairs)
        top_candidates_idx = np.argsort(-sims_hard[i])[:TOP_RERANK]
        candidates = [corpus_texts[j] for j in top_candidates_idx]
        for c in candidates:
            hard_pairs.append((h["question"], c))
        hard_pair_offsets.append((start, len(candidates), top_candidates_idx))
    hard_scores = ce.predict(hard_pairs, show_progress_bar=False, batch_size=32)
    hard_scores = np.array(hard_scores)
    hard_results = {}
    for i, h in enumerate(hard_qs):
        start, n_cands, top_idx = hard_pair_offsets[i]
        h_scores = hard_scores[start:start+n_cands]
        reranked_local = np.argsort(-h_scores)[:3]
        candidates = [corpus_texts[j] for j in top_idx]
        hard_results[h["id"]] = [
            {"chunk_id": corpus_ids[top_idx[j]], "text_preview": candidates[j][:100]}
            for j in reranked_local
        ]
    del ce; gc.collect()
    return {**metrics, "hard_questions": hard_results}

run_config("vn_embed_v2+vn_reranker", config_vn_reranker)

# -- Final output --
print(f"\n{'='*60}", flush=True)
print("FINAL RESULTS", flush=True)
print(f"{'='*60}", flush=True)

qwen3_hit5 = results[0].get("hit@5", 0) if "error" not in results[0] else 0
winner_acc = "qwen3_baseline"
winner_acc_val = qwen3_hit5
for r in results:
    if "error" not in r and r.get("hit@5", 0) > winner_acc_val:
        winner_acc = r["config"]
        winner_acc_val = r["hit@5"]

winner_speed = min(results, key=lambda r: r.get("time_s", 9999))["config"]

best_hard = {}
for r in results:
    if "error" not in r and r.get("hit@5", 0) == winner_acc_val:
        best_hard = r.get("hard_questions", {})
        break

final = {
    "configs": [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in results],
    "hard_questions": best_hard,
    "winner_by_accuracy": winner_acc,
    "winner_by_speed": winner_speed,
    "qwen3_known_benchmark": {
        "hit@5": 0.6167, "hit@10": 0.6667,
        "partial@5": 0.6778, "partial@10": 0.725,
        "query_ms_mean": 199.56
    },
    "note": "e2rank_skipped: too slow on CPU (Qwen3-0.6B arch, not VN-finetuned). vn_embed_v2+vn_reranker: CrossEncoder batch-reranked 800 pairs once.",
    "eval_size": N_EVAL,
    "corpus_size": len(corpus_texts)
}

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)
print(f"\nWrote {OUT_FILE}", flush=True)
print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
