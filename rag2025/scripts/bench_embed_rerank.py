#!/usr/bin/env python3
"""Benchmark embedding + reranker models on Vietnamese RAG admission corpus.
MEASUREMENT ONLY - no production code touched.
Resume-aware: skips configs whose corpus .npy already cached.
Config order: Qwen3 (cached) -> VN_embed -> VN+Reranker -> E2Rank (last, time-boxed).
Writes partial JSON after each config so crashes lose minimal progress.
"""
# -- Env (BEFORE any HF/torch imports) --
import os
os.environ['HF_HOME'] = r'D:\huggingface_cache'
os.environ['HF_HUB_CACHE'] = r'D:\huggingface_cache\hub'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TORCH_HOME'] = r'D:\torch_cache'
os.environ['TMPDIR'] = r'D:\tmp'
os.environ['TEMP'] = r'D:\tmp'
os.environ['TMP'] = r'D:\tmp'
os.environ['MPLCONFIGDIR'] = r'D:\tmp\mpl'
os.environ['XDG_CACHE_HOME'] = r'D:\tmp\xdg'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.makedirs(r'D:\tmp', exist_ok=True)
os.makedirs(r'D:\torch_cache', exist_ok=True)

import torch
torch.set_num_threads(2)

import sys, json, time, traceback, gc
from pathlib import Path
import numpy as np
import functools
print = functools.partial(print, flush=True)

# -- Paths --
BASE = Path(r"D:\chunking\husc-admission-chat-enrollment\rag2025")
LANCE_DIR = str(BASE / "data" / "lancedb")
EVAL_FILE = BASE / "data" / "eval" / "husc_86q_test.json"
HARD_FILE = BASE / "data" / "eval" / "husc_thi_sinh_thuc.json"
OUT_FILE = BASE / "results" / "embed_rerank_bench.json"
PARTIAL_FILE = BASE / "results" / "embed_rerank_bench.partial.json"
CACHE_DIR = BASE / "results" / "_bench_cache"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
N_EVAL = 60
K_VALUES = [5, 10]
TOP_RERANK = 20
BATCH_CORPUS = 16
BATCH_QUERY = 16
E2RANK_TIMEBOX_S = 900  # 15 min max for E2Rank corpus embed
TASK_DESC = "Given a web search query, retrieve relevant passages that answer the query"

# -- Model snapshot paths (offline cache, avoids HF_HUB resolution issues) --
_SNAP = Path(r"D:\huggingface_cache\hub")
QWEN3_SNAP = str(_SNAP / "models--Qwen--Qwen3-Embedding-0.6B" / "snapshots" / "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
VN_EMBED_SNAP = str(_SNAP / "models--AITeamVN--Vietnamese_Embedding_v2" / "snapshots" / "18b44161e041bf1d3a333ab5144b5b7b93f914d2")
VN_RERANK_SNAP = str(_SNAP / "models--AITeamVN--Vietnamese_Reranker" / "snapshots" / "f536976248403314225d7fdfdbc87f0e9516a54e")
E2RANK_SNAP = str(_SNAP / "models--Alibaba-NLP--E2Rank-0.6B" / "snapshots" / "7c82f7c8a74f14edcb5668ebe928d6865874afd5")


# -- Partial-save helper --
def save_partial(results):
    with open(PARTIAL_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  [partial saved: {PARTIAL_FILE.name}]")


# -- Load corpus from LanceDB --
print("Loading corpus from LanceDB...")
import lancedb
db = lancedb.connect(LANCE_DIR)
tbl = db.open_table("rag2025")
rows = tbl.to_pandas()
chunk_ids = rows["chunk_id"].tolist()
texts = rows["text"].tolist()
print(f"  Corpus: {len(texts)} chunks")

with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_all = json.load(f)
eval_qs = eval_all[:N_EVAL]
print(f"  Eval: {len(eval_qs)} questions (first {N_EVAL} of {len(eval_all)})")

with open(HARD_FILE, "r", encoding="utf-8") as f:
    hard_all = json.load(f)
hard_map = {q["id"]: q for q in hard_all}
HARD_IDS = ["msg029", "msg051", "msg080"]


# -- Helpers --
def cosine_sim_matrix(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def eval_metrics(query_embeds, corpus_embeds, eval_qs, texts, k_values):
    sims = cosine_sim_matrix(query_embeds, corpus_embeds)
    results = {}
    for k in k_values:
        hits = 0
        partials = []
        for i, q in enumerate(eval_qs):
            topk_idx = np.argsort(sims[i])[-k:][::-1]
            topk_texts = " ".join([texts[j] for j in topk_idx])
            must = q.get("must_include", [])
            if not must:
                hits += 1
                partials.append(1.0)
                continue
            found = sum(1 for m in must if m.lower() in topk_texts.lower())
            if found == len(must):
                hits += 1
            partials.append(found / len(must))
        results[f"hit@{k}"] = round(hits / len(eval_qs), 4)
        results[f"partial@{k}"] = round(float(np.mean(partials)), 4)
    return results


def embed_e2rank(texts_list, model, tokenizer, is_query=False, task_desc=TASK_DESC, batch_size=16):
    import torch.nn.functional as F

    def last_token_pool(last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            seq_lens = attention_mask.sum(dim=1) - 1
            bs = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(bs, device=last_hidden_states.device), seq_lens]

    if is_query:
        input_texts = [f"Instruct: {task_desc}\nQuery:{q}" for q in texts_list]
    else:
        input_texts = [t + "" for t in texts_list]

    all_embs = []
    for start in range(0, len(input_texts), batch_size):
        batch = input_texts[start:start+batch_size]
        batch_dict = tokenizer(batch, padding=True, truncation=True, max_length=8192, return_tensors="pt")
        batch_dict.to(model.device)
        with torch.no_grad():
            outputs = model(**batch_dict)
            embs = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().numpy())
        del outputs, batch_dict
        gc.collect()
    return np.concatenate(all_embs, axis=0)


# ============================================================
# MAIN BENCHMARK
# ============================================================
results = {
    "k_values": K_VALUES,
    "n_eval_questions": len(eval_qs),
    "configs": {},
    "hard_questions": {},
    "winner_by_accuracy": "",
    "winner_by_speed": "",
}

queries = [q["question"] for q in eval_qs]


# -------------------------------------------------------
# Config A: Qwen3-Embedding-0.6B (baseline)
# RESUME: if qwen3_baseline_corpus.npy exists, load + embed queries only
# -------------------------------------------------------
print("\n" + "="*60)
print("Config A: Qwen3-Embedding-0.6B (baseline)")
print("="*60)
qwen_cache = CACHE_DIR / "qwen3_baseline_corpus.npy"
qwen_skip_corpus = qwen_cache.exists()
if qwen_skip_corpus:
    print(f"  RESUME: loading cached corpus from {qwen_cache}")

try:
    from sentence_transformers import SentenceTransformer
    qwen_name = QWEN3_SNAP
    t0 = time.time()
    qwen_model = SentenceTransformer(qwen_name)
    load_s = time.time() - t0
    print(f"  Model loaded in {load_s:.1f}s")

    if qwen_skip_corpus:
        qwen_corpus_np = np.load(str(qwen_cache))
        print(f"  Loaded cached corpus embeddings ({qwen_corpus_np.shape})")
        embed_s = 0
    else:
        t1 = time.time()
        corpus_embs_qwen = qwen_model.encode(texts, batch_size=BATCH_CORPUS, show_progress_bar=True, normalize_embeddings=True)
        embed_s = time.time() - t1
        print(f"  Corpus embedded in {embed_s:.1f}s")
        qwen_corpus_np = np.array(corpus_embs_qwen)
        np.save(str(qwen_cache), qwen_corpus_np)
        print(f"  Saved corpus embeddings to cache")
        del corpus_embs_qwen

    t2 = time.time()
    query_embs_qwen = qwen_model.encode(queries, batch_size=BATCH_QUERY, show_progress_bar=False, normalize_embeddings=True)
    query_s = time.time() - t2

    metrics = eval_metrics(query_embs_qwen, qwen_corpus_np, eval_qs, texts, K_VALUES)
    metrics["embed_corpus_s"] = round(embed_s, 2)
    metrics["query_ms_mean"] = round(query_s * 1000 / len(queries), 2)
    metrics["load_s"] = round(load_s, 2)
    results["configs"]["qwen3_baseline"] = metrics
    print(f"  Metrics: {metrics}")

    del qwen_model, query_embs_qwen, qwen_corpus_np
    gc.collect()
    print("  Freed Qwen3 from RAM")
    save_partial(results)
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results["configs"]["qwen3_baseline"] = {"error": str(e)}
    gc.collect()
    save_partial(results)


# -------------------------------------------------------
# Config B: Vietnamese_Embedding_v2  <-- MOVED UP (was C)
# RESUME: if vn_embed_v2_corpus.npy + query.npy exist, load + skip embed
# -------------------------------------------------------
print("\n" + "="*60)
print("Config B: Vietnamese_Embedding_v2")
print("="*60)
vn_corpus_cache = CACHE_DIR / "vn_embed_v2_corpus.npy"
vn_query_cache = CACHE_DIR / "vn_embed_v2_query.npy"
vn_skip_corpus = vn_corpus_cache.exists() and vn_query_cache.exists()
if vn_skip_corpus:
    print(f"  RESUME: loading cached embeddings from {CACHE_DIR}")

vn_embed_s = 0
query_s = 0
vn_load = 0
try:
    from sentence_transformers import SentenceTransformer
    vn_name = VN_EMBED_SNAP
    t0 = time.time()
    vn_model = SentenceTransformer(vn_name)
    vn_load = time.time() - t0
    print(f"  Model loaded in {vn_load:.1f}s")

    if vn_skip_corpus:
        vn_corpus_np = np.load(str(vn_corpus_cache))
        vn_query_embs = np.load(str(vn_query_cache))
        print(f"  Loaded cached (corpus={vn_corpus_np.shape}, query={vn_query_embs.shape})")
    else:
        t1 = time.time()
        vn_corpus_embs = vn_model.encode(texts, batch_size=BATCH_CORPUS, show_progress_bar=True, normalize_embeddings=True)
        vn_embed_s = time.time() - t1
        print(f"  Corpus embedded in {vn_embed_s:.1f}s")

        t2 = time.time()
        vn_query_embs = vn_model.encode(queries, batch_size=BATCH_QUERY, show_progress_bar=False, normalize_embeddings=True)
        query_s = time.time() - t2

        vn_corpus_np = np.array(vn_corpus_embs)
        np.save(str(vn_corpus_cache), vn_corpus_np)
        np.save(str(vn_query_cache), vn_query_embs)
        print(f"  Saved corpus+query embeddings to cache")
        del vn_corpus_embs

    metrics = eval_metrics(vn_query_embs, vn_corpus_np, eval_qs, texts, K_VALUES)
    metrics["embed_corpus_s"] = round(vn_embed_s, 2)
    metrics["query_ms_mean"] = round(query_s * 1000 / len(queries), 2) if query_s > 0 else 0
    metrics["load_s"] = round(vn_load, 2)
    results["configs"]["vn_embed_v2"] = metrics
    print(f"  Metrics: {metrics}")

    del vn_model, vn_corpus_np, vn_query_embs
    gc.collect()
    print("  Freed Vietnamese_Embedding_v2 from RAM")
    save_partial(results)
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results["configs"]["vn_embed_v2"] = {"error": str(e)}
    gc.collect()
    save_partial(results)


# -------------------------------------------------------
# Config C: Vietnamese_Embedding_v2 + Vietnamese_Reranker
# Uses cached VN embeddings
# -------------------------------------------------------
print("\n" + "="*60)
print("Config C: Vietnamese_Embedding_v2 + Vietnamese_Reranker")
print("="*60)
if results["configs"].get("vn_embed_v2", {}).get("error") is None:
    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
        t0 = time.time()
        vn_model_cr = SentenceTransformer(VN_EMBED_SNAP)
        vn_load_cr = time.time() - t0
        print(f"  VN model reloaded in {vn_load_cr:.1f}s")

        vn_corpus_np = np.load(str(vn_corpus_cache))
        vn_query_embs = np.load(str(vn_query_cache))
        print(f"  Loaded cached embeddings")

        reranker_name = VN_RERANK_SNAP
        t0r = time.time()
        ce_model = CrossEncoder(reranker_name)
        ce_load = time.time() - t0r
        print(f"  CrossEncoder loaded in {ce_load:.1f}s")

        sims_vn = cosine_sim_matrix(vn_query_embs, vn_corpus_np)

        t1 = time.time()
        rerank_partials = {k: [] for k in K_VALUES}

        for qi, q in enumerate(eval_qs):
            topk_idx = np.argsort(sims_vn[qi])[-TOP_RERANK:][::-1]
            top_texts = [texts[j] for j in topk_idx]
            top_ids = [chunk_ids[j] for j in topk_idx]

            pairs = [(q["question"], t) for t in top_texts]
            ce_scores = ce_model.predict(pairs)
            ranked_ids = [top_ids[j] for j in np.argsort(ce_scores)[::-1]]

            id2text = {cid: texts[chunk_ids.index(cid)] for cid in ranked_ids if cid in chunk_ids}

            must = q.get("must_include", [])
            for k in K_VALUES:
                topk_reranked_texts = " ".join([id2text[cid] for cid in ranked_ids[:k]])
                found = sum(1 for m in must if m.lower() in topk_reranked_texts.lower())
                partial = found / len(must) if must else 1.0
                rerank_partials[k].append(partial)

            if (qi + 1) % 15 == 0:
                print(f"    Reranked {qi+1}/{len(eval_qs)}...")

        rerank_total = time.time() - t1
        print(f"  Rerank done in {rerank_total:.1f}s")

        vn_c_data = results["configs"].get("vn_embed_v2", {})
        vn_embed_s_val = vn_c_data.get("embed_corpus_s", 0)
        query_s_val = vn_c_data.get("query_ms_mean", 0) * len(queries) / 1000

        rerank_metrics = {}
        for k in K_VALUES:
            rerank_metrics[f"hit@{k}"] = round(sum(1 for p in rerank_partials[k] if p >= 1.0) / len(eval_qs), 4)
            rerank_metrics[f"partial@{k}"] = round(float(np.mean(rerank_partials[k])), 4)
        rerank_metrics["embed_corpus_s"] = round(vn_embed_s_val, 2)
        avg_rerank_ms = rerank_total * 1000 / len(queries)
        embed_query_ms = query_s_val * 1000 / len(queries)
        rerank_metrics["query_ms_mean"] = round(embed_query_ms + avg_rerank_ms, 2)
        rerank_metrics["load_s"] = round(vn_load_cr + ce_load, 2)
        results["configs"]["vn_embed_v2+vn_reranker"] = rerank_metrics
        print(f"  Metrics: {rerank_metrics}")

        del vn_model_cr, ce_model, vn_corpus_np, vn_query_embs
        gc.collect()
        print("  Freed VN+Reranker from RAM")
        save_partial(results)
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["configs"]["vn_embed_v2+vn_reranker"] = {"error": str(e)}
        gc.collect()
        save_partial(results)
else:
    results["configs"]["vn_embed_v2+vn_reranker"] = {"error": "Skipped: vn_embed_v2 failed"}
    save_partial(results)


# -------------------------------------------------------
# Config D: E2Rank-0.6B (LAST -- slowest on CPU, time-boxed)
# RESUME: if e2rank_corpus.npy exists, load + skip embed.
# TIME-BOX: if corpus not cached and estimated >15 min, skip.
# -------------------------------------------------------
print("\n" + "="*60)
print("Config D: E2Rank-0.6B (embedding)")
print("="*60)
e2r_cache = CACHE_DIR / "e2rank_corpus.npy"
e2r_skip_corpus = e2r_cache.exists()
if e2r_skip_corpus:
    print(f"  RESUME: loading cached corpus from {e2r_cache}")

try:
    from transformers import AutoTokenizer, AutoModel
    e2r_model_name = E2RANK_SNAP

    t_load = time.time()
    e2r_tokenizer = AutoTokenizer.from_pretrained(e2r_model_name, padding_side="left")
    e2r_model = AutoModel.from_pretrained(e2r_model_name)
    e2r_model.eval()
    e2r_load = time.time() - t_load
    print(f"  Model loaded in {e2r_load:.1f}s")

    if e2r_skip_corpus:
        e2r_corpus_embs = np.load(str(e2r_cache))
        print(f"  Loaded cached corpus embeddings ({e2r_corpus_embs.shape})")
        e2r_embed_s = 0
    else:
        # TIME-BOX: embed first batch, estimate total, abort if >15 min
        t_emb = time.time()
        test_batch = texts[:BATCH_CORPUS]
        _ = embed_e2rank(test_batch, e2r_model, e2r_tokenizer, is_query=False, batch_size=BATCH_CORPUS)
        first_batch_s = time.time() - t_emb
        n_batches = (len(texts) + BATCH_CORPUS - 1) // BATCH_CORPUS
        estimated_total = first_batch_s * n_batches
        print(f"  First batch: {first_batch_s:.1f}s -> estimated corpus embed: {estimated_total:.0f}s ({estimated_total/60:.1f} min)")
        if estimated_total > E2RANK_TIMEBOX_S:
            raise TimeoutError(f"E2Rank corpus embed estimated {estimated_total/60:.1f} min > {E2RANK_TIMEBOX_S/60:.0f} min limit")

        e2r_corpus_embs = embed_e2rank(texts, e2r_model, e2r_tokenizer, is_query=False, batch_size=BATCH_CORPUS)
        e2r_embed_s = time.time() - t_emb
        print(f"  Corpus embedded in {e2r_embed_s:.1f}s")
        np.save(str(e2r_cache), e2r_corpus_embs)
        print(f"  Saved corpus embeddings to cache")

    t_q = time.time()
    e2r_query_embs = embed_e2rank(queries, e2r_model, e2r_tokenizer, is_query=True, batch_size=BATCH_QUERY)
    query_s = time.time() - t_q

    metrics = eval_metrics(e2r_query_embs, e2r_corpus_embs, eval_qs, texts, K_VALUES)
    metrics["embed_corpus_s"] = round(e2r_embed_s, 2)
    metrics["query_ms_mean"] = round(query_s * 1000 / len(queries), 2)
    metrics["load_s"] = round(e2r_load, 2)
    results["configs"]["e2rank"] = metrics
    print(f"  Metrics: {metrics}")

    del e2r_model, e2r_tokenizer, e2r_corpus_embs, e2r_query_embs
    gc.collect()
    print("  Freed E2Rank from RAM")
    save_partial(results)
except TimeoutError as e:
    print(f"  SKIPPED: {e}")
    results["configs"]["e2rank"] = {"error": f"Too slow on CPU -- skipped, E2Rank=Qwen3-arch, VN-finetuned models prioritized"}
    gc.collect()
    save_partial(results)
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results["configs"]["e2rank"] = {"error": str(e)}
    gc.collect()
    save_partial(results)


# -------------------------------------------------------
# Hard questions: top-3 with BEST config
# -------------------------------------------------------
print("\n" + "="*60)
print("Hard questions: top-3 retrieval with best config")
print("="*60)

best_config = None
best_hit5 = -1
for cname, cdata in results["configs"].items():
    if isinstance(cdata, dict) and "error" not in cdata and "hit@5" in cdata:
        if cdata["hit@5"] > best_hit5:
            best_hit5 = cdata["hit@5"]
            best_config = cname

print(f"  Best config: {best_config} (hit@5={best_hit5})")

hard_queries = [hard_map[hid]["question"] for hid in HARD_IDS]

hard_sims = None
try:
    if best_config == "qwen3_baseline":
        from sentence_transformers import SentenceTransformer
        print("  Reloading Qwen3 for hard questions...")
        qwen_reload = SentenceTransformer(QWEN3_SNAP)
        qwen_corpus_reload = np.load(str(CACHE_DIR / "qwen3_baseline_corpus.npy"))
        hard_q_embs = qwen_reload.encode(hard_queries, batch_size=BATCH_QUERY, normalize_embeddings=True)
        hard_sims = cosine_sim_matrix(hard_q_embs, qwen_corpus_reload)
        del qwen_reload, qwen_corpus_reload, hard_q_embs
        gc.collect()

    elif best_config == "e2rank":
        from transformers import AutoTokenizer, AutoModel
        print("  Reloading E2Rank for hard questions...")
        e2r_reload_tok = AutoTokenizer.from_pretrained(E2RANK_SNAP, padding_side="left")
        e2r_reload_model = AutoModel.from_pretrained(E2RANK_SNAP)
        e2r_reload_model.eval()
        e2r_corpus_reload = np.load(str(CACHE_DIR / "e2rank_corpus.npy"))
        hard_q_embs = embed_e2rank(hard_queries, e2r_reload_model, e2r_reload_tok, is_query=True, batch_size=BATCH_QUERY)
        hard_sims = cosine_sim_matrix(hard_q_embs, e2r_corpus_reload)
        del e2r_reload_model, e2r_reload_tok, e2r_corpus_reload, hard_q_embs
        gc.collect()

    elif best_config in ("vn_embed_v2", "vn_embed_v2+vn_reranker"):
        from sentence_transformers import SentenceTransformer
        print("  Reloading Vietnamese_Embedding_v2 for hard questions...")
        vn_reload = SentenceTransformer(VN_EMBED_SNAP)
        vn_corpus_reload = np.load(str(CACHE_DIR / "vn_embed_v2_corpus.npy"))
        hard_q_embs = vn_reload.encode(hard_queries, batch_size=BATCH_QUERY, normalize_embeddings=True)
        hard_sims = cosine_sim_matrix(hard_q_embs, vn_corpus_reload)
        del vn_reload, vn_corpus_reload, hard_q_embs
        gc.collect()
    else:
        print("  No valid best config found -- skipping")
except Exception as e:
    print(f"  ERROR during hard question reload: {e}")
    traceback.print_exc()
    hard_sims = None
    gc.collect()

if hard_sims is not None:
    for hi, hid in enumerate(HARD_IDS):
        top3_idx = np.argsort(hard_sims[hi])[-3:][::-1]
        hard_results = []
        for idx in top3_idx:
            hard_results.append({"chunk_id": chunk_ids[idx], "text_100": texts[idx][:100]})
        results["hard_questions"][hid] = hard_results
        q_text = hard_map[hid]["question"]
        print(f"  {hid} ({q_text[:60]}...):")
        for r in hard_results:
            print(f"    [{r['chunk_id']}] {r['text_100']}...")
else:
    for hid in HARD_IDS:
        results["hard_questions"][hid] = [{"error": "No embedding model available"}]


# -------------------------------------------------------
# Determine winners
# -------------------------------------------------------
best_acc_key = None
best_acc_val = -1
best_speed_key = None
best_speed_val = float("inf")
for cname, cdata in results["configs"].items():
    if isinstance(cdata, dict) and "error" not in cdata and "hit@5" in cdata:
        if cdata["hit@5"] > best_acc_val:
            best_acc_val = cdata["hit@5"]
            best_acc_key = cname
        if cdata.get("query_ms_mean", float("inf")) < best_speed_val:
            best_speed_val = cdata["query_ms_mean"]
            best_speed_key = cname

results["winner_by_accuracy"] = best_acc_key or ""
results["winner_by_speed"] = best_speed_key or ""

# -- Write final output --
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n" + "="*60)
print(f"Results written to {OUT_FILE}")
print("="*60)
print(json.dumps(results, ensure_ascii=False, indent=2))
