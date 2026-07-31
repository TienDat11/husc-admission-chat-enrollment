#!/usr/bin/env python3
"""Generates bench_embed_rerank.py"""
from pathlib import Path

out = Path("D:/chunking/husc-admission-chat-enrollment/rag2025/scripts/bench_embed_rerank.py")
out.parent.mkdir(parents=True, exist_ok=True)

code = '''#!/usr/bin/env python3
"""Benchmark embedding + reranker models on Vietnamese RAG admission corpus.
MEASUREMENT ONLY - no production code touched.
"""
import os, sys, json, time, traceback
from pathlib import Path

# -- Env (set BEFORE any HF/torch imports) --
os.environ["HF_HOME"] = r"D:\\huggingface_cache"
os.environ["HF_HUB_CACHE"] = r"D:\\huggingface_cache\\hub"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np

# -- Paths --
BASE = Path(r"D:\\chunking\\husc-admission-chat-enrollment\\rag2025")
LANCE_DIR = str(BASE / "data" / "lancedb")
EVAL_FILE = BASE / "data" / "eval" / "husc_86q_test.json"
HARD_FILE = BASE / "data" / "eval" / "husc_thi_sinh_thuc.json"
OUT_FILE = BASE / "results" / "embed_rerank_bench.json"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
N_EVAL = 60
K_VALUES = [5, 10]
TOP_RERANK = 20

TASK_DESC = "Given a web search query, retrieve relevant passages that answer the query"

# -- Load corpus from LanceDB --
print("Loading corpus from LanceDB...")
import lancedb
db = lancedb.connect(LANCE_DIR)
tbl = db.open_table("rag2025")
rows = tbl.to_pandas()
chunk_ids = rows["chunk_id"].tolist()
texts = rows["text"].tolist()
print(f"  Corpus: {len(texts)} chunks")

# -- Load eval set --
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_all = json.load(f)
eval_qs = eval_all[:N_EVAL]
print(f"  Eval: {len(eval_qs)} questions (first {N_EVAL} of {len(eval_all)})")

# -- Load hard questions --
with open(HARD_FILE, "r", encoding="utf-8") as f:
    hard_all = json.load(f)
hard_map = {q["id"]: q for q in hard_all}
HARD_IDS = ["msg029", "msg051", "msg080"]

# -- Helpers --
def cosine_sim_matrix(a, b):
    """a: (M, D), b: (N, D) -> (M, N)"""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def eval_metrics(query_embeds, corpus_embeds, eval_qs, texts, k_values):
    """Compute HIT@K and PARTIAL@K using must_include recall."""
    sims = cosine_sim_matrix(query_embeds, corpus_embeds)  # (NQ, NC)
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


def embed_e2rank(texts_list, is_query=False, task_desc=TASK_DESC, batch_size=32):
    """Embed texts with E2Rank using transformers AutoModel + last_token_pool.
    Appends  to each text. Queries get instruct prefix."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    model_name = "Alibaba-NLP/E2Rank-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    def last_token_pool(last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            seq_lens = attention_mask.sum(dim=1) - 1
            bs = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(bs, device=last_hidden_states.device), seq_lens]

    if is_query:
        input_texts = [f"Instruct: {task_desc}\\nQuery:{q} " for q in texts_list]
    else:
        input_texts = [t + " " for t in texts_list]

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
# -------------------------------------------------------
print("\\n" + "="*60)
print("Config A: Qwen3-Embedding-0.6B (baseline)")
print("="*60)
try:
    from sentence_transformers import SentenceTransformer
    qwen_name = "Qwen/Qwen3-Embedding-0.6B"
    t0 = time.time()
    qwen_model = SentenceTransformer(qwen_name)
    load_s = time.time() - t0
    print(f"  Model loaded in {load_s:.1f}s")

    t1 = time.time()
    corpus_embs_qwen = qwen_model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embed_s = time.time() - t1
    print(f"  Corpus embedded in {embed_s:.1f}s")

    t2 = time.time()
    query_embs_qwen = qwen_model.encode(queries, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    query_s = time.time() - t2

    metrics = eval_metrics(query_embs_qwen, np.array(corpus_embs_qwen), eval_qs, texts, K_VALUES)
    metrics["embed_corpus_s"] = round(embed_s, 2)
    metrics["query_ms_mean"] = round(query_s * 1000 / len(queries), 2)
    metrics["load_s"] = round(load_s, 2)
    results["configs"]["qwen3_baseline"] = metrics
    print(f"  Metrics: {metrics}")

    qwen_corpus_np = np.array(corpus_embs_qwen)
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results["configs"]["qwen3_baseline"] = {"error": str(e)}
    qwen_model = None
    qwen_corpus_np = None

# -------------------------------------------------------
# Config B: E2Rank-0.6B (embedding mode)
# -------------------------------------------------------
print("\\n" + "="*60)
print("Config B: E2Rank-0.6B (embedding)")
print("="*60)
try:
    t_load = time.time()
    e2r_corpus_embs = embed_e2rank(texts, is_query=False)
    e2r_load = time.time() - t_load
    print(f"  Corpus embedded in {e2r_load:.1f}s")

    t_q = time.time()
    e2r_query_embs = embed_e2rank(queries, is_query=True)
    query_s = time.time() - t_q

    metrics = eval_metrics(e2r_query_embs, e2r_corpus_embs, eval_qs, texts, K_VALUES)
    metrics["embed_corpus_s"] = round(e2r_load, 2)
    metrics["query_ms_mean"] = round(query_s * 1000 / len(queries), 2)
    metrics["load_s"] = 0  # load included in embed time for E2Rank
    results["configs"]["e2rank"] = metrics
    print(f"  Metrics: {metrics}")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results["configs"]["e2rank"] = {"error": str(e)}
    e2r_corpus_embs = None

# -------------------------------------------------------
# Config C: Vietnamese_Embedding_v2
# -------------------------------------------------------
print("\\n" + "="*60)
print("Config C: Vietnamese_Embedding_v2")
print("="*60)
try:
    from sentence_transformers import SentenceTransformer
    vn_name = "AITeamVN/Vietnamese_Embedding_v2"
    t0 = time.time()
    vn_model = SentenceTransformer(vn_name)
    vn_load = time.time() - t0
    print(f"  Model loaded in {vn_load:.1f}s")

    t1 = time.time()
    vn_corpus_embs = vn_model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    vn_embed_s = time.time() - t1
    print(f"  Corpus embedded in {vn_embed_s:.1f}s")

    t2 = time.time()
    vn_query_embs = vn_model.encode(queries, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    query_s = time.time() - t2

    metrics = eval_metrics(vn_query_embs, vn_corpus_embs, eval_qs, texts, K_VALUES)
    metrics["embed_corpus_s"] = round(vn_embed_s, 2)
    metrics["query_ms_mean"] = round(query_s * 1000 / len(queries), 2)
    metrics["load_s"] = round(vn_load, 2)
    results["configs"]["vn_embed_v2"] = metrics
    print(f"  Metrics: {metrics}")

    vn_corpus_np = np.array(vn_corpus_embs)
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results["configs"]["vn_embed_v2"] = {"error": str(e)}
    vn_model = None
    vn_corpus_np = None

# -------------------------------------------------------
# Config C+R: Vietnamese_Embedding_v2 + Vietnamese_Reranker
# -------------------------------------------------------
print("\\n" + "="*60)
print("Config C+R: Vietnamese_Embedding_v2 + Vietnamese_Reranker")
print("="*60)
if vn_model is not None and vn_corpus_np is not None:
    try:
        from sentence_transformers import CrossEncoder
        reranker_name = "AITeamVN/Vietnamese_Reranker"
        t0 = time.time()
        ce_model = CrossEncoder(reranker_name)
        ce_load = time.time() - t0
        print(f"  CrossEncoder loaded in {ce_load:.1f}s")

        # Get top-TOP_RERANK from VN embedding similarity
        sims_vn = cosine_sim_matrix(vn_query_embs, vn_corpus_np)

        t1 = time.time()
        rerank_partials = {k: [] for k in K_VALUES}

        for qi, q in enumerate(eval_qs):
            topk_idx = np.argsort(sims_vn[qi])[-TOP_RERANK:][::-1]
            top_texts = [texts[j] for j in topk_idx]
            top_ids = [chunk_ids[j] for j in topk_idx]

            # Rerank with CrossEncoder
            pairs = [(q["question"], t) for t in top_texts]
            ce_scores = ce_model.predict(pairs)
            ranked_ids = [top_ids[j] for j in np.argsort(ce_scores)[::-1]]

            # Build id->text map for fast lookup
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

        rerank_metrics = {}
        for k in K_VALUES:
            rerank_metrics[f"hit@{k}"] = round(sum(1 for p in rerank_partials[k] if p >= 1.0) / len(eval_qs), 4)
            rerank_metrics[f"partial@{k}"] = round(float(np.mean(rerank_partials[k])), 4)
        rerank_metrics["embed_corpus_s"] = round(vn_embed_s, 2)
        # query time = embedding time + average rerank time per query
        avg_rerank_ms = rerank_total * 1000 / len(queries)
        embed_query_ms = query_s * 1000 / len(queries)
        rerank_metrics["query_ms_mean"] = round(embed_query_ms + avg_rerank_ms, 2)
        rerank_metrics["load_s"] = round(vn_load + ce_load, 2)
        results["configs"]["vn_embed_v2+vn_reranker"] = rerank_metrics
        print(f"  Metrics: {rerank_metrics}")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["configs"]["vn_embed_v2+vn_reranker"] = {"error": str(e)}
else:
    results["configs"]["vn_embed_v2+vn_reranker"] = {"error": "Skipped: vn_embed_v2 failed"}

# -------------------------------------------------------
# Hard questions: top-3 retrieval with BEST embedding config
# -------------------------------------------------------
print("\\n" + "="*60)
print("Hard questions: top-3 retrieval with best config")
print("="*60)

# Find best config by hit@5
best_config = None
best_hit5 = -1
for cname, cdata in results["configs"].items():
    if isinstance(cdata, dict) and "error" not in cdata and "hit@5" in cdata:
        if cdata["hit@5"] > best_hit5:
            best_hit5 = cdata["hit@5"]
            best_config = cname

print(f"  Best config: {best_config} (hit@5={best_hit5})")

hard_queries = [hard_map[hid]["question"] for hid in HARD_IDS]

if best_config == "qwen3_baseline" and qwen_model is not None and qwen_corpus_np is not None:
    hard_q_embs = qwen_model.encode(hard_queries, normalize_embeddings=True)
    hard_sims = cosine_sim_matrix(hard_q_embs, qwen_corpus_np)
elif best_config == "e2rank" and e2r_corpus_embs is not None:
    hard_q_embs = embed_e2rank(hard_queries, is_query=True)
    hard_sims = cosine_sim_matrix(hard_q_embs, e2r_corpus_embs)
elif best_config in ("vn_embed_v2", "vn_embed_v2+vn_reranker") and vn_model is not None and vn_corpus_np is not None:
    hard_q_embs = vn_model.encode(hard_queries, normalize_embeddings=True)
    hard_sims = cosine_sim_matrix(hard_q_embs, vn_corpus_np)
else:
    hard_sims = None

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

# -- Write output --
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\\n" + "="*60)
print(f"Results written to {OUT_FILE}")
print("="*60)
print(json.dumps(results, ensure_ascii=False, indent=2))
'''

out.write_text(code, encoding='utf-8')
print(f"Written {out} ({out.stat().st_size} bytes)")
