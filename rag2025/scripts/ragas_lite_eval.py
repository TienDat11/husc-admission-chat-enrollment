#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAGAS-lite evaluator for the 86Q thesis benchmark (NO fabrication).

Computes two RAGAS metrics that the available data CAN support honestly:
  (1) answer_relevancy  : reverse-question generation (LLM) + cosine to original
                          question, using the SAME local embedding (Qwen3-0.6B)
                          that the retrieval pipeline used.  -> all answered Q.
  (2) faithfulness      : LLM claim decomposition + per-claim NLI judge against
                          the reconstructed retrieved context. -> only questions
                          whose chunk_ids ALL resolve to text (honest subset).

context_precision / context_recall are NOT computed: they require gold-relevant
chunk labels / gold answers that the 86Q set does not have. Reported as N/A.

Outputs: results/ragas_lite_86q.json  (per-question + aggregate, by intent)
Checkpointing: writes after every question so a mid-run failure keeps progress.
"""
import os, json, glob, time, re, math, sys

from dotenv import load_dotenv
load_dotenv()

RESULT_IN = "results/86q_thi_sinh_results_qwen.json"
OUT = "results/ragas_lite_86q.json"
EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"
BASE = os.getenv("NS24_BASE_URL")
KEY = os.getenv("NS24_API_KEY")
GEN = os.getenv("NS24_GEN_MODEL", "gpt-6-luna")

# ---------- 1. corpus chunk_id -> text ----------
def build_corpus():
    cmap = {}
    for f in glob.glob("data/chunked/*.jsonl"):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            cid = o.get("id")
            txt = o.get("text") or o.get("text_plain") or ""
            if cid:
                cmap[cid] = txt
    raw = "data/raw/husc_2026_operational_qa.json"
    if os.path.exists(raw):
        data = json.load(open(raw, encoding="utf-8"))
        if isinstance(data, dict):
            items = sum([v for v in data.values() if isinstance(v, list)], [])
        else:
            items = data
        for o in items:
            if isinstance(o, dict):
                cid = o.get("id") or o.get("chunk_id")
                txt = o.get("text") or o.get("answer") or o.get("content") or ""
                if cid and cid not in cmap:
                    cmap[cid] = txt
    return cmap

# ---------- 2. embeddings (local) ----------
def get_embedder():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB_MODEL)
    def encode(texts):
        return m.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return encode

def cos(a, b):
    import numpy as np
    return float(np.dot(a, b))  # already normalized

# ---------- 3. LLM ----------
def get_llm():
    from openai import OpenAI
    c = OpenAI(base_url=BASE, api_key=KEY)
    def chat(prompt, max_tokens=400, temperature=0.0, retries=3):
        for i in range(retries):
            try:
                r = c.chat.completions.create(
                    model=GEN,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature, max_tokens=max_tokens,
                )
                return r.choices[0].message.content.strip()
            except Exception as e:
                if i == retries - 1:
                    raise
                time.sleep(2 * (i + 1))
    return chat

# ---------- 4. metric: answer_relevancy ----------
GEN_Q_PROMPT = """Cho câu trả lời sau, hãy sinh CHÍNH XÁC 3 câu hỏi mà câu trả lời này trả lời trực tiếp.
Mỗi câu hỏi một dòng, không đánh số, không giải thích thêm.

Câu trả lời:
\"\"\"{answer}\"\"\""""

def answer_relevancy(chat, encode, question, answer):
    out = chat(GEN_Q_PROMPT.format(answer=answer[:1500]), max_tokens=200)
    gen_qs = [q.strip(" -•\t") for q in out.splitlines() if q.strip()]
    gen_qs = [q for q in gen_qs if len(q) > 5][:3]
    if not gen_qs:
        return None, []
    embs = encode([question] + gen_qs)
    qv = embs[0]
    sims = [cos(qv, embs[i + 1]) for i in range(len(gen_qs))]
    return sum(sims) / len(sims), [round(s, 4) for s in sims]

# ---------- 5. metric: faithfulness ----------
CLAIM_PROMPT = """Tách câu trả lời sau thành các mệnh đề khẳng định nguyên tử (atomic claims), mỗi mệnh đề là một sự kiện độc lập có thể kiểm chứng.
Mỗi mệnh đề một dòng, không đánh số. Bỏ qua câu chào, lời mời, lời khuyên chung chung.

Câu trả lời:
\"\"\"{answer}\"\"\""""

NLI_PROMPT = """Bối cảnh (context) được trích từ tài liệu tuyển sinh:
\"\"\"{context}\"\"\"

Mệnh đề cần kiểm tra: "{claim}"

Mệnh đề này CÓ được suy ra trực tiếp từ bối cảnh trên không?
Chỉ trả lời đúng một từ: CO hoặc KHONG."""

def faithfulness(chat, answer, context):
    out = chat(CLAIM_PROMPT.format(answer=answer[:1800]), max_tokens=400)
    claims = [c.strip(" -•\t") for c in out.splitlines() if len(c.strip()) > 8]
    if not claims:
        return None, 0, 0
    supported = 0
    for cl in claims:
        v = chat(NLI_PROMPT.format(context=context[:6000], claim=cl), max_tokens=8)
        if v and v.strip().upper().startswith("CO"):
            supported += 1
    return supported / len(claims), supported, len(claims)

# ---------- main ----------
def main():
    cmap = build_corpus()
    print(f"[corpus] {len(cmap)} chunks indexed", flush=True)
    d = json.load(open(RESULT_IN, encoding="utf-8"))
    res = d["results"]

    # resume
    done = {}
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT, encoding="utf-8"))
            done = {r["id"]: r for r in prev.get("per_question", [])}
            print(f"[resume] {len(done)} already scored", flush=True)
        except Exception:
            pass

    encode = get_embedder()
    chat = get_llm()
    print("[init] embedder + llm ready", flush=True)

    per_q = []
    for idx, r in enumerate(res):
        qid = r["id"]
        if qid in done:
            per_q.append(done[qid])
            continue
        rec = {"id": qid, "intent": r.get("intent"), "has_answer": r.get("has_answer"),
               "answer_relevancy": None, "faithfulness": None,
               "faith_supported": None, "faith_total": None, "ctx_full": False}
        ans = r.get("answer") or ""
        q = r.get("question") or ""
        if r.get("has_answer") and ans.strip():
            try:
                ar, sims = answer_relevancy(chat, encode, q, ans)
                rec["answer_relevancy"] = round(ar, 4) if ar is not None else None
            except Exception as e:
                rec["ar_error"] = str(e)[:120]
            cids = r.get("chunk_ids") or []
            texts = [cmap[c] for c in cids if cmap.get(c)]
            if cids and len(texts) == len(cids) and texts:
                rec["ctx_full"] = True
                ctx = "\n\n---\n\n".join(texts)
                try:
                    fa, sup, tot = faithfulness(chat, ans, ctx)
                    rec["faithfulness"] = round(fa, 4) if fa is not None else None
                    rec["faith_supported"] = sup
                    rec["faith_total"] = tot
                except Exception as e:
                    rec["faith_error"] = str(e)[:120]
        per_q.append(rec)
        # checkpoint each question
        _write(per_q, len(res))
        print(f"[{idx+1}/{len(res)}] {qid} AR={rec['answer_relevancy']} "
              f"F={rec['faithfulness']} ctx_full={rec['ctx_full']}", flush=True)

    _write(per_q, len(res), final=True)
    print("[done]", flush=True)

def _agg(per_q, key, cond=lambda r: True):
    vals = [r[key] for r in per_q if r.get(key) is not None and cond(r)]
    if not vals:
        return {"mean": None, "n": 0}
    return {"mean": round(sum(vals) / len(vals), 4), "n": len(vals),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}

def _write(per_q, total, final=False):
    by_intent = {}
    intents = set(r.get("intent") for r in per_q)
    for it in intents:
        sub = [r for r in per_q if r.get("intent") == it]
        by_intent[it] = {
            "n": len(sub),
            "answer_relevancy": _agg(sub, "answer_relevancy"),
            "faithfulness": _agg(sub, "faithfulness"),
        }
    out = {
        "metadata": {
            "source": RESULT_IN, "embedding_model": EMB_MODEL,
            "judge_model": "conversational LLM via OpenAI-compatible endpoint",
            "n_total": total, "n_scored": len(per_q),
            "note": "answer_relevancy on all answered Q; faithfulness only on "
                    "questions whose retrieved chunk_ids ALL resolve to text "
                    "(ctx_full). context_precision/recall = N/A (no gold labels).",
            "final": final,
        },
        "aggregate": {
            "answer_relevancy": _agg(per_q, "answer_relevancy"),
            "faithfulness_ctx_full": _agg(per_q, "faithfulness",
                                          cond=lambda r: r.get("ctx_full")),
            "n_ctx_full": sum(1 for r in per_q if r.get("ctx_full")),
        },
        "by_intent": by_intent,
        "per_question": per_q,
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
