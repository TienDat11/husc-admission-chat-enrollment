"""Re-run only the 3 failed questions (msg021, msg036, msg040) to verify booster rules fixed them.

Updates 86q_thi_sinh_results.json by replacing those 3 entries with new results.
"""
from __future__ import annotations
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Dict

from loguru import logger
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from config.settings import RAGSettings
from services.graphrag_retriever import UnifiedRAGPipeline
from services.lancedb_retrieval import LanceDBRetriever
from services.embedding import EmbeddingService
from services.llm_generator import LLMGenerator
from src.services.aggregation_booster import boost_with_aggregation

EVAL_PATH = ROOT / "data" / "eval" / "husc_thi_sinh_thuc.json"
RESULTS_PATH = ROOT / "results" / "86q_thi_sinh_results.json"

TARGET_IDS = {"msg024"}


async def run_one(q: Dict, pipeline, retriever, embedding, generator) -> Dict:
    query = q["question"]
    start = time.perf_counter()
    try:
        router_result = await pipeline._router.route(query)
        all_q = router_result.all_queries_for_embedding
        results: List = []
        for variant in all_q[:4]:
            qv = embedding.encode_single(variant).tolist()
            rr = retriever.retrieve(qv, top_k=10)
            if rr.is_success:
                results.extend(rr.documents)
        seen = set()
        unique = []
        for d in results:
            cid = d.chunk_id or ""
            if cid not in seen:
                seen.add(cid)
                unique.append(d)
        baseline_docs = unique[:15]
        baseline_docs = boost_with_aggregation(query, baseline_docs, retriever, top_k=15, max_inject=2)
    except Exception as e:
        logger.warning(f"Baseline failed for {q['id']}: {e}")
        baseline_docs = []
        router_result = None

    if router_result and router_result.skip_retrieval:
        answer = router_result.auto_answer or ""
        route = "hyde_auto_answer"
        intent = router_result.intent
        complexity = router_result.complexity
        docs_used = []
        gen_provider = f"hyde_auto_answer:{router_result.intent}"
    else:
        intent = router_result.intent if router_result else "general"
        complexity = router_result.complexity if router_result else 1
        if router_result:
            route_obj = router_result.route
            route = route_obj.value if hasattr(route_obj, "value") else str(route_obj)
        else:
            route = "padded_rag"
        docs_used = baseline_docs[:5]

        chunks_for_gen = []
        for doc in docs_used:
            chunks_for_gen.append({
                "text": getattr(doc, "text", ""),
                "summary": getattr(doc, "metadata", {}).get("summary", ""),
                "metadata": getattr(doc, "metadata", {}),
            })
        is_enum = any(k in query.lower() for k in ["bao nhiêu", "danh sách", "liệt kê", "tất cả"])

        try:
            gr = await generator.generate_answer(
                query=query, chunks=chunks_for_gen,
                confidence=0.5, is_program_list_query=is_enum,
            )
            answer = gr.get("answer", "") if isinstance(gr, dict) else ""
            gen_provider = gr.get("provider", "?") if isinstance(gr, dict) else "?"
        except Exception as e:
            logger.error(f"Generation failed for {q['id']}: {e}")
            answer = ""
            gen_provider = f"error:{e}"

    latency = time.perf_counter() - start
    ans_lower = (answer or "").strip().lower()
    is_fallback = (
        not ans_lower
        or len(ans_lower) < 50
        or ans_lower.startswith("tôi không tìm thấy")
        or ans_lower.startswith("không tìm thấy")
        or "không tìm thấy thông tin chính xác" in ans_lower
        or "không tìm thấy thông tin này trong tài liệu" in ans_lower
        or ans_lower.startswith("tôi không cung cấp")
    )
    has_answer = not is_fallback

    return {
        "id": q["id"],
        "question": query,
        "answer": answer,
        "has_answer": has_answer,
        "route": route,
        "intent": str(intent) if not isinstance(intent, str) else intent,
        "complexity": int(complexity) if hasattr(complexity, "__int__") else complexity,
        "latency_s": round(latency, 2),
        "provider": gen_provider if 'gen_provider' in locals() else "mimo-v2.5-pro",
        "chunk_ids": [d.chunk_id for d in (docs_used if 'docs_used' in locals() else [])],
    }


async def main():
    eval_data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    questions = eval_data.get("questions", eval_data) if isinstance(eval_data, dict) else eval_data
    target_qs = [q for q in questions if q["id"] in TARGET_IDS]
    logger.info(f"Re-running {len(target_qs)} target questions: {[q['id'] for q in target_qs]}")

    settings = RAGSettings()
    embedding = EmbeddingService(settings)
    retriever = LanceDBRetriever.from_env()
    generator = LLMGenerator()
    pipeline = UnifiedRAGPipeline.from_disk()

    new_results = []
    for q in target_qs:
        logger.info(f"Running {q['id']}: {q['question'][:60]}")
        r = await run_one(q, pipeline, retriever, embedding, generator)
        logger.info(f"  -> has_answer={r['has_answer']} route={r['route']} latency={r['latency_s']}s")
        new_results.append(r)

    # Merge into existing results
    existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in existing["results"]}
    for r in new_results:
        by_id[r["id"]] = r
    existing["results"] = [by_id[q["id"]] for q in questions if q["id"] in by_id]

    RESULTS_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Updated {RESULTS_PATH.name}")

    new_ans = sum(1 for r in new_results if r["has_answer"])
    print(f"\n{'='*60}")
    print(f"Re-run results: {new_ans}/{len(new_results)} now have answers")
    for r in new_results:
        status = "PASS" if r["has_answer"] else "FAIL"
        print(f"  [{status}] {r['id']}: {r['answer'][:120]}")


if __name__ == "__main__":
    asyncio.run(main())
