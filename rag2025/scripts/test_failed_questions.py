"""
Test riêng các câu hỏi đã fix:
- Q1, Q15, Q17: ngành đào tạo (HyDE auto-answer)
- Q15: vague-reject
- Q65: học bổng (sau khi append chunk hocbong_husc_2026_full)

Cập nhật kết quả vào 86q_thi_sinh_results.json (giữ nguyên các câu khác)
rồi regenerate PDF report.
"""
from __future__ import annotations
import asyncio
import json
import sys
import time
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config.settings import RAGSettings
from services.graphrag_retriever import UnifiedRAGPipeline
from services.lancedb_retrieval import LanceDBRetriever
from services.embedding import EmbeddingService
from services.llm_generator import LLMGenerator
from src.services.aggregation_booster import boost_with_aggregation


# Câu hỏi cần test lại (theo thứ tự id trong eval/husc_thi_sinh_thuc.json)
TARGETS = [
    {"id": "msg023", "question": "Em thi THPT tổ hợp khác. Mà em xét học bạ tổ hợp khác được chứ ạ?"},
    {"id": "msg055", "question": "Trường có group hay zalo tư vấn tuyển sinh, hướng dẫn đăng ký xét tuyển không ạ, e muốn được tư vấn?"},
]

RESULTS_FILE = ROOT / "results" / "86q_thi_sinh_results.json"


async def run_one(q, pipeline, retriever, embedding, generator):
    query = q["question"]
    start = time.perf_counter()

    # Step 1: route
    router_result = await pipeline._router.route(query)

    # Short-circuit: HyDE auto-answer / vague-reject
    if router_result.skip_retrieval and router_result.auto_answer:
        elapsed = time.perf_counter() - start
        return {
            "id": q["id"],
            "question": query,
            "answer": router_result.auto_answer,
            "has_answer": True,
            "route": "hyde_auto_answer",
            "complexity": router_result.complexity,
            "intent": router_result.intent,
            "provider": f"hyde_auto_answer:{router_result.intent}",
            "latency_s": round(elapsed, 2),
            "chunk_ids": [],
            "chunk_sources": [],
        }

    # Step 2: baseline retrieval (multi-variant)
    all_q = router_result.all_queries_for_embedding
    results = []
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

    # Step 3: pipeline
    rag_result = await pipeline.query(
        user_query=query,
        baseline_docs=baseline_docs,
        top_k=5,
        router_result=router_result,
    )
    route = rag_result.route
    complexity = rag_result.router_result.complexity
    intent = rag_result.router_result.intent
    top_docs = rag_result.documents

    chunks_for_gen = []
    for doc in top_docs:
        chunks_for_gen.append({
            "text": getattr(doc, "text", ""),
            "summary": getattr(doc, "metadata", {}).get("summary", ""),
            "metadata": getattr(doc, "metadata", {}),
        })

    is_enum = any(k in query.lower() for k in ["bao nhiêu", "danh sách", "liệt kê", "tất cả"])

    gr = await generator.generate_answer(
        query=query, chunks=chunks_for_gen,
        confidence=0.5, is_program_list_query=is_enum,
    )
    answer = gr.get("answer", "")
    provider = gr.get("provider", "?")

    elapsed = time.perf_counter() - start
    # has_answer: chỉ flag fallback khi câu trả lời CHỈ là 1 dòng
    # "Tôi không tìm thấy" / "Tôi không cung cấp" / nội dung quá ngắn (<50 chars).
    # KHÔNG dùng "không đủ" làm sentinel — false positive với phrase
    # "không đủ điểm/điều kiện" trong câu trả lời thật.
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
        "complexity": complexity,
        "intent": intent,
        "provider": provider,
        "latency_s": round(elapsed, 2),
        "chunk_ids": [getattr(d, "chunk_id", "") for d in top_docs][:5],
        "chunk_sources": [getattr(d, "source", "") for d in top_docs][:5],
    }


async def main():
    settings = RAGSettings()
    pipeline = UnifiedRAGPipeline.from_disk()
    retriever = LanceDBRetriever.from_env()
    embedding = EmbeddingService(settings)
    generator = LLMGenerator()

    new_results = []
    for q in TARGETS:
        logger.info(f"Running {q['id']}: {q['question'][:60]}")
        try:
            r = await run_one(q, pipeline, retriever, embedding, generator)
            new_results.append(r)
            logger.info(
                f"  -> {('PASS' if r['has_answer'] else 'FAIL')} | route={r['route']} | "
                f"latency={r['latency_s']}s | provider={r['provider']}"
            )
        except Exception as e:
            logger.error(f"  -> ERROR: {e}")
            new_results.append({
                "id": q["id"], "question": q["question"], "answer": f"ERROR: {e}",
                "has_answer": False, "route": "error", "complexity": 0, "intent": "error",
                "provider": f"error:{e}", "latency_s": 0, "chunk_ids": [], "chunk_sources": [],
            })

    # Patch the existing results file: replace results for the 2 targeted IDs
    # (msg023, msg055).
    with RESULTS_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    target_ids = {"msg023", "msg055"}
    by_id = {r["id"]: r for r in new_results if r["id"] in target_ids}
    bonus = [r for r in new_results if r["id"] not in target_ids]

    patched = []
    for r in data["results"]:
        if r["id"] in by_id:
            patched.append(by_id[r["id"]])
        else:
            patched.append(r)
    # Append any bonus tests at the end (none for this run)
    patched.extend(bonus)

    data["results"] = patched
    data["metadata"]["completed"] = len(patched)
    data["metadata"]["total"] = len(patched)
    data["metadata"]["last_patched_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data["metadata"]["patched_ids"] = sorted(target_ids)

    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    has_answer = sum(1 for r in patched if r.get("has_answer"))
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total: {len(patched)} | has_answer: {has_answer} ({has_answer/len(patched)*100:.1f}%)")
    logger.info(f"Patched 4 questions; bonus added: {len(bonus)}")


if __name__ == "__main__":
    asyncio.run(main())
