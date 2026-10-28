"""
Run pipeline RAG cho 86 câu thực tế của thí sinh và sinh 2 báo cáo:
1) results/86q_thi_sinh_chuyen_mon.md     — báo cáo kỹ thuật chi tiết
2) results/86q_thi_sinh_thay_tuyen_sinh.md — báo cáo cho thầy tuyển sinh

Phase E: Slice E.4 deliverable cho dataset thi-sinh-thuc.
"""
from __future__ import annotations
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import RAGSettings
from services.graphrag_retriever import UnifiedRAGPipeline
from services.lancedb_retrieval import LanceDBRetriever
from services.embedding import EmbeddingService
from services.llm_generator import LLMGenerator
from src.services.aggregation_booster import boost_with_aggregation


EVAL_PATH = Path(__file__).parent.parent / "data" / "eval" / "husc_thi_sinh_thuc.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"


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

    try:
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
    except Exception as e:
        logger.warning(f"Pipeline failed for {q['id']}: {e}")
        route = "error"
        complexity = 0
        intent = "error"
        top_docs = baseline_docs[:5]

    # Short-circuit: HyDE auto-answer (intro/contact/list_url) hoặc vague-reject
    # → bỏ qua retrieval/generation, dùng câu trả lời static do router cung cấp
    if router_result and getattr(router_result, "skip_retrieval", False) and router_result.auto_answer:
        elapsed = time.perf_counter() - start
        provider = f"hyde_auto_answer:{router_result.intent}"
        return {
            "id": q["id"],
            "question": query,
            "answer": router_result.auto_answer,
            "has_answer": True,
            "route": "hyde_auto_answer",
            "complexity": router_result.complexity,
            "intent": router_result.intent,
            "provider": provider,
            "latency_s": round(elapsed, 2),
            "chunk_ids": [],
            "chunk_sources": [],
        }

    chunks_for_gen = []
    for doc in top_docs:
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
        answer = gr.get("answer", "")
        provider = gr.get("provider", "?")
    except Exception as e:
        answer = f"ERROR: {e}"
        provider = f"error:{e}"

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
    chunk_ids = [getattr(d, "chunk_id", "") for d in top_docs]
    chunk_sources = [getattr(d, "source", "") for d in top_docs]
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
        "chunk_ids": chunk_ids[:5],
        "chunk_sources": chunk_sources[:5],
    }


async def main(embedding_override: str = "default", out_suffix: str = ""):
    # Embedding override (for benchmark across models)
    if embedding_override == "qwen":
        import os
        os.environ["EMBEDDING_PROVIDER"] = "qwen"
        os.environ["EMBEDDING_MODEL"] = os.environ.get("QWEN_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
        os.environ["EMBEDDING_DIM"] = os.environ.get("QWEN_EMBEDDING_DIM", "1024")
        logger.info(f"Embedding override: qwen -> {os.environ['EMBEDDING_MODEL']}")
    elif embedding_override == "bge":
        import os
        os.environ["EMBEDDING_PROVIDER"] = "bge"
        os.environ["EMBEDDING_MODEL"] = os.environ.get("BGE_EMBEDDING_MODEL", "BAAI/bge-m3")
        os.environ["EMBEDDING_DIM"] = os.environ.get("BGE_EMBEDDING_DIM", "1024")
        logger.info(f"Embedding override: bge -> {os.environ['EMBEDDING_MODEL']}")

    suffix_part = f"_{out_suffix}" if out_suffix else ""

    with EVAL_PATH.open("r", encoding="utf-8") as f:
        questions = json.load(f)
    logger.info(f"Loaded {len(questions)} questions")

    settings = RAGSettings()
    pipeline = UnifiedRAGPipeline.from_disk()
    retriever = LanceDBRetriever.from_env()
    embedding = EmbeddingService(settings)
    generator = LLMGenerator()

    out = []
    for i, q in enumerate(questions):
        logger.info(f"{i+1}/{len(questions)}: {q['id']}")
        try:
            r = await run_one(q, pipeline, retriever, embedding, generator)
            out.append(r)
            logger.info(f"  -> {chr(72)+chr(65)+chr(83)+chr(95)+chr(65)+chr(78)+chr(83) if r['has_answer'] else chr(70)+chr(65)+chr(76)+chr(76)+chr(66)+chr(65)+chr(67)+chr(75)} | route={r['route']} | latency={r['latency_s']}s")
            # Save partial after each query
            RESULTS_DIR.mkdir(exist_ok=True)
            with (RESULTS_DIR / f"86q_thi_sinh_results{suffix_part}.json").open("w", encoding="utf-8") as f:
                json.dump({"metadata": {
                    "ner_model": settings.NS24_NER_MODEL,
                    "hyde_model": settings.NS24_HYDE_MODEL,
                    "gen_model": settings.NS24_GEN_MODEL,
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "total": len(questions),
                    "completed": len(out),
                }, "results": out}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"  -> ERROR: {e}")
            out.append({"id": q["id"], "question": q["question"], "answer": f"ERROR: {e}",
                        "has_answer": False, "route": "error", "complexity": 0, "intent": "error",
                        "provider": f"error:{e}", "latency_s": 0, "chunk_ids": [], "chunk_sources": []})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="86Q thi-sinh evaluation")
    parser.add_argument("--embedding", choices=["default", "qwen", "bge"], default="default",
                        help="Embedding override: default=use .env / qwen / bge")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix for output: 86q_thi_sinh_results_{suffix}.json")
    args = parser.parse_args()
    asyncio.run(main(embedding_override=args.embedding, out_suffix=args.out_suffix))
