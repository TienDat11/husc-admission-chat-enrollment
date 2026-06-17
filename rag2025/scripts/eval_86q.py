"""
86Q Evaluation Script – Unified RAG Pipeline (PaddedRAG + GraphRAG)

Tests the complete UnifiedRAGPipeline (SmartRouter → LanceDB baseline → GraphRAG fusion)
against 86 questions. Reports accuracy by category, route distribution, and saves
full answers for admissions office review.

Usage:
    cd rag2025
    set PYTHONPATH=src
    set HF_HOME=D:/chunking/.cache/huggingface
    python scripts/eval_86q.py
    python scripts/eval_86q.py --limit 10          # quick test
    python scripts/eval_86q.py --force-route graph_rag  # force all to graph_rag
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from loguru import logger

# Load .env BEFORE any service imports (os.getenv reads process env, not pydantic .env)
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import RAGSettings
from services.graphrag_retriever import UnifiedRAGPipeline, QueryRoute
from services.lancedb_retrieval import LanceDBRetriever, RetrievedDocument
from services.embedding import EmbeddingService
from services.llm_generator import LLMGenerator
from services.query_router import SmartQueryRouter
from src.services.aggregation_booster import boost_with_aggregation


EVAL_PATH = Path(__file__).parent.parent / "data" / "eval" / "husc_86q_test.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_questions() -> List[Dict]:
    if not EVAL_PATH.exists():
        logger.error(f"Eval file not found: {EVAL_PATH}")
        return []
    with EVAL_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_must_include(answer: str, must_include: List[str]) -> bool:
    answer_lower = answer.lower()
    for token in must_include:
        if token.lower() in answer_lower:
            continue
        # Numeric check: remove dots/commas for comparison
        if token.replace(".", "").replace(",", "").isdigit():
            if token.replace(".", "").replace(",", "") in answer_lower.replace(".", "").replace(",", ""):
                continue
        return False
    return True


async def eval_single(
    question: Dict,
    pipeline: UnifiedRAGPipeline,
    retriever: LanceDBRetriever,
    embedding: EmbeddingService,
    generator: LLMGenerator,
    force_route: Optional[str] = None,
) -> Dict[str, Any]:
    q = question["question"]
    must = question.get("must_include", [])
    category = question.get("category", "unknown")

    start = time.perf_counter()

    # Step 1: Get baseline docs from LanceDB
    try:
        router_result = await pipeline._router.route(q)
        all_queries = router_result.all_queries_for_embedding
        results = []
        for variant in all_queries[:4]:
            query_vector = embedding.encode_single(variant).tolist()
            retrieval_result = retriever.retrieve(query_vector, top_k=10)
            if retrieval_result.is_success:
                results.extend(retrieval_result.documents)
        # Deduplicate by chunk_id
        seen = set()
        unique = []
        for doc in results:
            cid = doc.chunk_id or ""
            if cid not in seen:
                seen.add(cid)
                unique.append(doc)
        baseline_docs = unique[:15]
        # Phase E.2: inject canonical aggregation chunks for enumeration / temporal / liet_ke queries
        baseline_docs = boost_with_aggregation(q, baseline_docs, retriever, top_k=15, max_inject=2)
    except Exception as e:
        logger.warning(f"Baseline retrieval failed for {question['id']}: {e}")
        baseline_docs = []
        router_result = None

    # Step 2: Run through pipeline (reuse router_result from Step 1)
    try:
        rag_result = await pipeline.query(
            user_query=q,
            baseline_docs=baseline_docs,
            top_k=5,
            router_result=router_result,
        )

        # Apply force_route override if requested
        if force_route == "graph_rag" and baseline_docs:
            docs, ppr_scores = await pipeline._graphrag.retrieve(
                query=q,
                router_result=rag_result.router_result,
                baseline_docs=baseline_docs,
                top_k=5,
            )
            rag_result = type(rag_result)(
                query=q,
                route="graph_rag",
                documents=docs,
                router_result=rag_result.router_result,
                ppr_scores=ppr_scores,
                latency_ms=rag_result.latency_ms,
                confidence=docs[0].score if docs else 0.0,
            )
        elif force_route == "padded_rag":
            rag_result.route = "padded_rag"
            rag_result.documents = baseline_docs[:5]

        route_used = rag_result.route
        ppr_scores = rag_result.ppr_scores
        top_docs = rag_result.documents
        complexity = rag_result.router_result.complexity
        intent = rag_result.router_result.intent
    except Exception as e:
        logger.warning(f"Pipeline failed for {question['id']}: {e}")
        route_used = "error"
        ppr_scores = {}
        top_docs = baseline_docs[:5]
        complexity = 0
        intent = "error"

    # Step 3: Generate answer
    chunks_for_gen = []
    for doc in top_docs:
        chunk = {
            "text": getattr(doc, "text", ""),
            "summary": getattr(doc, "metadata", {}).get("summary", ""),
            "metadata": getattr(doc, "metadata", {}),
        }
        chunks_for_gen.append(chunk)

    # Detect enumeration queries for longer answer
    is_enum = "bao nhiêu ngành" in q.lower() or "danh sách" in q.lower() or "liệt kê" in q.lower() or "tất cả" in q.lower()

    try:
        gen_result = await generator.generate_answer(
            query=q,
            chunks=chunks_for_gen,
            confidence=rag_result.confidence if 'rag_result' in dir() else 0.5,
            is_program_list_query=is_enum,
        )
        answer = gen_result.get("answer", "")
        provider = gen_result.get("provider", "unknown")
    except Exception as e:
        logger.warning(f"Generation failed for {question['id']}: {e}")
        answer = ""
        provider = f"error:{e}"

    elapsed = time.perf_counter() - start

    # Check accuracy
    passed = check_must_include(answer, must) if must else False
    has_answer = bool(answer and "không tìm thấy" not in answer.lower() and "không đủ dữ liệu" not in answer.lower())

    # Extract chunk source metadata
    chunk_sources = []
    for doc in top_docs:
        src = getattr(doc, "source", "") or getattr(doc, "metadata", {}).get("source", "")
        chunk_sources.append(src)

    # Count PPR hits (chunks with ppr_score > 0)
    ppr_hits = sum(1 for score in ppr_scores.values() if score > 0) if ppr_scores else 0

    # Top chunk confidence score
    top_confidence = top_docs[0].score if top_docs else 0.0

    return {
        "id": question["id"],
        "category": category,
        "weakness_type": question.get("weakness_type", ""),
        "question": q,
        "ground_truth": question.get("ground_truth_answer", ""),
        "answer": answer,
        "must_include": must,
        "passed": passed,
        "has_answer": has_answer,
        "route": route_used,
        "complexity": complexity,
        "intent": intent,
        "ner_seed_count": len(ppr_scores),
        "ppr_hits": ppr_hits,
        "ppr_seed_count": len(ppr_scores),
        "top_confidence": round(top_confidence, 4),
        "provider": provider,
        "latency_s": round(elapsed, 2),
        "num_chunks": len(top_docs),
        "chunk_sources": chunk_sources,
    }


async def run_eval(
    limit: int = 0,
    force_route: Optional[str] = None,
    embedding_override: str = "default",
    out_suffix: str = "",
) -> None:
    logger.info("Loading 86Q test file...")
    questions = load_questions()
    if not questions:
        logger.error("No questions loaded, aborting")
        return

    if limit:
        questions = questions[:limit]
    logger.info(f"Loaded {len(questions)} questions")

    # Embedding override (for benchmark across models)
    if embedding_override == "qwen":
        import os
        os.environ["EMBEDDING_PROVIDER"] = "qwen"
        os.environ["EMBEDDING_MODEL"] = os.environ.get("QWEN_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
        os.environ["EMBEDDING_DIM"] = os.environ.get("QWEN_EMBEDDING_DIM", "1024")
        logger.info(f"Embedding override: qwen → {os.environ['EMBEDDING_MODEL']}")
    elif embedding_override == "bge":
        import os
        os.environ["EMBEDDING_PROVIDER"] = "bge"
        os.environ["EMBEDDING_MODEL"] = os.environ.get("BGE_EMBEDDING_MODEL", "BAAI/bge-m3")
        os.environ["EMBEDDING_DIM"] = os.environ.get("BGE_EMBEDDING_DIM", "1024")
        logger.info(f"Embedding override: bge → {os.environ['EMBEDDING_MODEL']}")

    # Initialize services
    settings = RAGSettings()
    logger.info(f"NER model: {settings.RAMCLOUDS_NER_MODEL}")
    logger.info(f"HYDE model: {settings.RAMCLOUDS_HYDE_MODEL}")
    logger.info(f"GEN model: {settings.RAMCLOUDS_GEN_MODEL}")
    logger.info(f"GraphRAG threshold: {settings.GRAPHRAG_SIMPLE_THRESHOLD}")

    pipeline = UnifiedRAGPipeline.from_disk()
    retriever = LanceDBRetriever.from_env()
    embedding = EmbeddingService(settings)
    generator = LLMGenerator()

    if force_route:
        logger.info(f"Force route override: {force_route}")

    # Run evaluation
    results = []
    for i, q in enumerate(questions):
        logger.info(f"Evaluating {i+1}/{len(questions)}: {q['id']} ({q.get('category','?')})")
        try:
            result = await eval_single(q, pipeline, retriever, embedding, generator, force_route)
            status = "PASS" if result["passed"] else ("ANS" if result["has_answer"] else "FAIL")
            logger.info(
                f"  → {status} | route={result['route']} | c={result['complexity']} | "
                f"{result['latency_s']}s | chunks={result['num_chunks']} | provider={result['provider']}"
            )
            results.append(result)
        except Exception as e:
            logger.error(f"  → ERROR: {e}")
            results.append({
                "id": q["id"],
                "category": q.get("category", "unknown"),
                "weakness_type": q.get("weakness_type", ""),
                "question": q["question"],
                "ground_truth": q.get("ground_truth_answer", ""),
                "answer": f"ERROR: {e}",
                "must_include": q.get("must_include", []),
                "passed": False,
                "has_answer": False,
                "route": "error",
                "complexity": 0,
                "intent": "error",
                "ner_seed_count": 0,
                "ppr_hits": 0,
                "ppr_seed_count": 0,
                "top_confidence": 0.0,
                "provider": f"error:{e}",
                "latency_s": 0,
                "num_chunks": 0,
                "chunk_sources": [],
            })

    # Calculate metrics
    by_category: Dict[str, Dict] = {}
    route_counts: Dict[str, int] = {"padded_rag": 0, "graph_rag": 0, "error": 0}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0, "has_answer": 0, "latency": []}
        by_category[cat]["total"] += 1
        if r["passed"]:
            by_category[cat]["passed"] += 1
        if r["has_answer"]:
            by_category[cat]["has_answer"] += 1
        by_category[cat]["latency"].append(r["latency_s"])
        route = r.get("route", "error")
        if route in route_counts:
            route_counts[route] += 1
        else:
            route_counts[route] = 1

    # Print results
    print("\n" + "=" * 90)
    print("  86Q EVALUATION REPORT – HUSC Unified RAG Pipeline (PaddedRAG + GraphRAG)")
    print("=" * 90)
    print(f"  Models: NER={settings.RAMCLOUDS_NER_MODEL} | HYDE={settings.RAMCLOUDS_HYDE_MODEL} | GEN={settings.RAMCLOUDS_GEN_MODEL}")
    print(f"  Embedding: {settings.EMBEDDING_MODEL} (dim={settings.EMBEDDING_DIM})")
    print(f"  Route distribution: padded_rag={route_counts.get('padded_rag',0)} | graph_rag={route_counts.get('graph_rag',0)}")
    if force_route:
        print(f"  Force route: {force_route}")
    print()

    total_pass = 0
    total_has_ans = 0
    total_questions = len(results)

    for cat in ["simple", "multihop", "temporal"]:
        if cat not in by_category:
            continue
        c = by_category[cat]
        acc = c["passed"] / c["total"] * 100 if c["total"] else 0
        ans_rate = c["has_answer"] / c["total"] * 100 if c["total"] else 0
        avg_lat = sum(c["latency"]) / len(c["latency"]) if c["latency"] else 0
        total_pass += c["passed"]
        total_has_ans += c["has_answer"]
        print(f"  {cat:<12} {c['total']:>3}Q  pass={c['passed']:>3} ({acc:5.1f}%)  answer={c['has_answer']:>3} ({ans_rate:5.1f}%)  avg={avg_lat:.1f}s")

    overall_acc = total_pass / total_questions * 100 if total_questions else 0
    overall_ans = total_has_ans / total_questions * 100 if total_questions else 0
    all_lat = [r["latency_s"] for r in results]
    avg_total = sum(all_lat) / len(all_lat) if all_lat else 0
    print(f"  {'OVERALL':<12} {total_questions:>3}Q  pass={total_pass:>3} ({overall_acc:5.1f}%)  answer={total_has_ans:>3} ({overall_ans:5.1f}%)  avg={avg_total:.1f}s")
    print("=" * 90)

    # Per-question detail for failures
    failed = [r for r in results if not r["passed"]]
    if failed:
        print(f"\n  FAILURES ({len(failed)}):")
        for r in failed[:40]:
            ans_preview = r["answer"][:100].replace("\n", " ")
            print(f"    {r['id']:>6} [{r['category']}] route={r['route']} missing={r['must_include']} | {ans_preview}...")

    # GraphRAG-specific questions
    graph_q = [r for r in results if r["route"] == "graph_rag"]
    if graph_q:
        graph_pass = sum(1 for r in graph_q if r["passed"])
        print(f"\n  GRAPHRAG DETAIL: {len(graph_q)} questions routed to graph_rag, {graph_pass} passed ({graph_pass/len(graph_q)*100:.1f}%)")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    suffix_part = f"_{out_suffix}" if out_suffix else ""
    out_path = RESULTS_DIR / f"86q_results{suffix_part}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "ner_model": settings.RAMCLOUDS_NER_MODEL,
                "hyde_model": settings.RAMCLOUDS_HYDE_MODEL,
                "gen_model": settings.RAMCLOUDS_GEN_MODEL,
                "embedding_model": settings.EMBEDDING_MODEL,
                "embedding_dim": settings.EMBEDDING_DIM,
                "graphrag_threshold": settings.GRAPHRAG_SIMPLE_THRESHOLD,
                "total_questions": total_questions,
                "route_distribution": route_counts,
                "force_route": force_route,
            },
            "results": results,
            "summary": {
                cat: {
                    "accuracy": by_category[cat]["passed"] / by_category[cat]["total"] * 100,
                    "answer_rate": by_category[cat]["has_answer"] / by_category[cat]["total"] * 100,
                    "avg_latency": sum(by_category[cat]["latency"]) / len(by_category[cat]["latency"]),
                }
                for cat in by_category
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Also save a readable report for admissions office
    report_path = RESULTS_DIR / f"86q_bao_cao_tuyen_sinh{suffix_part}.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Báo cáo đánh giá hệ thống tư vấn tuyển sinh HUSC\n\n")
        f.write(f"## Tổng quan\n\n")
        f.write(f"- **Tổng số câu hỏi**: {total_questions}\n")
        f.write(f"- **Trả lời đúng (must_include)**: {total_pass}/{total_questions} ({overall_acc:.1f}%)\n")
        f.write(f"- **Có câu trả lời**: {total_has_ans}/{total_questions} ({overall_ans:.1f}%)\n")
        f.write(f"- **Routing**: PaddedRAG={route_counts.get('padded_rag',0)} | GraphRAG={route_counts.get('graph_rag',0)}\n\n")
        f.write(f"## Theo loại câu hỏi\n\n")
        f.write(f"| Loại | Số câu | Pass | Có TL | Độ chính xác |\n")
        f.write(f"|------|--------|------|-------|-------------|\n")
        for cat in ["simple", "multihop", "temporal"]:
            if cat not in by_category:
                continue
            c = by_category[cat]
            acc = c["passed"] / c["total"] * 100 if c["total"] else 0
            ans_rate = c["has_answer"] / c["total"] * 100 if c["total"] else 0
            cat_vi = {"simple": "Đơn giản", "multihop": "Nhiều bước", "temporal": "Theo thời gian"}.get(cat, cat)
            f.write(f"| {cat_vi} | {c['total']} | {c['passed']} | {c['has_answer']} ({ans_rate:.0f}%) | {acc:.1f}% |\n")
        f.write(f"\n## Chi tiết từng câu hỏi\n\n")
        for r in results:
            status_icon = "✅" if r["passed"] else ("⚠️" if r["has_answer"] else "❌")
            route_vi = "GraphRAG" if r["route"] == "graph_rag" else "PaddedRAG"
            f.write(f"### {status_icon} {r['id']} [{r['category']}] - {route_vi}\n\n")
            f.write(f"**Câu hỏi**: {r['question']}\n\n")
            f.write(f"**Câu trả lời**: {r['answer']}\n\n")
            if r.get("ground_truth"):
                f.write(f"**Đáp án đúng**: {r['ground_truth']}\n\n")
            if not r["passed"] and r["must_include"]:
                f.write(f"**Thiếu**: {r['must_include']}\n\n")
    print(f"  Admissions report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="86Q evaluation with UnifiedRAG Pipeline")
    parser.add_argument("--limit", type=int, default=0, help="Max questions (0=all)")
    parser.add_argument("--force-route", choices=["padded_rag", "graph_rag"], default=None,
                        help="Force all queries to use specified route")
    parser.add_argument("--embedding", choices=["default", "qwen", "bge"], default="default",
                        help="Embedding override: default=use .env / qwen=Qwen3-Embedding-0.6B / bge=BAAI/bge-m3")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix for output file: results/86q_results_{suffix}.json (default: 86q_results.json)")
    args = parser.parse_args()
    asyncio.run(run_eval(limit=args.limit, force_route=args.force_route,
                         embedding_override=args.embedding, out_suffix=args.out_suffix))
