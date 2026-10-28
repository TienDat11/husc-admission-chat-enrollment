"""
Eval V3: gọi /v3/query qua HTTP để đo Hybrid (Dense+BM25 RRF) + Aggregation Booster
+ Reranker (Qwen3-Reranker-8B + lost-in-middle) + Generator (gpt-6-luna, max_tokens=6000).

Yêu cầu: uvicorn server đã chạy ở http://localhost:8000
    cd rag2025 && uvicorn src.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import httpx
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "data" / "eval" / "husc_thi_sinh_thuc.json"
RESULTS_DIR = ROOT / "results"
DEFAULT_API = "http://localhost:8000/v3/query"


async def run_one(client: httpx.AsyncClient, q: Dict[str, Any], api: str,
                  top_k: int, candidate_pool: int) -> Dict[str, Any]:
    """POST a single query to /v3/query and capture full response + has_answer."""
    query = q["question"]
    payload = {
        "query": query,
        "top_k": top_k,
        "candidate_pool": candidate_pool,
        "apply_lost_in_middle": True,
    }
    start = time.perf_counter()
    try:
        r = await client.post(api, json=payload, timeout=180.0)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning(f"Query failed for {q['id']}: {exc}")
        elapsed = time.perf_counter() - start
        return {
            "id": q["id"],
            "question": query,
            "answer": f"ERROR: {exc}",
            "has_answer": False,
            "route": "error",
            "stages": {},
            "provider": f"error:{exc}",
            "latency_s": round(elapsed, 2),
            "chunks_used": 0,
            "sources": [],
            "confidence": 0.0,
        }

    answer = data.get("answer", "")
    ans_lower = answer.strip().lower()
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

    elapsed = data.get("latency_ms", (time.perf_counter() - start) * 1000) / 1000.0
    stages = data.get("pipeline_stages", {})

    return {
        "id": q["id"],
        "question": query,
        "answer": answer,
        "has_answer": has_answer,
        "route": "v3_hybrid",
        "stages": stages,
        "provider": data.get("provider", "?"),
        "latency_s": round(elapsed, 2),
        "chunks_used": data.get("chunks_used", 0),
        "sources": data.get("sources", []),
        "confidence": data.get("confidence", 0.0),
    }


async def main(api: str, out_suffix: str, top_k: int, candidate_pool: int,
               limit: int) -> None:
    if not EVAL_PATH.exists():
        logger.error(f"Eval file not found: {EVAL_PATH}")
        sys.exit(1)

    with EVAL_PATH.open("r", encoding="utf-8") as f:
        questions = json.load(f)
    if limit:
        questions = questions[:limit]
    logger.info(f"Loaded {len(questions)} questions; API={api}")

    suffix_part = f"_{out_suffix}" if out_suffix else ""
    out_path = RESULTS_DIR / f"86q_thi_sinh_results{suffix_part}.json"
    RESULTS_DIR.mkdir(exist_ok=True)

    out: list = []
    async with httpx.AsyncClient() as client:
        # Health check first
        try:
            r = await client.get(api.replace("/v3/query", "/health"), timeout=10.0)
            logger.info(f"Health check: {r.status_code} {r.text[:120]}")
        except Exception as exc:
            logger.error(f"Server not reachable at {api}: {exc}")
            sys.exit(2)

        for i, q in enumerate(questions):
            logger.info(f"{i + 1}/{len(questions)}: {q['id']}")
            r = await run_one(client, q, api, top_k=top_k,
                              candidate_pool=candidate_pool)
            out.append(r)
            stat = "HAS_ANS" if r["has_answer"] else "FALLBACK"
            logger.info(
                f"  -> {stat} | latency={r['latency_s']}s "
                f"| chunks={r['chunks_used']} | stages={r.get('stages', {})}"
            )

            # Partial save every question
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metadata": {
                            "api_endpoint": api,
                            "top_k": top_k,
                            "candidate_pool": candidate_pool,
                            "total": len(questions),
                            "completed": len(out),
                        },
                        "results": out,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    logger.info(f"Done. Output: {out_path}")
    # Print summary
    has_ans = sum(1 for r in out if r["has_answer"])
    logger.info(
        f"Summary: has_answer={has_ans}/{len(out)} "
        f"({has_ans / len(out) * 100:.1f}%)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="86Q via /v3/query")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--out-suffix", default="v3")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(
        main(
            api=args.api,
            out_suffix=args.out_suffix,
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            limit=args.limit,
        )
    )
