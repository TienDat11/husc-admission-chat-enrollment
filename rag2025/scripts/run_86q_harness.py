"""
Canonical 86-Question Evaluation Harness

Scope: HUSC 2026 tuyển sinh only
Question count: EXACTLY 86 (q001-q086)
Scoring: SCOPE_PASS, LANG_PASS, YEAR_PASS, GROUNDED_PASS, CLARITY_PASS

Usage:
    python scripts/run_86q_harness.py [--questions PATH] [--output PATH]
"""
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# HUSC 26-program canonical list for scope checking
HUSC_PROGRAMS = [
    "công nghệ thông tin", "kỹ thuật phần mềm", "khoa học máy tính",
    "quản trị và phân tích dữ liệu", "kỹ thuật điện tử viễn thông",
    "công nghệ sinh học", "kỹ thuật hóa học", "khoa học vật liệu",
    "kiến trúc", "địa kỹ thuật xây dựng", "kỹ thuật trắc địa bản đồ",
    "quản lý tài nguyên và môi trường", "khoa học môi trường",
    "toán học", "vật lý học", "hóa học", "sinh học",
    "địa lý học", "lịch sử", "văn học", "hán nôm", "triết học",
    "đông phương học", "báo chí", "truyền thông số",
    "xã hội học", "công tác xã hội", "quản lý văn hóa",
]

# Non-HUSC programs that commonly hallucinate
NON_HUSC_PROGRAMS = [
    "y khoa", "dược học", "răng hàm mặt", "kinh tế",
    "quản trị kinh doanh", "luật", "giáo dục tiểu học",
    "giáo dục mầm non", "giáo dục trung học",
]


def load_86_questions(path: str) -> List[Dict]:
    """Load exactly 86 questions from the test JSON file."""
    with open(path, encoding="utf-8") as f:
        all_questions = json.load(f)

    # Filter to exactly q001-q086
    questions = []
    for q in all_questions:
        qid = q.get("id", "")
        if qid.startswith("q0"):
            try:
                num = int(qid[2:])
                if 1 <= num <= 86:
                    questions.append(q)
            except ValueError:
                continue

    questions.sort(key=lambda q: q["id"])

    if len(questions) != 86:
        logger.warning(f"Expected 86 questions, found {len(questions)}")

    return questions


def score_scope(answer: str) -> bool:
    """Check: answer references only HUSC 26-program catalog."""
    answer_lower = answer.lower()
    for prog in NON_HUSC_PROGRAMS:
        if prog in answer_lower:
            return False
    return True


def score_lang(answer: str) -> bool:
    """Check: answer is 100% Vietnamese (allow proper nouns/abbreviations)."""
    # Detect English sentences (3+ consecutive English-only words)
    # Exclude numbers, currency codes, and common abbreviations
    english_pattern = re.compile(r'(?:^|[\s(])([A-Za-z]{3,}(?:\s+[A-Za-z]{3,}){2,})(?:[\s).,]|$)')
    matches = english_pattern.findall(answer)
    for m in matches:
        # Allow common abbreviations/proper nouns
        allowed = {"THPT", "HUSC", "IELTS", "TOEFL", "VSTEP", "VNĐ",
                   "KTX", "CNTT", "KTPM", "NV", "CCCD", "DHT", "A00", "A01"}
        words = m.split()
        if all(w in allowed for w in words):
            continue
        return False
    return True


def score_year(answer: str, question: str) -> bool:
    """Check: answer uses 2026 data (not 2025 unless comparison asked)."""
    answer_lower = answer.lower()
    question_lower = question.lower()

    is_comparison = any(kw in question_lower for kw in
                        ["so sánh", "so sanh", "đối chiếu", "thay đổi",
                         "tăng", "giảm", "so với", "năm trước", "năm cũ"])

    # If comparison query, 2025 is OK
    if is_comparison:
        return True

    # If answer mentions 2025 without comparison context, fail
    if "2025" in answer_lower and "2026" not in answer_lower:
        return False

    return True


def score_grounded(answer: str, chunks: List[Dict]) -> bool:
    """Check: answer claims are supported by retrieved chunks."""
    if not chunks:
        return False

    answer_lower = answer.lower()

    # Extract key numbers from answer
    numbers_in_answer = re.findall(r'\d+[.,]?\d*', answer)
    if not numbers_in_answer:
        return True  # No specific claims to verify

    # Check if at least one chunk mentions a number from the answer
    chunk_texts = " ".join(c.get("text", "") + " " + c.get("summary", "")
                           for c in chunks).lower()

    matches = 0
    for num in numbers_in_answer:
        clean = num.replace(",", ".")
        if clean in chunk_texts:
            matches += 1

    # Pass if at least 50% of numbers are grounded
    return matches >= max(1, len(numbers_in_answer) * 0.5)


def score_clarity(answer: str) -> bool:
    """Check: answer is specific, not generic fallback."""
    generic_phrases = [
        "liên hệ nhà trường để biết thêm",
        "vui lòng liên hệ",
        "hãy liên hệ",
        "thông tin chưa có",
        "không có thông tin",
        "chưa được cập nhật",
    ]
    answer_lower = answer.lower()

    # Generic phrase only fails if answer is very short (< 30 words)
    for phrase in generic_phrases:
        if phrase in answer_lower and len(answer.split()) < 30:
            return False

    return True


async def run_single_question(
    question: Dict,
    pipeline: Any,
    embedder: Any,
    hybrid: Any,
) -> Dict:
    """Run a single question through the full RAG pipeline and score it."""
    from services.lancedb_retrieval import LanceDBRetriever, LanceDBRetrieverConfig
    from services.llm_generator import LLMGenerator

    q_text = question["question"]
    q_id = question["id"]

    try:
        # Step 1: Embed query
        from config.settings import RAGSettings
        settings = RAGSettings()
        query_vector = embedder.encode(q_text)

        # Step 2: Retrieve
        from services.hybrid_search import HybridSearchService
        from services.lancedb_retrieval import LanceDBRetriever, LanceDBRetrieverConfig

        retriever_config = LanceDBRetrieverConfig(
            uri=settings.LANCEDB_URI,
            table_name=settings.LANCEDB_TABLE,
            embedding_dim=settings.EMBEDDING_DIM,
        )
        retriever = LanceDBRetriever(config=retriever_config)

        hybrid_svc = HybridSearchService(retriever, settings)
        hybrid_svc.build_bm25_index()

        retrieval_result = await hybrid_svc.retrieve(
            query=q_text,
            query_vector=query_vector.tolist(),
            top_k=5,
        )

        chunks = []
        for doc in retrieval_result.documents[:5]:
            chunks.append({
                "text": doc.text,
                "summary": doc.metadata.get("summary", ""),
                "source": doc.source,
                "chunk_id": doc.chunk_id,
                "metadata": doc.metadata,
            })

        # Step 3: Generate answer
        generator = LLMGenerator()
        gen_result = await generator.generate_answer(
            query=q_text,
            chunks=chunks,
            confidence=retrieval_result.confidence,
        )
        answer = gen_result.get("answer", "")

    except Exception as e:
        logger.error(f"Pipeline error for {q_id}: {e}")
        answer = f"[ERROR: {e}]"
        chunks = []

    # Step 4: Score
    scores = {
        "SCOPE_PASS": score_scope(answer),
        "LANG_PASS": score_lang(answer),
        "YEAR_PASS": score_year(answer, q_text),
        "GROUNDED_PASS": score_grounded(answer, chunks),
        "CLARITY_PASS": score_clarity(answer),
    }

    fail_reasons = []
    for k, v in scores.items():
        if not v:
            fail_reasons.append(f"{k}: FAIL")

    return {
        "question_id": q_id,
        "question": q_text,
        "system_answer": answer,
        "retrieved_chunk_ids": [c.get("chunk_id", "") for c in chunks[:3]],
        "scores": scores,
        "fail_reasons": fail_reasons,
    }


async def run_harness(questions_path: str, output_path: str) -> Dict:
    """Run the full 86-question harness."""
    questions = load_86_questions(questions_path)
    logger.info(f"Loaded {len(questions)} questions")

    # Initialize embedder
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("dangvantuan/sentence-camembert-large")

    results = []
    for i, q in enumerate(questions):
        logger.info(f"Processing {q['id']} ({i+1}/{len(questions)})")
        result = await run_single_question(q, pipeline=None, embedder=embedder, hybrid=None)
        results.append(result)

    # Calculate summary
    total = len(results)
    summary = {}
    for criterion in ["SCOPE_PASS", "LANG_PASS", "YEAR_PASS", "GROUNDED_PASS", "CLARITY_PASS"]:
        passed = sum(1 for r in results if r["scores"].get(criterion, False))
        rate = passed / total if total > 0 else 0
        summary[f"{criterion}_rate"] = round(rate, 4)

    blocking = []
    if summary["SCOPE_PASS_rate"] < 0.95:
        blocking.append("SCOPE_PASS below 95%")
    if summary["LANG_PASS_rate"] < 1.0:
        blocking.append("LANG_PASS below 100%")
    if summary["YEAR_PASS_rate"] < 0.90:
        blocking.append("YEAR_PASS below 90%")
    if summary["GROUNDED_PASS_rate"] < 0.85:
        blocking.append("GROUNDED_PASS below 85%")
    if summary["CLARITY_PASS_rate"] < 0.80:
        blocking.append("CLARITY_PASS below 80%")

    harness_result = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question_count": total,
        "scope": "HUSC_2026",
        "results": results,
        "summary": {
            **summary,
            "overall_pass": len(blocking) == 0,
            "blocking_failures": blocking,
        },
    }

    # Write output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(harness_result, f, ensure_ascii=False, indent=2)

    logger.info(f"Harness result written to {output}")
    logger.info(f"Summary: {summary}")
    logger.info(f"Overall pass: {len(blocking) == 0}")
    if blocking:
        logger.warning(f"Blocking failures: {blocking}")

    return harness_result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="86-Question Evaluation Harness")
    parser.add_argument("--questions", default="data/eval/husc_86q_test.json",
                        help="Path to test questions JSON")
    parser.add_argument("--output", default="results/harness_result.json",
                        help="Output path for harness result")
    args = parser.parse_args()

    questions_path = str(Path(__file__).parent.parent / args.questions)
    output_path = str(Path(__file__).parent.parent / args.output)

    asyncio.run(run_harness(questions_path, output_path))


if __name__ == "__main__":
    main()
