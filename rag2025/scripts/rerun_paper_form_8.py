"""Re-run specific question IDs to update paper-form answers post-corpus-fix."""
import os
import sys
import asyncio
import json
from pathlib import Path

os.environ.setdefault('HF_HOME', 'D:/chunking/.cache/huggingface')
os.environ.setdefault('HF_HUB_CACHE', 'D:/chunking/.cache/huggingface/hub')

sys.path.insert(0, 'src')

from dotenv import load_dotenv
load_dotenv('.env')

from config.settings import RAGSettings
from services.graphrag_retriever import UnifiedRAGPipeline
from services.lancedb_retrieval import LanceDBRetriever
from services.embedding import EmbeddingService
from services.llm_generator import LLMGenerator
from src.services.aggregation_booster import boost_with_aggregation


async def go():
    settings = RAGSettings()
    pipeline = UnifiedRAGPipeline.from_disk()
    retriever = LanceDBRetriever.from_env()
    embedding = EmbeddingService(settings)
    generator = LLMGenerator()

    target_ids = ['msg005', 'msg013', 'msg040', 'msg046', 'msg053', 'msg055', 'msg058', 'msg074']
    eval_data = json.load(open('data/eval/husc_thi_sinh_thuc.json', encoding='utf-8'))
    questions = {q['id']: q['question'] for q in eval_data}

    results = {}
    for qid in target_ids:
        if qid not in questions:
            print(f"SKIP {qid}: not in eval set")
            continue
        query = questions[qid]
        print(f"\n========== {qid} ==========")
        print(f"Q: {query[:80]}")
        try:
            rr = await pipeline._router.route(query)
            qv = embedding.encode_single(query).tolist()
            res = retriever.retrieve(qv, top_k=10)
            docs = res.documents[:15] if res.is_success else []
            docs = boost_with_aggregation(query, docs, retriever, top_k=15, max_inject=2)

            rag_result = await pipeline.query(
                user_query=query, baseline_docs=docs, top_k=5, router_result=rr
            )
            chunks_for_gen = [
                {"text": getattr(d, 'text', ''), "summary": '', "metadata": getattr(d, 'metadata', {})}
                for d in rag_result.documents
            ]
            gen = await generator.generate_answer(
                query=query, chunks=chunks_for_gen, confidence=0.5, is_program_list_query=False
            )
            answer = gen.get('answer', '')
            results[qid] = answer
            print(f"NEW ANSWER (first 250 chars):\n{answer[:250]}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[qid] = None

    # Save patches
    with open('logs/msg_paper_patches.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\n=== Saved {len([v for v in results.values() if v])} patches to logs/msg_paper_patches.json ===")


if __name__ == '__main__':
    asyncio.run(go())
