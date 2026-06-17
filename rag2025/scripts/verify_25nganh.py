"""Verify 25 ngành fix - test 6 critical questions using eval_86q pattern."""
import asyncio
import json, sys, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.graphrag_retriever import UnifiedRAGPipeline
from services.lancedb_retrieval import LanceDBRetriever
from services.embedding import EmbeddingService
from services.llm_generator import LLMGenerator
from config.settings import RAGSettings
from services.aggregation_booster import boost_with_aggregation

settings = RAGSettings()
pipeline = UnifiedRAGPipeline.from_disk()
embedding_svc = EmbeddingService(settings)
retriever = LanceDBRetriever.from_env()
generator = LLMGenerator()

test_msgs = [
    {'id': 'msg001', 'text': 'HUSC có bao nhiêu ngành tuyển sinh năm 2026?'},
    {'id': 'msg007', 'text': 'Em thi được 20 điểm khối C01 thì đỗ ngành nào ở HUSC?'},
    {'id': 'msg015', 'text': 'Cho em hỏi thông tin các ngành đào tạo của trường Đại học Khoa học Huế'},
    {'id': 'msg017', 'text': 'Trường Đại học Khoa học Huế tuyển sinh bao nhiêu ngành năm 2026? Các ngành là gì?'},
    {'id': 'msg022', 'text': 'Em thi khối C00 thì có những ngành nào ở HUSC?'},
    {'id': 'msg049', 'text': 'Mỗi tổ hợp môn chỉ được dùng cho 1 phương thức xét tuyển thôi phải không?'},
]

async def verify():
    results = []
    for msg in test_msgs:
        t0 = time.time()
        q = msg['text']

        # Step 1: Route
        router_result = await pipeline._router.route(q)
        all_queries = router_result.all_queries_for_embedding

        # Step 2: Retrieve from LanceDB (same pattern as eval_86q)
        baseline_docs = []
        for variant in all_queries[:4]:
            query_vector = embedding_svc.encode_single(variant).tolist()
            retrieval_result = retriever.retrieve(query_vector, top_k=10)
            if retrieval_result.is_success:
                baseline_docs.extend(retrieval_result.documents)

        # Deduplicate by chunk_id
        seen_ids = set()
        unique_docs = []
        for doc in baseline_docs:
            if doc.chunk_id not in seen_ids:
                seen_ids.add(doc.chunk_id)
                unique_docs.append(doc)
        baseline_docs = unique_docs[:15]

        # Inject aggregation chunks
        baseline_docs = boost_with_aggregation(q, baseline_docs, retriever, top_k=15, max_inject=2)

        # Step 3: Pipeline query
        rag_result = await pipeline.query(
            user_query=q,
            baseline_docs=baseline_docs,
            router_result=router_result,
        )
        top_docs = rag_result.documents[:5]

        # Step 4: Generate answer
        chunks = []
        for doc in top_docs:
            chunks.append({
                'text': doc.text,
                'source': doc.source,
                'chunk_id': doc.chunk_id,
            })

        is_enum = 'bao nhiêu ngành' in q.lower() or 'liệt kê' in q.lower()
        gen_result = await generator.generate_answer(
            query=q,
            chunks=chunks,
            confidence=rag_result.confidence,
            is_program_list_query=is_enum,
        )
        answer = gen_result.get('answer', '')
        elapsed = time.time() - t0

        has_25 = '25 ngành' in answer
        has_26 = '26 ngành' in answer or '26 chương trình' in answer
        status = 'OK_25' if has_25 and not has_26 else ('STILL_26' if has_26 else 'NO_MENTION')
        results.append({
            'id': msg['id'],
            'status': status,
            'has_25': has_25,
            'has_26': has_26,
            'elapsed': round(elapsed, 1),
            'answer': answer[:200],
        })
        print(f"{msg['id']}: {status} | 25={has_25} 26={has_26} | {elapsed:.1f}s")

    out_path = Path(__file__).parent.parent / "results" / "verify_25nganh_v2.json"
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok_count = sum(1 for r in results if r['status'] in ('OK_25', 'NO_MENTION'))
    print(f"\nResult: {ok_count}/{len(results)} correct")
    print(f"Saved to {out_path}")

asyncio.run(verify())
