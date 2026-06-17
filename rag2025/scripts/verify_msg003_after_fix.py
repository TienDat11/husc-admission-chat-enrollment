"""Quick test msg003 after corpus fix - direct pipeline."""
import os
import sys
import asyncio
import json

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

    queries = [
        ("msg003", "Em muốn đăng ký xét tuyển vào trường  mình, phải làm sao ạ?"),
        ("msg010", "Xét tuyển thẳng đăng ký  như thế nào?"),
    ]

    for qid, query in queries:
        print(f"\n========== {qid} ==========")
        print(f"Q: {query}")
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
        chunk_ids = [getattr(d, 'chunk_id', '') for d in rag_result.documents]
        print(f"CHUNK_IDS: {chunk_ids}")
        print(f"\nANSWER:\n{gen.get('answer', '')}")


if __name__ == '__main__':
    asyncio.run(go())
