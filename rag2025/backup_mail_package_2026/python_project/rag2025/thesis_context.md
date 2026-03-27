# Thesis Context - PaddedRAG System

## Tổng quan hệ thống

Dự án này triển khai hệ thống tư vấn tuyển sinh thông minh cho Trường Đại Học Khoa Học – Đại Học Huế (HUSC), sử dụng kiến trúc Retrieval-Augmented Generation (RAG).

## Thông tin hệ thống cũ (PaddedRAG/Naive RAG)

### Dataset
- **Loại dữ liệu**: Tài liệu tuyển sinh đại học năm 2025
- **Nguồn**: Thông tư Bộ GDĐT, quy chế tuyển sinh, FAQ tuyển sinh
- **Định dạng**: JSONL (JSON Lines)
- **Số lượng**: 11 files chunked (chunked_1.jsonl đến chunked_11.jsonl)
- **Metadata**: source, issued_date, effective_date, info_type, audience, unit, year

### Embedding Model
- **Primary**: `intfloat/e5-small-v2` (384-dimension)
- **Multilingual**: `intfloat/multilingual-e5-base` (768-dimension)
- **BGE-M3**: `BAAI/bge-m3` (1024-dimension, multi-vector)
- **Tokenizer**: tiktoken (cl100k_base)
- **Batch size**: 32
- **Normalization**: L2-normalize embeddings

### Vector Store
- **Primary**: Qdrant (http://localhost:6333)
- **Fallback**: NumPy local store (index/vector_store.npz)
- **Collection**: rag2025

### Chunking Strategy
- **Chunk size**: 350 tokens (configurable 300-500)
- **Overlap**: 70 tokens
- **Profiles**: auto, faq, policy
- **Min tokens**: 50

### Retrieval Parameters
- **Confidence threshold**: 0.35 (adaptive: 0.35/<500 docs, 0.45/500-5k, 0.55/>5k)
- **Top-K Dense**: 20
- **Top-K Sparse** (BM25): 20
- **Max rerank**: 50
- **Semantic compression**: Top-3 hits

### LLM Backend
- **Models**: gemini-2.0-flash-exp, OpenAI, Z.AI (GLM-4.5)
- **Temperature**: 0.1
- **Force RAG-only mode**: Có thể bật

### Reranker
- **Model**: BAAI/bge-reranker-base (cross-encoder)

## Cấu trúc project

```
rag2025/
├── src/
│   ├── main.py                 # Entry point
│   ├── chunker.py              # Adaptive chunking
│   ├── normalize_data.py       # Data normalization
│   ├── validate_jsonl.py       # JSONL validation
│   └── services/
│       ├── embedding.py         # Embedding service
│       ├── vector_store.py     # Vector storage
│       ├── qdrant_retrieval.py # Qdrant-specific retrieval
│       ├── bge_retriever.py    # BGE retriever
│       ├── retriever.py        # Main retriever
│       ├── llm_generator.py    # LLM response generation
│       └── query_enhancer.py   # Query enhancement (HyDE)
├── config/
│   ├── settings.py             # Pydantic settings
│   ├── chunk_profiles.yaml     # Chunking profiles
│   └── rag_chunk_schema.json   # JSON Schema
├── scripts/
│   ├── ingest_all_chunks.py    # Ingest chunks to vector DB
│   ├── build_index.py          # Build FAISS index
│   ├── check_qdrant.py         # Check Qdrant status
│   └── enhance_chunks.py      # Enhance chunks with metadata
├── data/
│   ├── chunked/                # Chunked documents
│   ├── normalized/             # Normalized data
│   └── validated/              # Validated data
├── prompts/
│   ├── generation_system_prompt.txt
│   └── hyde_system_prompt.txt
├── index/                      # Vector indices
├── tests/                      # Unit tests
├── venv/                       # Virtual environment
├── thesis_latex.tex            # Thesis LaTeX
└── requirements.txt            # Dependencies
```

## Ghi chú quan trọng

1. **Hệ thống hiện tại là Naive RAG** - chưa có GraphRAG
2. **Chunking strategy** là token-based với overlap - đây là PaddedRAG (thêm padding để giảm context fragmentation)
3. **Chưa có Neo4j** - đây là yêu cầu chuyển đổi từ cô giáo hướng dẫn
4. **Cần chạy thực nghiệm** để so sánh Naive RAG vs GraphRAG


## Thực nghiệm và Đánh giá

### Evaluation Script
 **Không có evaluation script có sẵn** để tính MRR, NDCG, hay faithfulness
 **Cần tự viết evaluation script** hoặc để trống số liệu ở Chương 3
 **Lý do**: Hệ thống hiện tại tập trung vào phát triển chức năng, chưa có script đánh giá tự động

### Ghi chú cho Chương 3
 Nếu không có số liệu MRR/NDCG/faithfulness, có thể:
  1. Tự viết evaluation script để thu thập số liệu
  2. Để trống phần số liệu trong bảng so sánh
  3. Sử dụng các đánh giá định tính thay thế (ví dụ: accuracy, precision, recall)

### Các file có thể liên quan đến evaluation:
 `debug_rag_hallucination.py` - Debug script cho hallucination
 `verify_api.py` - Script test API
 Các file test trong thư mục `tests/` - Unit tests chức năng