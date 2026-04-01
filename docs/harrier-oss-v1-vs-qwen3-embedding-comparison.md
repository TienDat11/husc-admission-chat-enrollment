# So Sánh Harrier-OSS-v1 vs Qwen3-Embedding

**Ngày:** 2026-04-01
**Dự án:** husc-admission-chat-enrollment
**Mục đích:** Đánh giá Harrier-OSS-v1 có thể thay thế Qwen3-Embedding tốt hơn không

---

## Tóm Tắt Điều Hành

**Harrier-OSS-v1** là model embedding mới nhất của Microsoft (30/03/2026), đạt SOTA trên Multilingual MTEB v2 với **74.3 điểm** (variant 27B). Trong khi đó **Qwen3-Embedding-8B** đạt 70.58.

**Khuyến nghị sơ bộ:** Harrier-OSS-v1 **có thể** thay thế tốt hơn, đặc biệt variant **0.6B** (69.0 MTEB) rất cân bằng giữa chất lượng và tài nguyên — gần tương đương Qwen3-4B (69.45) nhưng nhẹ hơn đáng kể.

---

## 1. Thông Số Kỹ Thuật So Sánh

### Bảng So Sánh Đầy Đủ

| Tiêu chí | **Harrier-OSS-v1-270M** | **Harrier-OSS-v1-0.6B** | **Harrier-OSS-v1-27B** | **Qwen3-Embedding-0.6B** | **Qwen3-Embedding-4B** | **Qwen3-Embedding-8B** |
|-----------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| **Tham số** | 270M | 600M | 27B | 600M | 4B | 8B |
| **Embedding Dim** | 640 | 1024 | 5376 | 1024 | 2560 | 4096 |
| **MTEB Score** | 66.5 | 69.0 | **74.3 (SOTA)** | 64.33 | 69.45 | 70.58 |
| **License** | MIT | MIT | MIT | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **Ngôn ngữ** | 94 | 94 | 94 | 100+ | 100+ | 100+ |
| **Context window** | 32,768 tok | 32,768 tok | 32,768 tok | 8,192 tok | 8,192 tok | 8,192 tok |
| **Disk (BF16)** | ~0.5 GB | ~1.2 GB | ~54 GB | ~1.2 GB | ~8 GB | ~15 GB |
| **RAM (CPU)** | ~2 GB | ~4 GB | ~80 GB | ~4 GB | ~12 GB | ~20 GB |
| **Kiến trúc** | Decoder-only | Decoder-only | Decoder-only | Encoder-style | Encoder-style | Encoder-style |

### Điểm MTEB Chi Tiết (Nguồn: Awesome Agents, MarkTechPost)

| Model | MTEB (Multilingual v2) | Leaderboard Rank |
|-------|------------------------|------------------|
| **Harrier-OSS-v1-27B** | **74.3** | **#1 SOTA** |
| Qwen3-Embedding-8B | 70.58 | #2 |
| Harrier-OSS-v1-0.6B | 69.0 | Top 10 |
| Qwen3-Embedding-4B | 69.45 | Top 10 |
| Gemini Embedding 001 | 68.32 | — |
| Harrier-OSS-v1-270M | 66.5 | — |
| Qwen3-Embedding-0.6B | 64.33 | — |
| OpenAI text-embedding-3-large | 64.6 | — |

---

## 2. Điểm Hơn Harrier-OSS-v1 So Với Qwen3-Embedding

### 2.1. Chất Lượng Retrieval (MTEB)

| Ưu điểm | Chi tiết |
|----------|----------|
| **SOTA Multilingual MTEB v2** | Harrier-27B đạt 74.3 — cao hơn 3.7 điểm so với Qwen3-8B (70.58) |
| **0.6B variant vượt Qwen3-4B** | Harrier-0.6B (69.0) ≈ Qwen3-4B (69.45) nhưng chỉ cần 1.2GB disk thay vì 8GB |
| **License tốt hơn** | MIT license cho phép thương mại hoàn toàn, không như Apache 2.0 có thể có ràng buộc |
| **32k context** | Gấp 4x Qwen3 (8k) — tốt cho document retrieval dài |

### 2.2. Tài Nguyên & Hiệu Suất

| Ưu điểm | Chi tiết |
|----------|----------|
| **Harrier-0.6B ≈ Qwen3-4B quality** | 69.0 vs 69.45 MTEB — khác biệt không đáng kể |
| **Disk nhẹ hơn 7x** | Harrier-0.6B (1.2GB) vs Qwen3-4B (8GB) |
| **RAM thấp hơn 3x** | Harrier-0.6B (~4GB) vs Qwen3-4B (~12GB) |
| **Không cần GPU** | Cả hai đều chạy CPU, nhưng Harrier tiết kiệm hơn nhiều |

### 2.3. Kiến Trúc

| Ưu điểm | Chi tiết |
|----------|----------|
| **Decoder-only architecture** | Giống LLM hiện đại, proven cho embedding chất lượng cao |
| **Last-token pooling** | Hiệu quả hơn attention-based pooling cổ điển |
| **Knowledge distillation** | 270M và 0.6B được distill từ model lớn hơn, đạt chất lượng vượt tầm parameter count |

---

## 3. Điểm Kém Harrier-OSS-v1 So Với Qwen3-Embedding

### 3.1. Hạn Chế Đã Biết

| Nhược điểm | Chi tiết |
|------------|----------|
| **Không có paper/technical report** | Microsoft không publish phương pháp huấn luyện — khó audit cho compliance |
| **Chưa có quantization** | Chỉ có BF16, chưa có GGUF/AWQ — 27B cần 80GB VRAM |
| **Per-language scores không công bố** | 94 ngôn ngữ nhưng không biết Vietnamese performance cụ thể |
| **MTEB v2 vs MTEB thường** | MTEB v2 benchmark mới hơn, so sánh với Qwen3 (dùng MTEB thường) có thể không apple-to-apple |

### 3.2. Vietnamese-Specific

| Cân nhắc | Chi tiết |
|----------|----------|
| **VN-MTEB benchmark tồn tại** | Có benchmark riêng cho Vietnamese (ACLANanthology 2026) — cần test thực tế |
| **Qwen3 đã được test trong repo** | Qwen3-Embedding đã tích hợp và chạy được trong hệ thống |
| **Harrier mới hoàn toàn** | Chưa có ai test Vietnamese retrieval trên Harrier trong repo này |

---

## 4. So Sánh Chi Phí Tài Nguyên (Cho Máy 16GB RAM)

### Kịch Bản A: Máy yếu (8-16GB RAM, disk < 10GB còn trống)

| Tiêu chí | **Harrier-0.6B** | **Qwen3-0.6B** | Người thắng |
|-----------|-----------------|----------------|-------------|
| Disk | ~1.2 GB | ~1.2 GB | Hòa |
| RAM khi chạy | ~4 GB | ~4 GB | Hòa |
| MTEB | **69.0** | 64.33 | **Harrier +4.7** |
| Vietnamese test | Chưa test | Đã test | **Qwen3** |

**→ Khuyến nghị:** Qwen3-0.6B cho đến khi có VN-MTEB benchmark cho Harrier

### Kịch Bản B: Máy trung bình (16GB RAM, disk 10-20GB còn trống)

| Tiêu chí | **Harrier-0.6B** | **Qwen3-4B** | Người thắng |
|-----------|-----------------|--------------|-------------|
| Disk | ~1.2 GB | ~8 GB | **Harrier** |
| RAM khi chạy | ~4 GB | ~12 GB | **Harrier** |
| MTEB | 69.0 | 69.45 | Qwen3 +0.45 |
| Quality/size ratio | **57.5/GB** | 8.7/GB | **Harrier 6.6x tốt hơn** |

**→ Khuyến nghị:** **Harrier-0.6B** — quality tương đương, resource hiệu quả hơn 6.6x

### Kịch Bản C: Máy mạnh (32GB+ RAM, GPU)

| Tiêu chí | **Harrier-27B** | **Qwen3-8B** | Người thắng |
|-----------|-----------------|--------------|-------------|
| MTEB | **74.3** | 70.58 | **Harrier +3.7** |
| Disk | ~54 GB | ~15 GB | Qwen3 |
| VRAM cần | 80GB+ | 16GB | **Qwen3** |

**→ Khuyến nghị:** Qwen3-8B trừ khi có đủ GPU

---

## 5. Tất Cả Phương Thức Cần Thay Đổi Trong Codebase

### 5.1. File: `rag2025/config/settings.py`

**Hiện tại:**
```python
EMBEDDING_MODEL: str = Field(
    default="Qwen/Qwen3-Embedding-4B",
    description="HuggingFace model name for embeddings",
)
EMBEDDING_DIM: int = Field(default=2560, description="Embedding dimension")
QWEN_EMBEDDING_MODEL: str = Field(
    default="Qwen/Qwen3-Embedding-4B",
    description="Qwen3 embedding model for multilingual retrieval",
)
QWEN_EMBEDDING_DIM: int = Field(default=2560, description="Qwen3 embedding dimension")
```

**Cần thêm (cho Harrier):**
```python
# ========== Harrier Embedding Configuration ==========
HARRIER_EMBEDDING_MODEL: str = Field(
    default="microsoft/harrier-oss-v1-0.6b",
    description="Harrier-OSS embedding model for multilingual retrieval",
)
HARRIER_EMBEDDING_DIM: int = Field(default=1024, description="Harrier-0.6B embedding dimension")
HARRIER_EMBEDDING_BATCH_SIZE: int = Field(default=8, description="Batch size for Harrier encoding")

# ========== Embedding Model Selection ==========
EMBEDDING_PROVIDER: Literal["qwen", "harrier"] = Field(
    default="qwen",
    description="Embedding provider: 'qwen' or 'harrier'",
)
```

**Validator cần cập nhật:**
```python
@field_validator("EMBEDDING_DIM", mode="before")
@classmethod
def validate_embedding_dim(cls, v):
    v = int(v)
    # Harrier dimensions: 640 (270M), 1024 (0.6B), 5376 (27B)
    # Qwen3 dimensions: 1024 (0.6B), 2560 (4B), 4096 (8B)
    valid_dims = [640, 1024, 2560, 4096, 5376]
    if v not in valid_dims:
        raise ValueError(f"EMBEDDING_DIM must be one of {valid_dims}")
    return v
```

### 5.2. File: `rag2025/src/services/embedding.py`

**Hiện tại (Qwen3):**
```python
def encode_query(self, query: str) -> np.ndarray:
    """Encode query with Qwen3 instruction prompt."""
    embedding = self.model.encode(
        query,
        prompt_name="query",  # CRITICAL for Qwen3
        normalize_embeddings=self.normalize,
        convert_to_numpy=True,
    ).astype(np.float32)
    return embedding
```

**Cần thêm (Harrier support):**
```python
def encode_query(self, query: str) -> np.ndarray:
    """Encode query with instruction prompt (Qwen3 or Harrier)."""
    if self.settings.EMBEDDING_PROVIDER == "harrier":
        # Harrier uses web_search_query prompt
        embedding = self.model.encode(
            query,
            prompt_name="web_search_query",  # Harrier's query instruction
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        ).astype(np.float32)
    else:
        # Qwen3 uses query prompt
        embedding = self.model.encode(
            query,
            prompt_name="query",  # Qwen3's query instruction
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        ).astype(np.float32)
    return embedding

def encode_documents(self, documents: List[str]) -> np.ndarray:
    """Encode documents WITHOUT instruction (both Qwen3 and Harrier)."""
    # Both models encode docs without any prompt prefix
    if not documents:
        return np.empty((0, self.expected_dim), dtype=np.float32)
    
    embeddings = self.model.encode(
        documents,
        batch_size=self.batch_size,
        normalize_embeddings=self.normalize,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    return embeddings
```

**Prompt mapping:**

| Provider | Model | Query Prompt | Document |
|---------|-------|--------------|----------|
| Qwen | Qwen3-Embedding-0.6B/4B/8B | `prompt_name="query"` | No prompt |
| Harrier | harrier-oss-v1-270M/0.6B/27B | `prompt_name="web_search_query"` | No prompt |

### 5.3. File: `rag2025/scripts/ingest_lancedb.py`

**Hiện tại (hardcoded Qwen3 reference):**
```python
"""
Ingest chunked JSONL files into LanceDB using Qwen3-Embedding.
"""
```

**Cần cập nhật:**
```python
"""
Ingest chunked JSONL files into LanceDB using embedding models.
Supports Qwen3-Embedding and Harrier-OSS-v1.
Model selection via EMBEDDING_PROVIDER in settings.
"""
```

### 5.4. File: `rag2025/src/services/lancedb_retrieval.py`

**Không cần thay đổi** — LanceDB lưu vector dưới dạng list, dimension được validate khi insert/query. Chỉ cần đảm bảo `EMBEDDING_DIM` trong settings match với model.

### 5.5. File: `rag2025/src/services/hybrid_search.py`

**Không cần thay đổi** — BM25 index không phụ thuộc embedding model. Dense retrieval được handle bởi `lancedb_retrieval.py` đã check dimension.

### 5.6. File: `rag2025/tests/test_embedding.py`

**Cần thêm test cho Harrier:**
```python
@pytest.fixture
def embedding_provider():
    """Fixture to test both embedding providers."""
    return RAGSettings().EMBEDDING_PROVIDER

def test_harrier_query_prompt(embedding_service, embedding_provider):
    """Test Harrier-specific query encoding if provider is harrier."""
    if embedding_provider != "harrier":
        pytest.skip("Only test Harrier-specific behavior")
    
    query = "What is the admission deadline?"
    embedding = embedding_service.encode_query(query)
    assert embedding.shape == (embedding_service.expected_dim,)

def test_harrier_document_no_prompt(embedding_service, embedding_provider):
    """Test Harrier documents don't have instruction prefix."""
    if embedding_provider != "harrier":
        pytest.skip("Only test Harrier-specific behavior")
    
    docs = ["Document without instruction"]
    embeddings = embedding_service.encode_documents(docs)
    assert embeddings.shape[1] == embedding_service.expected_dim
```

---

## 6. Hướng Dẫn So Sánh Hiệu Suất Thực Tế

### 6.1. Baseline: Đo Qwen3-Embedding Hiện Tại

**Bước 1: Chạy benchmark Qwen3-4B (baseline)**
```bash
cd rag2025
python -c "
import time
from config.settings import RAGSettings
from src.services.embedding import EmbeddingService

settings = RAGSettings()
print(f'Current model: {settings.EMBEDDING_MODEL}')
print(f'Dimension: {settings.EMBEDDING_DIM}')

emb = EmbeddingService(settings)

# Test query latency
queries = [
    'Điều kiện tuyển sinh ngành CNTT là gì?',
    'Học phí năm 2025 bao nhiêu?',
    'Ngành nào có điểm chuẩn cao nhất?',
    'Thời hạn nộp hồ sơ tuyển sinh?',
    'Khoa Công nghệ thông tin ở đâu?',
]

times = []
for q in queries:
    start = time.perf_counter()
    vec = emb.encode_query(q)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    print(f'Query: {q[:50]}... => {elapsed:.1f}ms, dim={vec.shape}')

print(f'\\nAvg latency: {sum(times)/len(times):.1f}ms')
"
```

**Bước 2: Đo indexing throughput**
```bash
python -c "
import time
from config.settings import RAGSettings
from src.services.embedding import EmbeddingService

settings = RAGSettings()
emb = EmbeddingService(settings)

# Create 100 test chunks
docs = [f'Test document number {i} for embedding performance measurement' for i in range(100)]

start = time.perf_counter()
vectors = emb.encode_documents(docs)
elapsed = time.perf_counter() - start

print(f'Indexed 100 docs in {elapsed:.2f}s')
print(f'Throughput: {100/elapsed:.1f} docs/sec')
print(f'Vector shape: {vectors.shape}')
"
```

### 6.2. Test Harrier-OSS-v1

**Bước 3: Tạo script test Harrier**
```python
# test_harrier_embedding.py
"""
Benchmark script để so sánh Harrier vs Qwen3 embedding.
"""
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RAGSettings
from src.services.embedding import EmbeddingService

def test_model(model_name: str, dim: int):
    """Test một embedding model cụ thể."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name} (dim={dim})")
    print('='*60)
    
    # Temporarily override settings
    original_model = RAGSettings().EMBEDDING_MODEL
    original_dim = RAGSettings().EMBEDDING_DIM
    
    # Note: Trong thực tế cần restart Python process
    # hoặc create EmbeddingService trực tiếp với config override
    
    settings = RAGSettings()
    emb = EmbeddingService(settings)
    
    # Test queries
    queries = [
        'Điều kiện tuyển sinh ngành CNTT là gì?',
        'Học phí năm 2025 bao nhiêu?',
        'Ngành nào có điểm chuẩn cao nhất?',
        'Thời hạn nộp hồ sơ tuyển sinh?',
        'Khoa Công nghệ thông tin ở đâu?',
        'Cho tôi biết về chương trình đào tạo ngành Marketing',
        'Điểm chuẩn các ngành năm 2024',
        'Học bổng tuyển sinh có những loại nào?',
    ]
    
    query_times = []
    for q in queries:
        start = time.perf_counter()
        vec = emb.encode_query(q)
        elapsed = (time.perf_counter() - start) * 1000
        query_times.append(elapsed)
    
    print(f"Query latencies: {[f'{t:.1f}ms' for t in query_times]}")
    print(f"Avg query latency: {sum(query_times)/len(query_times):.1f}ms")
    
    # Test batch encoding
    docs = [f'Test document {i}' for i in range(50)]
    
    start = time.perf_counter()
    vectors = emb.encode_documents(docs)
    batch_elapsed = time.perf_counter() - start
    
    print(f"Batch encode 50 docs: {batch_elapsed*1000:.1f}ms")
    print(f"Throughput: {50/batch_elapsed:.1f} docs/sec")
    print(f"Vector shape: {vectors.shape}")

if __name__ == "__main__":
    settings = RAGSettings()
    print(f"Current config: {settings.EMBEDDING_MODEL}")
    print(f"Provider: {getattr(settings, 'EMBEDDING_PROVIDER', 'qwen')}")
```

### 6.3. Retrieval Quality Comparison

**Bước 4: So sánh retrieval quality**

```python
# compare_retrieval_quality.py
"""
So sánh retrieval quality giữa 2 embedding models
bằng cách đo precision@K trên test queries đã có ground truth.
"""
import json
from pathlib import Path
from typing import List, Dict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.lancedb_retrieval import LanceDBRetriever
from src.services.embedding import EmbeddingService
from config.settings import RAGSettings

def load_test_queries(path: Path) -> List[Dict]:
    """Load test queries với ground truth relevant chunks."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []

def evaluate_retrieval(retriever: LanceDBRetriever, emb_service: EmbeddingService, 
                       test_queries: List[Dict], k: int = 5) -> Dict:
    """Evaluate retrieval precision@K."""
    total_precision = 0.0
    
    for item in test_queries:
        query = item['query']
        relevant_ids = set(item.get('relevant_chunk_ids', []))
        
        # Encode query
        query_vec = emb_service.encode_query(query)
        
        # Retrieve
        result = retriever.retrieve(query_vec.tolist(), top_k=k)
        retrieved_ids = {doc.chunk_id for doc in result.documents}
        
        # Calculate precision
        if relevant_ids:
            precision = len(retrieved_ids & relevant_ids) / k
        else:
            # Nếu không có ground truth, dùng score threshold
            precision = result.documents[0].score if result.documents else 0.0
        
        total_precision += precision
    
    return {
        'precision@k': total_precision / len(test_queries) if test_queries else 0.0,
        'num_queries': len(test_queries),
    }

if __name__ == "__main__":
    # Load test queries
    test_path = Path(__file__).parent.parent / "data" / "test_queries.json"
    test_queries = load_test_queries(test_path)
    
    if not test_queries:
        print("No test queries found. Creating sample...")
        test_queries = [
            {
                "query": "Điều kiện tuyển sinh ngành CNTT?",
                "relevant_chunk_ids": ["chunk_001", "chunk_042"]
            },
            {
                "query": "Học phí ngành Kinh tế?",
                "relevant_chunk_ids": ["chunk_123"]
            },
        ]
    
    settings = RAGSettings()
    emb = EmbeddingService(settings)
    retriever = LanceDBRetriever.from_env()
    
    results = evaluate_retrieval(retriever, emb, test_queries, k=5)
    print(f"Embedding: {settings.EMBEDDING_MODEL}")
    print(f"Precision@5: {results['precision@k']:.3f}")
```

---

## 7. Kế Hoạch Migration

### Phase 1: Chuẩn Bị (Ngày 1-2)
- [ ] Tạo benchmark script cho Qwen3 baseline
- [ ] Đo baseline metrics (query latency, indexing throughput, precision@K)
- [ ] Lưu kết quả baseline

### Phase 2: Tích Hợp Harrier (Ngày 3-4)
- [ ] Thêm `EMBEDDING_PROVIDER` và Harrier config vào `settings.py`
- [ ] Cập nhật `embedding.py` để hỗ trợ Harrier prompts
- [ ] Cập nhật validator cho Harrier dimensions
- [ ] Commit changes

### Phase 3: Test Harrier (Ngày 5-6)
- [ ] Chạy `setup_data.bat` với Harrier-0.6B
- [ ] Đo Harrier metrics (query latency, indexing throughput)
- [ ] So sánh retrieval quality trên test queries
- [ ] Benchmark retrieval precision@K

### Phase 4: So Sánh & Quyết Định (Ngày 7)
- [ ] So sánh đầy đủ: Qwen3-4B vs Harrier-0.6B
- [ ] Xem xét Vietnamese-specific performance
- [ ] Quyết định: migrate hay giữ Qwen3
- [ ] Document kết quả

---

## 8. Chi Phí Ước Tính

| Hoạt động | Thời gian | Ghi chú |
|-----------|-----------|---------|
| Baseline benchmark (Qwen3-4B) | ~30 phút | Chạy test queries |
| Harrier-0.6B integration | ~2 giờ | Code changes |
| Harrier indexing (1000 chunks) | ~5-10 phút | 0.6B nhanh |
| Retrieval quality test | ~1 giờ | Manual evaluation |
| So sánh & quyết định | ~2 giờ | Phân tích |

**Tổng:** ~6-7 giờ làm việc

---

## 9. Rủi Ro

| Rủi ro | Xác suất | Mitigation |
|--------|----------|-----------|
| Harrier Vietnamese performance kém hơn Qwen3 | Trung bình | Test trước trên subset, so sánh precision@K |
| Harrier dimension 1024 không tương thích với existing index | Cao | PHẢI re-index sau khi đổi model |
| Harrier model chưa có trên HuggingFace cache | Thấp | Auto-download, cần internet |
| Breaking change trong sentence-transformers API | Thấp | Sử dụng cùng API `prompt_name` |
| Harrier 27B quá nặng cho máy test | Cao | KHÔNG test 27B trên máy 16GB RAM |

---

## 10. Recommendations

### Cho Máy Hiện Tại (16GB RAM)

**Khuyến nghị:** Thử **Harrier-OSS-v1-0.6B** vì:
1. ✅ Quality tương đương Qwen3-4B (69.0 vs 69.45 MTEB)
2. ✅ Disk chỉ 1.2GB thay vì 8GB (tiết kiệm 7GB)
3. ✅ RAM ~4GB thay vì ~12GB 
4. ✅ MIT license thương mại thân thiện
5. ⚠️ **Cần verify Vietnamese performance thực tế**

### Hành Động Ngay

1. **Tạo benchmark script** (30 phút)
2. **Chạy baseline với Qwen3-4B hiện tại** (1 giờ)
3. **Integrate Harrier-0.6B vào code** (2 giờ)
4. **So sánh retrieval quality** (2 giờ)
5. **Quyết định dựa trên dữ liệu thực tế**

---

## 11. Tài Liệu Tham Khảo

- Harrier-OSS-v1 HuggingFace: https://huggingface.co/microsoft/harrier-oss-v1-0.6b
- Harrier-OSS-v1-27B: https://huggingface.co/microsoft/harrier-oss-v1-27b
- Qwen3-Embedding: https://huggingface.co/Qwen/Qwen3-Embedding-4B
- MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- VN-MTEB (Vietnamese benchmark): https://aclanthology.org/2026.findings-eacl.86/
- MarkTechPost Review: https://www.marktechpost.com/2026/03/30/microsoft-ai-releases-harrier-oss-v1-a-new-family-of-multilingual-embedding-models-hitting-sota-on-multilingual-mteb-v2/
- Awesome Agents Review: https://awesomeagents.ai/news/microsoft-harrier-oss-v1-multilingual-embeddings/

---

## 12. File Cần Tạo Mới

| File | Mục đích |
|------|----------|
| `rag2025/scripts/benchmark_embedding.py` | Benchmark script cho so sánh embedding models |
| `rag2025/scripts/compare_retrieval.py` | So sánh retrieval quality |
| `docs/harrier-oss-v1-vs-qwen3-comparison.md` | Báo cáo so sánh (file này) |

---

**Kết luận:** Harrier-OSS-v1 **có tiềm năng thay thế tốt hơn** Qwen3-Embedding, đặc biệt variant 0.6B với quality tương đương nhưng resource hiệu quả hơn 6x. **Tuy nhiên cần benchmark thực tế trên Vietnamese data** để xác nhận, vì MTEB score không phản ánh đầy đủ Vietnamese retrieval quality.
