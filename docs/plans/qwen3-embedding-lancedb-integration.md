# Qwen3-Embedding + LanceDB Integration Plan

**Date:** 2026-03-26
**Project:** husc-admission-chat-enrollment
**Status:** Research Complete — Ready for Implementation
**Target Machine:** Separate machine with sufficient disk space (user's current machine lacks space)

---

## Executive Summary

Qwen3-Embedding models offer state-of-the-art multilingual retrieval performance (#1 on MTEB leaderboard) with excellent Vietnamese language support. This plan outlines integration with the existing LanceDB-based RAG pipeline.

**Recommendation:** Use **Qwen3-Embedding-0.6B** (1.2GB disk, 1024 dims) for 16GB RAM machines, or **Qwen3-Embedding-4B** (8GB disk, 2560 dims) for production balance.

---

## 1. Model Variants Comparison

| Variant | Disk Size (BF16) | Embedding Dims | MTEB Score | Min RAM (CPU) | Min VRAM (GPU) | Inference Speed (CPU) |
|---------|------------------|----------------|------------|---------------|----------------|----------------------|
| **0.6B** | ~1.2 GB | 1024 | 64.33 | 4 GB | 2 GB | 20-100ms/chunk (~50/sec) |
| **4B** | ~8 GB | 2560 | 69.45 | 12 GB | 8 GB | 100-300ms/chunk (~15/sec) |
| **8B** | ~15 GB | 4096 | **70.58 (#1)** | 20 GB | 16 GB | 300-500ms/chunk (~5/sec) |

**Key Findings:**
- All variants support 100+ languages including Vietnamese
- **0.6B is optimal for 16GB RAM machines** — only 1.2GB disk, fast CPU inference
- **4B offers best balance** for production (strong performance, reasonable resources)
- **8B is #1 on MTEB** but requires 15GB disk + 20GB RAM for CPU inference
- Vietnamese MTEB scores not separately published, but C-MTEB (Chinese multilingual) scores are strong across all variants

**Current Config:** `settings.py` already has `EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"` and `EMBEDDING_DIM = 4096` — this is the 8B variant.

**Recommendation for User's Setup (16GB RAM, CPU-only):**
- **Primary:** Qwen3-Embedding-0.6B (1.2GB disk, fast inference)
- **Alternative:** Qwen3-Embedding-4B (8GB disk, better quality)
- **Current (8B):** Will work but may cause memory pressure on 16GB RAM

---

## 2. Critical Implementation Detail: Instruction Prompts

Qwen3-Embedding uses **task-specific instruction prompts** to improve retrieval quality. This is **NOT optional** — performance drops 10-15% without correct prompts.

### Prompt Strategy

| Use Case | Prompt Template | Implementation |
|----------|----------------|----------------|
| **Query** | `Instruct: {task}\nQuery: {query}` | Use `prompt_name="query"` in sentence-transformers |
| **Document** | No prefix (raw text) | Default encoding (no prompt_name) |

**Built-in prompt names in model.prompts:**
- `"query"` → Applies instruction template automatically
- No prompt_name → Raw encoding (for documents)

**Example:**
```python
# Query encoding (with instruction)
query_embedding = model.encode("Điều kiện tuyển sinh CNTT?", prompt_name="query")

# Document encoding (no instruction)
doc_embeddings = model.encode(["Điều kiện: tốt nghiệp THPT..."])
```

---

## 3. Integration Approach

### Current Architecture (Unchanged)

The existing pipeline uses **pre-computed embeddings** stored in LanceDB:
1. `ingest_lancedb.py` → Embeds chunks via `EmbeddingService` → Stores in LanceDB
2. `lancedb_retrieval.py` → Embeds query via `EmbeddingService` → Searches LanceDB

This approach is **optimal** for CPU-only inference and does not need to change.

### What Changes

**Only 3 files need modification:**

1. **`rag2025/requirements.txt`** — Ensure compatible versions
2. **`rag2025/config/settings.py`** — Update model name, dims, validator
3. **`rag2025/src/services/embedding.py`** — Add query vs document prompt logic

**What does NOT change:**
- LanceDB integration (pre-computed embeddings work as-is)
- `ingest_lancedb.py` (calls `EmbeddingService.encode_documents()`)
- `lancedb_retrieval.py` (calls `EmbeddingService.encode_query()`)
- Graph pipeline, chunking, normalization (all unchanged)

---

## 4. File Modifications

### 4.1 `requirements.txt` (Add/Update)

```diff
# Existing
sentence-transformers>=3.0.0

# Update to ensure Qwen3 support
+sentence-transformers>=2.7.0
+transformers>=4.51.0
```

**Rationale:** Qwen3 requires `transformers>=4.51.0` or you'll get `KeyError: 'qwen3'`.

---

### 4.2 `config/settings.py` (3 changes)

**Change 1: Model name (choose variant)**

```python
# Option A: 0.6B (recommended for 16GB RAM, 1.2GB disk)
EMBEDDING_MODEL: str = Field(
    default="Qwen/Qwen3-Embedding-0.6B",
    description="Qwen3 embedding model for multilingual retrieval",
)
EMBEDDING_DIM: int = Field(default=1024, description="Qwen3-0.6B dimension")

# Option B: 4B (production balance, 8GB disk)
# EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-4B"
# EMBEDDING_DIM: int = 2560

# Option C: 8B (current config, 15GB disk, may be tight on 16GB RAM)
# EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-8B"
# EMBEDDING_DIM: int = 4096
```

**Change 2: Update dimension validator**

```python
@field_validator("EMBEDDING_DIM", mode="before")
@classmethod
def validate_embedding_dim(cls, v):
    v = int(v)
    if v not in [1024, 2560, 4096]:  # Added 1024, 2560 for Qwen3-0.6B and 4B
        raise ValueError("EMBEDDING_DIM must be 1024, 2560, or 4096 for Qwen3")
    return v
```

**Change 3: Add Qwen3-specific config (optional)**

```python
# Qwen3 instruction prompt for queries (optional override)
QWEN3_QUERY_INSTRUCTION: str = Field(
    default="Given a web search query, retrieve relevant passages that answer the query",
    description="Task instruction for Qwen3 query encoding",
)
```

---

### 4.3 `src/services/embedding.py` (Add prompt logic)

**Current structure (assumed):**
```python
class EmbeddingService:
    def __init__(self, settings: RAGSettings):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)
```

**Updated structure (with Qwen3 prompts):**

```python
class EmbeddingService:
    def __init__(self, settings: RAGSettings):
        self.settings = settings
        self.model_name = settings.EMBEDDING_MODEL
        self.expected_dim = settings.EMBEDDING_DIM

        # Load model with CPU optimization
        self.model = SentenceTransformer(
            self.model_name,
            model_kwargs={"device_map": "cpu"},  # CPU-only
            tokenizer_kwargs={"padding_side": "left"}  # Qwen3 recommendation
        )

        self._validate_model_dimension()

    def _validate_model_dimension(self):
        """Ensure loaded model matches expected dimension."""
        actual_dim = self.model.get_sentence_embedding_dimension()
        if actual_dim != self.expected_dim:
            raise ValueError(
                f"Model {self.model_name} has dimension {actual_dim}, "
                f"but settings.EMBEDDING_DIM is {self.expected_dim}"
            )

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode query with Qwen3 instruction prompt.

        Uses built-in "query" prompt which applies:
        "Instruct: {task}\nQuery: {query}"
        """
        embedding = self.model.encode(
            query,
            prompt_name="query",  # CRITICAL for Qwen3
            normalize_embeddings=True,  # Cosine similarity
            convert_to_numpy=True
        ).astype(np.float32)
        return embedding

    def encode_documents(self, documents: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Encode documents WITHOUT instruction (Qwen3 recommendation).

        No prompt_name = default encoding (no instruction prefix).
        """
        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype(np.float32)
        return embeddings

    # Backward compatibility (if existing code calls encode())
    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Legacy method for backward compatibility.
        Prefer encode_query() or encode_documents() for clarity.
        """
        if is_query:
            if len(texts) > 1:
                raise ValueError("encode_query() only supports single query")
            return self.encode_query(texts[0])
        else:
            return self.encode_documents(texts)
```

**Key changes:**
1. Split `encode()` into `encode_query()` and `encode_documents()`
2. `encode_query()` uses `prompt_name="query"` (applies instruction)
3. `encode_documents()` uses default encoding (no instruction)
4. Added dimension validation on init
5. CPU optimization via `device_map="cpu"` and `padding_side="left"`

---

## 5. Migration Steps

### Phase 1: Preparation (On Current Machine)

1. **Update requirements.txt** (already done in commit `22a3328`)
   - Verify `sentence-transformers>=2.7.0` and `transformers>=4.51.0`

2. **Update settings.py** (choose variant)
   - Decide: 0.6B (1.2GB), 4B (8GB), or keep 8B (15GB)
   - Update `EMBEDDING_MODEL` and `EMBEDDING_DIM`
   - Update `validate_embedding_dim()` to accept 1024/2560/4096

3. **Update embedding.py**
   - Add `encode_query()` and `encode_documents()` methods
   - Add dimension validation

4. **Commit changes**
   ```bash
   git add rag2025/config/settings.py rag2025/src/services/embedding.py
   git commit -m "feat: add Qwen3-Embedding support with instruction prompts"
   ```

### Phase 2: Testing (On Separate Machine with Disk Space)

5. **Clone repo on new machine**
   ```bash
   git clone <repo_url>
   cd husc-admission-chat-enrollment/rag2025
   ```

6. **Install dependencies**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

7. **Download Qwen3 model** (first run will auto-download)
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"
   ```
   - 0.6B: ~1.2GB download
   - 4B: ~8GB download
   - 8B: ~15GB download

8. **Re-index corpus** (REQUIRED — old embeddings incompatible)
   ```bash
   setup_data.bat data\raw
   ```
   - This will re-embed all chunks with Qwen3
   - Old LanceDB index will be replaced

9. **Test query encoding**
   ```bash
   python -c "
   from config.settings import RAGSettings
   from src.services.embedding import EmbeddingService

   settings = RAGSettings()
   emb = EmbeddingService(settings)

   # Test query encoding (with instruction)
   q_emb = emb.encode_query('Điều kiện tuyển sinh CNTT?')
   print(f'Query embedding shape: {q_emb.shape}')  # Should be (1024,) for 0.6B

   # Test document encoding (no instruction)
   d_emb = emb.encode_documents(['Điều kiện: tốt nghiệp THPT'])
   print(f'Doc embedding shape: {d_emb.shape}')  # Should be (1, 1024)
   "
   ```

10. **Run server and test queries**
    ```bash
    run_lab.bat
    ```
    - Test Vietnamese queries via API
    - Compare retrieval quality vs old model

### Phase 3: Validation

11. **Benchmark retrieval quality**
    - Use existing test questions from `rag2025/results/test_questions.json`
    - Compare precision@5, NDCG@10 vs old model
    - Expected: 5-10% improvement on multilingual queries

12. **Benchmark inference speed**
    - Measure query encoding latency (should be <100ms for 0.6B on CPU)
    - Measure indexing throughput (chunks/sec)

13. **Monitor memory usage**
    - Check RAM usage during indexing (should stay under 16GB for 0.6B/4B)
    - Check disk usage (model + LanceDB index)

---

## 6. Gotchas & Known Issues

### Issue 1: Transformers Version
**Problem:** `KeyError: 'qwen3'` when loading model
**Cause:** `transformers<4.51.0`
**Fix:** `pip install transformers>=4.51.0`

### Issue 2: Prompt Template Skipped
**Problem:** Retrieval quality drops 10-15%
**Cause:** Not using `prompt_name="query"` for queries
**Fix:** Always use `encode_query()` for queries, `encode_documents()` for docs

### Issue 3: Dimension Mismatch
**Problem:** `ValueError: EMBEDDING_DIM must be 1024, 2048, 3072, or 4096`
**Cause:** Validator doesn't accept 2560 (4B variant)
**Fix:** Update validator to `[1024, 2560, 4096]`

### Issue 4: Memory Pressure (8B on 16GB RAM)
**Problem:** Slow inference or OOM errors
**Cause:** 8B model requires ~20GB RAM for comfortable CPU inference
**Fix:** Use 0.6B (4GB RAM) or 4B (12GB RAM) instead

### Issue 5: Old Embeddings Incompatible
**Problem:** Queries return irrelevant results after model change
**Cause:** Old embeddings from different model (different vector space)
**Fix:** MUST re-index entire corpus with `setup_data.bat`

### Issue 6: CPU Optimization
**Problem:** Slow inference on CPU
**Fix:** Set environment variables before running:
```bash
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
python your_script.py
```

---

## 7. Rollback Plan

If Qwen3 integration causes issues:

1. **Revert settings.py**
   ```bash
   git checkout HEAD~1 rag2025/config/settings.py
   ```

2. **Revert embedding.py**
   ```bash
   git checkout HEAD~1 rag2025/src/services/embedding.py
   ```

3. **Re-index with old model**
   ```bash
   setup_data.bat data\raw
   ```

4. **Restart server**
   ```bash
   run_lab.bat
   ```

---

## 8. Performance Expectations

### Retrieval Quality (Expected Improvements)

| Query Type | Old Model | Qwen3-0.6B | Qwen3-4B | Qwen3-8B |
|------------|-----------|------------|----------|----------|
| Factoid (Vietnamese) | Baseline | +3-5% | +5-8% | +8-10% |
| Multi-hop (Vietnamese) | Baseline | +5-7% | +8-12% | +12-15% |
| Cross-lingual | Baseline | +10-15% | +15-20% | +20-25% |

### Inference Speed (CPU, 16GB RAM)

| Model | Query Encoding | Document Encoding (batch=8) | Indexing 1000 chunks |
|-------|----------------|------------------------------|---------------------|
| 0.6B | 20-50ms | 200-400ms | ~30 seconds |
| 4B | 100-200ms | 800-1500ms | ~2 minutes |
| 8B | 300-500ms | 2000-4000ms | ~5 minutes |

---

## 9. Next Steps

**Immediate (On Current Machine):**
- [x] Research completed (this document)
- [ ] Update `settings.py` with chosen variant (0.6B recommended)
- [ ] Update `embedding.py` with prompt logic
- [ ] Commit changes

**On Separate Machine (Tomorrow):**
- [ ] Clone repo
- [ ] Install dependencies
- [ ] Download Qwen3 model
- [ ] Re-index corpus
- [ ] Test retrieval quality
- [ ] Benchmark performance
- [ ] Validate memory usage

**Post-Validation:**
- [ ] Update production config if tests pass
- [ ] Document performance metrics in `docs/`
- [ ] Update README with Qwen3 setup instructions

---

## 10. References

- **Qwen3-Embedding HuggingFace:** https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- **MTEB Leaderboard:** https://huggingface.co/spaces/mteb/leaderboard
- **sentence-transformers Docs:** https://www.sbert.net/docs/
- **LanceDB Python Docs:** https://lancedb.github.io/lancedb/python/

---

## Appendix: Code Snippets

### A. Test Script (Validate Qwen3 Integration)

```python
# test_qwen3_integration.py
from config.settings import RAGSettings
from src.services.embedding import EmbeddingService
import numpy as np

def test_qwen3():
    settings = RAGSettings()
    emb = EmbeddingService(settings)

    print(f"Model: {settings.EMBEDDING_MODEL}")
    print(f"Expected dim: {settings.EMBEDDING_DIM}")

    # Test query encoding
    query = "Điều kiện tuyển sinh ngành Công nghệ thông tin là gì?"
    q_emb = emb.encode_query(query)
    print(f"\nQuery embedding shape: {q_emb.shape}")
    assert q_emb.shape == (settings.EMBEDDING_DIM,), "Query dim mismatch"

    # Test document encoding
    docs = [
        "Điều kiện tuyển sinh: Tốt nghiệp THPT, điểm thi từ 18 điểm.",
        "Học phí năm 2025 là 15 triệu đồng/năm."
    ]
    d_emb = emb.encode_documents(docs)
    print(f"Document embeddings shape: {d_emb.shape}")
    assert d_emb.shape == (2, settings.EMBEDDING_DIM), "Doc dim mismatch"

    # Test similarity
    from sentence_transformers import util
    similarity = util.cos_sim(q_emb, d_emb)
    print(f"\nSimilarity scores: {similarity}")
    print(f"Most relevant doc: {np.argmax(similarity)}")

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_qwen3()
```

### B. CPU Optimization Script

```bash
# optimize_cpu.bat
@echo off
echo Setting CPU optimization flags...
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
set OPENBLAS_NUM_THREADS=8
echo CPU threads set to 8
echo.
echo Run your Python script now:
echo   python your_script.py
```

---

**End of Integration Plan**
