# 🚀 PROMPT PHÁT TRIỂN RAG CHATBOT - FAST TRACK

## 🎯 MỤC TIÊU
Phát triển RAG backend với HYDE, BGE multi-vector, Qdrant, và kết nối với UI có sẵn tại `D:\chunking\rag2025_2\uni-guide-ai`

---

## 📋 YÊU CẦU CHÍNH

### 0️⃣ CẬP NHẬT REQUIREMENTS.TXT

**Thêm vào `requirements.txt`**:
```txt
# Vector Database
qdrant-client>=1.7.0

# LLM Providers
google-generativeai>=0.3.0
groq>=0.4.0
openai>=1.0.0  # For GLM-4 via Z.AI endpoint

# BGE Multi-Vector (AITeamVN)
transformers>=4.35.0
torch>=2.0.0

# Additional utilities
python-dotenv>=1.0.0
aiohttp>=3.9.0
tenacity>=8.2.0  # For retry logic
```

**Cài đặt**:
```bash
pip install -r requirements.txt
```

---

### 1️⃣ HYDE Query Enhancement (INPUT LAYER)

**Mục tiêu**: Chuyển đổi simple user query → QueryRequest object với enhanced query

**Flow**:
```
User Input: "điều kiện xét tuyển"
    ↓
HYDE Enhancement
    ↓
QueryRequest {
    query: "Trong năm 2025, điều kiện xét tuyển đại học bao gồm...",
    original_query: "điều kiện xét tuyển",
    top_k: 5 (auto-detect),
    force_rag_only: false,
    query_type: "admission_criteria" (auto-classify)
}
```

**Implementation**:

**File: `src/services/query_enhancer.py`**
```python
import os
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
import google.generativeai as genai
from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

class HYDEQueryEnhancer:
    """
    HYDE: Hypothetical Document Embeddings
    Converts user query → hypothetical answer → better retrieval
    """
    
    def __init__(self):
        # Priority: Gemini → GLM-4 → Groq
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.glm4_key = os.getenv("OPENAI_API_KEY")  # Z.AI endpoint
        self.groq_key = os.getenv("GROQ_API_KEY")
        
        # Configure clients
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite-preview-02-05')
        
        if self.glm4_key:
            self.glm4_client = AsyncOpenAI(
                api_key=self.glm4_key,
                base_url="https://open.bigmodel.cn/api/paas/v4"  # Z.AI/GLM-4 endpoint
            )
        
        if self.groq_key:
            self.groq_client = AsyncGroq(api_key=self.groq_key)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate hypothetical answer using LLM
        """
        prompt = f"""Bạn là chuyên gia tuyển sinh đại học Việt Nam.

Câu hỏi: {query}

Hãy viết một đoạn văn giả định (150-200 từ) trả lời câu hỏi này như thể bạn đang trích dẫn từ văn bản chính thức về quy chế tuyển sinh 2025.

Yêu cầu:
- Viết ngắn gọn, chính xác
- Dùng ngôn ngữ học thuật
- Đề cập các khái niệm chính: điều kiện, hồ sơ, điểm, ngành, trường...
- KHÔNG cần chính xác 100%, chỉ cần giả định hợp lý

Đoạn văn giả định:"""

        # Try Gemini first
        if self.gemini_key:
            try:
                response = await self.gemini_model.generate_content_async(prompt)
                return response.text.strip()
            except Exception as e:
                logger.warning(f"Gemini failed: {e}, trying GLM-4...")
        
        # Fallback to GLM-4
        if self.glm4_key:
            try:
                response = await self.glm4_client.chat.completions.create(
                    model="glm-4-plus",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"GLM-4 failed: {e}, trying Groq...")
        
        # Fallback to Groq
        if self.groq_key:
            response = await self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        
        raise Exception("All LLM providers failed")
    
    def classify_query_type(self, query: str) -> str:
        """
        Auto-classify query type for better routing
        """
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["điều kiện", "yêu cầu", "tuyển sinh"]):
            return "admission_criteria"
        elif any(kw in query_lower for kw in ["hồ sơ", "giấy tờ", "nộp"]):
            return "documents"
        elif any(kw in query_lower for kw in ["điểm", "thang điểm", "tính điểm"]):
            return "scoring"
        elif any(kw in query_lower for kw in ["ngành", "chuyên ngành", "đào tạo"]):
            return "major_info"
        elif any(kw in query_lower for kw in ["thời gian", "lịch", "deadline"]):
            return "timeline"
        else:
            return "general"
    
    def estimate_top_k(self, query: str, query_type: str) -> int:
        """
        Auto-adjust top_k based on query complexity
        """
        if query_type == "general" or len(query.split()) > 15:
            return 7  # Complex query needs more context
        elif query_type in ["admission_criteria", "documents"]:
            return 5  # Standard
        else:
            return 3  # Simple factual query
    
    async def enhance_query(
        self, 
        user_query: str,
        force_rag_only: bool = False
    ) -> Dict[str, Any]:
        """
        Main method: Convert user string → QueryRequest dict
        
        Returns:
            {
                "query": "enhanced query with hypothetical answer",
                "original_query": "user's original query",
                "top_k": 5,
                "force_rag_only": false,
                "query_type": "admission_criteria",
                "hypothetical_answer": "generated hypothetical answer"
            }
        """
        logger.info(f"Enhancing query: {user_query}")
        
        # 1. Classify query type
        query_type = self.classify_query_type(user_query)
        
        # 2. Estimate top_k
        top_k = self.estimate_top_k(user_query, query_type)
        
        # 3. Generate hypothetical answer
        try:
            hypothetical_answer = await self.generate_hypothetical_answer(user_query)
            
            # 4. Combine: original query + hypothetical answer
            enhanced_query = f"{user_query}\n\nThông tin liên quan: {hypothetical_answer}"
            
        except Exception as e:
            logger.error(f"HYDE failed: {e}, using original query")
            hypothetical_answer = ""
            enhanced_query = user_query
        
        return {
            "query": enhanced_query,
            "original_query": user_query,
            "top_k": top_k,
            "force_rag_only": force_rag_only,
            "query_type": query_type,
            "hypothetical_answer": hypothetical_answer
        }

# Global instance
query_enhancer = HYDEQueryEnhancer()
```

**✅ Checkpoint 1**: Test HYDE
```bash
# Test script: scripts/test_hyde.py
import asyncio
from src.services.query_enhancer import query_enhancer

async def test():
    result = await query_enhancer.enhance_query("điều kiện xét tuyển")
    print(result)

asyncio.run(test())
```

**Expected Output**:
```json
{
  "query": "điều kiện xét tuyển\n\nThông tin liên quan: Trong năm 2025, điều kiện xét tuyển...",
  "original_query": "điều kiện xét tuyển",
  "top_k": 5,
  "force_rag_only": false,
  "query_type": "admission_criteria",
  "hypothetical_answer": "Trong năm 2025, điều kiện xét tuyển..."
}
```

**Agent gợi ý**: `prompt-engineer`, `ai-engineer`

---

### 2️⃣ BGE Multi-Vector Retrieval (RETRIEVAL LAYER)

**Mục tiêu**: Thay thế hybrid (dense+BM25) bằng BGE multi-vector approach

**Model**: `AITeamVN/vietnamese-embedding` 

**File: `src/services/bge_retriever.py`**
```python
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from loguru import logger
import os

class BGEMultiVectorRetriever:
    """
    BGE Multi-Vector Retrieval with score boosting
    """
    
    def __init__(self):
        # Load BGE model (AITeamVN Vietnamese embedding)
        self.model = SentenceTransformer('BAAI/bge-m3')  # Fallback to bge-m3
        # TODO: Switch to AITeamVN/vietnamese-embedding when available
        
        # Qdrant client
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = "hue_admissions_2025_v2"
        
        # Weights for multi-vector fusion
        self.dense_weight = 0.7
        self.sparse_weight = 0.3
        
        # Reranker
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder('AITeamVN/Vietnamese_Reranker')
    
    async def retrieve(
        self, 
        query_enhanced: str,
        original_query: str,
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Main retrieval with score boosting
        
        Returns:
            (chunks, confidence_score)
        """
        logger.info(f"Retrieving for: {original_query}")
        
        # 1. Encode query
        query_vector = self.model.encode(query_enhanced, normalize_embeddings=True)
        
        # 2. Search Qdrant (get top_k * 2 for reranking)
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k * 2
        )
        
        # 3. Apply score boosting (CRITICAL!)
        boosted_results = self._apply_score_boosting(
            search_results, 
            original_query,
            query_vector
        )
        
        # 4. Rerank top results
        reranked_results = self._rerank(
            boosted_results[:top_k * 2],
            original_query,
            top_k
        )
        
        # 5. Calculate confidence
        confidence = self._calculate_confidence(reranked_results)
        
        # 6. Format chunks
        chunks = [
            {
                "id": r.id,
                "text": r.payload.get("text", ""),
                "metadata": r.payload.get("metadata", {}),
                "score": r.score,
                "boosted": r.payload.get("boosted", False)
            }
            for r in reranked_results
        ]
        
        return chunks, confidence
    
    def _apply_score_boosting(
        self, 
        results: List[Any],
        query: str,
        query_vector: np.ndarray
    ) -> List[Any]:
        """
        CRITICAL: Boost scores to avoid rejecting near-correct answers
        """
        query_keywords = set(query.lower().split())
        
        for result in results:
            original_score = result.score
            boost = 0.0
            
            # Boost 1: High semantic similarity but low score
            # Recalculate cosine similarity
            doc_vector = np.array(result.vector) if hasattr(result, 'vector') else None
            if doc_vector is not None:
                cosine_sim = np.dot(query_vector, doc_vector)
                if original_score < 0.6 and cosine_sim > 0.75:
                    boost += 0.15
                    logger.debug(f"Applied semantic boost +0.15 (cosine={cosine_sim:.3f})")
            
            # Boost 2: Exact keyword match
            text = result.payload.get("text", "").lower()
            keyword_matches = sum(1 for kw in query_keywords if kw in text)
            if keyword_matches >= len(query_keywords) * 0.7:  # 70% keywords match
                boost += 0.1
                logger.debug(f"Applied keyword boost +0.1 ({keyword_matches} matches)")
            
            # Boost 3: Source credibility (official documents)
            info_type = result.payload.get("metadata", {}).get("info_type", "")
            if info_type == "van_ban_phap_ly":
                boost += 0.05
                logger.debug(f"Applied source boost +0.05 (official doc)")
            
            # Apply boost
            if boost > 0:
                result.score = min(original_score + boost, 1.0)
                result.payload["boosted"] = True
                result.payload["original_score"] = original_score
                logger.info(f"Score boosted: {original_score:.3f} → {result.score:.3f}")
        
        # Re-sort by boosted scores
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def _rerank(
        self, 
        results: List[Any],
        query: str,
        top_k: int
    ) -> List[Any]:
        """
        Rerank using cross-encoder
        """
        if not results:
            return []
        
        # Prepare pairs for reranking
        pairs = [(query, r.payload.get("text", "")) for r in results]
        
        # Rerank scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine with original scores (weighted)
        for i, result in enumerate(results):
            original = result.score
            rerank = float(rerank_scores[i])
            
            # Weighted combination
            result.score = 0.6 * original + 0.4 * rerank
        
        # Sort and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _calculate_confidence(self, results: List[Any]) -> float:
        """
        Calculate ensemble confidence score
        """
        if not results:
            return 0.0
        
        # Weighted average of top 3 scores
        top_scores = [r.score for r in results[:3]]
        
        if len(top_scores) == 1:
            return top_scores[0]
        elif len(top_scores) == 2:
            return 0.7 * top_scores[0] + 0.3 * top_scores[1]
        else:
            return 0.5 * top_scores[0] + 0.3 * top_scores[1] + 0.2 * top_scores[2]

# Global instance
bge_retriever = BGEMultiVectorRetriever()
```

**✅ Checkpoint 2**: Test BGE Retrieval (sau khi migrate Qdrant)
```bash
# Start FastAPI
uvicorn src.main:app --reload --port 8000

# Test endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "điều kiện xét tuyển", "top_k": 5}'
```

**Agent gợi ý**: `search-specialist`, `data-scientist`

---

### 3️⃣ Qdrant Migration (VECTOR STORE)

**File: `scripts/migrate_to_qdrant.py`**
```python
import asyncio
import os
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from loguru import logger

async def migrate_to_qdrant():
    """
    Migrate from NumPy to Qdrant
    1. Delete old collection
    2. Create new collection
    3. Load all chunks from data/chunked/*.jsonl
    4. Embed and upload
    """
    
    # Connect to Qdrant
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    collection_name = "hue_admissions_2025_v2"
    
    # 1. Delete old collection if exists
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted old collection: {collection_name}")
    except Exception as e:
        logger.info(f"No old collection to delete: {e}")
    
    # 2. Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024,  # bge-m3 dimension
            distance=Distance.COSINE
        )
    )
    logger.info(f"Created new collection: {collection_name}")
    
    # 3. Load all chunks
    chunks_dir = Path("data/chunked")
    all_chunks = []
    
    for jsonl_file in sorted(chunks_dir.glob("chunked_*.jsonl")):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    all_chunks.append(chunk)
    
    logger.info(f"Loaded {len(all_chunks)} chunks")
    
    # 4. Embed and upload
    model = SentenceTransformer('BAAI/bge-m3')
    batch_size = 32
    
    points = []
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding & Uploading"):
        batch = all_chunks[i:i + batch_size]
        
        # Extract texts for embedding
        texts = [chunk.get("text", "") for chunk in batch]
        
        # Embed batch
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        
        # Create points
        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            point = PointStruct(
                id=i + j,
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk.get("id"),
                    "text": chunk.get("text", ""),
                    "summary": chunk.get("summary", ""),
                    "metadata": chunk.get("metadata", {}),
                    "faq_type": chunk.get("faq_type", ""),
                    "text_raw": chunk.get("text_raw", "")
                }
            )
            points.append(point)
        
        # Upload batch to Qdrant
        if len(points) >= batch_size:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            points = []
    
    # Upload remaining points
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
    
    # 5. Verify
    collection_info = client.get_collection(collection_name)
    logger.info(f"Migration complete! Total vectors: {collection_info.points_count}")

if __name__ == "__main__":
    asyncio.run(migrate_to_qdrant())
```

**✅ Checkpoint 3**: Run Migration
```bash
python scripts/migrate_to_qdrant.py

# Expected output:
# INFO: Deleted old collection: hue_admissions_2025_v2
# INFO: Created new collection: hue_admissions_2025_v2
# INFO: Loaded 110 chunks
# Embedding & Uploading: 100%|████████| 4/4 [00:15<00:00,  3.8s/it]
# INFO: Migration complete! Total vectors: 110
```

**Agent gợi ý**: `data-scientist`, `ai-engineer`

---

### 4️⃣ LLM Answer Generation (GENERATION LAYER)

**File: `src/services/llm_generator.py`**
```python
import os
from typing import List, Dict, Any
from openai import AsyncOpenAI
import google.generativeai as genai
from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

class LLMGenerator:
    """
    Generate final answer from retrieved chunks
    """
    
    def __init__(self):
        # Configure LLM clients
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.glm4_key = os.getenv("OPENAI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite-preview-02-05')
        
        if self.glm4_key:
            self.glm4_client = AsyncOpenAI(
                api_key=self.glm4_key,
                base_url="https://open.bigmodel.cn/api/paas/v4"
            )
        
        if self.groq_key:
            self.groq_client = AsyncGroq(api_key=self.groq_key)
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context from retrieved chunks
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            source = chunk.get("metadata", {}).get("source", "Không rõ nguồn")
            
            context_parts.append(f"[Đoạn {i}] (Nguồn: {source})\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        confidence: float
    ) -> Dict[str, Any]:
        """
        Generate answer with context from chunks
        """
        
        # Build context
        context = self._build_context(chunks)
        
        # Build prompt
        prompt = f"""Bạn là trợ lý tư vấn tuyển sinh đại học 2025 của Việt Nam.

**Context từ văn bản chính thức**:
{context}

**Câu hỏi của sinh viên**: {query}

**Hướng dẫn trả lời**:
- Trả lời DỰA HOÀN TOÀN trên context được cung cấp
- Trích dẫn nguồn văn bản (Thông tư số..., Quyết định số...)
- Nếu không có thông tin trong context: "Tôi không tìm thấy thông tin này trong tài liệu hiện có."
- Format câu trả lời rõ ràng với bullet points nếu cần
- Ngắn gọn, súc tích (200-300 từ)

**Confidence score của thông tin này**: {confidence:.2f}/1.0

**Câu trả lời**:"""

        # Try LLM providers with fallback
        try:
            if self.gemini_key:
                response = await self.gemini_model.generate_content_async(prompt)
                answer = response.text.strip()
                provider = "Gemini 2.0 Flash"
            elif self.glm4_key:
                response = await self.glm4_client.chat.completions.create(
                    model="glm-4-plus",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                answer = response.choices[0].message.content.strip()
                provider = "GLM-4"
            elif self.groq_key:
                response = await self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                answer = response.choices[0].message.content.strip()
                provider = "Llama-3.1"
            else:
                raise Exception("No LLM provider available")
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback: return chunks as plain text
            answer = "⚠️ Không thể tạo câu trả lời. Dưới đây là các đoạn văn liên quan:\n\n" + context
            provider = "Fallback"
        
        # Extract sources
        sources = list(set([
            chunk.get("metadata", {}).get("source", "Không rõ nguồn")
            for chunk in chunks
        ]))
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "provider": provider,
            "chunks_used": len(chunks)
        }

# Global instance
llm_generator = LLMGenerator()
```

**✅ Checkpoint 4**: Test LLM Generation
```bash
# Start FastAPI
uvicorn src.main:app --reload --port 8000

# Test full pipeline
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "điều kiện xét tuyển đại học 2025"}'

# Check response includes "answer" field
```

---

## 🔄 TESTING WORKFLOW (Sau mỗi bước phải chạy FastAPI)

### Test Flow Complete:

**Bước 1: Start FastAPI server**
```bash
# Terminal 1
uvicorn src.main:app --reload --port 8000 --log-level debug
```

**Bước 2: Test HYDE Enhancement**
```bash
# Terminal 2
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "điều kiện xét tuyển"}' | jq .

# Expected: Check "enhanced_query" có hypothetical answer
# Expected: Check "query_type" = "admission_criteria"
# Expected: Check "top_k_used" = 5 (auto-detected)
```

**Bước 3: Test BGE Retrieval + Score Boosting**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "hồ sơ xét tuyển"}' | jq '.chunks[] | {score, boosted, text: .text[:100]}'

# Expected: Các chunks có score > 0.6
# Expected: Có chunks được boosted (boosted=true)
# Expected: Chunks được sắp xếp theo score giảm dần
```

**Bước 4: Test LLM Generation**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tôi cần nộp những giấy tờ gì để xét tuyển?"}' | jq '.answer'

# Expected: Câu trả lời có cấu trúc rõ ràng
# Expected: Có trích dẫn nguồn (Thông tư số...)
# Expected: provider = "Gemini 2.0 Flash" hoặc fallback
```

**Bước 5: Test với UI có sẵn**
```bash
# Giả sử UI ở D:\chunking\rag2025_2\uni-guide-ai
# UI gọi endpoint: POST http://localhost:8000/query

# Test từ UI:
1. Mở UI trong browser
2. Nhập query: "điều kiện xét tuyển đại học 2025"
3. Kiểm tra:
   - Response time < 3s
   - Có câu trả lời đầy đủ
   - Hiển thị sources
   - Confidence score hiển thị
```

**Bước 6: Test Edge Cases**
```bash
# Test 1: Query mơ hồ
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "xét tuyển"}' | jq .

# Expected: HYDE enhance thành câu hỏi rõ ràng hơn

# Test 2: Query dài phức tạp
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tôi là học sinh lớp 12, điểm thi THPT 2025 chưa có, muốn xét tuyển vào ngành Y có được không và cần điều kiện gì?"}' | jq .

# Expected: top_k tự động tăng lên 7
# Expected: Câu trả lời chi tiết từ nhiều chunks

# Test 3: Query không có trong database
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "học phí ngành công nghệ thông tin là bao nhiêu?"}' | jq .

# Expected: "Tôi không tìm thấy thông tin này trong tài liệu hiện có"
# Expected: confidence score thấp
```

**Bước 7: Load Testing**
```bash
# Install ab (Apache Bench)
# Test 100 requests, 10 concurrent
ab -n 100 -c 10 -p query.json -T application/json http://localhost:8000/query

# query.json:
# {"query": "điều kiện xét tuyển"}

# Expected: 95%+ success rate
# Expected: Average response time < 2s
```

---

## 🎯 KẾT NỐI VỚI UI CÓ SẴN

**Giả sử UI ở `D:\chunking\rag2025_2\uni-guide-ai` có file config:**

```javascript
// uni-guide-ai/src/config/api.js
export const API_CONFIG = {
  baseURL: 'http://localhost:8000',
  endpoints: {
    query: '/query',
    health: '/health'
  }
}

// uni-guide-ai/src/services/ragService.js
import axios from 'axios';
import { API_CONFIG } from '../config/api';

export const ragService = {
  async query(userQuery) {
    const response = await axios.post(
      `${API_CONFIG.baseURL}${API_CONFIG.endpoints.query}`,
      {
        query: userQuery,  // Simple string, không cần QueryRequest phức tạp
        force_rag_only: false
      }
    );
    
    return {
      answer: response.data.answer,
      sources: response.data.sources,
      confidence: response.data.confidence,
      queryType: response.data.query_type
    };
  }
};
```

**Cập nhật CORS trong FastAPI nếu cần:**

```python
# src/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # UI ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🤖 AGENTS WORKFLOW (Chạy Song Song)

### Phase 1: Setup & Migration (30 phút)
**Agents**: `data-scientist` + `ai-engineer`
- [ ] Update `requirements.txt`
- [ ] Install dependencies
- [ ] Run Qdrant migration script
- [ ] Verify 110 chunks in Qdrant
- [ ] Test `/health` endpoint

**Checkpoint 1**:
```bash
uvicorn src.main:app --reload --port 8000
curl http://localhost:8000/health
# Expected: {"status": "healthy", "vectors_count": 110}
```

---

### Phase 2: HYDE Implementation (45 phút)
**Agents**: `prompt-engineer` + `ai-engineer`
- [ ] Create `src/services/query_enhancer.py`
- [ ] Implement HYDE with LLM fallback
- [ ] Add query type classification
- [ ] Add auto top_k estimation
- [ ] Unit tests

**Checkpoint 2**:
```bash
python scripts/test_hyde.py
# Expected: Enhanced query with hypothetical answer
```

---

### Phase 3: BGE Retrieval + Score Boosting (1 giờ)
**Agents**: `search-specialist` + `data-scientist`
- [ ] Create `src/services/bge_retriever.py`
- [ ] Load BGE model (bge-m3 hoặc AITeamVN)
- [ ] Implement Qdrant search
- [ ] **CRITICAL**: Implement score boosting (3 strategies)
- [ ] Implement reranking
- [ ] Integration tests

**Checkpoint 3**:
```bash
curl -X POST http://localhost:8000/query -d '{"query": "điều kiện"}' | jq '.chunks[] | {score, boosted}'
# Expected: Chunks có boosted=true, scores > 0.6
```

---

### Phase 4: LLM Generation (45 phút)
**Agents**: `ai-engineer` + `prompt-engineer`
- [ ] Create `src/services/llm_generator.py`
- [ ] Implement LLM client với fallback
- [ ] Context building from chunks
- [ ] Prompt engineering for Vietnamese
- [ ] Source extraction

**Checkpoint 4**:
```bash
curl -X POST http://localhost:8000/query -d '{"query": "hồ sơ xét tuyển"}' | jq '.answer'
# Expected: Structured answer với citations
```

---

### Phase 5: FastAPI Integration (30 phút)
**Agents**: `ai-engineer`
- [ ] Update `src/main.py` với `/query` endpoint mới
- [ ] Integrate HYDE → BGE → LLM pipeline
- [ ] Add CORS for UI
- [ ] Update response model
- [ ] Error handling

**Checkpoint 5**:
```bash
curl -X POST http://localhost:8000/query -d '{"query": "tuyển thẳng"}' | jq .
# Expected: Full response với all fields
```

---

### Phase 6: UI Connection Testing (30 phút)
**Agents**: `ai-engineer` + `error-detective`
- [ ] Update UI config (nếu cần)
- [ ] Test UI gọi API `/query`
- [ ] Verify response display trong UI
- [ ] Test 10 queries từ UI
- [ ] Performance tuning

**Checkpoint 6**:
```bash
# Start UI
cd D:\chunking\rag2025_2\uni-guide-ai
npm run dev

# Test trong browser
# Expected: Chatbot responses hoạt động tốt
```

---

### Phase 7: Final Testing & Optimization (30 phút)
**Agents**: `model-evaluator` + `error-detective`
- [ ] End-to-end testing với 20 queries
- [ ] Edge cases testing
- [ ] Performance benchmarking
- [ ] Error handling verification
- [ ] Documentation update

---

## ✅ CHECKLIST HOÀN THÀNH

### Core Features
- [ ] HYDE query enhancement working với 3 LLMs fallback
- [ ] BGE multi-vector retrieval from Qdrant
- [ ] **Score boosting implemented (CRITICAL)**
- [ ] LLM generation với structured answers
- [ ] `/query` endpoint accepts simple string
- [ ] CORS configured for UI
- [ ] UI successfully connects to API

### Testing
- [ ] All checkpoints passed
- [ ] 10+ test queries successful
- [ ] Edge cases handled
- [ ] Load testing passed (100 requests)
- [ ] UI integration tested

### Documentation
- [ ] README updated
- [ ] API docs (Swagger) accessible
- [ ] .env.example created (no exposed keys)

---

## 🚨 LƯU Ý QUAN TRỌNG

### 1. Score Boosting (MOST CRITICAL!)
Đây là yêu cầu **QUAN TRỌNG NHẤT**. Đừng skip! Implement đầy đủ 3 strategies:
- ✅ Semantic similarity boost (cosine > 0.75)
- ✅ Keyword matching boost (70%+ keywords)
- ✅ Source credibility boost (official docs)

### 2. HYDE phải connect vào query
HYDE không chỉ enhance query, mà còn:
- Auto-classify query type
- Auto-estimate top_k
- Convert simple string → full QueryRequest

### 3. Qdrant Migration
- **Phải xóa collection cũ hoàn toàn**
- Create mới với multi-vector config
- Verify 110 chunks uploaded

### 4. API Keys Security
```bash
# ROTATE NGAY! Keys đã exposed:
# - GEMINI_API_KEY
# - QDRANT_API_KEY
# - OPENAI_API_KEY (GLM-4)
# - GROQ_API_KEY
```

### 5. Testing After Each Phase
**KHÔNG được skip checkpoints!** Mỗi phase xong phải:
1. Start FastAPI server
2. Test endpoint với curl
3. Verify response format
4. Check logs cho errors
5. Move to next phase

---

## 📊 EXPECTED TIMELINE

| Phase | Time | Agents | Output |
|-------|------|--------|---------|
| 1. Setup & Migration | 30 min | data-scientist + ai-engineer | Qdrant ready |
| 2. HYDE | 45 min | prompt-engineer + ai-engineer | Query enhancement |
| 3. BGE + Boosting | 60 min | search-specialist + data-scientist | Retrieval working |
| 4. LLM Generation | 45 min | ai-engineer + prompt-engineer | Answers generated |
| 5. FastAPI Integration | 30 min | ai-engineer | API endpoint ready |
| 6. UI Connection | 30 min | ai-engineer + error-detective | Full pipeline |
| 7. Testing & Optimization | 30 min | model-evaluator + error-detective | Production ready |

**Total**: ~4 giờ (với agents song song: 3-3.5 giờ)

---

## 🚀 BẮT ĐẦU NGAY!

**Copy prompt này vào Claude Code CLI và run:**

```bash
cd D:\chunking\rag2025_2
code .

# Paste prompt vào Claude Code CLI
# Let agents work in parallel!
```

**Agents sẽ tự động:**
1. Phân tích code structure
2. Implement từng phase
3. Test sau mỗi bước
4. Fix errors tự động
5. Optimize performance

**Bạn chỉ cần:**
- Monitor progress
- Test checkpoints
- Verify với UI
- Deploy khi ready!

🎯 **Good luck! Let's build this RAG system FAST!** 🚀

**File: `src/main.py`** (Update `/query` endpoint)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from src.services.query_enhancer import query_enhancer
from src.services.bge_retriever import bge_retriever
from src.services.llm_generator import llm_generator
from loguru import logger

app = FastAPI(
    title="RAG API 2025 - HYDE + BGE + Qdrant",
    description="Advanced RAG with query enhancement and multi-vector retrieval",
    version="2.0.0"
)

# Simple request model (user chỉ cần gửi string)
class SimpleQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's question")
    force_rag_only: Optional[bool] = Field(default=False, description="Force RAG only mode")

# Response model
class QueryResponse(BaseModel):
    # Original query info
    original_query: str
    enhanced_query: str
    query_type: str
    
    # Answer
    answer: str
    sources: List[str]
    confidence: float
    
    # Metadata
    top_k_used: int
    chunks_used: int
    provider: str
    
    # Retrieved chunks (optional, for debugging)
    chunks: Optional[List[Dict[str, Any]]] = None

@app.post("/query", response_model=QueryResponse)
async def query(request: SimpleQueryRequest):
    """
    Main RAG endpoint with HYDE enhancement
    
    Flow:
    1. HYDE: user query → enhanced QueryRequest
    2. BGE: retrieve chunks with score boosting
    3. LLM: generate answer from chunks
    """
    
    try:
        logger.info(f"Received query: {request.query}")
        
        # Step 1: HYDE Enhancement
        enhanced_request = await query_enhancer.enhance_query(
            user_query=request.query,
            force_rag_only=request.force_rag_only
        )
        
        logger.info(f"Query type: {enhanced_request['query_type']}, top_k={enhanced_request['top_k']}")
        
        # Step 2: BGE Multi-Vector Retrieval with Score Boosting
        chunks, confidence = await bge_retriever.retrieve(
            query_enhanced=enhanced_request["query"],
            original_query=enhanced_request["original_query"],
            top_k=enhanced_request["top_k"]
        )
        
        logger.info(f"Retrieved {len(chunks)} chunks, confidence={confidence:.3f}")
        
        # Step 3: LLM Answer Generation
        result = await llm_generator.generate_answer(
            query=enhanced_request["original_query"],
            chunks=chunks,
            confidence=confidence
        )
        
        logger.info(f"Generated answer using {result['provider']}")
        
        # Step 4: Build response
        return QueryResponse(
            original_query=enhanced_request["original_query"],
            enhanced_query=enhanced_request["query"],
            query_type=enhanced_request["query_type"],
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            top_k_used=enhanced_request["top_k"],
            chunks_used=result["chunks_used"],
            provider=result["provider"],
            chunks=chunks  # Include for debugging
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check with Qdrant connection"""
    try:
        from src.services.bge_retriever import bge_retriever
        collection_info = bge_retriever.qdrant_client.get_collection(
            bge_retriever.collection_name
        )
        
        return {
            "status": "healthy",
            "qdrant_connected": True,
            "vectors_count": collection_info.points_count,
            "collection": bge_retriever.collection_name
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/")
async def root():
    """API info"""
    return {
        "name": "RAG API 2025",
        "version": "2.0.0",
        "features": [
            "HYDE query enhancement",
            "BGE multi-vector retrieval",
            "Score boosting for near-matches",
            "Qdrant vector store",
            "Multi-LLM fallback (Gemini → GLM-4 → Groq)"
        ],
        "endpoints": {
            "POST /query": "Main RAG query endpoint (simple string input)",
            "GET /health": "Health check with Qdrant status",
            "GET /docs": "Swagger API documentation"
        }
    }
    