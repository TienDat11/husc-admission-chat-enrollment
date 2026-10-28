# Tóm tắt dự án: Chatbot tuyển sinh HUSC

## 1) Bức tranh tổng quan
Dự án xây dựng một hệ thống chatbot tư vấn tuyển sinh cho Trường Đại học Khoa học Huế (HUSC), tập trung vào hai mục tiêu: (1) trả lời nhanh, đúng ngữ cảnh tuyển sinh; (2) giảm rủi ro hallucination bằng cơ chế retrieval + guardrail + groundedness scoring.

Hệ thống gồm 3 phần chính:
- Frontend chat web (`uni-guide-ai`) cho trải nghiệm người dùng.
- Backend API FastAPI (`rag2025/src/main.py`) xử lý truy vấn, retrieval, generation, guardrail.
- Bộ đánh giá chất lượng notebook (`packages/rag-chatbot-husc/src/notebooks/eval_core.py`) để đo accuracy/recall/hallucination và so sánh route v1/v2.

---

## 2) Pipeline xử lý chính (end-to-end)

### 2.1 Pipeline `/query` (HYDE multi-variant RAG)
User question
→ Rate limit + cache check + guardrail precheck
→ HYDE query enhancement (sinh nhiều biến thể truy vấn)
→ Embedding cho từng variant
→ Retrieval từ LanceDB (hoặc Hybrid dense+BM25 khi bật)
→ Merge + deduplicate + rerank
→ LLM generation theo context chunks
→ Groundedness score + status_code/data_gap_hints
→ Trả response (answer, sources, confidence)

**Bằng chứng mã nguồn:**
- Endpoint và flow `/query`: `rag2025/src/main.py:486`
- Query enhancement HYDE: `rag2025/src/services/query_enhancer.py:1`
- LanceDB retrieval: `rag2025/src/services/lancedb_retrieval.py:1`
- Hybrid search (dense+BM25+RRF): `rag2025/src/services/hybrid_search.py:1`
- Guardrail (scope/PII/no-result): `rag2025/src/services/guardrail.py:24`
- Query cache TTL: `rag2025/src/services/query_cache.py:7`

### 2.2 Pipeline `/v2/query` (Smart Router + Unified RAG)
User question
→ Rate limit + cache check
→ Baseline dense retrieval
→ Smart Router (Step-back + HyDE + classification)
→ Route:
  - `padded_rag` cho truy vấn đơn giản
  - `graph_rag` cho truy vấn multi-hop/so sánh
→ (Nếu graph_rag) PPR fusion từ knowledge graph + baseline docs
→ LLM generation + groundedness warning
→ Trả response kèm `router_info`, `route`, `graph_stats`

**Bằng chứng mã nguồn:**
- Endpoint `/v2/query`: `rag2025/src/main.py:919`
- Router thông minh: `rag2025/src/services/query_router.py:127`
- Unified pipeline + GraphRAG fusion: `rag2025/src/services/graphrag_retriever.py:209`

### 2.3 Pipeline cập nhật tri thức graph
Admin request `/v2/graph/update`
→ Validate payload chunks
→ Incremental graph update (không cần rebuild full)
→ Persist graph mới

**Bằng chứng mã nguồn:**
- Endpoint graph update: `rag2025/src/main.py:1075`
- Incremental update logic: `rag2025/src/services/graphrag_retriever.py:323`

---

## 3) Nhóm chức năng hiện tại

## 3.1 API & Backend Orchestration
- FastAPI app, CORS, startup services, health/metrics.
- Endpoint chính: `/query`, `/v2/query`, `/v2/graph/update`, `/v2/graph/stats`.
- Rate limiting theo IP và in-memory metrics.

**Bằng chứng:** `rag2025/src/main.py:477`, `rag2025/src/main.py:486`, `rag2025/src/main.py:919`, `rag2025/src/main.py:1075`

## 3.2 Retrieval & Reranking
- LanceDB dense retrieval với metadata filter.
- Hybrid dense + BM25 + Reciprocal Rank Fusion (RRF).
- Reranker cross-encoder để cải thiện thứ tự chunks.

**Bằng chứng:** `rag2025/src/services/lancedb_retrieval.py:71`, `rag2025/src/services/hybrid_search.py:19`, `rag2025/src/main.py:703`

## 3.3 Router & GraphRAG
- Smart routing dựa trên complexity/intent.
- Step-back prompting + HyDE document generation.
- GraphRAG fusion dùng PPR score theo entity seed.

**Bằng chứng:** `rag2025/src/services/query_router.py:1`, `rag2025/src/services/graphrag_retriever.py:75`, `rag2025/src/services/graphrag_retriever.py:153`

## 3.4 LLM Orchestration
- UnifiedLLMClient theo mô hình OpenAI-compatible.
- Provider fallback chain, retry/backoff, JSON mode.

**Bằng chứng:** `rag2025/src/services/llm_client.py:1`, `rag2025/src/services/llm_client.py:85`, `rag2025/src/services/llm_client.py:167`

## 3.5 Guardrail & Trust Layer
- Chặn truy vấn ngoài scope tuyển sinh HUSC.
- Phát hiện dữ liệu nhạy cảm (PII) trước khi xử lý.
- Phân loại no-result thành `HUSC_ENTITY_NOT_FOUND` hoặc `INSUFFICIENT_DATA`.
- Groundedness warning khi câu trả lời bám nguồn thấp.

**Bằng chứng:** `rag2025/src/services/guardrail.py:24`, `rag2025/src/main.py:524`, `rag2025/src/main.py:715`, `rag2025/src/main.py:1029`

## 3.6 Frontend Chat Experience
- Chat UI với optimistic message send.
- Hiển thị nguồn (`sources`) và confidence.
- Surface `status_code`, `status_reason`, `data_gap_hints` để minh bạch lỗi dữ liệu.
- Suggested questions, theme toggle.

**Bằng chứng:** `uni-guide-ai/src/components/chat/ChatLayout.tsx:17`, `uni-guide-ai/src/components/chat/ChatLayout.tsx:42`, `uni-guide-ai/src/components/chat/SuggestedQuestions.tsx:11`, `uni-guide-ai/src/components/chat/ThemeToggle.tsx:5`, `uni-guide-ai/src/lib/api.ts:45`

## 3.7 Evaluation & Quality Control
- Notebook eval hỗ trợ gọi cả `/query` và `/v2/query`.
- Chuẩn hóa output để so sánh route.
- Tính recall proxy, hallucination flag, diagnostic report markdown.

**Bằng chứng:** `packages/rag-chatbot-husc/src/notebooks/eval_core.py:62`, `packages/rag-chatbot-husc/src/notebooks/eval_core.py:142`, `packages/rag-chatbot-husc/src/notebooks/eval_core.py:219`, `packages/rag-chatbot-husc/src/notebooks/eval_core.py:235`, `packages/rag-chatbot-husc/src/notebooks/eval_core.py:244`

---

## 4) Điểm mạnh kỹ thuật để trình bày với nhà tuyển dụng
1. **Routing thông minh giữa 2 chiến lược RAG** (padded vs graph), giúp cân bằng tốc độ và chiều sâu suy luận.
2. **Thiết kế trust-aware**: guardrail + PII detection + groundedness scoring giúp chatbot an toàn, thực dụng trong bối cảnh giáo dục.
3. **Khả năng mở rộng dữ liệu dài hạn** qua incremental graph update, phù hợp chu kỳ thay đổi dữ liệu tuyển sinh hằng năm.
4. **Khả năng đo lường rõ ràng**: có pipeline eval, metric và diagnostic để theo dõi chất lượng theo từng route.
5. **Kiến trúc tách lớp rõ**: frontend, API orchestration, retrieval, router, graph, eval độc lập tương đối, dễ mở rộng.

---

## 5) Hạn chế hiện tại (nên nói thẳng khi phỏng vấn)
1. Nhiều thành phần cache/metrics đang ở mức in-memory, cần externalized storage nếu scale production lớn.
2. Một số route/flow vẫn phụ thuộc độ ổn định của provider LLM bên ngoài.
3. Độ chính xác retrieval và groundedness phụ thuộc mạnh vào chất lượng/chuẩn hóa nguồn dữ liệu ingest.
4. Cần tiếp tục mở rộng bộ benchmark thực chiến theo mùa tuyển sinh mới.

---

## 6) Tuyên bố giá trị dự án (phiên bản ngắn)
Đây là một chatbot tuyển sinh theo hướng “evidence-aware RAG”: không chỉ trả lời mà còn cung cấp nguồn, mức tin cậy, cảnh báo thiếu dữ liệu, và cơ chế route thông minh để xử lý cả câu hỏi đơn lẫn câu hỏi đa chiều. Dự án cho thấy năng lực thiết kế AI application end-to-end: từ retrieval, graph reasoning, guardrail, API engineering đến UX và quality evaluation.
