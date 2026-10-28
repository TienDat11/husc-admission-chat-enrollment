# Source Map — HUSC RAG Chatbot (Snapshot 2026-05-21)

> Provider/model names in the diagram below were updated to the 2026-10-27 ns24
> state (see the Update section at the bottom); the rest is the 2026-05-21 snapshot.

## Pipeline Architecture

```
user_query
  → main.py (FastAPI endpoint)
    → guardrail.precheck (profanity/injection filter)
    → [MISSING: ambiguity_detector — Phase 3]
    → QueryRouter.route(query)
        → NER extraction (role=ner → gemini-3.5-flash-lite on ns24)
        → HyDE document generation (role=hyde → gemini-3.5-flash-lite on ns24)
        → Intent classification + complexity scoring
        → Returns: RouterResult{all_queries_for_embedding, route, intent, complexity}
    → EmbeddingService.encode_single(variant) [Qwen3-Embedding-0.6B, 1024-dim]
    → LanceDBRetriever.retrieve(vector, top_k=10) [per variant, max 4 variants]
    → Dedup by chunk_id → top 15
    → boost_with_aggregation(query, docs, retriever, top_k=15, max_inject=2)
    → UnifiedRAGPipeline.query(baseline_docs=..., top_k=5)
        → GraphRAG fusion (if route=graph_rag): NetworkX graph traversal
        → PPR reranking
        → Returns: RAGResult{documents, route, router_result}
    → LLMGenerator.generate_answer(query, chunks, confidence, is_program_list_query)
        → role=gen → gpt-6-luna on ns24 (primary)
        → claude-haiku-5-5 on ns24 (same-gateway fallback, no cross-provider)
        → Returns: {answer, provider}
  → Response to user
```

## Key Files & Classes

| Component | File | Key Function/Class |
|-----------|------|-------------------|
| Entry point | `src/main.py` | FastAPI app, `/query` endpoint |
| Router | `src/services/query_router.py` | `QueryRouter.route(query)` |
| NER | `src/services/ner_service.py` | `NERService.extract_entities()` |
| HyDE | `src/services/query_enhancer.py` | `HYDEQueryEnhancer` |
| Embedding | `src/services/embedding.py` | `EmbeddingService.encode_single()` |
| LanceDB retrieval | `src/services/lancedb_retrieval.py` | `LanceDBRetriever.from_env()` |
| GraphRAG | `src/services/graphrag_retriever.py` | `UnifiedRAGPipeline.from_disk()` |
| Hybrid search | `src/services/hybrid_search.py` | Sparse+dense fusion |
| Aggregation boost | `src/services/aggregation_booster.py` | `boost_with_aggregation()` |
| LLM generation | `src/services/llm_generator.py` | `LLMGenerator().generate_answer()` |
| LLM client | `src/services/llm_client.py` | Role-aware ns24 gateway client (single provider) |
| Settings | `config/settings.py` | `RAGSettings` (pydantic) |
| Graph domain | `src/domain/graph.py` | `KnowledgeGraph`, `KnowledgeGraphBuilder` |

## Data Layout

```
rag2025/data/
├── chunked/                          # Source JSONL chunks
│   ├── chunked_1.jsonl ... chunked_25.jsonl  # PDF-extracted (pdf_1, pdf_2, pdf_3)
│   ├── chunked_2026_husc.jsonl       # 32 ngành-specific chunks (husc_nganh_*)
│   ├── husc_2026_aggregation.jsonl   # 35 aggregation chunks (diem_chuan, so_sanh, liet_ke)
│   ├── husc_2026_operational_qa.jsonl # 24 operational Q-A chunks
│   └── husc_2026_chunks.jsonl        # Duplicate of chunked_2026_husc (legacy)
├── eval/
│   ├── husc_thi_sinh_thuc.json       # 86 câu user thực (no must_include)
│   └── husc_86q_test.json            # 303 câu golden (with must_include)
├── graph/
│   ├── knowledge_graph.graphml       # NetworkX graph (entities + triples)
│   └── entity_index.json             # chunk_id → entities mapping
└── lancedb/                          # LanceDB vector store
    └── rag2025/                      # Table name: "rag2025"
        └── (lance files)             # ~1583 rows, 1024-dim vectors
```

## Chunk ID Prefixes & Sources

| Prefix | Source File | Content |
|--------|------------|---------|
| `husc_nganh_<mã>_2026` | chunked_2026_husc.jsonl | Info per-ngành (chỉ tiêu, tổ hợp, mô tả) |
| `tuyensinh_overview_2026` | chunked_2026_husc.jsonl | Tổng quan HUSC 2026 |
| `liet_ke_<n>_nganh_2026` | husc_2026_aggregation.jsonl | Danh sách aggregation |
| `bang_xep_hang_chi_tieu_2026` | husc_2026_aggregation.jsonl | Bảng xếp hạng |
| `diem_chuan_2025_full` | husc_2026_aggregation.jsonl | ⚠️ SAI DATA — cần fix |
| `so_sanh_hocphi_2025_vs_2026` | husc_2026_aggregation.jsonl | So sánh học phí |
| `qa_cong_thuc_diem_*` | husc_2026_operational_qa.jsonl | ⚠️ CÔNG THỨC SAI |
| `qa_<topic>_2026` | husc_2026_operational_qa.jsonl | Operational Q-A |
| `chunked_<N>_chunk_<M>` | chunked_<N>.jsonl | PDF raw chunks |
| `husc_news_*` | chunked_11.jsonl | News/thông báo HUSC |

## Scripts Reference

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `scripts/build_graph.py` | Build knowledge graph | `data/chunked/chunked_*.jsonl` | `data/graph/knowledge_graph.graphml` + `entity_index.json` |
| `scripts/build_graph.py --incremental` | Append new chunks to graph | Same | Same (incremental) |
| `scripts/ingest_aggregation_to_lancedb.py` | Ingest aggregation chunks | `data/chunked/husc_2026_aggregation.jsonl` | LanceDB table `rag2025` |
| `scripts/ingest_operational_qa_v2.py` | Ingest operational Q-A | `data/chunked/husc_2026_operational_qa.jsonl` | LanceDB table `rag2025` |
| `scripts/ingest_all_chunks.py` | Full ingest all chunks | All `data/chunked/*.jsonl` | LanceDB table `rag2025` |
| `scripts/eval_thi_sinh_thuc.py` | 86Q eval (real users) | `data/eval/husc_thi_sinh_thuc.json` | `results/86q_thi_sinh_results.json` |
| `scripts/gen_pdf_report.py` | Generate PDF reports | `results/86q_thi_sinh_results.json` | `results/86q_thi_sinh_report_full.pdf` + `*_thay_tuyen_sinh.pdf` |
| `scripts/verify_25nganh.py` | Quick 6Q smoke test | Hardcoded queries | stdout pass/fail |

## Models & Providers

| Role | Model | Provider | Endpoint |
|------|-------|----------|----------|
| All LLM roles (default/router/ner/hyde/guardrail/gen) | gemini-3.5-flash-lite / deepseek-v4.1-flash; gen: gpt-6-luna / claude-haiku-5-5 | ns24 | NS24_API_KEY @ https://api.ns24.io/v1 |

## Current Eval State (2026-05-21)

- **86Q eval**: 75/86 has_answer (87.2%)
- **11 fails**: chunking_gap (7), retrieval_miss (2), llm_hallucination (2)
- **4 Q-A mismatch**: msg061 (điểm→học phí bleed), msg019 (placeholder guess), msg043 (550k wrong), msg001 (no enumeration)
- **Target**: ≥82/86 (95.3%) after Phase 1+2

## Known Issues (MUST FIX)

1. ⚠️ `diem_chuan_2025_full` chunk has WRONG data (CNTT=23.5, should be 17.50)
2. ⚠️ 5 công thức chunks missing `/(H1+H2+H3)×3` factor
3. ⚠️ `chunked_24.jsonl` lines 371-374: cross-unit contamination (unit="HUSC" but actually Y-Dược/Du lịch/KT-CN)
4. Generation prompt missing: enumeration rule, khối→ngành→tiền ordering, relevance rules
5. No temporal fallback: "chưa công bố" without showing previous year data

## Auto-Crawl Infrastructure (EXISTING but INCOMPLETE)

| Component | File | Status |
|-----------|------|--------|
| IngestPipeline (7-step) | `src/infrastructure/auto_crawl.py` | ✅ Exists (613 lines) |
| CrawlerState model | `src/infrastructure/auto_crawl.py:50` | ✅ Exists |
| Hash dedup | `src/infrastructure/auto_crawl.py:159` | ✅ Exists |
| Supersession logic | `src/infrastructure/auto_crawl.py:198` | ✅ Exists |
| Snapshot/rollback | `src/infrastructure/auto_crawl.py:227` | ✅ Exists |
| Scheduler helper | `src/infrastructure/auto_crawl.py:584` | ✅ Exists |
| crawler_state.json | `rag2025/crawler_state.json` | ⚠️ max_id=0 (should be 73) |
| crawl_hash_log.json | `rag2025/data/crawl_hash_log.json` | ✅ Exists (empty) |
| **crawl_fn (HTTP fetch)** | — | ❌ MISSING — no implementation |
| **scripts/crawl_incremental.py** | — | ❌ MISSING — entry point not created |
| **Windows Task Scheduler** | — | ❌ NOT REGISTERED |

### Pipeline Flow (auto_crawl.py):
```
crawl_fn() → raw pages [(url, content)]
  → Step 1: _step_crawl (call crawl_fn)
  → Step 2: hash_dedup (SHA256, skip duplicates)
  → Step 3: chunk (split into JSONL chunks)
  → Step 4: metadata enrichment
  → Step 5: supersession (mark old versions)
  → Step 6: embed → LanceDB upsert
  → Step 7: graph update (incremental)
```

### Thongbao Coverage:
- IDs in LanceDB: 59-73 (crawled manually, in chunked_12 → chunked_22)
- Latest: id=73 (2026-05-18)
- Next expected: id=74 (after THPT exam results)
- Target: auto-detect id=74+ and ingest without manual intervention

```bash
# Ingest
python scripts/ingest_aggregation_to_lancedb.py    # aggregation chunks
python scripts/ingest_operational_qa_v2.py          # operational Q-A chunks
python scripts/build_graph.py --incremental         # rebuild graph (new chunks only)

# Eval
python scripts/eval_thi_sinh_thuc.py               # 86Q full eval (~75 min)
python scripts/gen_pdf_report.py                    # regenerate PDF reports
python scripts/verify_25nganh.py                    # quick 6Q smoke test (~5 min)

# Verify
python -c "import lancedb; db=lancedb.connect('./data/lancedb'); print(db.table_names()); t=db.open_table('rag2025'); print(f'Rows: {t.count_rows()}')"
```

## Subagent Routing Guide

| Task Type | Agent | Model | Skills |
|-----------|-------|-------|--------|
| Fix chunk data (jsonl edit) | `oh-my-claudecode:executor` | sonnet | Read → Edit → verify with `python -c "import json; ..."` |
| Fix prompt (txt edit) | `oh-my-claudecode:executor` | sonnet | Read → Edit → verify line count |
| Ingest to LanceDB | `oh-my-claudecode:executor` | sonnet | Bash: `cd rag2025 && python scripts/ingest_*.py` |
| Build graph | `oh-my-claudecode:executor` | sonnet | Bash: `cd rag2025 && python scripts/build_graph.py --incremental` |
| Run eval (long) | `oh-my-claudecode:executor` | sonnet | Bash (background): `cd rag2025 && python scripts/eval_thi_sinh_thuc.py` |
| Review/verify results | `oh-my-claudecode:architect` | opus | Read results.json → compare with acceptance criteria |
| Create test file | `oh-my-claudecode:executor` | sonnet | Write test skeleton → pytest verify |
| Research/explore | `oh-my-claudecode:explore` | sonnet | Grep/Glob/Read for discovery |

## Update 2026-10-27 — LLM gateway migration

The diagram above is the 2026-05-21 snapshot and is kept as a historical record.
Current state differs in exactly one layer:

- **LLM provider**: one gateway, ns24 (`NS24_BASE_URL`, OpenAI-compatible). The
  RamClouds primary, the Groq fallback and the Z.AI path are all removed; there is
  no cross-provider fallback anywhere on the runtime path.
- **Role → model** (from `rag2025/config/ns24_models.py`, enforced at startup):
  default / router / ner / hyde / guardrail = `gemini-3.5-flash-lite` →
  `deepseek-v4.1-flash`; gen = `gpt-6-luna` → `claude-haiku-5-5`.
- **Credential**: `NS24_API_KEY` is the only LLM credential. Unset ⇒ every role
  builds no provider and generation degrades to the static fallback answer.
- **Code**: `src/services/llm_client.py` (role-aware client, same-gateway
  fallback), `config/ns24_models.py` (whitelist + prices), `config/request_context.py`
  (request-id contextvar for token attribution).

