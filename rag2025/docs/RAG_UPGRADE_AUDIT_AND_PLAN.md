# RAG Upgrade Mission — Audit, Research, and Production Upgrade Plan

## 0) Execution Notes
- Loaded required skill: `amp-workflow-sop` (mandatory).
- `@oracle` and `@librarian` tool calls failed in this environment (`model_not_supported`, `auth_unavailable`), so audit/research below was completed via direct code inspection + Exa + Context7.

---

## 1) Architecture Overview (Current)

### 1.1 Module map
- API orchestration: `src/main.py`
- Chunking: `src/chunker.py`
- Embeddings: `src/services/embedding.py`
- Retrieval:
  - LanceDB dense retrieval: `src/services/lancedb_retrieval.py`
  - Hybrid retriever (dense+BM25+rerank): `src/services/retriever.py` (not primary path in `/query`)
  - GraphRAG fusion + routing: `src/services/graphrag_retriever.py`
- Generation: `src/services/llm_generator.py`, `src/services/llm_client.py`
- Guardrail/cache: `src/services/guardrail.py`, `src/services/query_cache.py`
- Infra adapter: `src/infrastructure/lancedb_adapter.py`
- Ingestion/build:
  - LanceDB ingest: `scripts/ingest_lancedb.py`
  - Graph build: `scripts/build_graph.py`
  - Chunk enhancement script: `scripts/enhance_chunks.py`
- Config: `config/settings.py`

### 1.2 Data flow
1. Ingestion: JSONL chunk files loaded (`scripts/ingest_lancedb.py`) and embedded with SentenceTransformer.
2. Chunking (offline / debug): `Chunker` with profile-based splitting (`auto/faq/policy`) and metadata/breadcrumb extraction.
3. Retrieval (`/query`):
   - HYDE generates variants.
   - Each variant embedded by SentenceTransformer.
   - Dense retrieval from LanceDB.
   - Manual merge + dedup + rerank.
4. Generation: LLM answers with context chunks.
5. `/v2/query`: baseline dense retrieval + router -> optional GraphRAG fusion.

### 1.3 Bottlenecks / anti-patterns
- Heavy startup model loading in API process (`SentenceTransformer`, reranker, graph load).
- `/query` does per-variant sequential retrieval and per-request embedding calls.
- In-memory cache is single-process only (`QueryCache`).
- Retrieval dedup previously relied on missing key (`id`) causing collisions.
- Ingestion script previously destructive table reset.

---

## 2) Code Quality Audit (Findings)

### 2.1 Critical
1. **Missing LOG_LEVEL setting causes runtime failure**
   - `src/main.py` references `settings.LOG_LEVEL` but `config/settings.py` had no field.
   - Impact: startup crash / misconfigured logging.
   - Status: **fixed**.

2. **Unsafe CORS config (`*` + credentials)**
   - `src/main.py` had `allow_origins=[..., "*"]` + `allow_credentials=True`.
   - Impact: standards violation, browser auth-cookie/security issues.
   - Status: **fixed** by disabling credentials when wildcard is present.

3. **Dedup key mismatch in retrieval merge path**
   - `src/main.py` dedup used `chunk.get("id")`, but retrieval docs carry `chunk_id`/`point_id`.
   - Impact: false dedup, unstable ranking, possible context loss.
   - Status: **fixed**.

### 2.2 High
4. **Ingestion script dropped table each run**
   - `scripts/ingest_lancedb.py` dropped existing table.
   - Impact: destructive reingest, no incremental path, production risk.
   - Status: **fixed** to append when table exists.

5. **Hard-coded external absolute paths in script**
   - `scripts/enhance_chunks.py` points to `D:\chunking\rag2025_2\...`.
   - Impact: non-portable, broken in CI/prod.
   - Recommendation: parameterize via CLI args + `Path` from repo root.

6. **Provider/config divergence across services**
   - `query_enhancer.py` and `llm_generator.py` manually read env and build clients.
   - `llm_client.py` already provides unified fallback client.
   - Impact: inconsistent fallback behavior and observability.

### 2.3 Medium
7. **Deprecated FastAPI lifecycle API** (`@app.on_event`) in `main.py`.
8. **Query cache unbounded growth** (`query_cache.py` dict no max-size / eviction sweep).
9. **Chunk splitter separator join logic can degrade chunk boundary quality** (`chunker.py` joins parts without separator retention).
10. **Test suite drift** (`tests/test_api.py`, `tests/test_embedding.py`) mismatches current API/model dims.

---

## 3) RAG-specific Assessment

### Chunking
- Current defaults: 350 tokens, overlap 70 (`settings.py`) with profile chunking in `chunker.py`.
- Good: profile strategy + metadata preservation + breadcrumbs.
- Gaps:
  - No semantic sentence boundary model.
  - No parent-child chunk graph for retrieval.
  - No automatic quality eval loop (coverage/fragmentation metrics).

### Embeddings
- Uses `Qwen/Qwen3-Embedding-8B` (4096-dim): strong multilingual choice for Vietnamese.
- Gaps:
  - No embedding cache/fingerprint reuse.
  - No asynchronous batching queue for high-QPS ingestion.

### Retrieval
- `/query` = mostly dense + custom rerank; `/v2` adds router+GraphRAG.
- Gaps:
  - Hybrid dense+sparse not first-class in online path.
  - No explicit retrieval-time latency budget per stage.
  - Score calibration across variants/rerank not normalized.

### Context window utilization
- Has chunk cap (15/30) before generation.
- Gaps:
  - No structured context compaction (MMR/diversity/section-aware packing).
  - No citation alignment check between answer spans and chunks.

---

## 4) AMP-specific / Integration / Config Notes
- Integration points:
  - External LLM providers (ramclouds, Groq, OpenAI-compatible, Z.AI).
  - LanceDB local embedded storage.
  - Graph artifacts in `data/graph`.
- Risks:
  - Mixed env naming and fallback ownership across modules.
  - Error exposure mode exists but not consistently enforced in all endpoint error paths.

---

## 5) Research Summary (Exa + Context7)

## 5.1 Grounded best practices (high-confidence)
1. **LanceDB production path** (Context7):
   - Build vector index (`IvfPq`/`HnswPq`) and scalar/FTS index.
   - Use `query_type="hybrid"` + RRF/reranker for robust recall.
   - Use filtering via SQL clauses and index filtered fields.
2. **RAG architecture trend (2024–2025)**:
   - Move from naive linear RAG to modular pipeline:
     query transform -> hybrid retrieve -> rerank -> grounded generation -> evaluate loop.
   - GraphRAG reserved for multi-hop/comparative queries, not all traffic.
3. **Evaluation-first operations**:
   - Track Recall@K/HitRate + MRR/nDCG + stage latency.
   - Keep an eval dataset and run regression checks on each retrieval change.

## 5.2 Gap analysis (current vs best practice)
- Current strengths: profile chunking, HYDE variants, reranking, GraphRAG route.
- Major gaps:
  - No first-class online hybrid retrieval in `/query`.
  - Limited observability and no standardized retrieval quality metrics endpoint.
  - Ingestion/upsert/versioning strategy is still script-heavy and partially non-portable.

---

## 6) Implemented Upgrades (This pass)

### Changed files
1. `config/settings.py`
   - Added `LOG_LEVEL` with constrained enum + default `INFO`.

2. `src/main.py`
   - Hardened CORS credentials behavior when wildcard origin present.
   - Fixed chunk dedup key to use `chunk_id`/`point_id` fallback chain.

3. `scripts/ingest_lancedb.py`
   - Replaced destructive table drop flow with append-if-exists behavior.

### 6.1 GraphRAG hardening pass (finalized)
4. `src/domain/graph.py`
   - Finalized deterministic PPR cache keying using sorted valid seeds + `(alpha, max_iter)`.
   - Added explicit cache reset API (`clear_caches`) and validated reuse/clear behavior.
   - Changed PPR non-convergence fallback to **empty scores** (instead of uniform-all-nodes), preventing graph noise from overriding dense retrieval.

5. `src/services/graphrag_retriever.py`
   - Extended GraphRAG retriever config with `ppr_alpha` and wired it to settings.
   - Seed extraction now uses merged context (original query + step-back + top HyDE variants) with deduped seed node_ids.
   - Fusion behavior finalized: when no graph scores exist, fallback to baseline dense-normalized score (no blind penalty).
   - Pipeline factory now consumes `GRAPHRAG_SIMPLE_THRESHOLD` (router threshold) and `GRAPHRAG_PPR_ALPHA` (PPR damping) from settings.

6. `src/main.py` (`/v2/query`)
   - Implemented effective `force_route` behavior:
     - `padded_rag`: bypass GraphRAG fusion and return baseline docs.
     - `graph_rag`: force GraphRAG fusion pass over baseline docs when available.

7. `tests/test_graphrag.py` (new targeted tests)
   - Added focused GraphRAG tests for:
     - PPR cache determinism/reuse + cache clear semantics.
     - PPR non-convergence behavior (empty fallback).
     - Fusion fallback to baseline when graph signals are absent.
   - Current status: `3 passed`.

---

## 7) Upgrade Roadmap (Impact / Effort)

### Phase A (high impact, low-medium effort)
1. Promote hybrid retrieval into `/query` path:
   - LanceDB vector + FTS index.
   - `query_type="hybrid"` + RRFReranker.
2. Standardize LLM provider usage through `UnifiedLLMClient` in enhancer/generator.
3. Add retrieval metrics middleware + per-stage timing logs.

### Phase B (high impact, medium effort)
4. Incremental ingestion with content hash dedup + upsert policy.
5. Embedding cache (hash(text)->vector) and batch worker.
6. Parent-child chunk retrieval and context packing.

### Phase C (medium impact, medium-high effort)
7. Unified online router between dense/hybrid/graph routes with SLA budget.
8. Add citation span attribution + hallucination guard checks before response.

---

## 8) Migration Guide (Safe rollout)

1. **Preflight**
   - Backup LanceDB directory.
   - Export current `.env` and validate required keys.
2. **Deploy step 1 (already done in code)**
   - Pull new config containing `LOG_LEVEL`.
   - Restart API and verify `/health`.
3. **Ingestion policy transition**
   - Run updated `ingest_lancedb.py` on staging first.
   - Confirm row growth behavior is expected (append mode).
4. **Canary rollout**
   - Route a small fraction of traffic and compare retrieval hit-rate/latency.
5. **Rollback**
   - Revert code + restore LanceDB backup directory.

---

## 9) Test Cases (required for next pass)

### Unit
1. `test_settings_log_level_present` (settings loads LOG_LEVEL default and env override).
2. `test_query_dedup_uses_chunk_id` (dedup retains highest score by `chunk_id`).
3. `test_ingest_append_mode` (existing table receives added records, no drop).

### Integration
4. `test_query_returns_stable_unique_chunks` (variants with same chunk_id dedup correctly).
5. `test_cors_wildcard_disables_credentials` (middleware config assertion).
6. `test_incremental_ingest_idempotency` (same source ingested twice does not duplicate when hash-dedup is added).

### Retrieval quality regression
7. Build static eval set and track: HitRate@5, MRR@10, nDCG@10; fail CI on statistically significant drop.

---

## 10) Known Test Status Snapshot
- `pytest rag2025/tests/test_chunker.py -q`: 1 failure (existing assertion bug about numeric sparse terms).
- `pytest rag2025/tests/test_api.py -q`: 3 failures (tests outdated vs current API contract).
- These are pre-existing test drift issues and should be aligned in dedicated test-refactor task.
