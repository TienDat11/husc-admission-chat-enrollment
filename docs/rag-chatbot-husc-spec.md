# Spec: rag-chatbot-husc — Quality + Latency + Standardize (phase-1)

> Source of truth for phase-1 implementation.
> Authority:
> - `docs/superpowers/specs/2026-04-17-colab quality-eval-design.md` (sections 2-6)
> - `rag2025/.omc/plans/latency_reduction_plan_v2.md` (sections 7-9)
> - `rag2025/.omc/plans/standardize_frontend_backend_plan_v2.md` (sections 10-12)

## Phase: phase-1

### Sections implemented

#### Original scope — Colab Quality Eval

- **S2**: Notebook architecture (`colab_eval.ipynb`)
- **S3.1**: Evaluation metrics — accuracy, recall, latency_p95
- **S3.2**: Evaluation metrics — groundedness, hallucination_rate
- **S4**: Diagnostic report enrichment
- **S5**: Route parity (force_route vs auto-route)
- **S6**: Reproducibility (seeds, bootstrap CI, NI margins)

#### Expanded scope (2026-05-24) — Latency reduction
Source: `rag2025/.omc/plans/latency_reduction_plan_v2.md`

- **S7**: Phase 0 verification + Quick wins (HyDE auto-answer, provider direct)
  - Sub: S7.1 benchmark scripts, S7.2 HyDE patterns expansion, S7.3 provider direct
- **S8**: Streaming + Parallelization (SSE endpoint, merged router, model swap)
  - Sub: S8.1 SSE backend `/query/stream`, S8.2 merged router with Pydantic, S8.3 generation model swap
- **S9**: Caching + Observability (semantic cache, OpenTelemetry tracing)
  - Sub: S9.1 semantic cache SQLite, S9.2 OTel tracing, S9.3 (optional) pre-compute FAQ

#### Expanded scope (2026-05-24) — Standardize FE/BE
Source: `rag2025/.omc/plans/standardize_frontend_backend_plan_v2.md`

- **S10**: Frontend foundation (code splitting, OpenAPI codegen, Lighthouse CI, security)
  - Sub: S10.1 Vite manualChunks, S10.2 OpenAPI types codegen + CI gate, S10.3 Lighthouse CI, S10.4 CSP/XSS hardening
- **S11**: Backend modular refactor (`main.py` 1156 LOC → 8 modules)
  - Sub: S11.1 skeleton + tests, S11.2 migrate `/health` `/` `/debug`, S11.3 migrate `/v2/graph` `/v2/query`, S11.4 migrate `/query` (last)
- **S12**: i18n + Streaming integration (vi/en, locale-aware guardrails, SSE consumer)
  - Sub: S12.1 react-i18next vi/en, S12.2 BE prompts + locale field, S12.3 locale-aware guardrails, S12.4 FE SSE consumer with fallback

### Files in scope

#### Original phase-1 (Colab eval)
- `packages/rag-chatbot-husc/src/notebooks/eval_core.py`
- `packages/rag-chatbot-husc/src/notebooks/colab_eval.ipynb`
- `packages/rag-chatbot-husc/src/notebooks/colab_eval_v5_thesis.ipynb`
- `packages/rag-chatbot-husc/tests/notebooks/test_eval_core.py`

#### Latency reduction (S7-S9)
- `rag2025/scripts/benchmark_provider_latency.py` (new)
- `rag2025/scripts/benchmark_streaming_support.py` (new)
- `rag2025/src/services/query_router.py` (HyDE patterns + merged router)
- `rag2025/src/services/llm_client.py` (chat_stream method, connection pool)
- `rag2025/src/services/llm_generator.py` (locale-aware guardrails)
- `rag2025/src/services/semantic_cache.py` (new — S9.1)
- `rag2025/src/observability/tracing.py` (new — S9.2)
- `rag2025/tests/services/test_*.py` (1:1 mapping)
- `rag2025/.env` (feature flags)

#### Standardize FE/BE (S10-S12)
**Frontend (`uni-guide-ai/`)**:
- `src/App.tsx` (lazy loading)
- `src/main.tsx`, `src/lib/api.ts`, `src/lib/api-types.ts` (OpenAPI codegen output)
- `src/i18n/config.ts`, `src/i18n/locales/{vi,en}/*.json` (new)
- `src/components/chat/ChatLayout.tsx` (SSE consumer + i18n)
- `vite.config.ts` (manualChunks + CSP)
- `.lighthouserc.json` (new)
- `package.json` (scripts: types:gen, prebuild)

**Backend (`rag2025/src/`)**:
- `main.py` (1156 LOC → ≤ 100 LOC)
- `app/lifecycle.py`, `app/middleware.py`, `app/dependencies.py`, `app/state.py`, `app/i18n.py` (new)
- `routers/{health,root,debug,graph,unified_query,query,stream,admin}.py` (new)
- `models/{query,unified,chunk}.py` (new)
- `observability/{metrics,tracing,rate_limit}.py` (new)
- `prompts/generation_system_prompt_{vi,en}.txt` (new)
- `tests/{app,routers,models,observability}/test_*.py` (1:1 mapping)

**Cross-cutting**:
- `.github/workflows/lighthouse.yml` (new)
- `.github/workflows/api-contract.yml` (new)
- `api/openapi.json` (generated)
- `nginx.conf` (SSE timeout config)

### Out of scope (current phase-1)

- Backend pipeline LLM provider migration beyond what plans specify
- E2E tests against live Colab environment (notebook-runtime only)
- Production deployment to HUSC infrastructure (separate phase)
- Pre-compute top 100 FAQ optimization (deferred to Sprint 3 optional, S9.3)
- Next.js App Router rewrite (rejected per ADR Section 10 of standardize plan; Q3 2026 re-eval)
- Tiếng Trung/Lào multi-language (out of scope; future phase)
- Migration to managed vector DB (Pinecone/Qdrant) — deferred, SQLite first

### Hard constraints (do NOT violate)

- **C1**: Giữ `max_tokens=1500` ở generation. User đã tune. (latency plan A1 REMOVED)
- **C2**: 86Q vi `has_answer ≥ 97.7%` — HARD GATE block PR
- **C3**: 10Q en `has_answer ≥ 80%` (Sprint 3)
- **C4**: p50 latency tăng > 5% so baseline → block PR
- **C5**: Hot-path purity — không thêm middleware nặng vào `/query`
- **C6**: No mock Gateway in E2E
- **C7**: Reuse Qwen3-Embedding-0.6B, không thêm model dependency
- **C8**: TDD V5-R030 — mỗi src file có test 1:1
- **C9**: SSE streaming MUST có 2-pass design (stream raw → buffer → guardrails → emit `replace`)
- **C10**: `/query` migrate LAST trong Sprint 2 (riskiest cuối)

### Conflict resolution

Plan latency Sprint 2 (SSE B1) **trùng** Plan standardize Sprint 3 (B4 streaming).
→ **Plan standardize là authoritative** cho streaming spec (event format, nginx config, FE consumer fallback).
Plan latency reference plan standardize cho streaming details.

### Success criteria (phase-1 completion)

- [ ] 86Q vi eval: has_answer ≥ 97.7%, p50 < 15s production timeout
- [ ] 10Q en eval: has_answer ≥ 80%
- [ ] Lighthouse Performance ≥ 90 (Sprint 3 final)
- [ ] `main.py` ≤ 100 LOC, modular routers + tests 1:1
- [ ] TTFT < 6s p95, streaming working with fallback
- [ ] Semantic cache hit rate > 30% on replay
- [ ] OpenTelemetry traces visible in Jaeger/Langfuse
- [ ] Independent review gate passed (reviewer + verifier + critic, V5-R055)

### Roadmap reference

6 sprints total, ~14 days effort. See:
- `rag2025/.omc/plans/latency_reduction_plan_v2.md` — Sprints 1-3 (latency)
- `rag2025/.omc/plans/standardize_frontend_backend_plan_v2.md` — Sprints 1-3 (standardize)
- `rag2025/.omc/plans/temporal_reingest_plan.md` — S13 temporal correctness (8 phases, ~14 days)

Sprints chạy parallel với decision tree từ Phase 0 verification.

---

## S13 — Temporal correctness (added 2026-05-25)

Source of truth: `rag2025/.omc/plans/temporal_reingest_plan.md` (Critic-APPROVED v2.1).

### Sections implemented

- **S13.1**: Audit + snapshot baseline (Phase 0)
- **S13.2**: Schema canonicalization v3 + dual-read adapter + grep gate (Phase 1)
- **S13.3**: 3-way chunker pipeline + arbiter + idempotency gate (Phase 2)
- **S13.4**: Blue-green reingest + atomic `/admin/reload-table` + RequestTracker (Phase 3)
- **S13.5**: Temporal router + retriever hard filter + NER year entity (Phase 4)
- **S13.6**: Generation grounding + remove anti-fallback-retry + major code validator (Phase 5)
- **S13.7**: Frontend year banner + `/api/meta` endpoint (Phase 6)
- **S13.8**: Yearly rotation + auto-trigger 2027 + Slack notify + freshness alert (Phase 7)

### Files in scope

See plan §6 "Files touched". Honors phase-1 paths in `spec-index.yaml`.

### Hard constraints (in addition to C1-C10)

- **C11**: Whitelist ngành 2026 = CHỈ trong đề án id=74. Ngành ngoài id=74 → graceful "không tuyển 2026" message.
- **C12**: LLM judge cho chunker arbiter = `deepseek-v4-pro` (qua UnifiedLLMClient, temperature=0).
- **C13**: Lifelong audit retention cho mọi `husc_v{year}_legacy` (KHÔNG drop after 30 days).
- **C14**: 2027 lifecycle auto-trigger: crawler detect `id > max_known_id` HOẶC content match `\b2027\b` → fire `new_year_detected` event → `yearly_rotation.py` chạy auto.

### Acceptance gates

- 86Q replay: Type-1 ≤ 3, Type-4 = 0, Type-5 ≤ 3, Type-6 = 0.
- p50 latency ≤ 60s.
- Idempotency: 2 consecutive Phase 2 runs → identical manifest hash.
- Reload SLA: < 30s total (drain ≤ 25s + swap ≤ 1s + healthcheck ≤ 4s).
- 30Q synthetic temporal: recall@5 ≥ 0.9 current/historical; ambiguous fallback < 30%.
- Grep gate: 0 raw `metadata["source"]` / `metadata.get("source")` access (use `_get_source_label`).
