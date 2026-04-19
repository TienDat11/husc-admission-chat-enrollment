# Colab Eval Parity Fix Plan v4 (Consensus)

> Ralplan consensus plan — fixes notebook `colab_eval.ipynb` to include mandatory comparison matrix, route parity, and pipeline coverage.

## RALPLAN-DR

### Principles
1. **Inference over raw delta**: pass/fail dựa trên CI95 bootstrap / non-inferiority, không chỉ delta thô.
2. **Slice-first safety**: kết luận tổng chỉ hợp lệ khi slice bắt buộc đạt power + multiple comparisons correction.
3. **Machine-checkable governance**: artifact schema version-pin, evidence lineage checksum.
4. **Notebook determinism contract**: restart kernel + run all, no hidden state, clean-env rerun.
5. **Escalation before ambiguity**: rollback trigger + thiếu evidence phải có authority/SLA/reopen conditions.

### Decision Drivers (Top 3)
1. Auditability của kết luận parity/improvement.
2. Regression containment speed.
3. Scope-fit cho vòng fix notebook (không over-scope).

### Chosen Option
**Option A: Notebook-only hard-gate** — mở rộng `colab_eval.ipynb` trực tiếp.
- Pros: đúng scope, triển khai nhanh, artifact thống nhất.
- Cons: cần kỷ luật cell/metadata.
- Alternative rejected: Option B (external orchestrator) — vượt scope/effort hiện tại.

---

## User Stories

### US1 — Mandatory Comparison Matrix (BGE/Harrier × GraphRAG/PaddedRAG)

**Files:** `packages/rag-chatbot-husc/src/notebooks/eval_core.py`, `colab_eval.ipynb`

**Acceptance Criteria:**
1. Notebook chạy đủ 4 tổ hợp: (BGE, padded_rag), (BGE, graph_rag), (Harrier, padded_rag), (Harrier, graph_rag).
2. Mỗi tổ hợp có `run_id`, `embedding_model`, `force_route`, `n_queries`, `timestamp`.
3. Thiếu bất kỳ tổ hợp nào → FAIL-FAST.
4. Bảng tổng hợp bắt buộc: accuracy, groundedness, hallucination_rate, recall, latency_p95.

### US2 — Route Parity via force_route + Auto-route Shadow

**Files:** `packages/rag-chatbot-husc/src/notebooks/eval_core.py`, `colab_eval.ipynb`

**Acceptance Criteria:**
1. Controlled lane: gọi `/v2/query` với `force_route="padded_rag"` và `force_route="graph_rag"`.
2. Parity lane: gọi `/v2/query` KHÔNG ép route (auto-route) trên cùng query set.
3. Route mismatch report: log requested vs actual route; global mismatch ≤ 1.0%.
4. Per-slice mismatch ≤ 2.0%.

### US3 — Metrics & Artifact Schema Contract

**Files:** `packages/rag-chatbot-husc/src/notebooks/eval_core.py`, `colab_eval.ipynb`

**Acceptance Criteria:**
1. Output files mỗi run: `eval_predictions_<run_id>.jsonl`, `eval_scored_<run_id>.jsonl`, `eval_summary_<run_id>.csv`.
2. Required fields per scored record: `question`, `answer`, `ground_truth_answer`, `category`, `score_exact`, `recall`, `hallucination`, `groundedness_score`, `route`, `latency_ms`, `embedding_model`, `run_id`.
3. Summary CSV required columns: `category`, `route`, `embedding`, `count`, `accuracy`, `recall`, `hallucination_rate`, `groundedness_avg`, `latency_p95`.
4. Winner-by-metric table + overall recommendation. Conflict resolution rule: if must-pass metrics disagree on winner (e.g., BGE wins accuracy but Harrier wins recall), report "no clear winner" and list per-metric winners; do NOT force a single winner when must-pass metrics conflict.

### US4 — Decision Table (Must-pass vs Support-only)

**Acceptance Criteria:**
1. Pre-registered decision table trước khi chạy eval:

| Metric | Gate Type | NI Margin | Rationale |
|---|---|---|---|
| accuracy (EM) | must-pass | -1.0 pp | Core quality — below this = user-visible regression |
| groundedness | must-pass | -2.0 pp | Faithfulness — below = hallucination risk |
| hallucination_rate | must-pass | +2.0 pp | Safety — above = trust erosion |
| recall | must-pass | -1.5 pp | Retrieval coverage — below = missing facts |
| latency_p95 | support-only | +20% | Performance — degradation noted but doesn't block |
| partial_credit | support-only | -5.0 pp | Informational — tracks nuance |

2. Margins locked pre-run; post-hoc adjustment = PROTOCOL_VIOLATION.
3. PASS requires ALL must-pass metrics pass at every mandatory slice.
4. Support-only fail generates mitigation ticket with owner + ETA.

### US5 — Reproducibility + Determinism Contract

**Acceptance Criteria:**
1. Seed chuẩn: `42`; rerun seeds: `[42, 43, 44]`.
2. Rerun subset: `min(N, max(50, N * 0.2))` where `N = len(rows)` from `load_test_questions()` (the loaded test_questions.json dataset).
3. Minimum 3 reruns.
4. Stable when: `stddev(accuracy) ≤ 0.5pp`, `stddev(groundedness) ≤ 0.7pp`.
5. Determinism: restart kernel + run all x2 → metric drift ≤ 0.1pp.
6. Evidence map must pin: `git_sha`, `dataset_hash`, `config_snapshot`, `runtime_info`.

### US6 — Diagnostic Report Enrichment

**Files:** `packages/rag-chatbot-husc/src/notebooks/eval_core.py`

**Acceptance Criteria:**
1. `diagnostic_report.md` includes:
   - Executive summary with matrix results.
   - Per-route breakdown (padded_rag vs graph_rag).
   - Per-embedding breakdown (BGE vs Harrier).
   - Category × route × embedding cross-table.
   - Top 5 worst queries per configuration.
   - Roadmap cải tiến.
2. Parity checklist section: PASS/FAIL per contract item.

---

## Verification Matrix

| Requirement | Method | Threshold | Artifact |
|---|---|---|---|
| Matrix completeness | Count combos | 4/4, each has data | manifest per run |
| Route parity | Mismatch calculator | global ≤1.0%, slice ≤2.0% | route_parity_report |
| Metric integrity | Null/NaN checker | 0 nulls in must-pass | scored JSONL |
| Decision table | Pre-run lock check | Exists before eval starts | decision_table.json |
| Reproducibility | 3-seed rerun | stddev within thresholds | rerun_stability |
| Determinism | Restart+run x2 | drift ≤0.1pp | determinism_check |
| Evidence completeness | Checksum verifier | All 4 checksums present | evidence_map |
| Report sections | Section checker | All 6 sections present | diagnostic_report.md |

## Gate Formula

```
PASS iff:
  matrix_complete(4/4)
  AND all_must_pass_metrics_pass(per slice)
  AND route_parity_mismatch ≤ thresholds
  AND reproducibility_stable
  AND determinism_pass
  AND evidence_complete
  AND report_sections_complete
```

Any single fail → overall FAIL.

## ADR

**Decision:** Notebook-only hard-gate (Option A) with 6 user stories covering matrix, parity, metrics, decision table, reproducibility, and report enrichment.

**Drivers:** Auditability, regression containment, scope fit.

**Alternatives:** External orchestrator (Option B) — rejected for scope/effort.

**Consequences:** Stricter initial fail rate; higher confidence in conclusions.

**Follow-ups:** If governance failures repeat >2 cycles, upgrade to Option B (CI-native evaluator).

---

## Known Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Kernel hidden state | Results not reproducible across runs | US5: restart kernel + run all x2 + clean-env rerun; HIDDEN_STATE_FAIL gate |
| API route drift | `force_route` ignored or auto-router overrides | US2: route mismatch report; fail if global >1.0% mismatch |
| Bootstrap instability | CI intervals unreliable with small N | US5: minimum 50 samples per slice-arm; UNDERPOWERED flag if insufficient |
| Embedding model unavailable | One variant cannot run → incomplete matrix | US1: FAIL-FAST if any of 4 combos missing; pre-flight check cell |
| Config/env drift between runs | Artifacts not comparable | US5: evidence_map pins git_sha, dataset_hash, config_snapshot, runtime_info |

### Rollback Trigger Conditions (P5 elaboration)
- **Trigger**: any must-pass metric fails at any mandatory slice AND evidence_map incomplete.
- **Authority**: QA Lead declares containment; Engineering Manager approves rollback.
- **SLA**: 30min declare → 2h minimum evidence → 8h final decision.
- **Reopen**: full gate re-pass + evidence complete + sign-off from QA Lead + Eng Manager.

---

## Implementation Checklist (Model Tier Rules)

### For small models (minimax m2.7 / equivalent):
- [ ] Matrix A/B embeddings present (BGE vs Harrier)
- [ ] Route A/B via `force_route` present
- [ ] Evidence-first: every claim has command + output
- [ ] Parity checklist PASS/FAIL in report
- [ ] Known Risks + Next Fix Order section
- [ ] No `git add -A` — only stage target files

### For large models (Sonnet/Opus):
- [ ] All small-model rules above
- [ ] Anti-overengineering: no scope beyond spec
- [ ] Blast-radius gate before commit
- [ ] Claim → evidence traceability mapping
- [ ] Postmortem at session end (1 error + 1 prevention)
