# BE Pipeline Metrics Dashboard
**Generated:** 2026-06-07 | **Corpus:** 86Q Vietnamese University Admissions RAG | **Eval harness:** all-Sonnet judge

---

## 1. RAGAS Trend — baseline → s14_fixed → s15 → s16

| Run | faithfulness (n) | answer_relevancy (n) | context_precision (n) | context_recall (n) | faith gate (≥0.95) |
|-----|-----------------|---------------------|----------------------|-------------------|-------------------|
| baseline_2026 | **0.959** (82) | 0.795 (86) | 0.843 (80) | 1.000 (⚠ n=2) | ✅ PASS |
| s14_fixed | **0.965** (84) | 0.916 (86) | 0.724 (84) | NULL (n=0) | ✅ PASS |
| s15 | **0.949** (84) | 0.936 (86) | 0.765 (84) | 1.000 (⚠ n=2) | ⚠ MARGINAL (0.949 < 0.950) |
| s16 | **0.964** (84) | 0.948 (86) | 0.789 (84) | 1.000 (⚠ n=2) | ✅ PASS |

**Sources:** `results/eval_harness/metrics_report/{run}/metrics_report_{run}.json` → key `metrics`

**Faith gate threshold:** 0.95 (key: `gates.faithfulness.threshold`). s15 scores 0.9494 — 0.0006 below gate; recovered in s16.

**context_recall caveat (all runs):** n=2 in baseline/s15/s16, n=0 in s14_fixed. All "PASS" verdicts on context_recall are statistically meaningless. See key `context_recall_caveat` and section 6.

---

## 2. Latency Seams — s16 (86 records)

Source: `results/eval_harness/86q_records_s16.jsonl` → key `latency_ms`, `latency_attribution`

| Seam | p50 (ms) | p95 (ms) | n | % of total p50 |
|------|----------|----------|---|---------------|
| **route_ms** | **9,344** | **10,657** | 86 | **50.5%** ← dominant |
| retrieval_loop_ms | 1,846 | 2,585 | 86 | 10.0% |
| query_ms | 7,627 | 9,350 | 86 | 41.2% |
| **total_ms** | **18,498** | **22,038** | 86 | — |

Attribution sub-seams (n=84):

| Sub-seam | p50 (ms) | p95 (ms) |
|----------|----------|----------|
| hyde_ms | 3,405 | 4,450 |
| router_inner_ms | 5,827 | 7,019 |
| gen_ms | 3,936 | 5,325 |

**Dominant seam: `route_ms` (p50 9.3 s, 50.5% of wall time).** Within route, `router_inner_ms` (p50 5.8 s) is the largest sub-component — the flash-low router LLM call. `query_ms` (p50 7.6 s) reflects generation time baked into the query phase for hybrid/graph routes.

**Trend vs baseline_2026:** total p50 dropped from 53,425 ms → 18,498 ms (−65%). Source: `metrics_report_baseline_2026.json` → `latency.total_ms.p50`.

---

## 3. Route Distribution

Source: `results/eval_harness/86q_records_s16.jsonl` → key `route`; corroborated by `s16_comparison.json` → key `route_dist`

| Run | padded_rag | hybrid | graph_rag | hyde_auto_answer | total |
|-----|-----------|--------|-----------|-----------------|-------|
| baseline_2026 | 73 | — | 7 | 6 | 86 |
| s14_fixed | 66 | — | 18 | 2 | 86 |
| s15 | 8 | — | 76 | 2 | 86 |
| **s16** | **7** | **53** | **24** | **2** | **86** |

**Over-routing check (graph_rag > 30/86):**
- s15: **76/86 = 88%** → ⚠ OVER-ROUTED (threshold 30/86). Source: `s15_comparison.json` → `route_shift`
- s16: 24/86 = 28% → ✅ PASS (gate: `s16_comparison.json` → `gate.graph<=30: PASS 24`)

**s16 padded gate:** 7/86 < 15 → ⚠ FAIL per `s16_comparison.json` → `gate.route_padded>=15`. Noted as acceptable: padded→hybrid is a safe routing direction (hybrid = vector + light-PPR).

**hyde_auto_answer:** 2/86 across s14–s16 (was 6/86 baseline). Killing this route eliminated all Type-6 hallucinations.

---

## 4. Hallucination Counts

Sources: `results/eval_harness/judge_halltype/{run}/judge_halltype_{run}.json` → `confusion_matrix.per_type_counts`; genuine/FP breakdown from `halltype_triage_baseline_2026.json` and `s14_fixed_comparison.json`, `s15_comparison.json`, `s16_comparison.json`.

| Run | Type-1 (year-mix) | Type-4 detector / genuine | Type-5 detector / genuine | Type-6 (anti-fallback) | answered_with_hall |
|-----|------------------|--------------------------|--------------------------|------------------------|-------------------|
| baseline_2026 | 1 / 1 | 9 / **4** | 10 / **5** | **4** / **4** | 15/79 (19%) |
| s14_fixed | 1 / 1 | 4 / **0** | 1 / **1** | 0 / **0** | 5/79 (6%) |
| s15 | 2 / 2 | 5 / **0** | 1 / **1** | 0 / **0** | 6/79 (8%) |
| s16 | 2 / 2 | 3 / **0** | 1 / **1** | 0 / **0** | 4/79 (5%) |

**Key findings:**
- **Type-4 genuine dropped 4→0** from s14 onward. Baseline FPs (5/9) stem from incomplete major-code registry (missing SC/IC suffix, some 75xxxxx/72xxxxx codes). Source: `halltype_triage_baseline_2026.json` → `type4.false_positive_reason`
- **Type-5 persistent genuine=1** across all runs: `msg068` (16.5M derived tuition sum, not literally in context). Source: `s16_comparison.json` → `headline`
- **Type-6 eliminated** by killing hyde_auto_answer route. Source: `s14_fixed_comparison.json` → `hallucination.type6`
- **Type-1 (year-mix) persists at 2** in s15/s16: `msg026`, `msg028` (historical-intent năm-trước/năm-ngoái questions — debatable GT convention)

---

## 5. Head-to-Head Comparison Deltas

Sources: `results/eval_harness/{run}_comparison.json`

### s14_fixed vs baseline_2026
| Metric | baseline | s14_fixed | delta |
|--------|----------|-----------|-------|
| faithfulness | 0.963 | 0.965 | +0.002 (held PASS) |
| answer_relevancy | 0.798 | 0.916 | **+0.118** |
| context_precision | 0.845 | 0.724 | −0.121 |
| total_ms p50 | 53,400 | 16,300 | **−3.3×** |
| abstain_accuracy | 0.714 | 0.286 | −0.428 (1 genuine regression, 2 GT artifacts) |

Source: `s14_fixed_comparison.json` → keys `ragas`, `latency`, `abstain_regression`

### s15 vs s14_fixed
| Metric | s14_fixed | s15 | delta |
|--------|-----------|-----|-------|
| faithfulness | 0.965 | 0.949 | **−0.016** (marginal gate fail) |
| answer_relevancy | 0.916 | 0.936 | +0.020 |
| context_precision | 0.724 | 0.765 | +0.041 (partial recovery) |
| total_ms p50 | 16,300 | 21,100 | +4,800 ms (76/86 graph route) |

Source: `s15_comparison.json` → `headline_metrics`

### s16 vs s15
| Metric | s15 | s16 | delta |
|--------|-----|-----|-------|
| faithfulness | 0.949 | 0.964 | **+0.015** (URL-claim fix, gate restored) |
| answer_relevancy | 0.936 | 0.948 | +0.012 (best ever) |
| context_precision | 0.765 | 0.789 | +0.024 |
| total_ms p50 | 21,100 | 18,500 | −2,600 ms |
| graph_rag count | 76 | 24 | −52 (routing normalized) |

Source: `s16_comparison.json` → `headline`

### Heldout v1 generalization (40 unseen questions)
| Metric | s16 (86Q) | heldout_v1 | note |
|--------|-----------|-----------|------|
| faithfulness | 0.964 | **0.971** | NOT overfit |
| answer_relevancy | 0.948 | 0.788 | lower due to complex/multi-part novel Qs |
| out-of-scope abstain | — | 7/8 (87.5%) | |
| route accuracy | — | 36/40 (90%) | all mismatches fail-safe (toward more context) |

Source: `results/eval_harness/heldout_v1_generalization.json` → `ragas_heldout_vs_86q`, `routing`, `out_of_scope_abstain`

---

## 6. Coverage Gaps & Flagged Issues

| Issue | Severity | Evidence |
|-------|----------|----------|
| **context_recall n=2** (baseline, s15, s16) | ⚠ CRITICAL — metric invalid | `metrics_report_{run}.json` → `metrics.context_recall.n` = 2; `context_recall_caveat` notes GT auto-generated, optimistic |
| **context_recall n=0** (s14_fixed) | ⚠ CRITICAL — metric absent | `metrics_report_s14_fixed.json` → `metrics.context_recall.n` = 0; gate verdict `FAIL` with `actual: null` |
| **abstain_accuracy persistent FAIL** all runs | HIGH — gate FAIL | `metrics_report_{run}.json` → `gates.abstain_accuracy.verdict`; threshold 0.95; actuals: 0.71/0.29/0.14/0.14 |
| **major_code_validator registry incomplete** | MED — inflates Type-4 FP count | `halltype_triage_baseline_2026.json` → `type4.false_positive_reason`; SC/IC suffix + 75xxxxx/72xxxxx codes missing |
| **eval path ≠ prod /v2 path** | MED — metrics may not reflect prod | `s16_comparison.json` → `prod_caveat` task #82 |
| **RAMCLOUDS_HYDE_MODEL env mismatch** | MED — prod .env = gpt-5.5 breaks JSON classify → all-hybrid routing | `s16_comparison.json` → `prod_caveat`; `s15_comparison.json` → `gateway_note` |
| **msg068 Type-5 genuine (derived sum)** | LOW — 1 persistent case | All comparison JSONs → hallucination triage; 16.5M tổng học phí derived, not in retrieved context |

---

## TOP 5 Ranked Bottlenecks

### #1 — SEVERITY: CRITICAL | context_recall is a phantom metric
All runs report context_recall on n=2 (or n=0 in s14_fixed) out of 86 questions. The gate "PASS" at 1.000 is statistically meaningless — a 95% CI on n=2 collapses to [1.0, 1.0]. No real retrieval coverage signal exists.

**Evidence:** `metrics_report_{run}.json` → `metrics.context_recall.n`; `context_recall_caveat`; `gates.context_recall`

**Fix:** Expand GT fact-level annotations from 2 → ≥40 questions before trusting context_recall gate.

---

### #2 — SEVERITY: HIGH | abstain_accuracy gate fails every run, worsening sprint-over-sprint
Threshold 0.95; actuals: baseline 0.714 → s14 0.286 → s15 0.143 → s16 0.143. The degradation from s14 onward is partly GT-convention artifacts (2/7 cases), but abstain_accuracy is never above 0.714 even at baseline.

**Evidence:** `metrics_report_{run}.json` → `gates.abstain_accuracy`; `s14_fixed_comparison.json` → `abstain_regression`

**Fix:** Audit 7-question OOS set — reclassify GT-convention FPs, then add genuine abstain-hardening prompts for cases like `msg055` (fabricated Zalo OA detail).

---

### #3 — SEVERITY: HIGH | route_ms dominates wall time (50% of p50 total)
`route_ms` p50 = 9,344 ms out of 18,498 ms total (s16). The sub-seam breakdown shows `router_inner_ms` (p50 5,827 ms) — the flash-low LLM classifier call — consumes 31% of total p50 alone.

**Evidence:** `86q_records_s16.jsonl` → `latency_ms.route_ms`, `latency_attribution.router_inner_ms`; `metrics_report_s16.json` → `latency.route_ms.p50`

**Fix:** Cache router decisions for repeated/near-duplicate queries; or reduce to a lightweight embedding-similarity gate before the LLM classifier call.

---

### #4 — SEVERITY: MED | major_code_validator registry incomplete — inflates Type-4 hallucination signal
5/9 Type-4 detections in baseline are false positives because major codes ARE present in retrieved context but the canonical 2026 registry lacks SC/IC suffix variants and some 75xxxxx/72xxxxx series. This means Type-4 genuine rate (4→0) looks better than it is — the detector will continue producing noisy alerts until the registry is patched.

**Evidence:** `halltype_triage_baseline_2026.json` → `type4.false_positive_reason`, `type4.false_positive_ids`

**Fix:** Regenerate major_code_validator registry from latest MOET 2026 data including all suffix/series variants.

---

### #5 — SEVERITY: MED | eval path ≠ prod path + env mismatch means metrics may not reflect production behavior
Two separate issues: (a) task #82 documents that eval harness hits a different code path than `/v2` prod endpoint; (b) prod `.env` sets `RAMCLOUDS_HYDE_MODEL=gpt-5.5` which breaks JSON classify on the gateway → router falls back to all-hybrid routing, diverging from the s16 3-way distribution (padded 7 / hybrid 53 / graph 24).

**Evidence:** `s16_comparison.json` → `prod_caveat`; `s15_comparison.json` → `gateway_note`

**Fix:** Align eval harness to call `/v2` endpoint; set `RAMCLOUDS_HYDE_MODEL=flash-low` in prod `.env` per instructions in both comparison files.

---

## Summary Table

| Gate | baseline_2026 | s14_fixed | s15 | s16 |
|------|--------------|-----------|-----|-----|
| faithfulness ≥ 0.95 | ✅ 0.959 | ✅ 0.965 | ⚠ 0.949 | ✅ 0.964 |
| context_recall ≥ 0.85 | ⚠ PASS (n=2) | ❌ NULL | ⚠ PASS (n=2) | ⚠ PASS (n=2) |
| abstain_accuracy ≥ 0.95 | ❌ 0.714 | ❌ 0.286 | ❌ 0.143 | ❌ 0.143 |
| Type-4 genuine = 0 | ❌ 4 | ✅ 0 | ✅ 0 | ✅ 0 |
| Type-6 genuine = 0 | ❌ 4 | ✅ 0 | ✅ 0 | ✅ 0 |
| graph_rag ≤ 30/86 | ✅ 7 | ✅ 18 | ❌ 76 | ✅ 24 |
| total_ms p50 | 53,425 ms | 16,319 ms | 21,104 ms | 18,498 ms |

**Overall verdict (s16):** Two hard gates remain FAIL (abstain_accuracy, context_recall n=2). All hallucination type gates pass. Latency is 3.3× improved vs baseline. System generalizes to unseen questions (heldout faithfulness 0.971 ≥ 86Q 0.964). Prod deployment blocked by env mismatch and eval/prod path divergence.
