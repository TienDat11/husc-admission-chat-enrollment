# Ultrawork Report — Metric Hardening + Empty-Retrieval Guard

**Date:** 2026-06-08 | **Flow:** discover(sonnet ×2) → synthesis+plan(opus) → executor(sonnet ×2) → review(opus, self) → report(opus)
**Commits:** `fb40dd8`, `4cb7384`, `e0f8251`, `c92ecf1` (author TienDat11, no co-author)

## What the multi-dimensional QA actually found

The earlier UltraQA pass flagged 2 "CRITICAL/HIGH" weaknesses. Discovery (2 parallel sonnet agents, read-only) + my own re-verification proved **both are measurement artifacts, not pipeline bugs**:

1. **precision@5 = 0.20 is the GT ceiling, not a defect.** `fact_level_gt_v2.json` has mean **1.09** supporting chunks/question (50/55 Q have exactly 1) → mathematical max p@5 = 1.09/5 = **0.218**. The RAG achieves 0.20 = **92% of the achievable ceiling**. The real quality levers are p@1 = 0.582 and MRR = 0.741. "Fixing" p@5 by tuning the reranker would have chased a phantom and risked regressing recall (r@5 = 0.945).

2. **msg019/msg020 are correct clarifications, not hallucinations.** Both queries are vague fragments ("Trường có xét tổ hợp.....không?", "Ngành... xét những tổ hợp nào ạ?"). Route `hyde_auto_answer` deliberately returns a "hãy nói rõ hơn" clarification with 0 chunks — that is correct behaviour. The metric mislabelled it `empty_retrieval / CRITICAL`.

## What was genuinely fixed (committed `c92ecf1`)

- **T2 — empty-retrieval abstain guard** (`llm_generator.py:759`, the one real defect): a *substantive* answer produced with **0 retrieved chunks** is now replaced by the standard abstain string. The vague-clarification path is explicitly exempt (`_CLARIFY_MARKERS`). TDD: `test_empty_retrieval_guard.py` 3 cases green. This closes the residual hallucination risk the contact-keyword guard didn't cover.
- **T1 — honest metric classification** (`compute_offline_metrics.py`): robust abstain/clarification detection; `hyde_auto_answer` clarifications reclassified out of CRITICAL (`true_empty_retrieval = 0`); added `precision_ceiling` so the dashboard reports "p@5 0.20 = 92% of ceiling 0.218" instead of implying a defect. Ranked-weakness list now drops the 7 artifact rows.

## Genuine signal surfaced during review (NOT a code bug)

**7 abstain "misses" are real data-coverage gaps**, not pipeline faults. Questions where the user expected an answer but the corpus genuinely lacks it, so the system correctly abstained:
- msg011 (thí sinh tự do tốt nghiệp 2004 — học bạ?), msg039 (ưu tiên THPT vs GDTX), msg040 (thí sinh tự do không thi lại), msg045 (chuyển điểm năng khiếu vẽ từ trường khác), msg082 (hết hạn mà chưa duyệt minh chứng), + msg019/020 (vague).

Abstaining here is **correct, safe behaviour** — the fix is *content* (ingest these admission edge-cases into the corpus), not code. Logged for the data team.

## Verification (self-run, gateway-free)
- `test_empty_retrieval_guard + test_offline_metrics + test_abstain_hardening + test_booster_year_durability` → **23 passed**.
- Metric re-run: context_recall 1.0 (n=65), p@1 0.582, MRR 0.741, ceiling@5 0.218, true_empty_retrieval **0**.

## Residual / follow-ups (no code change needed now)
- Latency: route_ms = 50.6% of wall time (G4 cache already cuts repeats); deeper fix = embedding-similarity pre-gate (deferred, design-first).
- hybrid over-routing 53/86 — monitor.
- Corpus: ingest the 7 edge-case admission topics above to convert correct-abstains into correct-answers.
- abstain gate 0.95 unrealistic at n=7 — re-spec at larger n.

**Verdict:** RAG generation/retrieval has **no open correctness bug**. The 4 original UltraQA groups (year time-bomb, /v2 contract, router cache, eval gates) are fixed and committed; this ultrawork pass corrected the measurement to tell the truth and closed the one genuine hallucination edge.
