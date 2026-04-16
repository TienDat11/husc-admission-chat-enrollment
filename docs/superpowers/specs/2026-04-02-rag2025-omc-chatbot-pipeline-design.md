# OMC Chatbot Refactoring Pipeline Design (rag2025)

**Date:** 2026-04-02  
**Scope:** `rag2025/` only  
**Objective:** Execute full OMC pipeline from `PROMPT.md` with evidence-backed outputs, prioritizing **answer quality > cost > latency**.

## 1) Design Decision Summary

We will use **Approach 1: Full pipeline with hard quality gates per stage**.

Why this approach:
- Maximizes output quality while preventing drift from repo reality.
- Ensures each recommendation in final YAML is traceable to code evidence and/or vetted external research.
- Controls risk by stage-level pass/fail checks before downstream synthesis.

## 2) Execution Architecture

### Stage A — Self-Review (Code-first)
Primary mode: `/oh-my-claudecode:sciomc AUTO`

Inputs:
- `rag2025/` source, config, scripts, tests, docs.

Outputs:
- `chatbot-self-review.md` with:
  - current architecture summary,
  - 5–7 critical issues,
  - performance bottleneck analysis,
  - security-relevant findings.

Gate A (must pass):
- Architecture representation exists and maps to actual code paths.
- 5–7 issues are concrete, evidence-based, and impact-ranked.
- Bottleneck claims reference observed implementation behavior.

### Stage B — Research (External + relevance filter)
Primary mode: `/oh-my-claudecode:sciomc` with Exa-backed evidence gathering.

Research themes:
- GraphRAG vs traditional/hybrid RAG for structured admissions content.
- Multi-turn state management.
- Vietnamese retrieval/reranking and query rewriting.
- LLM routing strategies for quality/cost balance.

Source policy:
- Prefer official technical documentation and highly cited/high-visibility engineering writeups.
- Require cross-source agreement for critical architectural recommendations.
- Record rejected alternatives with reasons.

Outputs:
- `chatbot-research-exa.md` with source-ranked findings and applicability notes for `rag2025`.

Gate B (must pass):
- Each major recommendation has at least two independent sources.
- Vietnamese admissions applicability is explicitly argued.
- “Not selected” options are documented to avoid hidden assumptions.

### Stage C — Synthesis into executable plan
Primary mode: `/oh-my-claudecode:team`

Outputs:
- `.omc/plans/chatbot-refactor-plan.yaml` fully populated (no placeholders), preserving PROMPT schema.

Synthesis rules:
- Every target-state choice must map to current-state issue(s).
- Phase tasks are ordered by impact on response quality first.
- Cost controls are explicit (model-tier routing, selective heavy components).

Gate C (must pass):
- YAML schema complete and executable.
- Clear mapping from problems → interventions → measurable gate metrics.
- No unresolved placeholders (`TODO`, `TBD`, template braces).

### Stage D — Validation and feasibility challenge
Primary mode: `/oh-my-claudecode:ask` (Codex/Gemini challenge pass)

Outputs:
- `chatbot-validation.md` containing:
  - feasibility checks (dependency/runtime/integration),
  - quality-cost-latency trade-off review,
  - residual risks and mitigations.

Gate D (must pass):
- External challenge feedback is reconciled into the plan.
- Remaining risks are explicit and actionable.
- Final recommendation still honors priority order: quality > cost > latency.

## 3) Artifact Contract

All outputs under `.omc/plans/`:

1. `chatbot-self-review.md`
2. `chatbot-research-exa.md`
3. `chatbot-refactor-plan.yaml`
4. `chatbot-validation.md`

Traceability rule:
- Each high-impact YAML recommendation includes a pointer to either:
  - codebase evidence from Stage A, and/or
  - external source rationale from Stage B.

## 4) YAML Structure Commitments

The final YAML keeps prompt-defined blocks:
- `plan_metadata`
- `current_state`
- `target_state`
- `implementation_phases`
- `quality_gates`
- `omc_execution_commands`
- `skills_activation`

Population policy:
- Replace all template placeholders with concrete values for `rag2025`.
- Keep metrics measurable and admissions-context aware.
- Prioritize quality-improving tasks early (retrieval fidelity, reranking quality, multi-turn handling).

## 5) Operational Constraints

- Scope locked to `rag2025/`.
- Must run full PROMPT pipeline end-to-end.
- Exa research required, with credibility/popularity filtering.
- Any recommendation conflicting with repository constraints is rejected or down-scoped.

## 6) Risks and Mitigations

- Risk: research-plan mismatch with existing stack.  
  Mitigation: Gate A and Gate C require code-grounded mapping.

- Risk: over-engineering from broad SOTA ideas.  
  Mitigation: include explicit “not selected” section and feasibility challenge.

- Risk: quality gains with uncontrolled cost.  
  Mitigation: model-tier routing and staged rollout in implementation phases.

## 7) Completion Definition

Design is considered complete when:
- All four artifacts exist,
- YAML is fully concrete and executable,
- validation document confirms feasibility,
- quality-first priority is preserved across phase ordering and gate thresholds.

## Execution Outputs (2026-04-02)
- `.omc/plans/chatbot-self-review.md`
- `.omc/plans/chatbot-research-exa.md`
- `.omc/plans/chatbot-refactor-plan.yaml`
- `.omc/plans/chatbot-validation.md`
