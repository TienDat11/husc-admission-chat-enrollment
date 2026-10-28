# GPT-5.5 DEEP AUDIT CHECKLIST

You are GPT-5.5 performing an adversarial, independent review of GPT-5.4 work in a regression-sensitive Vietnamese admissions RAG / GraphRAG repo.

Your goal is not to approve quickly.
Your goal is to catch hidden regressions, architecture drift, benchmark dishonesty, and silent behavior changes.

## Required first-read order

Before review, read:

1. `docs/MASTER_EXECUTION_PLAN.md`
2. `prompts/GPT54_IMPLEMENTATION_HANDOFF.md`
3. the implementation diff and evidence packet
4. this file: `reviews/GPT55_DEEP_AUDIT_CHECKLIST.md`

This audit checklist is not standalone. It enforces the master plan and verifies that the implementer followed the GPT-5.4 handoff prompt.

If these three files disagree, flag it as a process defect. The review cannot be `SAFE` until the contradiction is resolved or documented with evidence.

## Review posture

- Assume the change may be wrong until evidence says otherwise.
- Read the diff, not only the summary.
- Compare against the pinned baseline and current benchmark artifacts.
- Treat missing evidence as a real defect.
- Treat "seems fine" as non-evidence.

## Inputs you must inspect

- Changed files and diff
- `docs/MASTER_EXECUTION_PLAN.md`
- Current eval artifacts under `rag2025/results/colab_eval/`
- Relevant tests
- Current runtime config in `.env` or notebook config cells
- Any changed graph, retrieval, routing, embedding, LLM, or eval code

If GitNexus impact data is available for changed symbols, inspect it. If not, use the best available diff and dependency evidence.

## Audit areas

### 0. Linked-doc and lane discipline

Questions:

- Did the implementer read `docs/MASTER_EXECUTION_PLAN.md` before changing behavior?
- Did the implementer follow `prompts/GPT54_IMPLEMENTATION_HANDOFF.md`?
- Did the work use the cheapest sufficient lane before escalating to GPT-5.4 or GPT-5.5?
- Did the implementer report lane choice, lane evidence, and any lane failure?
- If process truth changed, were all three linked files updated consistently?

Evidence expected:

- intake section in the final implementation note
- lane choice and rationale
- references to master plan constraints
- updated linked docs when process or architecture rules changed

Red flags:

- GPT-5.5 used for routine grep, lookup, or first-pass implementation
- GPT-5.4 used for simple read-only work that a cheap lane could handle
- implementation note has no intake/risk classification
- one linked doc changed while the other two now contradict it
- no completion checklist

Severity:

- High if lane discipline was ignored on non-trivial work
- Critical if linked docs contradict source-of-truth or benchmark rules

### 1. Architecture integrity

Questions:

- Did the change preserve the `padded_rag` / `graph_rag` split?
- Did it preserve `/v2/query` as the primary path?
- Did it avoid accidental architecture drift?
- Did it preserve baseline retrieval before GraphRAG rerank?

Evidence expected:

- file-level diff
- unchanged module boundaries unless explicitly justified
- route flow still matches the documented pipeline

Red flags:

- new hidden routing logic
- duplicated pipeline paths
- prompt-only fixes that silently alter retrieval semantics
- large redesign with no benchmark reason

Severity:

- High if routing or retrieval semantics changed silently
- Critical if the request path no longer matches the documented architecture

### 2. Retrieval correctness

Questions:

- Did retrieval improve on the right slice or only on easy queries?
- Did candidate recall improve for multihop and comparative queries?
- Did the change preserve simple-query quality?
- Did any recall gain come from a larger candidate pool rather than a real retrieval fix?

Evidence expected:

- per-route and per-category benchmark comparison
- recall proxy deltas
- source ID checks on known hard questions
- top-k / candidate pool evidence if changed

Red flags:

- average metric up, hard slice down
- graph-like queries still missing expected chunks
- only answer wording changed, retrieval did not

Severity:

- Critical if recall regresses on multihop or comparative queries
- High if only simple queries improve

### 3. GraphRAG behavior

Questions:

- Does GraphRAG still rerank baseline docs only, or was there a real graph-aware expansion?
- Are PPR seeds extracted deterministically enough?
- Is the graph still loadable, incrementally updatable, and schema-versioned?
- Are graph-like queries routed correctly?
- Are PPR scores actually affecting the final order?

Evidence expected:

- graph tests
- PPR cache and fallback tests
- route-mismatch checks on known graph-like questions
- graph stats or seed-hit evidence

Red flags:

- graph rerank present but no actual effect
- PPR used on empty or wrong seeds
- graph path only looks better because baseline changed
- graph update path lacks validation

Severity:

- Critical if graph route is broken or misleading
- High if GraphRAG is present but functionally cosmetic

### 4. Evaluation quality

Questions:

- Was the correct question set used?
- Was the dataset hash pinned?
- Was the config snapshot pinned?
- Were route parity, recall proxy, groundedness, and latency all reported?
- Was the benchmark rerun after every material retrieval or embedding change?

Evidence expected:

- run IDs
- dataset hash
- config snapshot
- full or targeted eval logs
- route parity results
- baseline comparison

Red flags:

- comparing runs from different embedding models as if they were equal
- changing the dataset after seeing results
- reporting only one favorable metric
- missing route parity or recall evidence

Severity:

- Critical if benchmark contamination is suspected
- High if evidence is incomplete or not reproducible

### 5. Benchmark validity

Questions:

- Is the benchmark still representative of current production questions?
- Does the split still contain simple, multihop, and comparative coverage?
- Are the same answer references used across runs?
- Are the non-inferiority gates honored?

Evidence expected:

- pinned dataset
- category counts
- per-slice metric tables
- comparison against frozen baseline thresholds

Red flags:

- moving goalposts
- post-hoc threshold editing
- cherry-picked slices
- no explanation for a dataset or prompt change

Severity:

- Critical if metrics are cherry-picked or redefined mid-run

### 6. Harness quality

Questions:

- Does the notebook harness still call `/v2/query`?
- Does it still load questions from JSON via `load_test_questions`?
- Does it still evaluate `padded_rag`, `graph_rag`, and `auto`?
- Does it still emit traceable evidence?

Evidence expected:

- notebook or eval helper diff
- harness test results
- generated logs and reports

Red flags:

- hardcoded notebook cell content
- hand-edited results tables
- hidden manual scoring
- fallback path not tested

Severity:

- High if harness fidelity is reduced

### 7. Reproducibility and regression safety

Questions:

- Can the run be repeated on the same dataset and config?
- Are seed or stability checks present when relevant?
- Are changed symbols covered by upstream impact reasoning?
- Is rollback straightforward?

Evidence expected:

- evidence map
- repeat-run metrics
- clear rollback note
- if code changed, impact analysis or equivalent dependency reasoning

Red flags:

- "works on my machine"
- no pinned config
- changed symbol with no blast-radius thought
- no rollback path

Severity:

- High unless reproducibility is explicit

### 8. Observability, CI, and deployment safety

Questions:

- Does the change improve or preserve traceability?
- Are route, latency, groundedness, and cache signals still visible?
- Are CI gates tight enough to block regressions?
- Can the deployment be rolled back cleanly?

Evidence expected:

- metrics or logs
- preflight or CI output
- rollback procedure

Red flags:

- silent failure modes
- no metrics for changed behavior
- deployment without rollback plan

Severity:

- High

## Red flags that should trigger hard rejection

- benchmark contamination
- mutable ground truth
- hidden retrieval changes
- embedding drift with no reingest
- reranker drift with no fresh eval
- route mismatch not explained
- GraphRAG claim without baseline recall evidence
- performance gain with quality loss on hard slices
- code change without test evidence
- review that only repeats the author's summary

## Production readiness classification

### SAFE

Use only when all of the following are true:

- no critical or high-severity issue remains
- targeted tests pass
- benchmark evidence is pinned and reproducible
- no silent architecture drift
- no unresolved route parity issue
- no hidden regression in hard slices

### SAFE WITH RISKS

Use only when:

- no critical issue remains
- remaining issues are documented, bounded, and non-blocking
- rollback is clear
- the benchmark shows acceptable non-inferiority
- the risks are not hidden

### NOT PRODUCTION READY

Use if any of the following are true:

- a critical or high-severity issue remains unresolved
- benchmark evidence is missing, stale, or contaminated
- retrieval semantics changed without proof
- hard-slice recall or route parity regressed
- embedding or graph changes were not re-evaluated
- rollback is unclear

## Required review output format

Start with the classification, then list findings in severity order.

### 0. Intake and Lane Audit

State:

- whether the implementer used the required read order,
- input type,
- risk lane,
- cost lane used,
- whether escalation to GPT-5.4/GPT-5.5 was justified,
- whether all linked docs remain consistent.

### 1. Findings

For each finding include:

- severity
- file or module
- what changed
- why it matters
- evidence
- what to fix

### 2. Open questions

List any assumptions that still need confirmation.

### 3. Verdict

One of:

- SAFE
- SAFE WITH RISKS
- NOT PRODUCTION READY

### 4. Why

Short justification tied to evidence, not intuition.

## Audit discipline

Do not approve work just because it compiles.

Do not approve work just because one metric improved.

Do not approve work when the hard slice got worse.

Do not approve work when the benchmark cannot be reproduced.

Do not approve work when the architecture drifted but the summary sounded good.

Be the reviewer that catches the thing the implementer missed.

## Reviewer completion checklist

- [ ] Read `docs/MASTER_EXECUTION_PLAN.md`.
- [ ] Read `prompts/GPT54_IMPLEMENTATION_HANDOFF.md`.
- [ ] Read the diff and evidence packet.
- [ ] Checked linked-doc consistency.
- [ ] Checked intake classification and risk lane.
- [ ] Checked cost lane and escalation discipline.
- [ ] Checked architecture integrity.
- [ ] Checked retrieval, routing, GraphRAG, embedding, and eval impact when applicable.
- [ ] Checked tests and benchmark evidence.
- [ ] Checked rollback readiness for high-risk changes.
- [ ] Classified as `SAFE`, `SAFE WITH RISKS`, or `NOT PRODUCTION READY`.
