# Planning Phase (Phase 1 — "Dọn Cỗ")

> Opus's job in planning: **receive a Batch Report, think once, output a complete self-contained plan.**
> Sub-agents collect. Opus synthesizes and decides. Never the other way around.

---

## Stage 1.1 — Discovery via Batch Collection

**Who does what:**
- Rush Workers (parallel, 2–5 depending on scope) → read codebase, docs, patterns
- Opus → does NOT read files itself; waits for the Batch Report

**Spawn Rush Workers with this template:**
```
You are Discovery Worker {ID} covering domain: {DOMAIN}.

FIRST: mode use rush

Mailbox: worker-discovery-{id}@amp.local
Report to: main-agent@amp.local

Collect findings on: {specific scope — files, APIs, patterns}
Use Librarian tool for multi-repo or Exa for OSS examples.

Return a structured report section:
## Worker-{ID}: {Domain}
- Key files: [list]
- Current state: [description]
- Constraints found: [list]
- Open questions: [list]

Do NOT write code. Collect only.
Send mail STATUS=DONE with your report section.
```

**Batch Discovery Report format** (workers each contribute a section):
```
# Batch Discovery Report – {timestamp}

## Worker-1: [Domain]
- Key files: ...
- Current state: ...
- Constraints: ...
- Open questions: ...

## Worker-2: [Domain]
...

## Unresolved Open Questions (for Opus)
- ?
```

Save to: `.spike/discovery-{timestamp}.md`

---

## Stage 1.2 — Synthesis + Decomposition + Final Plan (Opus, 1 request)

Opus receives the full Batch Discovery Report and returns a **single self-contained plan package** in one shot.
Oracle review is optional and only used when risk triggers apply.

This plan package is **authoritative** for all downstream workers.
No worker is allowed to reinterpret strategy after this stage.

**What Opus outputs**:
- `.spike/synthesis-{timestamp}.md`
- `.spike/execution-plan-{timestamp}.md`
- `.spike/ctx-summary-{timestamp}.md`  ← Dense context snapshot cho review session

```
## Synthesis Status
APPROVED | NEEDS_INPUT | NEEDS_REVISION

## Unified Understanding
[what the system is and what the problem requires]

## Recommended Approach
[chosen solution + why]

## Risk Matrix
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| ...  | ...        | ...    | ...       |

## Assumptions Confirmed
[list]

## Blocking Questions (max 3; only if NEEDS_INPUT)
- ?

## Oracle Verdict (optional; include only if risk-triggered)
APPROVED | NEEDS_REVISION — [rationale]

## Bead Assignments
Use this mandatory bead contract for every bead:
- Bead: `BEAD-{TRACK}-{NNN}`
- Worker type: `rush | smart | deep`
- Input: exact files/inputs
- Output: exact expected artifacts
- Done criteria: checkable conditions
- Estimated time: `<=4h` preferred (`<=8h` max)
- Dependencies: explicit list or `none`

Good bead example:
`BEAD-AUTH-01-001 | rush | Input: src/auth/login.ts | Output: login.ts with rate-limit middleware | Done: tests pass + 429 after 5 attempts | ETA: 1.5h | Dependencies: none`

Bad bead example:
`BEAD-AUTH-01-001 | Task: improve auth`

## Critical Path
[ordered list of sequential dependencies]

## Parallel Groups
Group A (can start immediately): ...
Group B (after Group A): ...

## Pre-written Spawn Prompts
[spawn prompt per track, copy-paste ready]

## L3/L4 Escalation Triggers
Escalate to `main-agent@amp.local` when:
- [specific condition 1]
- [specific condition 2]

## Plan Sovereignty Rules (mandatory for all workers)
- PLAN_ID and PLAN_VERSION must be present in every worker prompt
- Workers execute only assigned bead contract (no scope expansion)
- If task is ambiguous/missing: send `BLOCKED` with missing field list
- Do not invent files/APIs/steps not present in the plan
- Any plan change requires L3 approval before execution
```

### ctx-summary schema (mandatory — output này bắt buộc dù Opus hay GPT làm planning)

Lưu vào: `.spike/ctx-summary-{timestamp}.md`

```
# Context Summary — {timestamp}
# ⚡ Dense snapshot. Đọc file này là đủ để review plan mà không cần load lại toàn bộ discovery report.

## Codebase Snapshot
current_state: |
  [Mô tả ngắn gọn trạng thái hiện tại của codebase liên quan đến epic — tối đa 200 từ]

key_files:
  - path: src/xxx/yyy.ts
    role: [vai trò trong epic]
  - ...

tech_stack:
  runtime: [Node/Python/etc + version]
  framework: [...]
  relevant_deps: [chỉ liệt kê deps liên quan đến epic]

## Constraints & Invariants
- [constraint 1 — non-negotiable]
- [constraint 2]

## Decisions Made (không được reopen trong review)
- DECISION-001: [tên quyết định] → [lý do ngắn gọn]
- DECISION-002: ...

## Open Questions Resolved
- Q: [câu hỏi ban đầu] → A: [câu trả lời đã xác nhận]

## Risk Register (summary)
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| ... | ... | ... | ... |

## Plan Metadata
plan_id: PLAN-{timestamp}
plan_version: v1
total_beads: N
critical_path_length: N beads
parallel_groups: N groups
estimated_opus_requests: N (planning) + N (execution spawn) = N total
```


**Few-shot example — lean synthesis prompt to Opus (default):**
```
Here is the Batch Discovery Report: [report content]

Return a complete plan package in one response:
- status (APPROVED | NEEDS_INPUT | NEEDS_REVISION)
- synthesis + risk matrix
- bead assignments (full contract per bead)
- critical path + parallel groups
- pre-written spawn prompts
- L3/L4 escalation triggers

Use Oracle only if high-risk ambiguity remains. If status is NEEDS_INPUT, include max 3 blocker-critical questions.
```

---

### ⚡ GPT Planning Mode (khi dùng GPT-4.5 Pro thay Opus để tiết kiệm request)

Khi dùng GPT-4.5 Pro để tạo plan thay vì Opus:

**GPT phải output đủ 3 files:**
1. `.spike/synthesis-{timestamp}.md` — synthesis + risk matrix (format giống như Opus)
2. `.spike/execution-plan-{timestamp}.md` — full plan (format chuẩn, xem schema ở trên)
3. `.spike/ctx-summary-{timestamp}.md` — **BẮT BUỘC** theo ctx-summary schema ở trên

**Prompt template cho GPT-4.5 Pro:**
```
Here is the full Batch Discovery Report: [paste full report]

Your job: produce a complete plan package in one response.

OUTPUT 3 FILES with these exact headers:
--- FILE: .spike/synthesis-{timestamp}.md ---
[synthesis + risk matrix]

--- FILE: .spike/execution-plan-{timestamp}.md ---
[full plan with bead contracts, spawn prompts, L3/L4 triggers]

--- FILE: .spike/ctx-summary-{timestamp}.md ---
[dense context snapshot following the ctx-summary schema exactly]

Rules:
- ctx-summary must be self-contained — a reviewer reading ONLY this file + the execution plan
  must have 100% of context needed to evaluate correctness, risk, and completeness
- Do NOT summarize the plan in ctx-summary — it summarizes the CONTEXT (codebase state,
  constraints, decisions, resolved questions)
- ctx-summary target size: 300–600 words maximum
- Bead contracts must be fully specified (input/output/done-criteria/ETA/dependencies)
- Status: APPROVED | NEEDS_INPUT | NEEDS_REVISION
```

**Sau khi GPT output xong:** chạy skill `review-planning` (xem Stage 1.5).

---

## Stage 1.3 — Verification (conditional, no extra Opus by default)

Only run this stage if Stage 1.2 returns `NEEDS_INPUT` or includes unverified technical assumptions.

- Prefer Rush/Smart workers for quick probes (API checks, package availability, schema checks)
- Use Oracle only when architectural risk remains unresolved after probes
- Do **not** call Opus again unless verification changes the chosen architecture materially

Save to: `.spike/verification-{timestamp}.md`

---

## Stage 1.4 — Handoff Packaging (no Opus)

Main Agent prepares final artifacts for Execution Orchestrator without additional Opus turns.

Required artifacts:
- `.spike/synthesis-{timestamp}.md`
- `.spike/execution-plan-{timestamp}.md`
- `.spike/handoff-gate-{timestamp}.md`

### Mandatory handoff artifact schema

Create `.spike/handoff-gate-{timestamp}.md`:
```
status: READY_FOR_EXECUTION | BLOCK
plan_id: PLAN-{timestamp}
plan_version: v1
checks:
  - all_beads_have_input_output_done_criteria: PASS|FAIL
  - no_circular_deps: PASS|FAIL
  - oracle_verdict: APPROVED|NOT_REQUIRED|NEEDS_REVISION
  - plan_self_contained: PASS|FAIL
  - escalation_triggers_defined: PASS|FAIL
  - plan_sovereignty_rules_embedded: PASS|FAIL
blocking_reason: ""
ready_for_execution: true|false
```

Execution phase can start only when `ready_for_execution: true`.

### Circular dependency check

Use repo-defined dependency command first (from project docs/scripts). If absent, use an equivalent checker in the repo and record command/output in `.spike/verification-{timestamp}.md`.

---

## Stage 1.5 — Plan Review Gate (Opus, 1 request — kích hoạt bằng keyword)

> Chỉ chạy stage này khi plan được tạo bởi GPT hoặc khi có nghi ngờ về chất lượng plan.
> **Không tự động chạy** — kích hoạt bằng cách gõ: `review plan` hoặc `review planning`

**Đọc skill `review-planning` để thực hiện.**

Preconditions:
- `.spike/ctx-summary-{timestamp}.md` tồn tại
- `.spike/execution-plan-{timestamp}.md` tồn tại

**Opus input (1 request):**
```
Load: .spike/ctx-summary-{ts}.md + .spike/execution-plan-{ts}.md

Review the plan against the context summary.
Follow review-planning skill.
```

**Output choices:**
- `APPROVED` → Cập nhật `.spike/handoff-gate-{ts}.md` với `plan_review: APPROVED`; tiếp tục Execution
- `NEEDS_REVISION` → List specific issues; GPT hoặc Opus sửa lại; re-review nếu cần

**Budget impact:**
- APPROVED path: +1 Opus request (review only)
- NEEDS_REVISION path: +1 review + 1 revision = +2 Opus requests max

---

## LEAN vs FULL Planning Flow (Opus budget control)

### LEAN FLOW (default)
- Stage 1.1 Discovery (workers)
- Stage 1.2 Single Opus plan package
- Stage 1.3/1.4 packaging and gate
- Stage 1.5 Plan Review Gate (chỉ khi GPT làm planning)
- **Opus planning requests: 1 (nếu Opus làm planning) | 0+1 review (nếu GPT làm planning)**

Use LEAN when:
- Scope is clear
- Risk is low/medium
- No migration of core architecture

### FULL FLOW (exception path)
Use only if one of these is true:
- Stage 1.2 returns `NEEDS_REVISION` twice
- Safety/data-integrity risk remains unresolved
- Critical architecture pivot is required

FULL adds one extra Opus revision turn after verification.
- **Opus planning requests: 2 (max for most epics)**

---

## Pivot / Escalation Signals

Go back to Stage 1.2 (or enter FULL FLOW) when:
- Verification spike fails > 2 times
- Oracle identifies unmitigated critical risk
- Critical path exceeds timeline constraints

