# Request Budget Optimization (Opus 4.6, billed per request)

> Context: Shared provider, cost = number of turns with Opus (smart mode).
> Goal: maximize output quality per Opus request. Delegate everything delegatable.

---

## Core Techniques

### 1. Single-shot with full context upfront

Every Opus request should include:
- All relevant file paths (don't make Opus guess)
- Error logs / stacktraces in full
- Expected output format
- Constraints and environment details

**Example — good debugging prompt (1 request):**
```
File: src/auth/middleware.ts
Error: TypeError: Cannot read property 'userId' of undefined
  at line 42, in verifyToken()
Stack: [full stacktrace]

Steps to reproduce: POST /api/login with valid credentials

Environment: Node 22, Express 5, JWT library v9
Expected: middleware passes req.user to next()

Return: diagnosis + exact fix + verification step
```

**Example — bad (wastes 3+ requests):**
```
My auth middleware is broken
[no file, no error, no context]
```

---

### 2. Standard Output Contract

Always request this structure for complex Opus tasks:
```
1) Diagnosis / analysis — confirmed understanding + assumptions
2) Execution plan       — ordered steps (self-contained)
3) Code changes         — ready to apply, no placeholders
4) Verification steps   — how to confirm it works
5) Risks & rollback     — what could go wrong + how to undo
6) Status               — APPROVED | NEEDS_INPUT | NEEDS_REVISION
```

Add this to the start of important requests:
```
MODE: request-budget
POLICY: process all context in one pass, return complete answer.
Ask clarification only for blocker-critical missing info.
OUTPUT: diagnosis → plan → implementation → verification → risks → status
```

---

### 3. Batch related tasks

Instead of: 3 separate requests for auth, session, and user profile modules
Do: "Review and fix issues in auth/middleware.ts, session/store.ts, and users/profile.ts. Return full patch for all three."

---

### 4. Delegate to Rush Workers

If it's small, clear, and unambiguous → Rush Worker, not Opus.

| Task type | Use |
|-----------|-----|
| Read and summarize 5 files | Rush Workers → Batch Report → Opus |
| Add a type annotation to a function | Rush Worker |
| Design the auth architecture | Opus |
| Convert 10 CSS files to Tailwind | Rush Worker × 10 (parallel) |
| Debug a complex race condition | Opus or Deep Worker |

---

### 5. Context Window Notes

| Spec | Value |
|------|-------|
| Default context | 200K tokens |
| 1M context (beta) | Requires `anthropic-beta: context-1m-2025-08-07` header at provider level |
| Max output | 128K tokens |
| Effective in Amp (without beta header) | ~168K |

If you need 1M context: must be enabled at provider/gateway config level, not via prompt content.

---

## Request Budget by Epic Size

| Epic | Planning (Opus) | Execution (Opus) | Total |
|------|----------------|-----------------|-------|
| Fast Path | 1 | 0 | 1 |
| Small (2–4 beads) | 1 | 1 spawn + 0–1 L3/L4 | 2–3 |
| Medium (5–10 beads) | 1–2 | 1 spawn + 0–1 L3/L4 | 3–4 |
| Large (10+ beads) | 2 | 1 spawn + 1–3 L3/L4 | 4–6 |

### Daily Cap Policy (150 Opus/day)

- Daily hard cap: **150**
- Reserve 20 for emergencies (L4/pivot/hotfix)
- Normal operating target: **<=130/day**
- If projected usage >130: enforce LEAN flow + split epics + downgrade more beads to Rush

If you're projecting > 6 Opus requests on one epic: split into smaller epics or downgrade more beads to Rush/Deep workers.

---

## Opus Request Ladder (mandatory)

Before using another Opus request, ask in this order:
1. Can a Rush worker gather/verify this instead?
2. Can the Execution Orchestrator resolve this as L1/L2?
3. Can Oracle review this inline without another Opus planning turn?
4. Is this truly a plan change or final sign-off?

If the answer is "no" to #4, do **not** spend the Opus request.

---

## A/B Request Simulation (old vs V3 hardcore)

Assume 1 medium epic (8 beads, mixed complexity):

| Workflow | Planning Opus | Execution Opus | Total Opus |
|----------|----------------|----------------|------------|
| Legacy (multi-pass planning) | 3 | 1 spawn + 1 L3 avg | 5 |
| V3 LEAN (default) | 1 | 1 spawn + 0–1 L3 | 2–3 |

Estimated savings per medium epic: **40–60%** Opus requests.

At 150/day cap:
- Legacy throughput: ~30 medium-epic equivalents/day
- V3 LEAN throughput: ~50–75 medium-epic equivalents/day

(Real usage depends on L3/L4 frequency; this table is planning baseline.)

---

## Daily Allocator (150/day)

- 20 reserved: incident/L4/hotfix
- 110 allocated: planned epics (LEAN default)
- 20 flexible: spillover or high-risk FULL-flow epics

If remaining daily Opus budget < 25:
- Freeze FULL flow
- Allow only Fast Path + in-flight L3 safety escalations
- Push non-critical planning to next reset

---

## Anti-Patterns That Waste Requests

```
❌ Prompt too short, no context → multiple follow-up turns needed
❌ Opus reads codebase itself instead of using Rush Workers batch
❌ Opus monitors execution instead of delegating to Execution Orchestrator
❌ Execution Orchestrator pings Opus for L1/L2 issues
❌ No output contract → response is incomplete → need to ask again
❌ Parallel tool calls for the same research query
❌ Spawning Oracle/Librarian as sub-agents (costs a spawn request + overhead)
```
