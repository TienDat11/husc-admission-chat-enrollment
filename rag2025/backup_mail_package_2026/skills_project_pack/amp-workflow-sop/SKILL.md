---
name: amp-workflow-sop
description: >
  Use when running or improving multi-agent delivery in Amp CLI, especially for
  tasks requiring deterministic triage, planning-to-execution handoff, worker
  orchestration, escalation control, MCP research routing, git safety preflight,
  and Opus request-budget optimization.
---

# AMP Workflow SOP — Main Entry

> **Two core principles** (never violate):
> 1. **Planning**: Sub-agents collect → Opus receives **1 Batch Report** → creates plan **1 time**.
> 2. **Execution**: Opus spawns **1 Execution Orchestrator** → steps back. Only re-enters for L3/L4 or final sign-off.

---

## Load Order (follow in this order)

| Step | Situation | Load |
|------|-----------|------|
| 1 | Start any task | This file (mandatory) |
| 2 | Any coding / file-changing task | `./git.md` preflight first |
| 3 | Entering Planning phase (Phase 1) | `./planning.md` |
| 4 | Entering Execution phase (Phase 2) | `./execution.md` |
| 5 | Research/tool routing needed | `./tools.md` |
| 6 | Optimizing Opus request count | `./request-budget.md` |

---

## Routing Invariant (MANDATORY)

- All non-human handoffs route to: `main-agent@amp.local`
- Workers and Execution Orchestrator never message Opus directly
- Completion, escalations, and status updates must follow the same status schema

---

## Tool Routing — BẮT BUỘC

### ❌ TUYỆT ĐỐI KHÔNG dùng
- `web_search` (Amp default)
- `read_web_page` / web-fetch style default webpage tools

### ✅ PHẢI dùng thay thế
| Nhu cầu | Tool bắt buộc |
|---|---|
| Tìm kiếm web, research, news | **Exa MCP** |
| Tra cứu docs thư viện / framework | **Context7 MCP** |
| Đọc nội dung URL cụ thể | **Exa MCP** (fetch/search context mode) |

### MCP Routing Discipline (hard constraint)
- Mọi web/docs routing phải đi qua Exa MCP + Context7 MCP theo đúng nhu cầu.
- Không được tự ý fallback sang default web tools.
- Nếu Exa hoặc Context7 không available: trả `BLOCKED` + nêu thiếu MCP nào + yêu cầu user bật MCP.
- Mục tiêu bắt buộc: giảm retry để giảm request count trong smart mode billed-per-request.

### Agent Mail (optional)
- Chỉ bật workflow mailbox (`main-agent@amp.local`, `orchestrator-*`, `worker-*`) khi **Agent Mail MCP đã được cấu hình thực tế**.
- Nếu chưa có Agent Mail MCP: giữ routing nội bộ trong thread, không giả lập gửi mail.


---

## Stage 0 — Triage (ALWAYS run this first)

Before anything else, classify the task:

``` 
FAST PATH only if ALL conditions are true:
1) Single-file change
2) ≤ 2h estimate
3) No cross-file or cross-package dependency changes
4) Acceptance criteria are explicit
5) No missing external constraints (versions/env/API contracts)

If any condition is false OR uncertain:
  └─→ FULL PATH: read ./planning.md → then ./execution.md
```

**First response must include:**
```
1) Path selected: FAST | FULL
2) Rationale: up to 2 bullets
3) Blocking ambiguity (if any): one line
```

### Preflight Gate (before spawning workers for code changes)

1) Confirm issue exists and scope is clear (`gh issue view <number>`)
2) Run git preflight from `./git.md`
3) Confirm current branch is NOT `main`

If any gate fails: stop and request user confirmation/input before spawning any worker.

**Fast Path output contract** (1 Opus request, no follow-up):
```
1) What I understood
2) Changes made (files + what changed)
3) How to verify
4) Risks / rollback if any
```

---

## System Architecture

```
PLANNING (Opus leads)                    EXECUTION (Orchestrator leads)
━━━━━━━━━━━━━━━━━━━━━━━━━━              ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Rush Workers (parallel)                  Execution Orchestrator (smart)
  └─ read codebase                         ├─ Rush Workers (leaf tasks)
  └─ gather docs                           ├─ Smart Workers (complex)
  └─ find patterns                         └─ Deep Workers (algorithms)
        ↓                                        ↓ (only L3/L4 or done)
  Batch Report ──→ Opus                    ↑ Opus re-enters
  Opus: synthesis                    
  Opus: plan (self-contained)        
        ↓                            
  Handoff Gate                       
```

**Agent types at a glance:**

| Agent | Mode | Call via | Role |
|-------|------|----------|------|
| Main Agent | smart (Opus 4.6) | default | Planning only; L3/L4 decisions; final sign-off |
| Execution Orchestrator | smart (Opus 4.6) | Task tool (spawn once) | Runs entire execution phase autonomously |
| Oracle | reasoning | oracle tool (direct) | Reviews, risk analysis — NOT a sub-agent |
| Librarian | internal | librarian tool (direct) | Research routing — NOT a sub-agent |
| Rush Worker | rush (Haiku 4.5) | Task tool + "mode use rush" | Discovery collection; small clear tasks |
| Smart Worker | smart (Opus 4.6) | Task tool | Complex tasks needing full tools |
| Deep Worker | deep (GPT-5.3 Codex) | Task tool + "mode use deep" | Algorithms, extended reasoning |

> **Critical**: Oracle and Librarian are **tools**, never sub-agents.
> Mode must be **explicitly set** in spawn prompt — sub-agents don't inherit parent mode.

---

## Opus-Minimization Policy (default)

- Daily hard cap context: **150 Opus requests/day** (resets daily)
- Reserve policy: keep **20 requests** as safety buffer for incident/L4
- Default to **LEAN FLOW** for most epics: max 2 Opus planning requests + 1 execution spawn request
- Use **FULL FLOW** only for high-risk/ambiguous epics
- Oracle review is **conditional**, not mandatory (trigger only on L3/L4 risk, unsafe migration, or unresolved architecture trade-off)
- Never spend an Opus request on file discovery or trivial code edits (delegate to Rush/Smart/Deep workers)

### V3 HARDCORE — Plan Sovereignty Protocol

- **Smart mode/Opus plan is the single source of truth** after plan generation
- Workers (rush/smart/deep) are executors, not planners
- No worker may change scope, files, approach, or acceptance criteria outside the plan
- If plan is missing detail: worker must return `BLOCKED` (never improvise)
- Any deviation requires L3 approval via `main-agent@amp.local` before continuing
- Execution Orchestrator must apply Plan Compliance Gate before accepting `DONE`

```
Fast Path:      1
Small Epic:     2–3
Medium Epic:    3–4
Large Epic:     4–6
```

See `./request-budget.md` for techniques.

---

## Escalation Path

```
Worker → Execution Orchestrator → (L3/L4 only) → Opus → Human
```

| Level | Who handles | SLA |
|-------|-------------|-----|
| L1 Minor | Worker self-resolve | 30 min |
| L2 Moderate | Execution Orchestrator | 1h |
| L3 Major (plan change needed) | Opus | 2h |
| L4 Critical (pivot) | Opus + Human | Immediate |

---

## Naming & Artifact Conventions

```
Track:   TRACK-{DOMAIN}-{NN}         e.g. TRACK-AUTH-01
Bead:    BEAD-{TRACK-ID}-{NNN}       e.g. BEAD-AUTH-01-001
Worker:  worker-{track}-{id}@amp     e.g. worker-auth-01@amp

.spike/
├── discovery-{ts}.md       ← Batch Report from Rush Workers
├── synthesis-{ts}.md       ← Opus synthesis + Oracle verdict
├── execution-plan-{ts}.md  ← Self-contained plan (worker types + spawn prompts)
├── progress-{ts}.md        ← Live updates by Execution Orchestrator
└── epic-completion-{ts}.md ← Final report
```
