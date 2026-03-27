# Execution Phase (Phase 2 — "Ăn Cỗ")

> Opus's job in execution: **spawn 1 Execution Orchestrator, then stop.**
> Orchestrator handles everything. Opus re-enters only for L3/L4 blocker or final sign-off.

---

## Stage 2.0 — Spawn Execution Orchestrator (Opus, 1 request max by default)

This is Opus's **only request** in the execution phase (unless L3/L4 arises).

Precondition:
- `.spike/handoff-gate-{timestamp}.md` exists with `ready_for_execution: true`

**Execution Orchestrator spawn template:**
```
You are the Execution Orchestrator for Epic {EPIC-ID}.

Load Orchestrator Skill.
Your mailbox: orchestrator-{epic-id}@amp.local
Report to main-agent@amp.local ONLY for: L3/L4 blocker | epic completion

Authoritative plan metadata:
- PLAN_ID: {plan-id}
- PLAN_VERSION: {plan-version}

Your execution plan (fully self-contained):
{full content of .spike/execution-plan-{timestamp}.md}

Plan sovereignty contract (mandatory):
- The plan is the single source of truth for all workers
- Do NOT reinterpret scope/architecture/acceptance criteria
- If any bead lacks required detail, mark BLOCKED and escalate with missing fields
- No worker may execute unplanned files/APIs/steps
- Any plan change requires L3 approval before work continues

RESPONSIBILITIES (in order):
1. Register worker mailboxes via Agent Mail MCP
   Format: worker-{track}-{id}@amp.local
2. Spawn workers per plan — spawn prompts are pre-written in the plan
3. Monitor Agent Mail; update bead status: PENDING→IN_PROGRESS→DONE/BLOCKED/FAILED
4. Handle L1/L2 blockers autonomously (see escalation triggers in plan)
5. Spawn next parallel group when dependencies clear
6. When all beads DONE:
   - generate .spike/epic-completion-{timestamp}.md
   - send final report to main-agent@amp.local
7. If any bead can be downgraded from Smart → Rush without losing correctness, do it before spawn

DO NOT ask Opus for clarification — the plan is self-contained.
DO NOT ping Opus for L1/L2 — resolve yourself.

L3 ESCALATION (send when plan change needed):
  TO: main-agent@amp.local
  SEVERITY: L3
  PLAN_ID: {plan-id}
  PLAN_VERSION: {plan-version}
  BEAD: {id}
  SITUATION: {description}
  PROPOSED SOLUTION: {your recommendation}
  AWAITING: approval

L4 ESCALATION (halt everything, send immediately):
  TO: main-agent@amp.local
  SEVERITY: L4 — ALL WORKERS PAUSED
  PLAN_ID: {plan-id}
  PLAN_VERSION: {plan-version}
  SITUATION: {full description}
  AWAITING: pivot decision
```

---

## Stage 2.1 — Worker Spawn Templates (used by Execution Orchestrator)

### Rush Worker

```
You are Rush Worker {ID} for {BEAD-ID}.

FIRST: mode use rush

Mailbox: {worker-mailbox}
Report to: orchestrator-{epic-id}@amp.local
PLAN_ID: {plan-id}
PLAN_VERSION: {plan-version}

Task: {bead-description}
Input files: {exact list}
Expected output: {exact output}
Done criteria: {checkable conditions}

Rules:
- Read ONLY listed files
- Execute ONLY assigned bead contract
- Do NOT spawn sub-agents
- Do NOT ask questions
- If any required detail is missing: STATUS=BLOCKED (never improvise)
- Do not invent files/APIs/steps outside plan
- Prefer deterministic edits/checks over analysis
- On finish: mail STATUS=DONE|BLOCKED|FAILED + output summary
```

### Smart Worker

```
You are Smart Worker {ID} for {BEAD-ID}.

Load Worker Skill.
Mailbox: {worker-mailbox}
Report to: orchestrator-{epic-id}@amp.local
PLAN_ID: {plan-id}
PLAN_VERSION: {plan-version}

Task: {bead-description}
Input: {bead-input}
Expected output: {exact output}
Done criteria: {checkable conditions}

Rules:
- Execute ONLY assigned bead contract
- No scope expansion or strategy rewrite
- If any required detail is missing: STATUS=BLOCKED (never improvise)
- Do not invent files/APIs/steps outside plan

On finish: mail STATUS=DONE|BLOCKED|FAILED + output + artifacts list
```

### Deep Worker

```
You are Deep Worker {ID} for {BEAD-ID}.

FIRST: mode use deep

Mailbox: {worker-mailbox}
Report to: orchestrator-{epic-id}@amp.local
PLAN_ID: {plan-id}
PLAN_VERSION: {plan-version}

Task: {bead-description}
Problem context: {full context + constraints}
Expected output: {exact output}
Done criteria: {checkable conditions}

Instructions:
- Reason deeply, but execute ONLY within assigned bead contract
- No scope expansion or architecture pivot without L3 approval
- If required detail is missing: STATUS=BLOCKED (never improvise)
- Do not invent files/APIs/steps outside plan
- Document reasoning in output
- On finish: mail STATUS=DONE|BLOCKED|FAILED + output + reasoning summary
```

---

## Stage 2.2 — Monitor & Handle Blockers

**Execution Orchestrator's loop:**
```
Poll Agent Mail
  ↓
DONE received?
  └─ Mark bead DONE; check if next group can start
BLOCKED received?
  └─ Assess: L1 (give context, retry) | L2 (coordinate workers) | L3/L4 (escalate)
FAILED received?
  └─ L1/L2: consult Oracle tool inline, retry with adjusted context
     L3/L4: escalate
```

**Deterministic recovery policy:**
- attempt <= 2: Orchestrator resolves L1/L2 autonomously
- 2nd BLOCKED/FAILED on same bead: treat as L2 and coordinate explicitly
- Any blocker requiring plan change/library/version swap: L3 escalation
- Any blocker affecting safety/data integrity/rollback reliability: L4 escalation + pause affected workers

**Worker mail format (mandatory):**
```
FROM: {worker-mailbox}
PLAN_ID: {plan-id}
PLAN_VERSION: {plan-version}
STATUS: DONE | BLOCKED | FAILED
BEAD: {bead-id}
ATTEMPT: {n}
EVIDENCE: {file:line | log excerpt}
OUTPUT: {artifacts / actual output}
BLOCKER: {if BLOCKED — specific description}
MISSING_FIELDS: {if BLOCKED by plan ambiguity}
NEXT: {suggested next action}
```

**L3/L4 escalation to main-agent format:**
```
TO: main-agent@amp.local
SEVERITY: L3 | L4
BEAD: {id}
SITUATION: {description}
PROPOSED SOLUTION: {recommendation}   ← always include for L3
WORKERS PAUSED: yes | no
AWAITING: decision
```

---

## Plan Compliance Gate (anti-hallucination, mandatory)

Before accepting any `STATUS=DONE`, Execution Orchestrator must verify:
1. `PLAN_ID` + `PLAN_VERSION` match active execution plan
2. Touched files are within bead Input/Output contract
3. Done criteria are explicitly evidenced
4. No unplanned API/library/scope change occurred

If any check fails:
- Mark bead `BLOCKED`
- Return `MISSING_FIELDS` or `DEVIATION_DETECTED`
- Escalate L3 for approval/replan (do not merge worker output directly)

---

## Stage 2.3 — Epic Completion

**Execution Orchestrator:**
- Verify all beads DONE
- Run integration check if specified in plan
- Update `.spike/progress-{timestamp}.md` on every status transition (PENDING→IN_PROGRESS→DONE/BLOCKED/FAILED)
- Generate `.spike/epic-completion-{timestamp}.md`
- Send final report to `main-agent@amp.local`

**Completion report format:**
```
# Epic Completion Report – {timestamp}
Epic: {EPIC-ID}
Beads completed: X/X
Artifacts created: [list with paths]
Known limitations: [list]
Suggested follow-ups: [list]
Opus requests used: N
```

**Main Agent final sign-off (1 request):**
- Review completion report
- Confirm no unresolved issues
- Human sign-off if needed

---

## Few-Shot Examples

### ✅ Good — Execution Orchestrator handling L2 blocker

Worker BEAD-AUTH-01-003 reports BLOCKED: "Cannot find the rate-limit config schema."

Execution Orchestrator:
1. Checks plan — no mention of config schema location
2. Calls Librarian tool: "Find rate-limit config schema in auth module"
3. Librarian returns: `src/config/rate-limit.schema.ts`
4. Sends worker follow-up: "Config schema is at src/config/rate-limit.schema.ts"
5. Worker retries → DONE
6. Does NOT ping Opus.

### ❌ Bad — Execution Orchestrator pinging Opus for L1

"Worker is blocked on a minor import path issue. Escalating to Opus."
→ This is L1. Orchestrator should resolve it directly or give worker the correct path.

### ✅ Good — L3 escalation format

```
TO: main-agent@amp.local
SEVERITY: L3
BEAD: BEAD-AUTH-01-005
SITUATION: The planned JWT library (jsonwebtoken v9) is incompatible with
  the Node 22 environment. Cannot install — peer dep conflict.
PROPOSED SOLUTION: Switch to jose v5 which is ESM-native and Node 22 compatible.
  Affected beads: BEAD-AUTH-01-005, BEAD-AUTH-01-006 (need updated imports)
WORKERS PAUSED: yes (those 2 beads)
AWAITING: approval to switch library
```
