# RAG2025 OMC Pipeline Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chạy full pipeline theo `.omc/skills/omc-chatbot-plan-generator/PROMPT.md` trên phạm vi `rag2025/`, tạo bộ artifacts thực chiến và YAML refactor plan có thể triển khai ngay.

**Architecture:** Pipeline gồm 4 stage tuần tự: Self-review (code-first) → Research (Exa-first) → Synthesis (YAML) → Validation (cross-model challenge). Mỗi stage có gate pass/fail rõ ràng; fail thì quay lại stage trước để bổ sung bằng chứng. Orchestration dùng OMC skills (`sciomc`, `team`, `ask`) và teammate chạy song song theo domain độc lập.

**Tech Stack:** OMC skills, Claude Code teammates, Exa MCP, Markdown/YAML artifacts, Python repo validation commands.

---

## File Structure Map

- Modify: `docs/superpowers/specs/2026-04-02-rag2025-omc-chatbot-pipeline-design.md` (thêm link sang artifact kết quả thực thi)
- Create: `docs/superpowers/plans/2026-04-02-rag2025-omc-chatbot-pipeline-execution.md`
- Create: `.omc/plans/chatbot-self-review.md`
- Create: `.omc/plans/chatbot-research-exa.md`
- Create: `.omc/plans/chatbot-refactor-plan.yaml`
- Create: `.omc/plans/chatbot-validation.md`
- Verify against: `rag2025/src/**`, `rag2025/tests/**`, `rag2025/requirements.txt`, `rag2025/run_lab.bat`, `rag2025/setup_data.bat`

---

### Task 1: Preflight scope lock + baseline evidence

**Files:**
- Modify: none
- Test: `rag2025` structural checks only

- [ ] **Step 1: Lock working scope to rag2025**

Run:
```bash
ls "D:/chunking/husc-admission-chat-enrollment/rag2025"
```
Expected: thư mục `src`, `tests`, `requirements.txt`, `.env.example`, `run_lab.bat`, `setup_data.bat` xuất hiện.

- [ ] **Step 2: Snapshot current state before OMC runs**

Run:
```bash
git -C "D:/chunking/husc-admission-chat-enrollment" status --short
```
Expected: hiển thị diff hiện tại để không nhầm artifact mới với thay đổi cũ.

- [ ] **Step 3: Create artifact directory**

Run:
```bash
mkdir -p "D:/chunking/husc-admission-chat-enrollment/.omc/plans"
```
Expected: command thành công không lỗi.

- [ ] **Step 4: Commit checkpoint (optional if repo policy allows)**

```bash
git -C "D:/chunking/husc-admission-chat-enrollment" add .omc/plans
git -C "D:/chunking/husc-admission-chat-enrollment" commit -m "chore: prepare omc plan artifact directory"
```
Expected: commit tạo checkpoint preflight.

---

### Task 2: Stage A — Self-review bằng OMC sciomc AUTO

**Files:**
- Create: `.omc/plans/chatbot-self-review.md`
- Test: Gate A checklist inside file

- [ ] **Step 1: Dispatch teammate A chạy self-review**

Command payload (qua OMC skill flow):
```text
/oh-my-claudecode:sciomc AUTO "Self-review current chatbot codebase architecture under rag2025, identify 5-7 critical issues, performance bottlenecks, security-relevant findings"
```
Expected: trả về findings có mapping tới file/module thực.

- [ ] **Step 2: Normalize output vào artifact markdown**

Write `.omc/plans/chatbot-self-review.md` with this exact structure:
```markdown
# Chatbot Self Review (rag2025)

## Architecture Summary
- Entry points:
- Retrieval pipeline:
- Reranking pipeline:
- Generation/response orchestration:

## Critical Issues (5-7)
1.
2.
3.
4.
5.

## Performance Bottlenecks
- 

## Security-Relevant Findings
- 

## Gate A Verdict
- [ ] Architecture mapped to concrete files
- [ ] 5-7 issues evidence-backed
- [ ] Bottlenecks tied to implementation behavior
Status: PASS/FAIL
```
Expected: file có đủ 4 section + Gate A verdict.

- [ ] **Step 3: Run Gate A fail-fast check**

Run:
```bash
python - <<'PY'
from pathlib import Path
p = Path("D:/chunking/husc-admission-chat-enrollment/.omc/plans/chatbot-self-review.md")
s = p.read_text(encoding="utf-8")
required = ["## Architecture Summary", "## Critical Issues (5-7)", "## Performance Bottlenecks", "## Security-Relevant Findings", "## Gate A Verdict"]
missing = [x for x in required if x not in s]
print("PASS" if not missing else f"FAIL missing: {missing}")
PY
```
Expected: `PASS`.

- [ ] **Step 4: Commit self-review artifact**

```bash
git -C "D:/chunking/husc-admission-chat-enrollment" add .omc/plans/chatbot-self-review.md
git -C "D:/chunking/husc-admission-chat-enrollment" commit -m "docs: add rag2025 omc self-review artifact"
```
Expected: artifact stage A được version hóa độc lập.

---

### Task 3: Stage B — Exa-backed research với credibility filter

**Files:**
- Create: `.omc/plans/chatbot-research-exa.md`
- Test: Gate B checklist inside file

- [ ] **Step 1: Dispatch teammate B nghiên cứu theo 5 theme độc lập**

Command payload:
```text
/oh-my-claudecode:sciomc "Research best chatbot pipeline for Vietnamese university admission; compare GraphRAG vs hybrid RAG, multi-turn state management, Vietnamese retrieval/reranking, LLM routing for quality-first"
```
Expected: có nguồn ngoài và kết luận theo từng theme.

- [ ] **Step 2: Enrich bằng Exa search ưu tiên nguồn phổ biến/được review**

Run Exa queries (agent tool), gom kết quả vào file với bảng nguồn:
```markdown
## Sources (ranked)
| Rank | Title | Domain | Why trusted | Relevance to rag2025 |
|---|---|---|---|---|
```
Expected: mỗi recommendation lớn có >=2 nguồn độc lập.

- [ ] **Step 3: Write research artifact with selected vs rejected options**

Mandatory sections in `.omc/plans/chatbot-research-exa.md`:
```markdown
# Chatbot Research (Exa) for rag2025

## Recommendation Summary
## Detailed Findings by Theme
## Not Selected (and why)
## Applicability to Vietnamese Admissions
## Gate B Verdict
- [ ] >=2 independent sources per major recommendation
- [ ] Applicability justified for Vietnamese admissions
- [ ] Rejected alternatives documented
Status: PASS/FAIL
```
Expected: có phần “Not Selected”.

- [ ] **Step 4: Run Gate B fail-fast check**

Run:
```bash
python - <<'PY'
from pathlib import Path
p = Path("D:/chunking/husc-admission-chat-enrollment/.omc/plans/chatbot-research-exa.md")
s = p.read_text(encoding="utf-8")
required = ["## Recommendation Summary", "## Detailed Findings by Theme", "## Not Selected (and why)", "## Applicability to Vietnamese Admissions", "## Gate B Verdict"]
missing = [x for x in required if x not in s]
print("PASS" if not missing else f"FAIL missing: {missing}")
PY
```
Expected: `PASS`.

- [ ] **Step 5: Commit research artifact**

```bash
git -C "D:/chunking/husc-admission-chat-enrollment" add .omc/plans/chatbot-research-exa.md
git -C "D:/chunking/husc-admission-chat-enrollment" commit -m "docs: add exa-backed research artifact for rag2025 chatbot"
```
Expected: stage B độc lập, traceable.

---

### Task 4: Stage C — Synthesis thành YAML plan executable

**Files:**
- Create: `.omc/plans/chatbot-refactor-plan.yaml`
- Test: YAML schema + placeholder checks

- [ ] **Step 1: Dispatch teammate C tổng hợp self-review + research**

Command payload:
```text
/oh-my-claudecode:team 2:claude "Synthesize rag2025 self-review and Exa research into executable YAML plan matching PROMPT.md schema"
```
Expected: draft YAML đầy đủ block yêu cầu.

- [ ] **Step 2: Write final YAML with prompt-compatible schema**

Required top-level keys:
```yaml
plan_metadata:
current_state:
target_state:
implementation_phases:
quality_gates:
omc_execution_commands:
skills_activation:
```
Expected: không thiếu key nào.

- [ ] **Step 3: Reject unresolved placeholders**

Run:
```bash
python - <<'PY'
from pathlib import Path
import re
p = Path("D:/chunking/husc-admission-chat-enrollment/.omc/plans/chatbot-refactor-plan.yaml")
s = p.read_text(encoding="utf-8")
bad = re.findall(r"\{\{[^}]+\}\}|\bTBD\b|\bTODO\b", s)
print("PASS" if not bad else f"FAIL placeholders: {sorted(set(bad))}")
PY
```
Expected: `PASS`.

- [ ] **Step 4: Validate YAML parseability**

Run:
```bash
python - <<'PY'
from pathlib import Path
import yaml
p = Path("D:/chunking/husc-admission-chat-enrollment/.omc/plans/chatbot-refactor-plan.yaml")
with p.open("r", encoding="utf-8") as f:
    yaml.safe_load(f)
print("PASS")
PY
```
Expected: `PASS`.

- [ ] **Step 5: Commit YAML artifact**

```bash
git -C "D:/chunking/husc-admission-chat-enrollment" add .omc/plans/chatbot-refactor-plan.yaml
git -C "D:/chunking/husc-admission-chat-enrollment" commit -m "docs: add executable rag2025 chatbot refactor yaml"
```
Expected: stage C hoàn tất độc lập.

---

### Task 5: Stage D — Validation with OMC ask (Codex + Gemini)

**Files:**
- Create: `.omc/plans/chatbot-validation.md`
- Test: Gate D checklist + decision consistency

- [ ] **Step 1: Run external challenge pass**

Commands:
```text
/oh-my-claudecode:ask codex "Review this rag2025 chatbot-refactor-plan.yaml for technical feasibility and cost optimization while preserving answer quality priority"
/oh-my-claudecode:ask gemini "Suggest improvements for Vietnamese NLP handling and evaluation robustness in this rag2025 plan"
```
Expected: nhận được critique + suggested adjustments.

- [ ] **Step 2: Reconcile feedback into validation artifact**

Write `.omc/plans/chatbot-validation.md` with sections:
```markdown
# Chatbot Plan Validation (rag2025)

## Feasibility Checks
## Quality-Cost-Latency Trade-off Review
## Adopted Changes After Challenge
## Rejected Suggestions (and why)
## Residual Risks and Mitigations
## Gate D Verdict
- [ ] External challenge feedback reconciled
- [ ] Remaining risks explicit and actionable
- [ ] Priority preserved: quality > cost > latency
Status: PASS/FAIL
```
Expected: có cả adopted + rejected.

- [ ] **Step 3: Run Gate D fail-fast check**

Run:
```bash
python - <<'PY'
from pathlib import Path
p = Path("D:/chunking/husc-admission-chat-enrollment/.omc/plans/chatbot-validation.md")
s = p.read_text(encoding="utf-8")
required = ["## Feasibility Checks", "## Quality-Cost-Latency Trade-off Review", "## Adopted Changes After Challenge", "## Rejected Suggestions (and why)", "## Residual Risks and Mitigations", "## Gate D Verdict"]
missing = [x for x in required if x not in s]
priority_ok = "quality > cost > latency" in s
print("PASS" if (not missing and priority_ok) else f"FAIL missing={missing}, priority_ok={priority_ok}")
PY
```
Expected: `PASS`.

- [ ] **Step 4: Commit validation artifact**

```bash
git -C "D:/chunking/husc-admission-chat-enrollment" add .omc/plans/chatbot-validation.md
git -C "D:/chunking/husc-admission-chat-enrollment" commit -m "docs: add codex-gemini validation artifact for rag2025 plan"
```
Expected: stage D hoàn tất.

---

### Task 6: Final integration + spec backlink

**Files:**
- Modify: `docs/superpowers/specs/2026-04-02-rag2025-omc-chatbot-pipeline-design.md`
- Test: artifact existence + consistency checks

- [ ] **Step 1: Add execution-result links into spec**

Append section to spec:
```markdown
## Execution Outputs (2026-04-02)
- `.omc/plans/chatbot-self-review.md`
- `.omc/plans/chatbot-research-exa.md`
- `.omc/plans/chatbot-refactor-plan.yaml`
- `.omc/plans/chatbot-validation.md`
```
Expected: spec liên kết đầy đủ artifact thực thi.

- [ ] **Step 2: Verify all artifacts exist**

Run:
```bash
python - <<'PY'
from pathlib import Path
base = Path("D:/chunking/husc-admission-chat-enrollment")
files = [
    ".omc/plans/chatbot-self-review.md",
    ".omc/plans/chatbot-research-exa.md",
    ".omc/plans/chatbot-refactor-plan.yaml",
    ".omc/plans/chatbot-validation.md",
    "docs/superpowers/specs/2026-04-02-rag2025-omc-chatbot-pipeline-design.md",
]
missing = [f for f in files if not (base / f).exists()]
print("PASS" if not missing else f"FAIL missing: {missing}")
PY
```
Expected: `PASS`.

- [ ] **Step 3: Final commit**

```bash
git -C "D:/chunking/husc-admission-chat-enrollment" add .omc/plans docs/superpowers/specs/2026-04-02-rag2025-omc-chatbot-pipeline-design.md
git -C "D:/chunking/husc-admission-chat-enrollment" commit -m "docs: finalize full omc pipeline outputs for rag2025 chatbot"
```
Expected: toàn bộ pipeline artifacts được chốt trong 1 commit cuối.

---

## Self-Review of This Plan

- **Spec coverage:** đã phủ đủ 4 stage trong spec (self-review, Exa research, synthesis YAML, validation challenge) + traceability + artifact contract.
- **Placeholder scan:** không dùng TBD/TODO hoặc template braces trong plan steps.
- **Type/signature consistency:** toàn bộ artifact paths và gate checks đồng nhất theo cùng tên file.
