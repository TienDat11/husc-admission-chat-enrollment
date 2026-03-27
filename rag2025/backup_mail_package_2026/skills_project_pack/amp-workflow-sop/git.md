# Git & Issue Conventions

> All code changes require a GitHub issue first. No issue = no code.
> One issue per branch. No direct commits to `main`.

---

## Branch Preflight (run before every coding session)

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
git checkout -b feature/<issue-number>-short-description
```

Self-check:
- [ ] `git branch --show-current` is NOT `main`
- [ ] Branch name matches `feature/*` or `fix/*`
- [ ] Branch contains the issue number

---

## Commit Format

```
type(#issue-number): short description

Types: feat | fix | refactor | docs | chore
Example: feat(#42): add chapter export feature
```

**Never** use `git add -A` or `git add .` — always stage specific files.

---

## Issue Template (minimum required fields)

```
Title: [Scope:] what this does   (e.g. "Auth: add rate limiting to login endpoint")

Description:
  What: [what changes]
  Why: [why it's needed]

Technical Details:
  Files: [affected files/modules]
  APIs: [relevant APIs]

Acceptance Criteria:
  - [ ] [specific, checkable condition]
  - [ ] [specific, checkable condition]
  - [ ] npm run lint passes

Dependencies: Requires: #XX | Part of: epic:name
```

---

## 3-Phase Delivery

**Before coding:**
- `gh issue view <number>`
- Confirm acceptance criteria and dependencies
- If ambiguous: ask user, don't guess

**Implementation:**
- Create branch from `main`
- Code + `npm run lint`
- Report changes → **wait for user confirm** before committing
- Never commit/push/close without explicit approval

**After confirmation:**
1. Comment on issue: files changed + what changed + testing done
2. Check off acceptance criteria in issue body
3. Stage specific files
4. Commit with `#issue-number` in message
5. Push branch
6. Close issue only after 1–5 complete

---

## Never Commit

`.env` | `node_modules/` | `dist/` | `data/`

---

## Safety Rules

- `git pull --ff-only origin main` before every new branch
- One issue = one branch, never mix
- Never rebase/force-push someone else's branch without consensus
- If you see unexpected changes in worktree: **don't touch them**, work only on your scope
