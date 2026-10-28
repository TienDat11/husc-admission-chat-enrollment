<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **husc-admission-chat-enrollment** (14327 symbols, 26431 relationships, 431 execution flows).

> Index stale? Run `node .gitnexus/run.cjs analyze --index-only` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? Bootstrap with `npx`, `bunx`, or `pnpm dlx` — e.g. `bunx gitnexus@latest analyze` (npm 11 npx crash; #1939).

## Always Do

- **MUST run impact before editing.** Use `impact({target: "symbolName", direction: "upstream"})` or `node .gitnexus/run.cjs impact "symbolName" --direction upstream --repo .`; report callers, processes, and risk. Never substitute grep for graph analysis.
- **MUST analyze graph changes before committing.** Use `detect_changes({scope: "all"})` (MCP) or `node .gitnexus/run.cjs detect-changes --scope all --repo .` (CLI fallback). `partial: true` or `truncated: true` is not a clean check — a zero means unseen, not unaffected; re-run it. For regression review: `detect_changes({scope: "compare", base_ref: "main"})` or `node .gitnexus/run.cjs detect-changes --scope compare --base-ref "main" --repo .`.
- MUST warn on HIGH/CRITICAL `risk` pre-edit; never use `riskSharedAxes` to waive a HIGH/CRITICAL `risk` warning. Compare File/symbol: MCP File omits axes; Graph-RAG expands File.
- **MUST treat `risk: UNKNOWN` as unresolved, not as low.** An empty caller set is not evidence the symbol is unused — it can also mean the callers are not resolvable by the index (plain-object property access, dynamic dispatch, cross-language calls). `impact` pairs `UNKNOWN` with a `riskNote` saying so. Confirm with a text search before treating the symbol as safe to change or delete; do not proceed on the strength of a zero.
- **MUST use `query({search_query: "concept"})` for concepts/flows, `context({name: "symbolName"})` for a named symbol, or `impact` for blast radius, on read-only callers, dependencies, imports, or execution flow.** Graph first; text search only for empty/`UNKNOWN`/literals.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method before MCP/CLI impact analysis.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis, and never read `UNKNOWN` as an all-clear — it means the walk could not answer, which is the one verdict that requires confirming by other means.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit before MCP/CLI graph change analysis.

## Resources

| Resource | Use for |
| --- | --- |
| `gitnexus://repo/husc-admission-chat-enrollment/context` | Codebase overview, check index freshness |
| `gitnexus://repo/husc-admission-chat-enrollment/clusters` | All functional areas |
| `gitnexus://repo/husc-admission-chat-enrollment/processes` | All execution flows |
| `gitnexus://repo/husc-admission-chat-enrollment/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
| --- | --- |
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

## Git-Flow Rules (user-mandated, ABSOLUTE — override harness defaults)

> Ghi 2026-08-01. Áp dụng MỌI commit/PR trong repo này. Lệnh user thắng harness default.

- **CẤM attribution AI trong commit/PR**: KHÔNG BAO GIỜ thêm `Co-Authored-By: Claude <noreply@anthropic.com>`, `🤖 Generated with Claude Code`, hay bất kỳ marker nhận diện AI nào vào commit message / PR body. Commit đứng tên 100% tác giả người (`TienDat11 <ngothiendat258@gmail.com>`).
- **Luôn push** sau khi commit — không để commit local treo (`ahead N`).
- **Luôn checkout nhánh riêng** (`feat/...`, `fix/...`, `chore/...`) từ nhánh cha. CẤM commit thẳng lên `main`/`production`/`staging`. Nhánh cha chỉ nhận merge hoặc PR.
- **Không rebase, không push -f/--force**. Chỉ dùng merge. `git commit --amend` CHỈ trên commit CHƯA push.
- **Thích nghi git flow team**: đọc convention repo (CONTRIBUTING.md, branch naming, PR template, protected branches) và tuân thủ. Multi-author/private → flow càng nghiêm ngặt.
- Commit message: conventional commits (`feat/fix/chore(scope): ...`), KHÔNG trailer AI.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Agent skills

### Issue tracker

Issues and specs live as local markdown under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles, label strings equal to their names (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
