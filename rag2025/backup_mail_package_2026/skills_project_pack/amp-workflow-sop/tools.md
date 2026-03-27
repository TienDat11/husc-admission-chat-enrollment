# MCP Tool Routing (via Librarian)

> **Golden rule**: Librarian decides which tool to call. Main Agent and workers
> call Librarian (as a tool directly — never spawn it as a sub-agent).
> Never call Context7 and Exa in parallel for the same query.

---

## Hard Constraints (mandatory)

- Use only **Context7 MCP** + **Exa MCP** for docs/web research routing.
- Do NOT use Docfork tools.
- Do NOT use default web tools (`web_search`, `read_web_page`) as fallback.
- If Context7/Exa is unavailable: return `BLOCKED` with missing MCP and required action.

## Routing Decision Tree

```
Classify the query:

Library / package API docs? (npm, PyPI, pub.dev)
  └─→ Context7                          [primary]

Specific doc URL to read in full?
  └─→ Exa search_and_contents (URL/domain-targeted retrieval)

Real-world code examples? (GitHub, OSS implementations)
  └─→ Exa get_code_context

Current events / news / real-time web?
  └─→ Exa web_search

Multi-repo / cross-codebase / framework comparison?
  └─→ Librarian deep mode (runs 2 sequential Context7 calls, then synthesizes)

General URL or fallback?
  └─→ Exa search_and_contents

If first tool result is INSUFFICIENT:
  1. Record `research_log`: tool + why insufficient
  2. Run exactly 1 additional tool (sequential, never parallel)
  3. If still insufficient: return BLOCKED (do not fallback to default web tools)
```

---

## Tool Reference

| Tool | Best for | Do NOT use for |
|------|---------|----------------|
| **Context7** | Package docs, function signatures, SDK options | Real-time info, code examples |
| **Exa** `search_and_contents` | Fetching full content from specific URLs/domains | Static library API docs |
| **Exa** `get_code_context` | OSS code search, real-world implementation patterns | News, current events |
| **Exa** `web_search` | Real-time web, news, current info | Static library API docs |
| **Librarian** | Multi-repo analysis, synthesizing from multiple sources | Single-source simple lookups |

---

## Routing Examples

```
"React 19 what's new in 2026?"
  → Exa web_search ✅

"useQuery options in @tanstack/react-query?"
  → Context7 ✅

"Who implemented RSC with Suspense on GitHub?"
  → Exa get_code_context ✅

"Compare Next.js vs Remix routing?"
  → Librarian deep mode:
      1. Context7 → Next.js routing docs
      2. Context7 → Remix routing docs
      3. Synthesize ✅ (2 calls max, sequential)

"Read https://docs.example.com/api-reference"
  → Exa search_and_contents ✅
```

---

## Oracle Routing

Oracle is a **reasoning review tool**, not a research tool. Use it for:
- Reviewing a plan or architecture
- Risk analysis
- Second opinion on a complex decision
- Debugging a subtle logic issue

**Good Oracle prompt structure:**
```
Review the following {plan | code section | architecture decision}:

{content — keep it focused, not the entire codebase}

Evaluate:
1. Correctness and edge cases
2. Unaddressed risks
3. Missing considerations
4. Specific recommendations

Return: verdict (APPROVED | NEEDS_REVISION | REJECTED)
        + top 3 action items
```

**Anti-patterns:**
```
❌ Calling Oracle for library docs lookups
❌ Asking Oracle to review 2000+ lines of code at once (give it a specific section)
❌ Spawning Oracle as a sub-agent via Task tool
❌ Calling Context7 + Exa in parallel for the same question
❌ Bypassing Librarian and calling tools directly from Main Agent for research
```
