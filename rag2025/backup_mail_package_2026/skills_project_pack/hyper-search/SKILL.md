---
name: hyper-search
description: "Orchestrates Context7 and Exa into a precision search pipeline for the Librarian agent. Routes each query to exactly one tool based on query type, applies multi-hop reasoning for complex research, and filters low-credibility sources. Use when Librarian performs deep research: library API docs, OSS code patterns, current tech news, academic papers, framework comparisons, or any task requiring maximum search accuracy."
allowed-tools:
  - mcp__Context7__resolve_library_id
  - mcp__Context7__query_docs
  - mcp__exa__web_search_exa
  - mcp__exa__get_code_context_exa
  - mcp__exa__search_and_contents
  - librarian
---

# Hyper-Search

Precision search orchestration for Librarian. One tool per query. Stop when sufficient.

---

## Routing Decision Tree

```
Query type?
│
├─ Library/package API, function signatures, options?
│    └─→ mcp__Context7__resolve_library_id → mcp__Context7__query_docs
│
├─ Specific doc page URL cần đọc full content?
│    └─→ mcp__exa__search_and_contents (URL/domain-targeted search/fetch)
│
├─ "How is X implemented?", OSS code examples, GitHub patterns?
│    └─→ mcp__exa__get_code_context_exa
│         query = "[function/class] [framework] [language]"
│
├─ General web search, find pages, company/product research?
│    └─→ mcp__exa__web_search_exa
│
├─ "Latest", "2025/2026", news, pricing, live status, quick facts?
│    └─→ mcp__exa__web_search_exa (general web search)
│
├─ Complex reasoning + web data required together?
│    └─→ mcp__exa__web_search_exa → synthesize manually
│
├─ Framework A vs B comparison?
│    └─→ Sequential: Context7(A) → Context7(B) → synthesize
│         max 2 calls, no third tool unless explicit gap
│
└─ MCP unavailable?
     └─→ BLOCKED: report missing MCP (Exa or Context7), request enablement.
```

**Stop rule**: If first tool returns sufficient results → output immediately. Only add second tool when explicit gaps remain. Max 3 tool calls total per user question.

---

## Multi-Hop Pattern (for dependent questions)

When answer to Q2 depends on Q1's result:

```
Example: "How does Remix handle deferred data loading?"

Hop 1: Context7 → remix → "defer Await streaming"
       → Found: uses defer() + <Await> component

Hop 2 (only if code example needed): Exa → "remix defer Await Suspense TypeScript"
       → Found: production implementation patterns

Synthesize → return unified answer
```

Do not pre-plan all hops. Evaluate after each hop whether to continue.

---

## Source Quality Check (before returning)

Reject results that are:
- ❌ Marked "deprecated" or "legacy" for an active-tech query
- ❌ Require login to verify
- ❌ Undated or older than 12 months for fast-moving tech (libraries, APIs, pricing)
- ❌ From tutorial aggregators (w3schools, tutorialspoint) for production questions

Prefer:
- ✅ Official docs domain (react.dev, nextjs.org, docs.python.org)
- ✅ GitHub repo of the library itself
- ✅ Dated within 6 months for evolving topics

---

## Output Format

```markdown
## Answer
[Direct answer]

## Details
[Key findings, code examples if applicable]

## Sources
[Tool used → what was found]

## Confidence
HIGH (multiple authoritative sources) | MEDIUM (one source) | LOW (indirect/outdated)

## Caveats
[Version dependencies, unverified claims, gaps]
```

---

## Anti-Patterns

```
❌ Context7 + Exa in parallel for same question
❌ Exa for static library API docs (use Context7)
❌ Context7 for code examples (gives docs, not GitHub implementations)
❌ >3 tool calls per question
❌ Retry same query >2× without reformulating
❌ Return raw tool output without synthesis
❌ Use Docfork/read_web_page/web_search fallback when Exa/Context7 is unavailable
```

---

## Reference Files

- `reference/routing-examples.md` — 30+ concrete routing decisions by query type
- `reference/source-credibility.md` — domain authority tiers by technology area
