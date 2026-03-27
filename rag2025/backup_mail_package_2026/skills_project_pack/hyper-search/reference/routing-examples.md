# Routing Examples — 30+ Concrete Query Decisions

## Context7 Patterns (Library/Package API)

| Query | Context7 Call |
|-------|--------------|
| "What are options for useSWR?" | resolve: "swr" → query: "useSWR options configuration" |
| "How to configure Prisma multi-tenant?" | resolve: "prisma" → query: "multi-tenant row-level security" |
| "Next.js middleware API" | resolve: "next.js" → query: "middleware request response API" |
| "Flutter riverpod StateProvider" | resolve: "riverpod" → query: "StateProvider usage watch read" |
| "Zod schema validation errors" | resolve: "zod" → query: "error handling ZodError format" |
| "tanstack-query v5 mutations" | resolve: "@tanstack/react-query" → query: "useMutation v5 API" |
| "Fastify route decorators" | resolve: "fastify" → query: "route decorator schema validation" |
| "Go fiber middleware" | resolve: "fiber" → query: "middleware request context" |

**Tip**: Always resolve library ID first — don't guess. The ID may differ from the package name.

---

## Exa Patterns (OSS Code / GitHub Examples)

| Query | Exa Search Term |
|-------|----------------|
| "How Next.js apps handle auth" | `"getServerSession next-auth App Router TypeScript"` |
| "Flutter clean architecture BLoC" | `"flutter bloc clean architecture repository usecase"` |
| "Go microservice with gRPC" | `"go grpc microservice protobuf server implementation"` |
| "React Server Components with database" | `"use server async function prisma React Server Component"` |
| "Python FastAPI with SQLAlchemy" | `"fastapi sqlalchemy async session dependency injection"` |
| "Rust tokio async patterns" | `"tokio::spawn async await rust production"` |
| "Kubernetes custom controller" | `"controller-runtime reconciler kubernetes operator golang"` |

**Token budget**:
- Quick example: `tokensNum: 3000`
- Deep dive: `tokensNum: 8000`
- Comprehensive: `tokensNum: 15000`

---

## Perplexity sonar-pro Patterns (Real-Time / Current)

| Query | Search Config |
|-------|--------------|
| "What changed in React 19?" | `search_recency_filter: "month"` |
| "Latest Bun.js features 2026" | `search_recency_filter: "month"` |
| "Current best practices CI/CD 2026" | `search_recency_filter: "year"` |
| "Is Vercel free tier still available?" | `search_recency_filter: "week", livecrawl: "preferred"` |
| "OpenAI API rate limits 2026" | `search_recency_filter: "month"` |
| "Cloudflare Workers pricing change" | `search_recency_filter: "week"` |

---

## Perplexity sonar-reasoning-pro Patterns (Complex Analysis)

Use **only** when the answer requires both real-time data AND multi-step reasoning:

| Query | Why reasoning-pro? |
|-------|--------------------|
| "Compare SvelteKit vs Next.js for e-commerce 2026" | Needs current perf data + architectural analysis |
| "Is Deno worth migrating to from Node?" | Needs ecosystem state + technical trade-offs |
| "Best database for time-series IoT data at scale" | Needs vendor landscape + technical deep-dive |
| "Should I use tRPC or GraphQL for my use case?" | Needs current ecosystem + contextual reasoning |

**Warning**: Each sonar-reasoning-pro call costs ~5-10x more than sonar-pro. Never use for simple factual questions.

---

## Multi-Hop Search Patterns

### Pattern A: Library + Usage Example
```
Q: "How to use React Query with Suspense?"
Hop 1: Context7 → tanstack/react-query → "suspense integration useSuspenseQuery"
Hop 2 (if needed): Exa → "useSuspenseQuery Suspense boundary TypeScript example"
```

### Pattern B: Error → Root Cause → Fix
```
Q: "Why does Prisma throw P2002 error?"
Hop 1: Context7 → prisma → "P2002 unique constraint error"
Hop 2 (if pattern unclear): Exa → "prisma P2002 unique constraint violation handle"
```

### Pattern C: Current State → Historical Context
```
Q: "What's the current recommendation for state management in React?"
Hop 1: Perplexity sonar-pro → "React state management recommendation 2026"
Hop 2 (if need code): Exa → "[recommended library] state management patterns"
```

### Pattern D: Framework Comparison
```
Q: "Next.js vs Remix routing comparison"
Hop 1: Context7 → next.js → "App Router routing"
Hop 2: Context7 → remix → "routing nested routes"
Synthesize: Compare findings
(No third tool needed unless very outdated docs)
```

---

## When NOT to Multi-Hop

These are SINGLE-TOOL queries — do not add more tools:

```
✅ Single tool sufficient:
- "What params does useState accept?" → Context7 only
- "Show me a zustand store example" → Exa only  
- "Did OpenAI release a new model today?" → Perplexity sonar-pro only
- "Read this URL: https://example.com/docs" → Perplexity web fetch only
```

---

## Query Reformulation (When First Search Fails)

If first search returns low-quality results:

| Problem | Reformulation Strategy |
|---------|----------------------|
| Too broad ("how does auth work") | Add specific technology ("how does NextAuth.js v5 JWT session work") |
| No results | Try synonym ("async iterator" → "generator async await") |
| Outdated results | Add year filter or "2026" to query |
| Wrong level | Add context ("for beginners" or "production-grade") |
| Wrong tool | Check classification — maybe switch tool type |

**Max 2 reformulations per sub-query then return best available.**
