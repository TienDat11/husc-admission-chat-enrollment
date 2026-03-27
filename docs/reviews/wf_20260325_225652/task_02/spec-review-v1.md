# Spec Review v1 — cross-encoder-reranker

**Phase:** spec_review

## Status: issues (5 critical corrections needed)

### Actual codebase state (verified):
- Current model: **Qwen/Qwen3-Reranker-8B** (NOT bge-reranker-base as spec states)
- RERANKER_ENABLED and RERANKER_WEIGHT already in settings.py (lines 56-57)
- RerankerService IS imported and wired in main.py (line 562)
- Missing: RERANKER_TOP_K (no candidate limiting before reranking)
- Missing: lost-in-middle reordering in rerank() method

### Issues:
1. Wrong current model stated (spec says bge-reranker-base, actual is Qwen3-Reranker-8B)
2. Proposes adding settings that already exist (RERANKER_ENABLED, RERANKER_WEIGHT)
3. Claims reranker not integrated - it IS integrated already at line 562
4. Line numbers wrong (spec says after 414, actual is after 562)
5. Lost-in-middle algorithm has logical error in insert() indexing
