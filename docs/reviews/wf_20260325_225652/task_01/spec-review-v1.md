# Spec Review v1 — hybrid-retrieval-integration

**Phase:** spec_review
**Reviewer note:** Reviewer was running in worktree (agent-a53cc17e) which does NOT have the docs/ directory from main branch commit 5327066. The spec file IS committed to main (verified: git log shows commit 5327066 'docs: add hybrid-retrieval-integration design spec'). Reviewer misidentified LanceDB as Qdrant — main.py uses lancedb_retrieval, not qdrant_retrieval. Key architectural questions are valid however.

## Status: Spec exists but reviewer was in wrong worktree context

## Valid issues raised:
1. BM25 index lifecycle (build on startup, lazy load, persistence strategy)
2. Corpus sync strategy
3. Fallback if BM25 build fails
4. Fusion weights should be configurable (currently hardcoded 0.6/0.4)
5. Missing settings for BM25 index path and persistence

## Notes for orchestrator:
- Spec IS at docs/specs/2026-03-25-hybrid-retrieval-integration-design.md (main branch)
- Active retriever is LanceDB (not Qdrant) — reviewer was confused by worktree
- Send author the valid issues, request revision
