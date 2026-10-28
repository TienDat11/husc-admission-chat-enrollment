# ChatGPT planning brief — HUSC admissions RAG

## Objective
Create an evidence-based implementation plan for upgrading the HUSC admissions assistant so students, teachers, and admissions staff can reliably find current admissions information, while staff can safely create, edit, archive, and publish source data.

## Repository scope
- Workspace: `D:/chunking/husc-admission-chat-enrollment`
- Main product surfaces: `rag2025/` ingestion and retrieval work; `uni-guide-ai/` web/admin UI; `new_data/` admissions PDFs; `docs/` existing specs and audits.
- Existing local quality assets: golden-set evaluation, chunking/ingestion tests, vector-parity checks, admin UI tests, and prior RAG/UX audit documents.

## Problems to solve
1. RAG quality: retrieval misses, stale/conflicting rules, table/PDF structure loss, weak citations, insufficient abstention, and limited evaluation feedback loops.
2. UX/UI: confusing student search flow, poor answer hierarchy, weak mobile/accessibility behavior, unclear source freshness and next actions.
3. Admissions operations: staff need role-aware CRUD, versioning, draft/review/publish lifecycle, audit trail, bulk import, validation, rollback, and safe re-indexing.
4. Trust and safety: Vietnamese terminology, year/effective-date handling, prompt injection resistance, privacy, authorization, and explainable answers.

## Planning requirements
Produce a phased plan with:
- current-state diagnosis tied to repository evidence;
- target architecture and explicit trade-offs;
- recommended latest stable libraries/versions as of the planning date, with primary-source links and migration risk;
- RAG design: ingestion, parsing, table-aware chunking, metadata, hybrid retrieval, reranking, query rewriting, graph/structured facts where justified, citations, freshness, conflict resolution, abstention;
- admin workflow and data model for admissions staff;
- UX information architecture and key student/teacher/admin journeys;
- observability, offline evaluation, online metrics, golden set expansion, and a realistic 99% satisfaction measurement definition (no unsupported guarantee);
- security, privacy, RBAC, auditability, and rollback;
- dependency-ordered epics/tickets suitable for `to-tickets` / issue-architect;
- acceptance criteria and verification commands for each phase;
- explicit MVP, follow-up, and non-goals.

## Constraints
Do not invent current behavior. Prefer existing repository conventions. Preserve Vietnamese admissions source fidelity. Never claim 99% satisfaction without a defined sample, baseline, confidence interval, and pass/fail threshold. Do not create external tracker issues without explicit approval.

## Output format requested from ChatGPT
1. Executive decision summary
2. Evidence table: problem → repository evidence → impact
3. Target architecture
4. UX/admin plan
5. RAG upgrade plan
6. Security and governance
7. Phased roadmap with dependencies
8. Ticket-ready issue list
9. Evaluation and release gates
10. Risks, open decisions, and questions for HUSC
