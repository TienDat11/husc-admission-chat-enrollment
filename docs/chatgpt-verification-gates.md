# Verification and acceptance gates

## RAG quality
- Maintain a versioned Vietnamese golden set covering factual lookup, multi-hop, tables, ambiguous year, conflicting sources, no-answer, prompt injection, and malformed queries.
- Report retrieval recall/precision by intent, citation correctness, answer faithfulness, abstention precision, latency p50/p95, and cost/query.
- Define the 99% goal as a measured satisfaction target on a pre-registered sample: task success, citation trust, and no critical misinformation; include confidence intervals and a separate critical-error ceiling. It is not a blanket guarantee.
- Release blocker: any fabricated deadline/score/eligibility or uncited high-impact answer.

## Data/admin
- CRUD contract tests for validation, effective dates, versioning, publish permissions, audit events, idempotent indexing, and rollback.
- Import fixtures for PDF/table extraction warnings, duplicate records, conflicting active records, and partial failures.
- Verify only approved/published records enter the student retrieval index.

## UX/accessibility
- Keyboard-only smoke path for search, answer citations, feedback, and admin CRUD/publish.
- Screen-reader labels, focus order, contrast, responsive mobile layout, loading/error/empty states.
- User journey smoke tests for student lookup, teacher comparison, staff edit/review/publish/rollback.

## Operations/security
- RBAC negative tests; audit log completeness; prompt injection and data exfiltration tests; rate limits and privacy checks.
- Re-index failure recovery and rollback drill; backup/restore evidence.
- Observability dashboards and alert thresholds for stale data, retrieval degradation, indexing failures, and critical answer feedback.

## Phase gate
A phase passes only when its behavior is demonstrated against representative fixtures, metrics are recorded, and no critical misinformation/security/accessibility blocker remains open.
