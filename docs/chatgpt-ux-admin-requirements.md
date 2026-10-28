# UX and admissions admin requirements

## Personas
- Student/applicant: asks in Vietnamese, often on mobile, needs direct answer, source/date, eligibility, deadlines, fees, and next action.
- Teacher/counselor: compares programs and policies, needs shareable cited answers and terminology clarity.
- Admissions staff: maintains official data, reviews changes, publishes by effective year, and corrects errors safely.
- System administrator: manages roles, audit, indexing, monitoring, and rollback.

## Student and teacher UX
- One prominent search/chat entry with suggested intents: ngành, phương thức, chỉ tiêu, học phí, mốc thời gian, hồ sơ, điểm chuẩn.
- Answers must show: short conclusion first, structured details, effective year/date, source links/citations, confidence or caveat, and next actions.
- Follow-up context must preserve the selected admission year and program.
- Explicit “I don’t have enough official information” state; never fabricate deadlines or scores.
- Compare mode for programs/methods; copy/share citation; feedback with reason categories.
- Mobile-first, keyboard accessible, Vietnamese typography, clear loading/error/empty states.

## Admissions admin UX
- Dashboard: draft/review/published/stale/conflicting documents and failed indexing jobs.
- CRUD for canonical structured records: programs, methods, quotas, thresholds, tuition, dates, documents, contacts, FAQs.
- Source document ingestion with preview, extraction warnings, table/PDF verification, metadata editor, and deduplication.
- Lifecycle: draft → review → approved → published → archived; effective year/date required; scheduled publish optional.
- Validation: required fields, date ordering, duplicate program codes, conflicting active records, unsupported claims.
- Preview exactly how a student answer will render before publish.
- Re-index only changed records/documents; show job status, errors, and rollback to last good version.
- RBAC: viewer, editor, reviewer, publisher, admin; separate publish permission from edit permission.
- Immutable audit log: actor, timestamp, before/after, reason, source, approval, publish/rollback event.

## Acceptance themes
- Students find authoritative answers quickly and can verify them.
- Staff can correct a record without database access or engineering help.
- Every published answer is traceable to current approved data.
- A bad import or publish can be rolled back without deleting history.
- Accessibility and Vietnamese language quality are release gates, not polish.
