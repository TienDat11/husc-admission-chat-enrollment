/**
 * Query key factory for the ['admin', ...] TanStack Query keys.
 * Shapes match the literal keys previously inlined in the admin pages.
 */
export const adminKeys = {
  all: ["admin"] as const,
  overview: () => ["admin", "overview"] as const,
  releases: () => ["admin", "releases"] as const,
  auditEvents: () => ["admin", "audit-events"] as const,
  workers: () => ["admin", "workers"] as const,
  jobs: () => ["admin", "jobs"] as const,
  job: (parseJobId: string) => ["admin", "job", parseJobId] as const,
  feedback: (sample: unknown) => ["admin", "feedback", sample] as const,
  unanswered: () => ["admin", "unanswered"] as const,
  unansweredAll: () => ["admin", "unanswered", "all"] as const,
  records: () => ["admin", "records"] as const,
  recordsReview: () => ["admin", "records", "review"] as const,
  recordsByStatus: (status: string) => ["admin", "records", { status }] as const,
  reviewDiff: (recordId: string | undefined) => ["admin", "review-diff", recordId] as const,
  vault: () => ["admin", "vault"] as const,
  vaultQuery: (filter: { text: string; year: unknown }) => ["admin", "vault", filter] as const,
  recordHistory: () => ["admin", "record-history"] as const,
  recordHistoryFor: (recordId: string | null) => ["admin", "record-history", recordId] as const,
} as const;
