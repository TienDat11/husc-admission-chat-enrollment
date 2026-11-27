/** Shared client constants. Components and lib modules import from here. */

/** Rows per page in admin list screens (Vault, PendingReview). */
export const PAGE_SIZE = 25;

/** Admin overview poll (sidebar review count). */
export const ADMIN_OVERVIEW_POLL_MS = 30_000;
/** Ops screen refresh (health, workers, jobs). */
export const OPS_REFRESH_MS = 15_000;
/** A worker whose heartbeat is older than this is treated as disconnected. */
export const WORKER_STALE_MS = 60_000;
/** Index job status re-read interval. */
export const INDEX_JOBS_REFETCH_MS = 5_000;
/** Parse job poll interval (AddInfo). */
export const PARSE_JOB_POLL_MS = 2_000;
/** Backend readiness poll interval. */
export const BACKEND_READINESS_POLL_MS = 2_000;
/** Retry hint when the backend is warming and sends no Retry-After. */
export const WARMING_RETRY_MS = 2_000;
/** Exponential backoff bounds for API retries. */
export const RETRY_BASE_MS = 1_000;
export const RETRY_MAX_MS = 30_000;

/** Number of most recent chat turns sent as history. */
export const CHAT_HISTORY_WINDOW = 4;

/** localStorage keys. */
export const STORAGE_KEYS = {
  theme: "theme",
} as const;
