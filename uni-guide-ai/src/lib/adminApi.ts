// Admin CMS client — ticket 19.
//
// Thin, typed wrapper over the ticket-12/13/19 admin endpoints. Two things it
// owns and the UI depends on:
//
// 1. The credential header. Identity comes from the credential alone, so this
//    is the only place a token is attached to a request; nothing else in the
//    admin surface builds a fetch call.
// 2. Turning a FastAPI failure into something a form can render. A 422 from
//    request validation carries `loc` paths (``["body","record","admission_year"]``),
//    which is what lets the form put the message *at the offending field*
//    instead of in a banner.
import { getApiBaseUrl } from "./api";

/** The header the backend reads (src/config/admin_identity.py). */
export const ADMIN_TOKEN_HEADER = "X-Admin-Token";

/**
 * Header the backend reads to deduplicate a retried write.
 *
 * A retried request only dedupes if it carries the *same* key as the attempt it
 * retries, so callers must mint one key per user intention (a click, a confirm)
 * and reuse it across every retry of that intention — a key regenerated inside
 * the retry helper would be theatre.
 */
export const IDEMPOTENCY_KEY_HEADER = "Idempotency-Key";
/** Auth headers. `null` means the request rides on the HttpOnly session cookie;
 * a token is the machine path (ops scripts) and is sent as `X-Admin-Token`. */
export function adminRequestHeaders(token: string | null): Record<string, string> {
  return token === null ? {} : { [ADMIN_TOKEN_HEADER]: token };
}

/** A fresh key for one user intention. */
export function newIdempotencyKey(): string {
  return crypto.randomUUID();
}

export type RecordStatus = "draft" | "review" | "approved" | "published" | "archived";

export type LifecycleAction =
  | "submit"
  | "approve"
  | "reject"
  | "publish"
  | "supersede"
  | "withdraw"
  | "revoke_supersession";

/** The grants the backend's role x action matrix names (AdminAction values). */
export type AdminActionName =
  | "records.read"
  | "records.create"
  | "records.edit"
  | "records.submit"
  | "records.approve"
  | "records.reject"
  | "records.publish"
  | "records.supersede"
  | "records.withdraw"
  | "records.revoke_supersession"
  | "records.rollback"
  | "records.override"
  | "roles.manage";

export interface AdminIdentity {
  actor_id: string;
  role: string;
  /** Every action the role holds, decided server-side. */
  actions: string[];
}

export interface CanonicalRecord {
  schema_version: string;
  record_id: string;
  record_type: string;
  version: number;
  admission_year: number;
  effective_from: string;
  effective_to: string | null;
  status: RecordStatus;
  authority_level: string;
  source_document_id: string | null;
  source_url: string | null;
  source_page: number | null;
  published_at: string | null;
  authored_by: string | null;
  supersedes: string | null;
  locale: string;
  text: string;
  /** The canonical schema is additive (`extra="allow"`): `values`, extractor
   * stamps and anything a later writer adds ride along here. */
  [extra: string]: unknown;
}

export interface LifecycleEvent {
  actor_id: string;
  role: string | null;
  action: string;
  entity: string;
  entity_id: string;
  before_version: number;
  after_version: number;
  before_status: RecordStatus;
  after_status: RecordStatus;
  reason: string | null;
  timestamp: string;
}

export interface LifecycleWriteResult {
  record: CanonicalRecord;
  event: LifecycleEvent;
}

export interface RecordFilters {
  nganh?: string;
  year?: number;
  status?: RecordStatus;
}

export interface RecordListResult {
  records: CanonicalRecord[];
  total: number;
}

/**
 * A failed admin request.
 *
 * ``fieldErrors`` is keyed by the form field name (``admission_year``), with
 * the ``body.record.`` prefix FastAPI's validation errors carry stripped off,
 * so a caller can render the message at the input it belongs to.
 */
export class AdminApiError extends Error {
  readonly status: number;
  readonly fieldErrors: Record<string, string>;

  constructor(status: number, message: string, fieldErrors: Record<string, string> = {}) {
    super(message);
    this.name = "AdminApiError";
    this.status = status;
    this.fieldErrors = fieldErrors;
  }
}

/**
 * True when the server is saying "you are not signed in": a 401, or the 422 the
 * backend returns when a request carries neither the token header nor the
 * session cookie (`{"detail":"no credential: ..."}`). Any other 422 is a real
 * validation error and must not be treated as anonymous.
 */
export function isMissingCredential(failure: unknown): boolean {
  if (!(failure instanceof AdminApiError)) return false;
  if (failure.status === 401) return true;
  if (failure.status === 403) {
    return /invalid or expired session|session no longer maps to a stored identity/i.test(failure.message);
  }
  return failure.status === 422 && /no credential/i.test(failure.message);
}

function describe(detail: unknown): { message: string; fieldErrors: Record<string, string> } {
  if (typeof detail === "string") return { message: detail, fieldErrors: {} };

  if (Array.isArray(detail)) {
    const fieldErrors: Record<string, string> = {};
    const formMessages: string[] = [];
    for (const entry of detail) {
      const error = entry as { loc?: unknown[]; msg?: unknown };
      const message = typeof error.msg === "string" ? error.msg : "Dữ liệu không hợp lệ.";
      const path = Array.isArray(error.loc)
        ? error.loc
            .slice(1) // drop the leading "body"
            .filter((part) => part !== "record")
            .map(String)
        : [];
      if (path.length === 0) {
        formMessages.push(message);
        continue;
      }
      const key = path.join(".");
      fieldErrors[key] = fieldErrors[key] ?? message;
    }
    return {
      message: formMessages.join("; ") || Object.values(fieldErrors)[0] || "Dữ liệu không hợp lệ.",
      fieldErrors,
    };
  }

  if (detail == null) return { message: "Yêu cầu thất bại.", fieldErrors: {} };
  return { message: JSON.stringify(detail), fieldErrors: {} };
}

export async function adminRequest<T>(
  token: string | null,
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers: Record<string, string> = {
    ...adminRequestHeaders(token),
    ...((init.headers as Record<string, string> | undefined) ?? {}),
  };
  if (init.body !== undefined) headers["Content-Type"] = "application/json";

  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    ...init,
    headers,
    credentials: "same-origin",
  });

  if (!response.ok) {
    let detail: unknown = null;
    try {
      detail = (await response.json())?.detail ?? null;
    } catch {
      detail = null;
    }
    const { message, fieldErrors } = describe(detail);
    throw new AdminApiError(response.status, message, fieldErrors);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

/** The caller's own identity and grants — what the UI hides buttons with. */
export async function fetchAdminIdentity(token: string | null): Promise<AdminIdentity> {
  return adminRequest<AdminIdentity>(token, "/admin/identity");
}

/** `POST /admin/login`: the server sets the HttpOnly session cookie. The password
 * is sent once and never stored; 401 = wrong credentials, 423 = locked out. */
export async function loginAdmin(actorId: string, password: string): Promise<void> {
  await adminRequest<void>(null, "/admin/login", {
    method: "POST",
    body: JSON.stringify({ actor_id: actorId, password }),
  });
}

/** `POST /admin/logout`: the server clears the session cookie. */
export async function logoutAdmin(): Promise<void> {
  await adminRequest<void>(null, "/admin/logout", { method: "POST" });
}

export async function listRecords(
  token: string | null,
  filters: RecordFilters = {},
): Promise<RecordListResult> {
  const params = new URLSearchParams();
  const nganh = filters.nganh?.trim();
  if (nganh) params.set("nganh", nganh);
  if (filters.year) params.set("year", String(filters.year));
  if (filters.status) params.set("status", filters.status);
  const query = params.toString();
  return adminRequest<RecordListResult>(token, `/admin/records${query ? `?${query}` : ""}`);
}

export async function fetchRecord(token: string | null, recordId: string): Promise<CanonicalRecord> {
  return adminRequest<CanonicalRecord>(
    token,
    `/admin/records/${encodeURIComponent(recordId)}`,
  );
}

/** The structured payload a record carries (`values` in the canonical schema). */
export function structuredValues(record: CanonicalRecord): Record<string, unknown> {
  for (const key of ["values", "fields", "facts"]) {
    const candidate = record[key];
    if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) {
      return candidate as Record<string, unknown>;
    }
  }
  return {};
}

export async function createRecord(
  token: string | null,
  record: Record<string, unknown>,
): Promise<LifecycleWriteResult> {
  return adminRequest<LifecycleWriteResult>(token, "/admin/records", {
    method: "POST",
    body: JSON.stringify({ record }),
  });
}

export async function editRecord(
  token: string | null,
  recordId: string,
  changes: Record<string, unknown>,
): Promise<LifecycleWriteResult> {
  return adminRequest<LifecycleWriteResult>(
    token,
    `/admin/records/${encodeURIComponent(recordId)}/edits`,
    { method: "POST", body: JSON.stringify(changes) },
  );
}

export async function transitionRecord(
  token: string | null,
  recordId: string,
  body: { action: LifecycleAction; expected_version: number; reason?: string },
  options: { idempotencyKey?: string } = {},
): Promise<LifecycleWriteResult> {
  return adminRequest<LifecycleWriteResult>(
    token,
    `/admin/records/${encodeURIComponent(recordId)}/transitions`,
    {
      method: "POST",
      body: JSON.stringify(body),
      ...(options.idempotencyKey
        ? { headers: { [IDEMPOTENCY_KEY_HEADER]: options.idempotencyKey } }
        : {}),
    },
  );
}

// ---------------------------------------------------------------------------
// Review diff + preview (ticket 25)
// ---------------------------------------------------------------------------

/** One field-level difference between the published version and the draft. */
export interface FieldChangeView {
  field: string;
  old: unknown;
  new: unknown;
  /** Server-side classification (src/domain/canonical/high_impact.py). */
  high_impact: boolean;
}

export interface RecordDiffView {
  record_id: string;
  /** null when the record has never been published — there is no baseline. */
  baseline_version: number | null;
  draft_version: number;
  high_impact_version: string;
  has_high_impact: boolean;
  changes: FieldChangeView[];
}

export interface ReviewDiffResult {
  record_id: string;
  draft_version: number;
  high_impact_version: string;
  diff: RecordDiffView;
  /** Errors that would block publish, keyed to the field the admin must fix. */
  validation_errors: { field: string; message: string }[];
}

/**
 * `POST /admin/records/{id}/review-preview` — the answer a student would get.
 *
 * `preview_status` is the whole contract here:
 * - `ok` — a model wrote `answer` grounded in `chunks`.
 * - `auto_answer` — the router answered without generation; `model_answer`
 *   is false, so the reviewer must not read `answer` as model-verified.
 * - `blocked_validation` — the draft cannot be published; `validation_errors`
 *   names the fields and no answer path ran.
 * - `blocked_pii` — the draft contains personal data (CCCD, email, phone); preview
 *   is blocked so no generated answer is shown or run.
 * - `insufficient_evidence` / `insufficient_evidence_provider` — retrieval or
 *   the LLM provider failed. `answer` is null and `chunks`/`sources` are
 *   still returned so the reviewer can inspect the grounding rather than be
 */
export interface ReviewPreviewResult {
  record_id: string;
  draft_version: number;
  baseline_version: number | null;
  high_impact_version: string;
  diff: RecordDiffView;
  validation_errors: { field: string; message: string }[];
  preview_status:
    | "ok"
    | "auto_answer"
    | "blocked_validation"
    | "insufficient_evidence"
    | "insufficient_evidence_provider"
    | "blocked_pii";
  answer: string | null;
  answer_status: string | null;
  model_answer: boolean | null;
  chunks: {
    chunk_id: string;
    text: string;
    score: number;
    source: string;
    metadata: Record<string, unknown>;
  }[];
  sources: {
    source_document_id: string | null;
    source: string;
    page: number | null;
    section: string | null;
    title: string | null;
    record_id: string | null;
    record_type: string | null;
    effective_from: string | null;
    effective_to: string | null;
    authority_level: string | null;
  }[];
}

/** `GET /admin/records/{id}/review-diff` — draft vs the published version. */
export async function fetchReviewDiff(
  token: string | null,
  recordId: string,
  version?: number,
): Promise<ReviewDiffResult> {
  const query = version ? `?version=${version}` : "";
  return adminRequest<ReviewDiffResult>(
    token,
    `/admin/records/${encodeURIComponent(recordId)}/review-diff${query}`,
  );
}

/** The student-facing answer this draft would produce, on the real path. */
export async function runReviewPreview(
  token: string | null,
  recordId: string,
  body: { query: string; version?: number; top_k?: number },
): Promise<ReviewPreviewResult> {
  return adminRequest<ReviewPreviewResult>(
    token,
    `/admin/records/${encodeURIComponent(recordId)}/review-preview`,
    { method: "POST", body: JSON.stringify(body) },
  );
}

/**
 * Download a source document's raw bytes (PDF, DOCX...) for opening in a new tab.
 * `GET /admin/source-documents/content?document_id=<document_id>` — returns bytes,
 * not JSON.
 *
 * Returns an object URL that the caller must pass to `URL.revokeObjectURL()`
 * once it is no longer needed. The fetch carries the admin credential so the
 * opened tab does not get a 401.
 */
export async function openSourceDocument(
  token: string | null,
  documentId: string,
): Promise<string> {
  const headers: Record<string, string> = {
    ...adminRequestHeaders(token),
  };
  const response = await fetch(
    `${getApiBaseUrl()}/admin/source-documents/content?document_id=${encodeURIComponent(
      documentId,
    )}`,
    { headers, credentials: "same-origin" },
  );

  if (!response.ok) {
    let detail: unknown = null;
    try {
      detail = (await response.json())?.detail ?? null;
    } catch {
      detail = null;
    }
    const { message } = describe(detail);
    throw new AdminApiError(response.status, message, {});
  }

  const blob = await response.blob();
  return URL.createObjectURL(blob);
}
// ---------------------------------------------------------------------------
// Releases + index jobs (ticket 26)
// ---------------------------------------------------------------------------

/** One release, as its manifest states it. `null` fields are "không ghi nhận". */
export interface ReleaseView {
  release_id: string;
  created_at: string | null;
  parent_release_id: string | null;
  lancedb_tag: string | null;
  lancedb_version: number | null;
  document_count: number;
  chunk_count: number;
}

/**
 * `GET /admin/releases` — the active release and the one behind it.
 *
 * `active_release_id` comes from the pointer the serving path reads, so it is
 * present even when `active` is `null` (the manifest could not be read).
 * `problems` is anything that could not be read: an unreadable pointer lands
 * there rather than being rendered as "chưa publish".
 */
export interface ReleaseStatus {
  active_release_id: string | null;
  previous_release_id: string | null;
  active: ReleaseView | null;
  previous: ReleaseView | null;
  problems: string[];
}

/** One half of retrying a failed job, and whether this process can run it. */
export interface RetryStep {
  step: string;
  runnable: boolean;
  command: string | null;
  reason: string | null;
}

/**
 * One index-job row.
 *
 * `state` is a JobState value (`pending`/`running`/`failed`/`active`) or `null`
 * when the row is a build on disk rather than a run. `saved` means "a build is
 * persisted"; `serving` means "this is the corpus the index answers from" and is
 * true only for `active`. The two are separate fields precisely so no label can
 * collapse them.
 */
export interface IndexJob {
  job_id: string;
  state: JobStateName | null;
  saved: boolean;
  serving: boolean;
  label: string;
  release_id: string | null;
  staged_dir: string | null;
  reason: string | null;
  retry: RetryStep[];
}

/** The four states the job state machine has. `saved` is deliberately not one. */
export type JobStateName = "pending" | "running" | "failed" | "active";

/** `GET /admin/index-jobs` — job state, staged builds, and what is owed. */
export interface IndexJobsStatus {
  jobs: IndexJob[];
  reindex_needed: boolean;
  reindex_reason: string;
  pending_outbox_entries: number;
  pending_affected_record_ids: string[];
  problems: string[];
}

/** How often the job screen re-reads its state. Polling, not a page reload. */
export const INDEX_JOBS_REFETCH_MS = 5000;

/** What `POST /admin/rollback` answers. */
export interface RollbackResult {
  release_id: string;
  previous_release_id: string;
  steps?: Record<string, string>;
  event_id?: string | null;
}

export async function fetchReleaseStatus(token: string | null): Promise<ReleaseStatus> {
  return adminRequest<ReleaseStatus>(token, "/admin/releases");
}

export async function fetchIndexJobs(token: string | null): Promise<IndexJobsStatus> {
  return adminRequest<IndexJobsStatus>(token, "/admin/index-jobs");
}

/**
 * Roll back to the previous release, or to a named one.
 *
 * `reason` is required by the route (`RollbackRequest.reason`, min length 1) and
 * is required here too — a rollback with no stated reason is not a thing this
 * client will send. The idempotency key is the caller's to mint and reuse, the
 * same way `transitionRecord` takes one.
 */
export async function rollbackRelease(
  token: string | null,
  body: { reason: string; target_release_id?: string },
  options: { idempotencyKey?: string } = {},
): Promise<RollbackResult> {
  return adminRequest<RollbackResult>(token, "/admin/rollback", {
    method: "POST",
    body: JSON.stringify(body),
    ...(options.idempotencyKey
      ? { headers: { [IDEMPOTENCY_KEY_HEADER]: options.idempotencyKey } }
      : {}),
  });
}
