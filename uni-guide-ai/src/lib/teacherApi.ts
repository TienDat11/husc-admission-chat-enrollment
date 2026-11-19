import {
  type AdminIdentity,
  type CanonicalRecord,
  type RecordFilters,
  type RecordListResult,
  type ReviewDiffResult,
  type ReviewPreviewResult,
  adminRequest,
  fetchAdminIdentity,
  fetchReviewDiff,
  IDEMPOTENCY_KEY_HEADER,
  listRecords,
  newIdempotencyKey,
  rollbackRelease,
  runReviewPreview,
  transitionRecord,
  adminRequestHeaders,
} from "./adminApi";
import { getApiBaseUrl } from "./api";
import type { components } from "./api-types";

type Schemas = components["schemas"];

// ---------------------------------------------------------------------------
// Ticket T09 — endpoints the existing admin client does not expose yet.
// ---------------------------------------------------------------------------

/** `GET /admin/unanswered/` — one open question the bot could not answer. */
export type UnansweredItem = Schemas["UnansweredItem"];
export type UnansweredList = Schemas["UnansweredList"];
/** `POST /admin/unanswered/{trace_id}/out-of-scope` result. */
export type OutOfScopeResult = Schemas["OutOfScopeResult"];
/** `POST /admin/unanswered/{trace_id}/promote-golden` request body. */
export type PromoteGoldenBody = Schemas["PromoteGoldenBody"];
/** `POST /admin/unanswered/{trace_id}/promote-golden` result. */
export type PromoteGoldenResult = Schemas["PromoteGoldenResult"];

async function teacherGet<T>(token: string | null, path: string): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    headers: adminRequestHeaders(token),
    credentials: "include",
  });
  if (!response.ok) throw new Error(`Không lấy được dữ liệu (lỗi ${response.status}).`);
  return (await response.json()) as T;
}

export function listUnanswered(
  token: string | null,
  resolved: boolean | null = null,
): Promise<UnansweredList> {
  const query = resolved == null ? "" : `?resolved=${resolved}`;
  return teacherGet<UnansweredList>(token, `/admin/unanswered/${query}`);
}

/** `POST /admin/unanswered/{trace_id}/out-of-scope` — mark a question out of scope (never deletes the row). */
export async function markOutOfScope(
  token: string | null,
  traceId: string,
): Promise<OutOfScopeResult> {
  return adminRequest<OutOfScopeResult>(
    token,
    `/admin/unanswered/${encodeURIComponent(traceId)}/out-of-scope`,
    { method: "POST" },
  );
}

/** `POST /admin/unanswered/{trace_id}/promote-golden` — create a draft record + golden query in one transaction. */
export async function promoteToGolden(
  token: string | null,
  traceId: string,
  body: PromoteGoldenBody,
): Promise<PromoteGoldenResult> {
  return adminRequest<PromoteGoldenResult>(
    token,
    `/admin/unanswered/${encodeURIComponent(traceId)}/promote-golden`,
    { method: "POST", body: JSON.stringify(body) },
  );
}

/** `GET /admin/audit/events` — the ledger, newest first (T09 history screen). */
export interface AuditEventRecord {
  seq: number;
  prev_hash: string;
  entry_hash: string;
  event: Record<string, unknown>;
}

export interface AuditEventList {
  events: AuditEventRecord[];
  count: number;
}

export function fetchAuditEvents(
  token: string | null,
  limit = 100,
): Promise<AuditEventList> {
  return teacherGet<AuditEventList>(token, `/admin/audit/events?limit=${limit}`);
}

// ---------------------------------------------------------------------------
// Ticket T09 — one-intention lifecycle wrappers so each screen stops
// hand-stitching three transitions.
// ---------------------------------------------------------------------------

/**
 * "Gửi duyệt": a freshly created draft moves straight to review.
 * The caller passes the id from the create call; the same key dedupes a
 * retry of that submit.
 */
export async function submitDraftForReview(
  token: string | null,
  recordId: string,
  expectedVersion: number,
  idempotencyKey: string,
): Promise<void> {
  await transitionRecord(
    token,
    recordId,
    { action: "submit", expected_version: expectedVersion },
    { idempotencyKey },
  );
}

/**
 * "Duyệt và xuất bản": approve then publish, the reviewer's pair of moves.
 * Both calls share the key, so a retry after a dropped response replays
 * instead of double-approving. The screen's confirmation box names this as
 * the one irreversible action.
 */
export async function approveAndPublish(
  token: string | null,
  recordId: string,
  version: number,
  idempotencyKey: string,
): Promise<void> {
  await transitionRecord(
    token,
    recordId,
    { action: "approve", expected_version: version },
    { idempotencyKey },
  );
  await transitionRecord(
    token,
    recordId,
    { action: "publish", expected_version: version + 1 },
    { idempotencyKey },
  );
}
/**
 * "Nhận nháp" (ticket T15): claim an unclaimed seed draft so the caller
 * becomes its author. The seed drafts carry authored_by = null, and
 * four-eyes refuses approval of a version nobody authored; this is the
 * fix that stamps the caller as author without touching the seed.
 *
 * The 409 the backend returns (non-draft, or already claimed) carries the
 * server's reason in its message, so callers render it verbatim.
 */
export async function claimDraft(
  token: string | null,
  recordId: string,
  expectedVersion: number,
  idempotencyKey: string,
): Promise<void> {
  await adminRequest<void>(
    token,
    `/admin/records/${encodeURIComponent(recordId)}/claim`,
    {
      method: "POST",
      body: JSON.stringify({ expected_version: expectedVersion }),
      headers: { [IDEMPOTENCY_KEY_HEADER]: idempotencyKey },
    },
  );
}
/**
 * Ticket T15 — batch "Duyệt và xuất bản": run approve+publish per record,
 * one idempotency key per record per intention so a retry replays.
 * A failure on one record is reported, never swallowed — the remaining
 * records still complete.
 */
export interface BatchApproveResult {
  /** Records whose publish calls all succeeded. */
  succeeded: string[];
  /** Records with at least one failed call: id + error. */
  failures: Array<{ recordId: string; error: string }>;
}

export async function approveAndPublishMany(
  token: string | null,
  records: Array<{ recordId: string; version: number }>,
  idempotencyKey: (recordId: string) => string,
): Promise<BatchApproveResult> {
  const succeeded: string[] = [];
  const failures: BatchApproveResult["failures"] = [];
  for (const record of records) {
    try {
      await approveAndPublish(
        token,
        record.recordId,
        record.version,
        idempotencyKey(record.recordId),
      );
      succeeded.push(record.recordId);
    } catch (failure: unknown) {
      failures.push({
        recordId: record.recordId,
        error: failure instanceof Error ? failure.message : "Hành động thất bại.",
      });
    }
  }
  return { succeeded, failures };
}

/** "Trả lại để sửa" — a reject always carries a reason (05 §4.4). */
export async function rejectWithReason(
  token: string | null,
  recordId: string,
  version: number,
  reason: string,
  idempotencyKey: string,
): Promise<void> {
  await transitionRecord(
    token,
    recordId,
    { action: "reject", expected_version: version, reason },
    { idempotencyKey },
  );
}

export type {
  AdminIdentity,
  CanonicalRecord,
  RecordFilters,
  RecordListResult,
  ReviewDiffResult,
  ReviewPreviewResult,
};
export {
  fetchAdminIdentity,
  fetchReviewDiff,
  listRecords,
  newIdempotencyKey,
  rollbackRelease,
  runReviewPreview,
};
