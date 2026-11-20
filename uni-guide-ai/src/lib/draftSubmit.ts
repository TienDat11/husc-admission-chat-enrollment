import { fetchRecord, newIdempotencyKey } from "@/lib/adminApi";
import { submitDraftForReview } from "@/lib/teacherApi";

/**
 * "Gửi duyệt" for one uploaded draft. The current version is read first so the
 * transition's optimistic-concurrency check matches the stored record. Refusals
 * (for example UNANCHORED_DRAFT) propagate as AdminApiError for the caller to show.
 */
export async function sendDraftForReview(recordId: string): Promise<void> {
  const record = await fetchRecord(null, recordId);
  await submitDraftForReview(null, recordId, record.version, newIdempotencyKey());
}
