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

/**
 * Runs `task` over `ids` with at most `limit` in flight and resolves once every
 * item has settled. `task` must not reject: a rejection would stop its lane.
 */
export async function forEachLimited(ids: readonly string[], limit: number, task: (id: string) => Promise<void>): Promise<void> {
  let next = 0;
  const lane = async () => {
    while (next < ids.length) {
      const id = ids[next];
      next += 1;
      await task(id);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, ids.length) }, lane));
}
