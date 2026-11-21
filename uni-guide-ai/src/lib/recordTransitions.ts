import type { QueryClient } from "@tanstack/react-query";

/**
 * Call after any record transition (submit, claim, approve, publish, reject).
 * The sidebar count on "Chờ duyệt" reads ["admin", "overview"], not the records
 * lists, so refreshing only the records would leave that count stale.
 */
export function invalidateRecordTransition(queryClient: QueryClient): void {
  void queryClient.invalidateQueries({ queryKey: ["admin", "records"] });
  void queryClient.invalidateQueries({ queryKey: ["admin", "overview"] });
}
