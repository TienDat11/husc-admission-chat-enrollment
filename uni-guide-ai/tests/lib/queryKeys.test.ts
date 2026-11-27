import { describe, expect, it } from "vitest";
import { adminKeys } from "@/lib/queryKeys";

describe("adminKeys", () => {
  it("produces the literal ['admin', ...] keys used by the admin pages", () => {
    expect(adminKeys.all).toEqual(["admin"]);
    expect(adminKeys.overview()).toEqual(["admin", "overview"]);
    expect(adminKeys.workers()).toEqual(["admin", "workers"]);
    expect(adminKeys.recordsReview()).toEqual(["admin", "records", "review"]);
    expect(adminKeys.recordsByStatus("review")).toEqual(["admin", "records", { status: "review" }]);
    expect(adminKeys.vaultQuery({ text: "a", year: 2026 })).toEqual(["admin", "vault", { text: "a", year: 2026 }]);
    expect(adminKeys.recordHistoryFor(null)).toEqual(["admin", "record-history", null]);
  });

  it("prefix keys are contained in their detail keys", () => {
    expect(adminKeys.job("p1")[0]).toBe(adminKeys.all[0]);
    expect(adminKeys.unansweredAll().slice(0, 2)).toEqual(adminKeys.unanswered());
  });
});
