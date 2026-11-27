import { describe, expect, it } from "vitest";
import type { CanonicalRecord } from "@/lib/adminApi";
import { recordTitle } from "@/components/teacher/review/recordTitle";

describe("recordTitle", () => {
  it("falls back to the raw record type when it has no label", () => {
    const record = { record_type: "ZZZ_TEST", admission_year: 2026 } as unknown as CanonicalRecord;
    expect(recordTitle(record)).toBe("ZZZ_TEST 2026");
  });
});
