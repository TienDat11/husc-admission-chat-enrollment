import { describe, expect, it } from "vitest";
import { formatViDate, formatViDateTime, getErrorMessage } from "@/lib/utils";
import { DEFAULT_ERROR_MESSAGE } from "@/lib/copy";

describe("formatViDate", () => {
  it("formats a valid date as dd/mm/yyyy", () => {
    expect(formatViDate(new Date(2026, 10, 27))).toBe("27/11/2026");
  });

  it("accepts ISO strings and epoch numbers", () => {
    expect(formatViDate(new Date(2026, 0, 5).toISOString())).toBe("05/01/2026");
    expect(formatViDate(new Date(2026, 0, 5).getTime())).toBe("05/01/2026");
  });

  it("returns the fallback for empty or invalid input", () => {
    expect(formatViDate(null)).toBe("—");
    expect(formatViDate(undefined)).toBe("—");
    expect(formatViDate("")).toBe("—");
    expect(formatViDate("not a date")).toBe("—");
    expect(formatViDate("nope", "n/a")).toBe("n/a");
  });
});

describe("formatViDateTime", () => {
  it("includes the date and time", () => {
    const out = formatViDateTime(new Date(2026, 10, 27, 13, 5));
    expect(out).toContain("27/11/2026");
    expect(out).toContain("13:05");
  });

  it("returns the fallback for invalid input", () => {
    expect(formatViDateTime("")).toBe("—");
    expect(formatViDateTime("garbage", "-")).toBe("-");
  });
});

describe("getErrorMessage", () => {
  it("returns the message of an Error", () => {
    expect(getErrorMessage(new Error("Không tìm thấy"))).toBe("Không tìm thấy");
  });

  it("returns a non-empty string as is", () => {
    expect(getErrorMessage("Lỗi mạng")).toBe("Lỗi mạng");
  });

  it("falls back to the default copy when nothing usable is given", () => {
    expect(getErrorMessage(null)).toBe(DEFAULT_ERROR_MESSAGE);
    expect(getErrorMessage(new Error("   "))).toBe(DEFAULT_ERROR_MESSAGE);
    expect(getErrorMessage({ message: "x" })).toBe(DEFAULT_ERROR_MESSAGE);
  });

  it("uses a caller-supplied fallback", () => {
    expect(getErrorMessage(undefined, "Không lưu được")).toBe("Không lưu được");
  });
});
