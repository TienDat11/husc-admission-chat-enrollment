import { describe, expect, it } from "vitest";
import { safeHref } from "@/lib/safeHref";

describe("safeHref", () => {
  it.each([
    "https://husc.hueuni.edu.vn/tuyen-sinh",
    "http://example.com/a?b=1",
    "mailto:tuyensinh@husc.edu.vn",
    "tel:+842343123456",
    "/quan-ly/cho-duyet",
  ])("allows %s", (href) => {
    expect(safeHref(href)).toBe(href);
  });

  it("trims surrounding whitespace", () => {
    expect(safeHref("  /quan-ly  ")).toBe("/quan-ly");
  });

  it.each([
    "//evil.example.com/x",
    "/\\evil.example.com",
    "javascript:alert(1)",
    "JaVaScRiPt:alert(1)",
    "java\tscript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "vbscript:msgbox(1)",
    "ftp://example.com",
    "relative/path",
    "",
    "   ",
  ])("rejects %j", (href) => {
    expect(safeHref(href)).toBeUndefined();
  });

  it("rejects null and undefined", () => {
    expect(safeHref(null)).toBeUndefined();
    expect(safeHref(undefined)).toBeUndefined();
  });
});
