import { describe, expect, it } from "vitest";
import { jobStageLabel, uploadWarningMessage } from "@/lib/uploadMessages";

describe("uploadWarningMessage", () => {
  it("maps TABLE_TIER_DISABLED to the friendly sentence and hides the env detail", () => {
    const raw =
      "TABLE_TIER_DISABLED: Docling đang tắt (PARSE_DOCLING_ENABLED=false); trang bảng 1 dùng văn bản pypdf, cần kiểm tra bảng thủ công";
    const message = uploadWarningMessage(raw);
    expect(message).toBe(
      "Bảng trong tệp được đọc ở chế độ văn bản — hãy kiểm tra số liệu trong bảng bằng tay.",
    );
    expect(message).not.toMatch(/TABLE_TIER|PARSE_DOCLING|=false|Docling/);
  });

  it("keeps only the human part of an unknown coded warning", () => {
    const message = uploadWarningMessage("SOME_NEW_CODE: Tệp có trang bị mờ (FOO_BAR=1)");
    expect(message).toBe("Tệp có trang bị mờ");
    expect(message).not.toMatch(/SOME_NEW_CODE|FOO_BAR|=1/);
  });

  it("falls back to a neutral line when only a bare code is left", () => {
    expect(uploadWarningMessage("UNKNOWN_CODE")).toBe("Cần kiểm tra lại tài liệu này.");
    expect(uploadWarningMessage("UNKNOWN_CODE: (ENV_NAME=true)")).toBe("Cần kiểm tra lại tài liệu này.");
  });

  it("leaves a plain human sentence unchanged", () => {
    expect(uploadWarningMessage("Trang 2 có chữ mờ, nên kiểm tra lại.")).toBe("Trang 2 có chữ mờ, nên kiểm tra lại.");
  });
});

describe("jobStageLabel", () => {
  it("shows done as Hoàn tất and parsed as Đã đọc xong", () => {
    expect(jobStageLabel("done")).toBe("Hoàn tất");
    expect(jobStageLabel("parsed")).toBe("Đã đọc xong");
  });

  it("never shows an unmapped or missing raw stage id", () => {
    expect(jobStageLabel("embedding_chunks")).toBe("Đang xử lý");
    expect(jobStageLabel(undefined)).toBe("Đang xử lý");
  });
});
