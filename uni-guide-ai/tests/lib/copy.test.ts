import { describe, expect, it } from "vitest";
import { DEFAULT_ERROR_MESSAGE, WARMING_BANNER } from "@/lib/copy";

describe("copy", () => {
  it("keeps the default error fallback text", () => {
    expect(DEFAULT_ERROR_MESSAGE).toBe("Thao tác thất bại. Vui lòng thử lại.");
  });

  it("renders the warming banner countdown", () => {
    expect(WARMING_BANNER.connecting(3)).toBe("Đang kết nối tới máy chủ... (3s)");
  });
});
