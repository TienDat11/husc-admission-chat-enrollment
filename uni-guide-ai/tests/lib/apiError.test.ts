import { createElement, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { parseApiError } from "@/lib/apiError";
import { DangerButton } from "@/components/ui-kit/DangerButton";
import { ExceptionBar } from "@/components/ui-kit/ExceptionBar";
import { GhostButton } from "@/components/ui-kit/GhostButton";
import { PrimaryButton } from "@/components/ui-kit/PrimaryButton";
import { SecondaryButton } from "@/components/ui-kit/SecondaryButton";
import { SidebarBadge } from "@/components/ui-kit/SidebarBadge";
import { StatusPill } from "@/components/ui-kit/StatusPill";

const at = { currentPath: "/quan-ly/kho" };

describe("parseApiError (PL-7 envelope)", () => {
  it("401 SESSION_EXPIRED returns to login with next", () => {
    const failure = parseApiError({ status: 401, body: { detail: { code: "SESSION_EXPIRED" } }, ...at });
    expect(failure).toMatchObject({
      kind: "session-expired",
      redirectTo: "/quan-ly/dang-nhap?next=%2Fquan-ly%2Fkho",
    });
  });

  it("423 ACCOUNT_LOCKED reads Retry-After seconds", () => {
    const failure = parseApiError({
      status: 423,
      body: { detail: { code: "ACCOUNT_LOCKED", message: "Khoá" } },
      retryAfterHeader: "900",
      ...at,
    });
    expect(failure).toMatchObject({ kind: "locked", retryAfterSeconds: 900 });
  });

  it("403 ACCOUNT_DISABLED shows the disabled message", () => {
    const failure = parseApiError({ status: 403, body: { detail: { code: "ACCOUNT_DISABLED" } }, ...at });
    expect(failure).toEqual({
      kind: "account-disabled",
      status: 403,
      message: "Tài khoản đã bị vô hiệu hóa",
    });
  });

  it("403 FORBIDDEN_ACTION is a forbidden state", () => {
    const failure = parseApiError({ status: 403, body: { detail: { code: "FORBIDDEN_ACTION" } }, ...at });
    expect(failure.kind).toBe("forbidden");
  });

  it("503 WRITES_FROZEN preserves the form", () => {
    const failure = parseApiError({
      status: 503,
      body: { detail: { code: "WRITES_FROZEN" } },
      retryAfterHeader: "30",
      ...at,
    });
    expect(failure).toMatchObject({ kind: "writes-frozen", preserveForm: true, retryAfterSeconds: 30 });
  });

  it("409 VERSION_CONFLICT offers reload with current_version", () => {
    const failure = parseApiError({
      status: 409,
      body: { detail: { code: "VERSION_CONFLICT", current_version: 4 } },
      ...at,
    });
    expect(failure).toMatchObject({ kind: "version-conflict", currentVersion: 4, canReload: true });
  });

  it("409 DOCUMENT_ALREADY_REGISTERED names the old document", () => {
    const failure = parseApiError({
      status: 409,
      body: { detail: { code: "DOCUMENT_ALREADY_REGISTERED", source_document_id: "doc-1" } },
      ...at,
    });
    expect(failure).toMatchObject({ kind: "duplicate-document", sourceDocumentId: "doc-1" });
  });

  it("415 UPLOAD_REJECTED and 400 VALIDATION_FAILED keep their details", () => {
    expect(parseApiError({ status: 415, body: { detail: { code: "UPLOAD_REJECTED" } }, ...at }).kind).toBe(
      "upload-rejected",
    );
    const validation = parseApiError({
      status: 400,
      body: {
        detail: {
          code: "VALIDATION_FAILED",
          field_errors: [{ field: "admission_year", code: "range", message: "Năm không hợp lệ" }],
        },
      },
      ...at,
    });
    expect(validation).toMatchObject({ kind: "validation", fieldErrors: { admission_year: "Năm không hợp lệ" } });
  });

  it("falls back to retry_after_ms when no header is sent", () => {
    const failure = parseApiError({
      status: 429,
      body: { detail: { code: "RATE_LIMITED", retry_after_ms: 2500 } },
      ...at,
    });
    expect(failure).toMatchObject({ kind: "rate-limited", retryAfterSeconds: 3 });
  });
});

describe("work-area invariant: one primary action", () => {
  const workarea = (...children: ReactElement[]) =>
    renderToStaticMarkup(createElement("section", { "data-workarea": "" }, ...children));
  const primaryCount = (html: string) => html.match(/data-variant="primary"/g)?.length ?? 0;

  it("counts one primary when the area also has secondary, ghost and danger actions", () => {
    const html = workarea(
      createElement(PrimaryButton, null, "Tải lên"),
      createElement(SecondaryButton, null, "Huỷ"),
      createElement(GhostButton, null, "Xem"),
      createElement(DangerButton, null, "Xoá"),
    );
    expect(primaryCount(html)).toBe(1);
  });

  it("exception bar and badges add no primary action", () => {
    const html = workarea(
      createElement(ExceptionBar, { title: "Không kết nối được", action: { label: "Thử lại", onClick: () => {} } }),
      createElement(SidebarBadge, { count: 2 }),
      createElement(StatusPill, { tone: "overdue", label: "Quá hạn" }),
    );
    expect(primaryCount(html)).toBe(0);
  });

  it("a second primary button is detected", () => {
    const html = workarea(createElement(PrimaryButton, null, "A"), createElement(PrimaryButton, null, "B"));
    expect(primaryCount(html)).toBe(2);
  });
});
