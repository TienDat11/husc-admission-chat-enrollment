// Admin error contract, PL-7 and the "Xử lý lỗi phía FE" rules.
//
// Every admin request reaches this module through `adminRequest` (adminApi.ts),
// which attaches the `ApiFailure` built here to the thrown AdminApiError. Screens
// switch on `kind`; none of them parses codes or headers itself.
//
// The backend answers either as `{detail: {code, message, ...}}` (HTTPException)
// or as the bare envelope. Both are read. `Retry-After` (seconds) is the header
// the backend sets on 423/429/503; `retry_after_ms` in the body is the fallback.
import { z } from "zod";

export type ApiFailure =
  | { kind: "session-expired"; status: 401; redirectTo: string; message: string }
  | { kind: "locked"; status: 423; retryAfterSeconds: number; message: string }
  | { kind: "account-disabled"; status: 403; message: string }
  | { kind: "forbidden"; status: 403; message: string }
  | {
      kind: "writes-frozen";
      status: 503;
      retryAfterSeconds: number | null;
      preserveForm: true;
      message: string;
    }
  | {
      kind: "version-conflict";
      status: 409;
      currentVersion: number | null;
      canReload: true;
      message: string;
    }
  | {
      kind: "duplicate-document";
      status: 409;
      sourceDocumentId: string | null;
      message: string;
    }
  | { kind: "upload-rejected"; status: 415; message: string }
  | { kind: "validation"; status: 400; fieldErrors: Record<string, string>; message: string }
  | { kind: "rate-limited"; status: 429; retryAfterSeconds: number | null; message: string }
  | { kind: "unavailable"; status: number; code: string | null; message: string }
  | { kind: "error"; status: number; code: string | null; message: string };

const MESSAGE: Record<string, string> = {
  SESSION_EXPIRED: "Phiên đăng nhập đã hết hạn.",
  ACCOUNT_DISABLED: "Tài khoản đã bị vô hiệu hóa",
  FORBIDDEN_ACTION: "Bạn không có quyền thực hiện thao tác này.",
  WRITES_FROZEN: "Hệ thống đang tạm khoá ghi để cập nhật. Dữ liệu bạn nhập vẫn được giữ lại.",
  VERSION_CONFLICT: "Bản ghi đã được cập nhật bởi người khác. Tải lại để xem bản mới nhất.",
  DOCUMENT_ALREADY_REGISTERED: "Tài liệu này đã được đăng ký trước đó.",
  UPLOAD_REJECTED: "Tệp không đúng định dạng được hỗ trợ.",
  VALIDATION_FAILED: "Dữ liệu không hợp lệ.",
  RATE_LIMITED: "Quá nhiều yêu cầu. Thử lại sau.",
};

// The envelope is parsed once, at the boundary. Unknown fields pass through
// untouched; a field that is absent or of the wrong type is simply undefined.
const FieldErrorSchema = z.object({
  field: z.string(),
  code: z.string().optional(),
  message: z.string().optional(),
});

const EnvelopeSchema = z
  .object({
    code: z.string().optional(),
    message: z.string().optional(),
    field_errors: z.array(FieldErrorSchema).optional(),
    retry_after_ms: z.number().optional(),
    current_version: z.number().optional(),
    source_document_id: z.string().optional(),
  })
  .passthrough();

const BodySchema = z.object({ detail: z.unknown().optional() }).passthrough();

type Envelope = z.infer<typeof EnvelopeSchema>;

/** The envelope the backend sent, from `detail` when nested there, else the body. */
function readEnvelope(body: unknown): { envelope: Envelope; detailText: string | null } {
  const parsedBody = BodySchema.safeParse(body);
  const detail = parsedBody.success ? parsedBody.data.detail : undefined;
  const nested = EnvelopeSchema.safeParse(detail);
  const flat = EnvelopeSchema.safeParse(body);
  const detailText = z.string().safeParse(detail);
  let envelope: Envelope = {};
  if (nested.success) envelope = nested.data;
  else if (flat.success) envelope = flat.data;
  return { envelope, detailText: detailText.success ? detailText.data : null };
}

/** `Retry-After` as whole seconds (the delta form the backend sends). */
function parseRetryAfter(header: string | null | undefined): number | null {
  if (header == null) return null;
  const seconds = Number(header.trim());
  return Number.isFinite(seconds) && seconds >= 0 ? Math.ceil(seconds) : null;
}

export interface ParseApiErrorInput {
  status: number;
  body: unknown;
  /** The raw `Retry-After` header, if any. */
  retryAfterHeader?: string | null;
  /** Where the user was, so a 401 can return them there after sign-in. */
  currentPath: string;
}

/** Turn one failed admin response into the ApiFailure a screen renders. */
export function parseApiError({
  status,
  body,
  retryAfterHeader,
  currentPath,
}: ParseApiErrorInput): ApiFailure {
  const { envelope, detailText } = readEnvelope(body);
  const code = envelope.code ?? null;
  const fallback = MESSAGE[code ?? ""] ?? "Yêu cầu thất bại.";
  const message = envelope.message ?? detailText ?? fallback;
  const retryFromBody =
    envelope.retry_after_ms !== undefined ? Math.ceil(envelope.retry_after_ms / 1000) : null;
  const retryAfterSeconds = parseRetryAfter(retryAfterHeader) ?? retryFromBody;

  if (status === 401 || code === "SESSION_EXPIRED") {
    return {
      kind: "session-expired",
      status: 401,
      redirectTo: `/quan-ly/dang-nhap?next=${encodeURIComponent(currentPath)}`,
      message: MESSAGE.SESSION_EXPIRED,
    };
  }
  if (status === 423) {
    return { kind: "locked", status: 423, retryAfterSeconds: retryAfterSeconds ?? 0, message };
  }
  if (status === 403 && code === "ACCOUNT_DISABLED") {
    return { kind: "account-disabled", status: 403, message: MESSAGE.ACCOUNT_DISABLED };
  }
  if (status === 403) {
    return { kind: "forbidden", status: 403, message: envelope.message ?? MESSAGE.FORBIDDEN_ACTION };
  }
  if (status === 503 && code === "WRITES_FROZEN") {
    return {
      kind: "writes-frozen",
      status: 503,
      retryAfterSeconds,
      preserveForm: true,
      message: MESSAGE.WRITES_FROZEN,
    };
  }
  if (status === 409 && code === "VERSION_CONFLICT") {
    return {
      kind: "version-conflict",
      status: 409,
      currentVersion: envelope.current_version ?? null,
      canReload: true,
      message: MESSAGE.VERSION_CONFLICT,
    };
  }
  if (status === 409 && code === "DOCUMENT_ALREADY_REGISTERED") {
    return {
      kind: "duplicate-document",
      status: 409,
      sourceDocumentId: envelope.source_document_id ?? null,
      message: MESSAGE.DOCUMENT_ALREADY_REGISTERED,
    };
  }
  if (status === 415 && code === "UPLOAD_REJECTED") {
    return { kind: "upload-rejected", status: 415, message: MESSAGE.UPLOAD_REJECTED };
  }
  if (status === 400 && code === "VALIDATION_FAILED") {
    const fieldErrors: Record<string, string> = {};
    for (const entry of envelope.field_errors ?? []) {
      fieldErrors[entry.field] = entry.message ?? MESSAGE.VALIDATION_FAILED;
    }
    return { kind: "validation", status: 400, fieldErrors, message: MESSAGE.VALIDATION_FAILED };
  }
  if (status === 429) {
    return { kind: "rate-limited", status: 429, retryAfterSeconds, message: MESSAGE.RATE_LIMITED };
  }
  if (status === 503) {
    return { kind: "unavailable", status: 503, code, message };
  }
  return { kind: "error", status, code, message };
}
