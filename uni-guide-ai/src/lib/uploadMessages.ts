// Friendly Vietnamese for upload/parse warnings and job stages shown in step 2.
// Codes, env var names and stage ids never reach the screen.

const WARNING_MESSAGES: Record<string, string> = {
  TABLE_TIER_DISABLED: "Bảng trong tệp được đọc ở chế độ văn bản — hãy kiểm tra số liệu trong bảng bằng tay.",
};

const FALLBACK_WARNING = "Cần kiểm tra lại tài liệu này.";

// A leading "CODE:" prefix, e.g. "TABLE_TIER_DISABLED: ...".
const CODE_PREFIX = /^([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\s*:\s*/;
// Env-style assignments such as "(PARSE_DOCLING_ENABLED=false)".
const ENV_ASSIGNMENT = /\(?\s*[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*=\S*\s*\)?/g;
// Any remaining SNAKE_CASE identifier, e.g. a bare code or an env name.
const SNAKE_CODE = /\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b/g;

export function uploadWarningMessage(raw: string): string {
  const prefix = CODE_PREFIX.exec(raw);
  if (prefix !== null) {
    const known = WARNING_MESSAGES[prefix[1]];
    if (known !== undefined) return known;
  }
  const body = prefix !== null ? raw.slice(prefix[0].length) : raw;
  const human = body
    .replace(ENV_ASSIGNMENT, "")
    .replace(SNAKE_CODE, "")
    .replace(/\s+/g, " ")
    .replace(/\s+([,.;:])/g, "$1")
    .trim();
  if (human === "" || /^[A-Z0-9_\s]+$/.test(human)) return FALLBACK_WARNING;
  return human;
}

const STAGE_LABELS: Record<string, string> = {
  parsed: "Đã đọc xong",
  done: "Hoàn tất",
};

export function jobStageLabel(stage: unknown): string {
  if (typeof stage !== "string") return "Đang xử lý";
  const label = STAGE_LABELS[stage];
  return label ?? "Đang xử lý";
}
