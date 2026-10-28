// Field-level validation for the admissions record form — ticket 19.
//
// The rules mirror the ones the backend enforces, and they exist for a
// different reason: the backend refuses bad data, this refuses *saving* it, so
// the admissions officer sees the problem at the input instead of after a
// round trip. The two are deliberately not deduplicated — the backend's rules
// are the contract, this is the form's copy of them, and the 422 path is what
// keeps them honest when they drift.
import { adminConfig } from "@/constants/adminConfig";
import { structuredValues, type CanonicalRecord, type RecordStatus } from "./adminApi";

/** Record kinds the canonical schema declares (RecordType, ticket 02). */
export const RECORD_TYPES = [
  "NGANH",
  "TO_HOP",
  "DIEM_CHUAN",
  "HOC_PHI",
  "THOI_GIAN",
  "TO_CHUC",
  "CHINH_SACH",
  "CHI_TIEU",
  "DIEU_KIEN",
] as const;

export const RECORD_TYPE_LABELS: Record<string, string> = {
  NGANH: "Ngành",
  TO_HOP: "Tổ hợp",
  DIEM_CHUAN: "Điểm chuẩn",
  HOC_PHI: "Học phí",
  THOI_GIAN: "Thời gian",
  TO_CHUC: "Tổ chức",
  CHINH_SACH: "Chính sách",
  CHI_TIEU: "Chỉ tiêu",
  DIEU_KIEN: "Điều kiện",
};

/** Source authority tiers (AuthorityLevel, ticket 02), highest first. */
export const AUTHORITY_LEVELS = [
  "husc_canonical",
  "husc_official",
  "hue_university",
  "ministry",
  "prior_year_official",
  "derived_faq",
] as const;

export const AUTHORITY_LABELS: Record<string, string> = {
  husc_canonical: "HUSC — canonical",
  husc_official: "HUSC — chính thức",
  hue_university: "Đại học Huế",
  ministry: "Bộ GD&ĐT",
  prior_year_official: "Chính thức năm trước",
  derived_faq: "Suy ra từ FAQ",
};

export const STATUS_LABELS: Record<RecordStatus, string> = {
  draft: "Nháp",
  review: "Chờ duyệt",
  approved: "Đã duyệt",
  published: "Đã công bố",
  archived: "Lưu trữ",
};

export const RECORD_STATUSES = Object.keys(STATUS_LABELS) as RecordStatus[];

/** Every field the form renders, as strings (an input's value is text). */
export interface RecordFormValues {
  record_id: string;
  record_type: string;
  admission_year: string;
  effective_from: string;
  effective_to: string;
  authority_level: string;
  source_document_id: string;
  source_url: string;
  source_page: string;
  ma_nganh: string;
  ten_nganh: string;
  text: string;
}

export type RecordFieldErrors = Partial<Record<keyof RecordFormValues, string>>;

export const EMPTY_RECORD_FORM: RecordFormValues = {
  record_id: "",
  record_type: "CHI_TIEU",
  // The admissions year the deployment is configured for, not the wall clock:
  // the office edits the year the system is running on.
  admission_year:
    adminConfig.find((entry) => entry.key === "CURRENT_ADMISSION_YEAR")?.value ?? "",
  effective_from: "",
  effective_to: "",
  authority_level: "husc_official",
  source_document_id: "",
  source_url: "",
  source_page: "",
  ma_nganh: "",
  ten_nganh: "",
  text: "",
};

const RECORD_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:%-]*$/;
const ISO_DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

/** A date input's value as a comparable string, or null when it is not a date. */
function isoDate(value: string): string | null {
  const trimmed = value.trim();
  if (!ISO_DATE_PATTERN.test(trimmed)) return null;
  const parsed = new Date(`${trimmed}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return null;
  // Round-trip check: rejects 2026-02-31, which Date would silently roll over.
  return parsed.toISOString().slice(0, 10) === trimmed ? trimmed : null;
}

export function validateRecordForm(values: RecordFormValues): RecordFieldErrors {
  const errors: RecordFieldErrors = {};

  const recordId = values.record_id.trim();
  if (!recordId) {
    errors.record_id = "Mã record không được để trống.";
  } else if (!RECORD_ID_PATTERN.test(recordId)) {
    errors.record_id =
      "Mã record chỉ gồm chữ, số và các ký tự . _ : % - (không dấu cách, không dấu tiếng Việt).";
  }

  if (!(RECORD_TYPES as readonly string[]).includes(values.record_type)) {
    errors.record_type = "Chọn loại dữ liệu.";
  }

  const year = Number(values.admission_year);
  if (!values.admission_year.trim()) {
    errors.admission_year = "Năm tuyển sinh không được để trống.";
  } else if (!Number.isInteger(year) || year < 2000 || year > 2100) {
    errors.admission_year = "Năm tuyển sinh phải là số nguyên từ 2000 đến 2100.";
  }

  const from = isoDate(values.effective_from);
  if (!values.effective_from.trim()) {
    errors.effective_from = "Ngày hiệu lực không được để trống.";
  } else if (!from) {
    errors.effective_from = "Ngày hiệu lực phải theo dạng YYYY-MM-DD.";
  }

  const to = values.effective_to.trim() ? isoDate(values.effective_to) : null;
  if (values.effective_to.trim() && !to) {
    errors.effective_to = "Ngày hết hiệu lực phải theo dạng YYYY-MM-DD.";
  } else if (from && to && from > to) {
    errors.effective_to = "Ngày hết hiệu lực phải sau hoặc bằng ngày hiệu lực.";
  }

  if (!(AUTHORITY_LEVELS as readonly string[]).includes(values.authority_level)) {
    errors.authority_level = "Chọn mức thẩm quyền của nguồn.";
  }

  const url = values.source_url.trim();
  if (url && !/^https?:\/\/\S+$/.test(url)) {
    errors.source_url = "URL nguồn phải bắt đầu bằng http:// hoặc https://.";
  }

  const page = values.source_page.trim();
  if (page && !(Number.isInteger(Number(page)) && Number(page) >= 1)) {
    errors.source_page = "Số trang phải là số nguyên lớn hơn 0.";
  }

  if (!values.text.trim()) {
    errors.text = "Nội dung không được để trống.";
  }

  return errors;
}

/** A stored record as form values. */
export function recordToFormValues(record: CanonicalRecord): RecordFormValues {
  const values = structuredValues(record);
  const asText = (value: unknown) =>
    typeof value === "string" ? value : value === null || value === undefined ? "" : String(value);

  return {
    record_id: record.record_id,
    record_type: record.record_type,
    admission_year: asText(record.admission_year),
    effective_from: asText(record.effective_from),
    effective_to: asText(record.effective_to),
    authority_level: record.authority_level,
    source_document_id: asText(record.source_document_id),
    source_url: asText(record.source_url),
    source_page: asText(record.source_page),
    ma_nganh: asText(values.ma_nganh),
    ten_nganh: asText(values.ten_nganh),
    text: record.text,
  };
}

/** The record body a create request sends: a draft, version 1. */
export function formValuesToNewRecord(values: RecordFormValues): Record<string, unknown> {
  const structured: Record<string, unknown> = {};
  if (values.ma_nganh.trim()) structured.ma_nganh = values.ma_nganh.trim();
  if (values.ten_nganh.trim()) structured.ten_nganh = values.ten_nganh.trim();

  const record: Record<string, unknown> = {
    record_id: values.record_id.trim(),
    record_type: values.record_type,
    version: 1,
    // A record created here is always a draft: publishing is a lifecycle
    // decision (submit -> approve -> publish), never a side effect of saving.
    status: "draft",
    admission_year: Number(values.admission_year),
    effective_from: values.effective_from.trim(),
    effective_to: values.effective_to.trim() || null,
    authority_level: values.authority_level,
    source_document_id: values.source_document_id.trim() || null,
    source_url: values.source_url.trim() || null,
    source_page: values.source_page.trim() ? Number(values.source_page) : null,
    text: values.text.trim(),
  };
  if (Object.keys(structured).length > 0) record.values = structured;
  return record;
}

/**
 * The change set an edit request sends.
 *
 * `values` is sent as the *whole* structured payload — the loaded record's
 * other keys merged with the two the form edits — because the backend replaces
 * that payload wholesale. Sending only the form's two keys would silently drop
 * the extractor's `chi_tieu`, `to_hop_codes`, … on every hand edit.
 */
export function formValuesToEdit(
  values: RecordFormValues,
  current: CanonicalRecord,
): Record<string, unknown> {
  const structured = { ...structuredValues(current) };
  if (values.ma_nganh.trim()) structured.ma_nganh = values.ma_nganh.trim();
  else delete structured.ma_nganh;
  if (values.ten_nganh.trim()) structured.ten_nganh = values.ten_nganh.trim();
  else delete structured.ten_nganh;

  return {
    expected_version: current.version,
    admission_year: Number(values.admission_year),
    effective_from: values.effective_from.trim(),
    effective_to: values.effective_to.trim() || null,
    authority_level: values.authority_level,
    source_document_id: values.source_document_id.trim() || null,
    source_url: values.source_url.trim() || null,
    source_page: values.source_page.trim() ? Number(values.source_page) : null,
    values: structured,
    text: values.text.trim(),
  };
}

/** Whether a record may be published: the schema refuses `published` without provenance. */
export function canPublish(record: CanonicalRecord): boolean {
  return Boolean((record.source_document_id ?? "").trim());
}
