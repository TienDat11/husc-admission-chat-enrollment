// Vietnamese labels for block warning codes. A code missing from this map is
// never shown raw: upper-case codes fall back to a neutral label.
export const BLOCK_WARNING_LABELS: Record<string, string> = {
  TABLE_TIER_DISABLED: "Bảng cần kiểm tra thủ công",
  TABLE_CELL_DROP_SUSPECTED: "Bảng thiếu dòng",
  OCR_LOW_CONFIDENCE: "Chữ khó đọc",
  LAYOUT_UNKNOWN: "Bố cục chưa xác định",
  UNANCHORED_DRAFT: "Chưa gán tọa độ",
  AMBIGUOUS_SUPERSEDED: "Thay thế không rõ",
};

export function warningLabel(warning: string): string {
  const label = BLOCK_WARNING_LABELS[warning];
  if (label !== undefined) return label;
  if (/^[A-Z][A-Z0-9_]*$/.test(warning)) return "Cần xem lại";
  return warning;
}
