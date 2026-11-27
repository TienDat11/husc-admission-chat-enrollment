import { RECORD_TYPE_LABELS } from "@/lib/adminValidation";
import type { CanonicalRecord } from "@/lib/adminApi";

export const recordTitle = (record: CanonicalRecord) =>
  `${RECORD_TYPE_LABELS[record.record_type] ?? record.record_type} ${record.admission_year}`;
