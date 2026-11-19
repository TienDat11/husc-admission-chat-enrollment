"use strict";

/**
 * ChunkHistogram — unit U7.
 *
 * Renders the chunk preview card list plus a token histogram from
 * ``POST /admin/previews``. Flag warnings without using colour as the sole
 * signal (each warning row also carries an icon + text label).
 */
import type { PreviewItem, PreviewResponse } from "@/lib/adminApi";

const BUCKETS: Array<{ key: string; label: string }> = [
  { key: "<50", label: "Dưới 50" },
  { key: "50-249", label: "50–249" },
  { key: "250-420", label: "250–420" },
  { key: "421-512", label: "421–512" },
  { key: ">512", label: "Trên 512" },
];

interface ChunkHistogramProps {
  preview: PreviewResponse;
}

export function ChunkHistogram({ preview }: ChunkHistogramProps) {
  const { items, histogram, warnings } = preview;
  const total = items.reduce((sum, it) => sum + it.token_count, 0);
  const avg = items.length ? Math.round(total / items.length) : 0;

  return (
    <section className="space-y-6" data-workarea="chunk-preview">
      <h2 className="text-lg font-semibold">Xem trước chunk</h2>

      {/* Histogram bar row */}
      <div className="grid grid-cols-5 gap-2 text-xs border rounded-lg p-2 bg-muted/50">
        {BUCKETS.map((bucket) => {
          const count = histogram[bucket.key] ?? 0;
          const pct = items.length ? Math.round((count / items.length) * 100) : 0;
          return (
            <div key={bucket.key} className="flex flex-col items-center gap-1">
              <span className="font-medium">{count}</span>
              <div
                className="w-full h-2 bg-muted rounded-full overflow-hidden"
                title={`${bucket.label}: ${count} chunk (${pct}%)`}
              >
                <div
                  className="h-full bg-primary"
                  style={{ width: `${Math.max(pct, 5)}%` }}
                />
              </div>
              <span className="text-muted-foreground">{bucket.label}</span>
            </div>
          );
        })}
      </div>

      {/* Summary chips */}
      <div className="flex gap-2 text-sm text-muted-foreground">
        <span>Tổng chunks: <span className="font-medium">{items.length}</span></span>
        <span>•</span>
        <span>Tổng token: <span className="font-medium">{total}</span></span>
        <span>•</span>
        <span>TB/trung bình: <span className="font-medium">{avg}</span></span>
      </div>

      {/* Warnings — colour + icon + text, never colour alone */}
      {warnings.length > 0 && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-3">
          <h3 className="text-sm font-medium text-destructive flex items-center gap-1">
            <span aria-label="Cảnh báo">⚠</span>
            Cảnh báo ({warnings.length})
          </h3>
          <ul className="mt-2 text-sm list-disc list-inside text-destructive/80 space-y-1">
            {warnings.map((w, i) => (
              <li key={i}>
                <span className="font-bold">[{w}]</span>{" "}
                {warningLabel(w)}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Chunk cards */}
      <div className="grid gap-3">
        {items.map((it) => (
          <div
            key={it.chunk_id}
            className="border rounded-lg p-3 bg-card text-sm"
          >
            <div className="flex justify-between items-start gap-2">
              <span className="font-mono text-xs text-muted-foreground">
                #{it.ordinal + 1}
              </span>
              <span className="font-mono text-xs text-muted-foreground">
                {it.token_count} token
              </span>
            </div>
            <p className="mt-1 leading-relaxed">{it.text}</p>
            {it.parent_unit_id && (
              <p className="mt-1 text-xs text-muted-foreground">
                Đơn vị cha: {it.parent_unit_id}
              </p>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

function warningLabel(code: string): string {
  const labels: Record<string, string> = {
    SHORT_FRAGMENT: "Đoạn văn quá ngắn — có thể bị chia không đúng ngữ cảnh.",
    MISSING_COMBINATION_CODE: "Thiếu mã tổ hợp (ví dụ A00, C03) trong đoạn.",
    UNANCHORED_LIST: "Danh sách chưa gán tọa độ (unanchored).",
    TABLE_CELL_DROP_SUSPECTED: "Bảng có thể thiếu dòng — kiểm tra dòng bị rơi.",
    OCR_LOW_CONFIDENCE: "Độ tin cậy OCR thấp — nên OCR lại trang.",
    LAYOUT_UNKNOWN: "Bố cục chưa xác định được.",
    AMBIGUOUS_SUPERSEDED: "Bản ghi bị thay thế chưa rõ ràng.",
  };
  return labels[code] ?? code;
}
