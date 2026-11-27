import type { CanonicalRecord } from "@/lib/adminApi";
import { handleTablistKey } from "@/lib/tablistKeys";
import { cn } from "@/lib/utils";
import { recordTitle } from "./recordTitle";

export type Column = "current" | "proposed" | "pdf";

const COLUMNS: { id: Column; tab: string }[] = [
  { id: "current", tab: "Đang chạy" },
  { id: "proposed", tab: "Đề xuất" },
  { id: "pdf", tab: "PDF gốc" },
];

type Change = { field: string; old?: unknown; new?: unknown };

type Props = {
  record: CanonicalRecord;
  column: Column;
  onColumnChange: (column: Column) => void;
  loading: boolean;
  failed: boolean;
  changes: ReadonlyArray<Change>;
  baselineVersion: unknown;
};

/** Three-column comparison: running version, proposed text, source PDF. Tabs on mobile. */
export function ReviewDiffPanel({ record, column, onColumnChange, loading, failed, changes, baselineVersion }: Props) {
  return (
    <section
      aria-label="So sánh bản đang chạy, đề xuất mới và bản gốc PDF"
      id="so-sanh-panel"
      role="tabpanel"
      className="rounded-[12px] border border-[#D6DEEA] bg-white p-5"
    >
      <h2 className="!m-0 font-display text-[18px] font-semibold text-[#0F1B2D]">{recordTitle(record)}</h2>

      <div
        role="tablist"
        aria-label="Chọn cột so sánh"
        className="mt-4 flex gap-2 md:hidden"
        onKeyDown={(event) => handleTablistKey(event, (index) => onColumnChange(COLUMNS[index].id))}
      >
        {COLUMNS.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-controls="so-sanh-panel"
            aria-selected={column === item.id}
            tabIndex={column === item.id ? 0 : -1}
            onClick={() => onColumnChange(item.id)}
            className={cn(
              "min-h-11 flex-1 rounded-[8px] border px-2 text-[13px] font-medium",
              column === item.id
                ? "border-brand-navy bg-brand-navy text-white"
                : "border-[#D6DEEA] bg-white text-[#0F1B2D]",
            )}
          >
            {item.tab}
          </button>
        ))}
      </div>

      {loading ? (
        <p role="status" className="mt-4 text-zinc-600">
          Đang tải so sánh…
        </p>
      ) : failed ? (
        <p role="alert" className="note bad mt-4">
          Không tải được bản so sánh.
        </p>
      ) : (
        <div className="mt-4 grid gap-4 md:grid-cols-3">
          <div className={cn("rounded-[8px] bg-[#F3F5F9] p-4", column === "current" ? "block" : "hidden", "md:block")}>
            <h3 className="!m-0 font-display text-[15px] font-semibold">Đang chạy trên Bot</h3>
            {baselineVersion == null ? (
              <p className="mt-2 text-[14px] text-[#474747]">Chưa có bản đang chạy để so sánh.</p>
            ) : changes.length === 0 ? (
              <p className="mt-2 text-[14px] text-[#474747]">Không có thay đổi.</p>
            ) : (
              <ul className="mt-2 list-none space-y-2 p-0 text-[14px]">
                {changes.map((change) => (
                  <li key={change.field}>
                    <span className="font-medium">{change.field}:</span>{" "}
                    {change.old != null ? String(change.old) : "chưa có"}
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className={cn("rounded-[8px] border border-brand-navy p-4", column === "proposed" ? "block" : "hidden", "md:block")}>
            <h3 className="!m-0 font-display text-[15px] font-semibold">Đề xuất mới</h3>
            <p className="mt-2 whitespace-pre-line text-[14px] leading-6 text-[#0F1B2D]">{record.text}</p>
            {changes.length > 0 ? (
              <ul className="mt-3 list-none space-y-1 p-0 text-[13px]">
                {changes.map((change) => (
                  <li key={change.field}>
                    {change.field}:{" "}
                    <span className="changed">{String(change.new ?? "")} (đã đổi)</span>
                  </li>
                ))}
              </ul>
            ) : null}
          </div>

          <div className={cn("rounded-[8px] bg-[#F3F5F9] p-4", column === "pdf" ? "block" : "hidden", "md:block")}>
            <h3 className="!m-0 font-display text-[15px] font-semibold">Bản gốc PDF</h3>
            <p className="mt-2 font-mono text-[13px] text-[#0F1B2D]">
              {record.source_document_id ?? "không ghi nhận"}
              {record.source_page ? ` · trang ${record.source_page}` : ""}
            </p>
            <p className="mt-2 text-[13px] text-[#474747]">Đối chiếu nội dung trực tiếp trên tệp gốc.</p>
            {record.source_url ? (
              <a
                href={record.source_url}
                target="_blank"
                rel="noreferrer"
                className="mt-2 inline-flex min-h-11 items-center text-[14px] font-semibold text-brand-navy underline"
              >
                Mở tài liệu gốc
              </a>
            ) : null}
          </div>
        </div>
      )}
    </section>
  );
}
