import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  listRecords,
  type CanonicalRecord,
  type RecordFilters,
  type RecordStatus,
} from "@/lib/adminApi";
import { RECORD_TYPE_LABELS } from "@/lib/adminValidation";
import { cn } from "@/lib/utils";

const YEARS = ["2026", "2025"];

type Bucket = "all" | "published" | "pending" | "archived" | "expiring";

const BUCKETS: { id: Bucket; label: string }[] = [
  { id: "all", label: "Tất cả" },
  { id: "published", label: "Bot đang sử dụng" },
  { id: "pending", label: "Chờ phát hành" },
  { id: "archived", label: "Đã ngừng sử dụng" },
  { id: "expiring", label: "Sắp hết hạn" },
];

const PENDING_STATUSES: RecordStatus[] = ["draft", "review", "approved"];
const PAGE_SIZE = 25;

/** A published record whose validity ends within the next 30 days. */
const isExpiring = (record: CanonicalRecord) => {
  if (record.status !== "published" || !record.effective_to) return false;
  const end = new Date(record.effective_to);
  const limit = new Date();
  limit.setDate(limit.getDate() + 30);
  return end.getTime() <= limit.getTime();
};

const matchesBucket = (record: CanonicalRecord, bucket: Bucket) => {
  switch (bucket) {
    case "all":
      return true;
    case "published":
      return record.status === "published";
    case "pending":
      return PENDING_STATUSES.includes(record.status);
    case "archived":
      return record.status === "archived";
    case "expiring":
      return isExpiring(record);
  }
};

const formatDate = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleDateString("vi-VN") : null;

/** What the Bot serves for this record: the badge text and its colour. */
const botStatus = (record: CanonicalRecord) => {
  if (record.status === "published")
    return { label: "Bot đang sử dụng", tone: "bg-[#E7F4E8] text-[#1B5E20]" };
  if (record.status === "archived")
    return { label: "Đã ngừng sử dụng", tone: "bg-[#E6E9EF] text-[#474747]" };
  return { label: "Chờ phát hành", tone: "bg-[#FFF3DC] text-[#6B3E00]" };
};

const BADGE = "inline-flex h-6 items-center rounded-full px-2.5 text-[12px] font-medium";
const EXPIRING_TONE = "bg-[#FFF3DC] text-[#6B3E00]";

/**
 * "Kho thông tin tuyển sinh" (frame M4NF9F): search, filter chips with counts,
 * one row per record, and the selected record's detail beside the table.
 * Year and search go to the server; type and status filter the loaded rows,
 * so the chip counts reflect the whole loaded set.
 */
const Vault = () => {
  const [text, setText] = useState("");
  const [year, setYear] = useState("");
  const [recordType, setRecordType] = useState("");
  const [bucket, setBucket] = useState<Bucket>("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [page, setPage] = useState(0);

  const list = useQuery({
    queryKey: ["admin", "vault", { text, year }],
    queryFn: () =>
      listRecords(null, {
        ...(text.trim() ? { nganh: text.trim() } : {}),
        ...(year ? { year: Number(year) } : {}),
      } satisfies RecordFilters),
    retry: false,
  });

  const loaded = list.data?.records ?? [];
  const typed = recordType
    ? loaded.filter((record) => record.record_type === recordType)
    : loaded;
  const shown = typed.filter((record) => matchesBucket(record, bucket));
  const selected = shown.find((record) => record.record_id === selectedId) ?? shown[0] ?? null;
  const pageCount = Math.max(1, Math.ceil(shown.length / PAGE_SIZE));
  const current = Math.min(page, pageCount - 1);
  const pageRows = shown.slice(current * PAGE_SIZE, (current + 1) * PAGE_SIZE);

  if (list.isPending) {
    return (
      <p role="status" aria-live="polite" className="text-zinc-600">
        Đang tải kho thông tin…
      </p>
    );
  }

  if (list.isError) {
    return (
      <div role="alert" className="note bad">
        Chưa tải được kho thông tin. Kiểm tra kết nối mạng rồi thử lại.
      </div>
    );
  }

  return (
    <section data-workarea="" aria-labelledby="vault-title" className="space-y-5">
      <header>
        <h1
          id="vault-title"
          className="!m-0 font-display text-[24px] font-bold leading-8 text-[#0F1B2D]"
        >
          Kho thông tin tuyển sinh
        </h1>
        <p className="mt-2 text-[15px] leading-6 text-[#474747]">
          Toàn bộ các mục thông tin chuẩn đang được Bot sử dụng hoặc lưu trữ theo từng năm.
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-3">
        <label htmlFor="kho-tim" className="sr-only">
          Tìm theo chữ
        </label>
        <input
          id="kho-tim"
          type="search"
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Tìm theo mã ngành hoặc tên ngành"
          className="!min-w-[16rem] flex-1 !border-[1px] !border-[#D6DEEA] bg-white px-4"
        />
        <label htmlFor="kho-nam" className="sr-only">
          Năm
        </label>
        <select
          id="kho-nam"
          value={year}
          onChange={(event) => setYear(event.target.value)}
          className="!border-[1px] !border-[#D6DEEA] bg-white px-4"
        >
          <option value="">Tất cả năm</option>
          {YEARS.map((value) => (
            <option key={value} value={value}>
              Năm {value}
            </option>
          ))}
        </select>
        <label htmlFor="kho-loai" className="sr-only">
          Loại
        </label>
        <select
          id="kho-loai"
          value={recordType}
          onChange={(event) => setRecordType(event.target.value)}
          className="!border-[1px] !border-[#D6DEEA] bg-white px-4"
        >
          <option value="">Tất cả loại</option>
          {Object.entries(RECORD_TYPE_LABELS).map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </div>

      <div role="group" aria-label="Lọc theo trạng thái" className="flex flex-nowrap gap-2 overflow-x-auto">
        {BUCKETS.map(({ id, label }) => {
          const active = bucket === id;
          const count = typed.filter((record) => matchesBucket(record, id)).length;
          return (
            <button
              key={id}
              type="button"
              aria-pressed={active}
              onClick={() => setBucket(id)}
              className={cn(
                "inline-flex min-h-11 items-center rounded-full border px-4 text-[14px] font-medium",
                active
                  ? "border-[#143980] bg-[#143980] text-white"
                  : "border-[#D6DEEA] bg-white text-[#0F1B2D] hover:bg-[#E6ECF7]",
              )}
            >
              {label} ({count})
            </button>
          );
        })}
      </div>

      {shown.length === 0 ? (
        <div className="note info" role="status">
          <p>
            Không có thông tin nào khớp bộ lọc hiện tại. Thử bỏ bớt bộ lọc, hoặc thêm tài liệu
            mới.
          </p>
          <Link to="/quan-ly/them" className="btn">
            Thêm tài liệu
          </Link>
        </div>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,7fr)_minmax(280px,3fr)]">
          <div className="table-wrap rounded-[12px] border border-[#D6DEEA] bg-white">
            <table>
              <thead>
                <tr>
                  <th scope="col">Nội dung</th>
                  <th scope="col">Áp dụng</th>
                  <th scope="col">Nguồn gốc</th>
                  <th scope="col">Trạng thái trên Bot</th>
                </tr>
              </thead>
              <tbody>
                {pageRows.map((record) => {
                  const active = selected?.record_id === record.record_id;
                  const status = botStatus(record);
                  const label = `${RECORD_TYPE_LABELS[record.record_type] ?? record.record_type} ${record.admission_year}`;
                  return (
                    <tr key={record.record_id} className={cn(active && "bg-[#E6ECF7]")}>
                      <td>
                        <button
                          type="button"
                          aria-pressed={active}
                          onClick={() => setSelectedId(record.record_id)}
                          className="block min-h-0 text-left"
                        >
                          <span className="block font-display text-[14px] font-semibold text-[#0F1B2D]">
                            {label}
                          </span>
                          <span className="block max-w-[28rem] truncate text-[13px] text-[#6B7686]">
                            {record.text}
                          </span>
                        </button>
                      </td>
                      <td className="text-[13px]">
                        Năm {record.admission_year}
                        <br />
                        {record.effective_to
                          ? `đến ${formatDate(record.effective_to)}`
                          : "không giới hạn"}
                      </td>
                      <td className="text-[13px]">
                        <span className="font-mono">
                          {record.source_document_id ?? "không ghi nhận"}
                        </span>
                        {record.source_page ? <> · trang {record.source_page}</> : null}
                      </td>
                      <td>
                        <span className={cn(BADGE, status.tone)}>{status.label}</span>
                        {isExpiring(record) ? (
                          <span className={cn(BADGE, EXPIRING_TONE, "ml-2")}>Sắp hết hạn</span>
                        ) : null}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="flex items-center justify-between gap-3 border-t border-[#E4E9F2] px-4 py-3 text-[13px] text-[#474747]">
              <span>
                Hiển thị {shown.length === 0 ? 0 : current * PAGE_SIZE + 1}–
                {Math.min((current + 1) * PAGE_SIZE, shown.length)} / {shown.length}
              </span>
              <div className="flex gap-2">
                <button
                  type="button"
                  className="btn"
                  disabled={current === 0}
                  onClick={() => setPage(current - 1)}
                >
                  Trước
                </button>
                <button
                  type="button"
                  className="btn"
                  disabled={current >= pageCount - 1}
                  onClick={() => setPage(current + 1)}
                >
                  Sau
                </button>
              </div>
            </div>
          </div>

          {selected ? (
            <aside
              aria-labelledby="kho-chi-tiet"
              className="self-start lg:sticky lg:top-4 rounded-[12px] border border-[#D6DEEA] bg-white p-5"
            >
              <h2
                id="kho-chi-tiet"
                className="!m-0 font-display text-[18px] font-semibold text-[#0F1B2D]"
              >
                {RECORD_TYPE_LABELS[selected.record_type] ?? selected.record_type}{" "}
                {selected.admission_year}
              </h2>
              <span className={cn(BADGE, botStatus(selected).tone, "mt-2")}>
                {botStatus(selected).label}
              </span>
              <div className="mt-4 whitespace-pre-line rounded-[8px] bg-[#F3F5F9] p-4 text-[15px] leading-6 text-[#0F1B2D]">
                {selected.text}
              </div>
              <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-[14px] text-[#0F1B2D]">
                <dt className="text-[#6B7686]">Phiên bản</dt>
                <dd>v{selected.version}</dd>
                <dt className="text-[#6B7686]">Hiệu lực</dt>
                <dd>
                  {formatDate(selected.effective_from) ?? "chưa rõ"} đến{" "}
                  {formatDate(selected.effective_to) ?? "không giới hạn"}
                </dd>
                <dt className="text-[#6B7686]">Nguồn</dt>
                <dd className="font-mono text-[13px]">
                  {selected.source_document_id ?? "không ghi nhận"}
                  {selected.source_page ? ` · trang ${selected.source_page}` : ""}
                </dd>
                <dt className="text-[#6B7686]">Người cập nhật</dt>
                <dd>{selected.authored_by ?? "không rõ"}</dd>
                <dt className="text-[#6B7686]">Xuất bản</dt>
                <dd>{formatDate(selected.published_at) ?? "chưa xuất bản"}</dd>
                {selected.supersedes ? (
                  <>
                    <dt className="text-[#6B7686]">Thay thế mục</dt>
                    <dd className="font-mono text-[13px]">{selected.supersedes}</dd>
                  </>
                ) : null}
              </dl>
              {selected.source_url ? (
                <a
                  href={selected.source_url}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-4 inline-flex min-h-11 items-center text-[14px] font-semibold text-[#143980] underline"
                >
                  Mở tài liệu gốc
                </a>
              ) : null}
            </aside>
          ) : null}
        </div>
      )}
    </section>
  );
};

export default Vault;
