import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  listRecords,
  type CanonicalRecord,
  type RecordFilters,
  type RecordStatus,
} from "@/lib/adminApi";
import { RECORD_TYPE_LABELS } from "@/lib/adminValidation";

const STATUS_LABELS: Record<string, string> = {
  draft: "Bản nháp",
  review: "Chờ duyệt",
  approved: "Đã duyệt",
  published: "Đang dùng",
  archived: "Đã lưu trữ",
};

const STATUS_CLASS: Record<string, string> = {
  draft: "s-muted",
  review: "s-info",
  approved: "s-info",
  published: "s-ok",
  archived: "s-muted",
};

/**
 * Ticket T09 — "Kho thông tin" (mockup 06, 05 §4.6): text search, the
 * always-visible filters (year / type / major / status), one row per record
 * with status + expiry + who updated it last, and the "Sắp hết hạn" tab.
 */
const Vault = () => {
  const [text, setText] = useState("");
  const [year, setYear] = useState("");
  const [recordType, setRecordType] = useState("");
  const [major, setMajor] = useState("");
  const [status, setStatus] = useState<RecordStatus | "">("");
  const [tab, setTab] = useState<"all" | "expiring">("all");

  const list = useQuery({
    queryKey: ["admin", "vault", { text, year, recordType, major, status }],
    queryFn: () =>
      listRecords(null, {
        ...(text.trim() ? { nganh: text.trim() } : {}),
        ...(year ? { year: Number(year) } : {}),
        ...(recordType ? { nganh: `${recordType}` } : {}),
        ...(major ? { nganh: major } : {}),
        ...(status ? { status: status as RecordStatus } : {}),
      } satisfies RecordFilters),
    retry: false,
  });

  const rows = list.data?.records ?? [];

  const isExpiring = (record: CanonicalRecord) => {
    if (record.status !== "published" || !record.effective_to) return false;
    const end = new Date(record.effective_to);
    const limit = new Date();
    limit.setDate(limit.getDate() + 30);
    return end.getTime() <= limit.getTime();
  };

  const expiring = useMemo(() => rows.filter(isExpiring), [rows]);

  const shown = tab === "expiring" ? expiring : rows;

  if (list.isPending) {
    return (
      <p role="status" aria-live="polite" className="text-zinc-600">
        <span aria-hidden="true">⏳</span> Đang tìm…
      </p>
    );
  }

  if (list.isError) {
    return (
      <div role="alert" className="note bad">
        <span aria-hidden="true">⚠</span> Không tải được danh sách thông tin.
        Kiểm tra kết nối mạng rồi thử lại.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Kho thông tin</h1>

      <div className="field">
        <label htmlFor="kho-tim">Tìm theo chữ</label>
        <input
          id="kho-tim"
          type="search"
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Ví dụ: học phí"
        />
      </div>

      <div className="row">
        <div className="field">
          <label htmlFor="kho-nam">Năm</label>
          <select id="kho-nam" value={year} onChange={(event) => setYear(event.target.value)}>
            <option value="">Tất cả</option>
            {["2026", "2025"].map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="kho-loai">Loại</label>
          <select
            id="kho-loai"
            value={recordType}
            onChange={(event) => setRecordType(event.target.value)}
          >
            <option value="">Tất cả</option>
            {Object.entries(RECORD_TYPE_LABELS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="kho-nganh">Ngành</label>
          <input
            id="kho-nganh"
            type="text"
            value={major}
            onChange={(event) => setMajor(event.target.value)}
            placeholder="Tên hoặc mã ngành"
          />
        </div>
        <div className="field">
          <label htmlFor="kho-trang-thai">Trạng thái</label>
          <select
            id="kho-trang-thai"
            value={status}
            onChange={(event) => setStatus(event.target.value as RecordStatus | "")}
          >
            <option value="">Tất cả</option>
            {Object.entries(STATUS_LABELS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div role="tablist" aria-label="Chế độ xem" className="actions">
        <button
          role="tab"
          aria-selected={tab === "all"}
          className={`btn ${tab === "all" ? "btn-primary" : ""}`}
          onClick={() => setTab("all")}
        >
          Tất cả
        </button>
        <button
          role="tab"
          aria-selected={tab === "expiring"}
          className={`btn ${tab === "expiring" ? "btn-primary" : ""}`}
          onClick={() => setTab("expiring")}
        >
          Sắp hết hạn ({expiring.length})
        </button>
      </div>

      {shown.length === 0 ? (
        <div className="note info" role="status">
          <p>
            Không có thông tin nào khớp bộ lọc hiện tại. Thử bỏ bớt bộ lọc,
            hoặc thêm tài liệu mới.
          </p>
          <Link to="/quan-ly/them" className="btn">
            Thêm tài liệu
          </Link>
        </div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Thông tin</th>
                <th scope="col">Trạng thái</th>
                <th scope="col">Áp dụng đến</th>
                <th scope="col">Người cập nhật</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((record: CanonicalRecord) => (
                <tr key={record.record_id}>
                  <td>
                    {RECORD_TYPE_LABELS[record.record_type] ?? record.record_type}{" "}
                    {record.admission_year}
                  </td>
                  <td>
{isExpiring(record) ? (
                      <span className="status s-warn">⚠ Sắp hết hạn</span>
                    ) : (
                      <span className={`status ${STATUS_CLASS[record.status]}`}>
                        {STATUS_LABELS[record.status]}
                      </span>
                    )}
                  </td>
                  <td>
                    {record.effective_to
                      ? new Date(record.effective_to).toLocaleDateString("vi-VN")
                      : "không giới hạn"}
                  </td>
                  <td>
                    {record.authored_by ?? "không rõ"},{" "}
                    {record.published_at
                      ? new Date(record.published_at).toLocaleDateString("vi-VN")
                      : "chưa xuất bản"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default Vault;
