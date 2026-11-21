import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useAdminSession } from "@/hooks/useAdminSession";
import { listUnanswered, markOutOfScope, type UnansweredItem } from "@/lib/teacherApi";
import { cn } from "@/lib/utils";

const REASON_LABELS: Record<string, string> = {
  OUT_OF_DOMAIN: "Ngoài phạm vi",
  LOW_CONFIDENCE: "Thí sinh đánh giá chưa tốt",
  NO_INFO: "Không có thông tin",
  GENERATION_FAILED: "Không sinh được câu trả lời",
};

const REASON_TONE: Record<string, string> = {
  OUT_OF_DOMAIN: "bg-[#E6E9EF] text-[#474747]",
  LOW_CONFIDENCE: "bg-[#FDECEA] text-[#8B1A1A]",
  GENERATION_FAILED: "bg-[#FDECEA] text-[#8B1A1A]",
  NO_INFO: "bg-[#FFF3DC] text-[#6B3E00]",
};

const DEFAULT_REASON = "Không có thông tin";

/** "Hôm nay 09:15", "Hôm qua 21:40", or the date for anything older. */
export function formatAskedAt(iso: string, now: Date = new Date()): string {
  const asked = new Date(iso);
  if (Number.isNaN(asked.getTime())) return "";
  const time = asked.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (asked.toDateString() === now.toDateString()) return `Hôm nay ${time}`;
  if (asked.toDateString() === yesterday.toDateString()) return `Hôm qua ${time}`;
  return `${asked.toLocaleDateString("vi-VN")} ${time}`;
}

const isOverdue = (item: UnansweredItem) =>
  item.is_overdue ?? new Date(item.sla_due_at ?? 0).getTime() < Date.now();

/**
 * "Câu hỏi chưa trả lời" (frame nNvBU): the open questions on the left, the
 * question being handled on the right. The primary action hands the question
 * to the Q&A form; "Ngoài phạm vi" marks it out of scope without deleting it.
 */
const Unanswered = () => {
  const { can } = useAdminSession();
  const canEdit = can("records.edit");
  const canCreate = can("records.create");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const list = useQuery({
    queryKey: ["admin", "unanswered"],
    queryFn: () => listUnanswered(null),
    retry: false,
  });

  const rows = (list.data?.items ?? []).filter((item) => !item.resolved);
  const selected = rows.find((item) => item.trace_id === selectedId) ?? rows[0] ?? null;

  const outOfScope = useMutation({
    mutationFn: (traceId: string) => markOutOfScope(null, traceId),
    onSuccess: () => list.refetch(),
  });

  if (list.isPending) {
    return (
      <p role="status" aria-live="polite" className="text-zinc-600">
        Đang lấy danh sách câu hỏi…
      </p>
    );
  }

  if (list.isError) {
    return (
      <div role="alert" className="note bad">
        Chưa lấy được danh sách câu hỏi. Kiểm tra kết nối mạng rồi thử lại.
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <header>
        <h1 className="!m-0 font-display text-[24px] font-bold leading-8 text-[#0F1B2D]">
          Câu hỏi chưa trả lời
        </h1>
        <p className="mt-2 text-[15px] leading-6 text-[#474747]">
          Các câu hỏi thực tế của thí sinh mà Bot từ chối trả lời hoặc chưa tìm thấy thông tin
          đã phát hành. Thông tin cá nhân (SĐT, CCCD, email) đã được tự động che.
        </p>
      </header>

      {rows.length === 0 ? (
        <p role="status" className="note ok">
          Bot đã trả lời được mọi câu hỏi trong 7 ngày qua.
        </p>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,7fr)]">
          <section
            aria-labelledby="cau-hoi-danh-sach"
            className="rounded-[12px] border border-[#D6DEEA] bg-white p-4"
          >
            <h2
              id="cau-hoi-danh-sach"
              className="!m-0 font-display text-[15px] font-semibold text-[#0F1B2D]"
            >
              Danh sách câu hỏi · {rows.length}
            </h2>
            <ul className="mt-3 max-h-[60vh] space-y-2 overflow-y-auto pr-1 lg:max-h-[calc(100vh-15rem)]">
              {rows.map((item) => {
                const active = selected?.trace_id === item.trace_id;
                return (
                  <li key={item.trace_id}>
                    <button
                      type="button"
                      aria-pressed={active}
                      onClick={() => setSelectedId(item.trace_id)}
                      className={cn(
                        "block w-full rounded-[8px] border p-3 text-left",
                        active
                          ? "border-[#143980] bg-[#E6ECF7]"
                          : "border-[#E4E9F2] bg-[#F7F9FC] hover:border-[#D6DEEA]",
                      )}
                    >
                      <span
                        className={cn(
                          "inline-flex h-6 items-center rounded-full px-2.5 text-[12px] font-medium",
                          REASON_TONE[item.status_code] ?? REASON_TONE.NO_INFO,
                        )}
                      >
                        {REASON_LABELS[item.status_code] ?? DEFAULT_REASON}
                      </span>
                      <span className="mt-2 block font-display text-[14px] font-semibold leading-5 text-[#0F1B2D]">
                        {item.question}
                      </span>
                      <span className="mt-1 block text-[12px] text-[#6B7686]">
                        Hỏi {item.occurrence_count} lần · {formatAskedAt(item.asked_at)}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </section>

          {selected ? (
            <section
              aria-labelledby="cau-hoi-xu-ly"
              className="rounded-[12px] border border-[#D6DEEA] bg-white p-5"
            >
              <h2
                id="cau-hoi-xu-ly"
                className="!m-0 font-display text-[18px] font-semibold text-[#0F1B2D]"
              >
                Xử lý câu hỏi đang chọn
              </h2>

              <div className="mt-4 rounded-[8px] bg-[#F3F5F9] p-4">
                <p className="text-[12px] font-medium text-[#6B7686]">
                  Câu hỏi gốc và lý do Bot từ chối
                </p>
                <p className="mt-1 text-[15px] leading-6 text-[#0F1B2D]">
                  “{selected.question}”
                </p>
                <p className="mt-2 text-[14px] leading-5 text-[#6B3E00]">
                  Lý do: {REASON_LABELS[selected.status_code] ?? DEFAULT_REASON}.{" "}
                  <span
                    aria-label={isOverdue(selected) ? "Quá hạn xử lý" : "Còn hạn xử lý"}
                    className={cn(
                      "font-medium",
                      isOverdue(selected) ? "text-[#8B1A1A]" : "text-[#1B5E20]",
                    )}
                  >
                    {isOverdue(selected) ? "Quá hạn xử lý" : "Còn hạn xử lý"}
                  </span>
                </p>
              </div>

              <div className="mt-4 rounded-[8px] border-2 border-[#143980] p-4">
                <h3 className="!m-0 font-display text-[15px] font-semibold text-[#143980]">
                  Chế độ 1: Tạo mục Hỏi - Đáp (HOI_DAP) chuẩn và gửi duyệt
                </h3>
                <p className="mt-2 text-[14px] leading-6 text-[#474747]">
                  Nhập câu trả lời chính thức, nguồn trích dẫn và năm áp dụng ở màn Hỏi - Đáp
                  để đưa câu hỏi này vào bộ câu hỏi chuẩn.
                </p>
                <div className="mt-4 flex flex-wrap justify-end gap-3">
                  {canCreate ? (
                    <Link
                      to={`/quan-ly/them/qa?question=${encodeURIComponent(selected.question)}&trace_id=${encodeURIComponent(selected.trace_id)}&promote=1`}
                      className="inline-flex min-h-11 items-center rounded-[8px] bg-[#143980] px-5 text-[14px] font-semibold text-white hover:bg-[#1a4a9e]"
                    >
                      Trả lời & Đưa vào bộ câu hỏi chuẩn
                    </Link>
                  ) : (
                    <button
                      type="button"
                      disabled
                      className="min-h-11 rounded-[8px] bg-[#F3F5F9] px-5 text-[14px] font-semibold text-[#6B7280]"
                    >
                      Trả lời & Đưa vào bộ câu hỏi chuẩn
                    </button>
                  )}
                </div>
              </div>

              <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-[8px] bg-[#F3F5F9] p-4">
                <p className="max-w-[28rem] text-[14px] leading-5 text-[#474747]">
                  Chế độ 2: câu hỏi không thuộc phạm vi tuyển sinh HUSC.
                </p>
                {selected.out_of_scope ? (
                  <span
                    aria-label="Đã đánh dấu ngoài phạm vi"
                    className="inline-flex h-6 items-center rounded-full bg-[#E6E9EF] px-2.5 text-[12px] font-medium text-[#474747]"
                  >
                    Đã đánh dấu ngoài phạm vi
                  </span>
                ) : (
                  <button
                    type="button"
                    className="min-h-11 rounded-[8px] border border-[#D6DEEA] bg-white px-4 text-[14px] font-semibold text-[#143980] hover:bg-[#E6ECF7] disabled:text-[#6B7280]"
                    disabled={!canEdit || outOfScope.isPending}
                    onClick={() => void outOfScope.mutate(selected.trace_id)}
                  >
                    Đánh dấu Ngoài phạm vi tuyển sinh HUSC (Bỏ qua)
                  </button>
                )}
              </div>
            </section>
          ) : null}
        </div>
      )}

      <p className="text-sm text-zinc-600">
        Về <Link to="/quan-ly">Tổng quan</Link>
      </p>
    </div>
  );
};

export default Unanswered;
