// Tổng quan vận hành — the three summary cards, the "Cần bạn xử lý ngay" list and
// the two-level approval panel. Counts come from GET /admin/overview, the active
// release from GET /admin/releases and the waiting records from GET /admin/records.
// The role toggle is a client-side view: the office view hides the approval-level
// panel. The backend still decides what each account may actually do.
import { useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { useAdminSession } from "@/hooks/useAdminSession";
import { AdminApiError, fetchReleaseStatus, listRecords } from "@/lib/adminApi";
import { fetchAdminOverview } from "@/lib/adminOpsApi";
import type { ApiFailure } from "@/lib/apiError";
import { ExceptionBar } from "@/components/ui-kit/ExceptionBar";
import { OVERDUE_LABEL, StatusPill } from "@/components/ui-kit/StatusPill";

function failureOf(error: unknown): ApiFailure | null {
  return error instanceof AdminApiError ? error.failure : null;
}

/** Release ids start with the publish timestamp (yyyymmdd…); show the day as dd/mm/yyyy. */
function releaseDay(releaseId: string | null): string | null {
  const match = releaseId ? /^(\d{4})(\d{2})(\d{2})/.exec(releaseId) : null;
  return match ? `${match[3]}/${match[2]}/${match[1]}` : null;
}

const VIEWS = [
  { value: "admin", label: "Góc nhìn Quản trị viên" },
  { value: "office", label: "Góc nhìn Thầy/cô tuyển sinh (ẩn chi tiết kỹ thuật)" },
] as const;
type View = (typeof VIEWS)[number]["value"];

type Action = { key: string; tag: ReactNode; text: string; to: string; cta: string };

const Overview = () => {
  const { can, identity } = useAdminSession();
  const canRead = can("records.read");
  const [view, setView] = useState<View>("admin");

  const overview = useQuery({
    queryKey: ["admin", "overview"],
    queryFn: () => fetchAdminOverview(null),
    enabled: canRead,
    retry: false,
  });
  const release = useQuery({
    queryKey: ["admin", "releases"],
    queryFn: () => fetchReleaseStatus(null),
    retry: false,
  });
  const waiting = useQuery({
    queryKey: ["admin", "records", "review"],
    queryFn: () => listRecords(null, { status: "review" }),
    enabled: canRead && view === "admin",
    retry: false,
  });

  const overviewFailure = failureOf(overview.error);
  const releaseFailure = failureOf(release.error);
  const counts = overview.data;
  const noActiveRelease = releaseFailure?.kind === "unavailable" && releaseFailure.code === "NO_ACTIVE_RELEASE";
  const activeDay = release.data && !noActiveRelease ? releaseDay(release.data.active_release_id) : null;
  const activeDocuments = release.data && !noActiveRelease ? (release.data.active?.document_count ?? null) : null;

  const blockers = counts?.open_blocker_count ?? 0;
  const overdue = counts?.unanswered_overdue_count ?? 0;
  const reviews = counts?.review_count ?? 0;
  const openQuestions = counts?.unanswered_open_count ?? 0;
  const needMe = blockers + overdue;

  const actions: Action[] = [
    blockers > 0
      ? {
          key: "blocker",
          tag: <span className="rounded-full bg-[#FDECEA] px-2.5 py-0.5 text-sm font-semibold text-[#9b1c1c]">Xung đột</span>,
          text: `${blockers} mâu thuẫn chưa phân xử`,
          to: "/quan-ly/cho-duyet",
          cta: "Phân xử xung đột",
        }
      : null,
    overdue > 0
      ? {
          key: "overdue",
          tag: <StatusPill tone="overdue" label={`${OVERDUE_LABEL}: ${overdue}`} />,
          text: `${overdue} câu hỏi bot chưa trả lời đã quá hạn`,
          to: "/quan-ly/cau-hoi",
          cta: "Trả lời câu hỏi",
        }
      : null,
  ].filter((action): action is Action => action !== null);

  const waitingItems = waiting.data?.records.slice(0, 3) ?? [];
  const waitingMore = (waiting.data?.total ?? 0) - waitingItems.length;

  return (
    <section data-workarea="" aria-labelledby="overview-title" className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="max-w-2xl space-y-2">
          <h1 id="overview-title" className="text-3xl font-bold text-[#0f1b2d]">
            Tổng quan vận hành
          </h1>
          <p className="text-base text-[#4a5568]">
            Các mục cần kiểm tra, duyệt hoặc bổ sung để Bot trả lời chính xác cho thí sinh.
          </p>
        </div>
        <div role="group" aria-label="Chọn góc nhìn" className="flex flex-wrap gap-2">
          {VIEWS.map((option) => (
            <button
              key={option.value}
              type="button"
              aria-pressed={view === option.value}
              onClick={() => setView(option.value)}
              className={`min-h-11 rounded-lg border px-4 text-sm font-semibold ${
                view === option.value
                  ? "border-[#143980] bg-[#143980] text-white"
                  : "border-[#d6deea] bg-white text-[#143980] hover:bg-[#eef3fb]"
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      {noActiveRelease ? (
        <ExceptionBar
          severity="warning"
          title="Chưa có bản phát hành đang hoạt động"
          detail="Bot chưa phục vụ thông tin nào. Kiểm tra lại mục Lịch sử."
        />
      ) : null}
      {overviewFailure ? <ExceptionBar title="Không tải được số liệu tổng quan" detail={overviewFailure.message} /> : null}
      {overview.isPending && canRead ? <p className="text-base">Đang tải...</p> : null}

      {counts ? (
        <ul className="m-0 grid list-none gap-4 p-0 md:grid-cols-3">
          <li className="space-y-2 rounded-xl border border-[#F0D9A6] bg-[#FFF3DC] p-5">
            <h2 className="text-lg font-semibold text-[#8A5A00]">
              Cần tôi xử lý · <span>{needMe}</span> mục
            </h2>
            <p className="text-base text-[#5c4a1c]">
              {blockers} xung đột số liệu · {overdue} câu hỏi quá hạn
            </p>
          </li>
          <li className="space-y-2 rounded-xl border border-[#C9D8EE] bg-[#E6ECF7] p-5">
            <h2 className="text-lg font-semibold text-[#143980]">
              Đang chờ duyệt &amp; cập nhật · <span>{reviews}</span> mục
            </h2>
            <p className="text-base text-[#2d4a7a]">{reviews} mục chờ quản trị viên thứ hai duyệt.</p>
          </li>
          <li className="space-y-2 rounded-xl border border-[#BFDCC4] bg-[#E7F4E8] p-5">
            <h2 className="text-lg font-semibold text-[#1E6B34]">
              Bot đang sử dụng · Phiên bản {activeDay ?? "chưa có"}
            </h2>
            <p className="text-base text-[#2f5a3a]">
              {noActiveRelease || activeDocuments === null
                ? "Chưa có phiên bản đang phục vụ."
                : `${activeDocuments} tài liệu đang phục vụ`}
              {" · "}
              {openQuestions} câu hỏi mới chưa trả lời được
            </p>
          </li>
        </ul>
      ) : null}

      {counts ? (
        <section aria-labelledby="overview-actions" className="space-y-3 rounded-xl border border-[#d6deea] bg-white p-6">
          <h2 id="overview-actions" className="text-xl font-semibold text-[#0f1b2d]">
            Cần bạn xử lý ngay
          </h2>
          {actions.length === 0 ? (
            <p className="text-base text-[#4a5568]">Hiện không có mục nào cần xử lý.</p>
          ) : (
            <ul className="m-0 list-none space-y-2 p-0">
              {actions.map((action) => (
                <li
                  key={action.key}
                  className="flex flex-wrap items-center justify-between gap-3 rounded-lg bg-[#f7f9fc] p-4"
                >
                  <div className="space-y-1">
                    {action.tag}
                    <p className="text-base font-medium">{action.text}</p>
                  </div>
                  <Link
                    to={action.to}
                    className="inline-flex min-h-11 items-center rounded-lg bg-[#143980] px-4 text-base font-semibold text-white hover:bg-[#0f2d66]"
                  >
                    {action.cta}
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </section>
      ) : null}

      {canRead && view === "admin" ? (
        <section aria-labelledby="overview-approval" className="space-y-3 rounded-xl border border-[#d6deea] bg-white p-6">
          <h2 id="overview-approval" className="text-xl font-semibold text-[#0f1b2d]">
            Chờ người quản trị thứ hai duyệt (duyệt hai cấp)
          </h2>
          {waiting.isPending ? <p className="text-base">Đang tải...</p> : null}
          {waiting.isError ? (
            <p className="text-base text-[#9b1c1c]">Không tải được danh sách chờ duyệt.</p>
          ) : null}
          {waiting.data && waiting.data.records.length === 0 ? (
            <p className="text-base text-[#4a5568]">Hiện không có mục nào chờ quản trị viên thứ hai duyệt.</p>
          ) : null}
          {waitingItems.map((record) => {
            const mine = record.authored_by !== null && record.authored_by === identity?.actor_id;
            const title = (record.text || record.record_id).trim().slice(0, 140);
            return (
              <div
                key={record.record_id}
                className="flex flex-wrap items-center justify-between gap-3 rounded-lg bg-[#f7f9fc] p-4"
              >
                <div className="space-y-1">
                  <span
                    className={`inline-block rounded-full px-2.5 py-0.5 text-sm font-semibold ${
                      mine ? "bg-[#FFF3DC] text-[#8A5A00]" : "bg-[#E6ECF7] text-[#143980]"
                    }`}
                  >
                    {mine ? "Cần người khác duyệt" : "Sẵn sàng cho người thứ hai duyệt"}
                  </span>
                  <p className="text-base font-semibold">{title}</p>
                  <p className="text-sm text-[#4a5568]">
                    {mine
                      ? "Tác giả: bạn. Bạn không thể tự duyệt mục này."
                      : `Tác giả: ${record.authored_by ?? "không ghi nhận"}.`}
                  </p>
                </div>
                <Link
                  to="/quan-ly/cho-duyet"
                  className="inline-flex min-h-11 items-center rounded-lg border border-[#143980] bg-white px-4 text-base font-semibold text-[#143980] hover:bg-[#eef3fb]"
                >
                  Mở màn hình duyệt
                </Link>
              </div>
            );
          })}
          {waitingMore > 0 ? (
            <p className="text-sm text-[#4a5568]">
              + {waitingMore} mục khác đang chờ quản trị viên thứ hai (xem trong Chờ duyệt).
            </p>
          ) : null}
        </section>
      ) : null}
    </section>
  );
};

export default Overview;
