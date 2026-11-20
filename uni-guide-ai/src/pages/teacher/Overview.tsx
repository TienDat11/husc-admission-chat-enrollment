// Tổng quan — the counts behind the three work cards, read from GET /admin/overview
// (Frame 5). The sidebar badge "Chờ duyệt (n)" reads the same review_count.
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { useAdminSession } from "@/hooks/useAdminSession";
import { AdminApiError, fetchReleaseStatus } from "@/lib/adminApi";
import { fetchAdminOverview } from "@/lib/adminOpsApi";
import type { ApiFailure } from "@/lib/apiError";
import { ExceptionBar } from "@/components/ui-kit/ExceptionBar";
import { OVERDUE_LABEL, StatusPill } from "@/components/ui-kit/StatusPill";

function failureOf(error: unknown): ApiFailure | null {
  return error instanceof AdminApiError ? error.failure : null;
}

/** Release ids start with the publish timestamp (yyyymmdd…); show the day, not the id. */
function releaseDay(releaseId: string | null): string | null {
  const match = releaseId ? /^(\d{4})(\d{2})(\d{2})/.exec(releaseId) : null;
  return match ? `${match[3]}/${match[2]}/${match[1]}` : null;
}

const Overview = () => {
  const { can } = useAdminSession();
  const overview = useQuery({
    queryKey: ["admin", "overview"],
    queryFn: () => fetchAdminOverview(null),
    enabled: can("records.read"),
    retry: false,
  });
  const release = useQuery({
    queryKey: ["admin", "releases"],
    queryFn: () => fetchReleaseStatus(null),
    retry: false,
  });

  const overviewFailure = failureOf(overview.error);
  const releaseFailure = failureOf(release.error);
  const counts = overview.data;
  const noActiveRelease = releaseFailure?.kind === "unavailable" && releaseFailure.code === "NO_ACTIVE_RELEASE";
  const liveDay = release.data && !noActiveRelease ? releaseDay(release.data.active_release_id) : null;

  const actions = counts
    ? [
        counts.unanswered_overdue_count > 0
          ? {
              key: "overdue",
              text: `${counts.unanswered_overdue_count} câu hỏi đã quá hạn, cần trả lời`,
              to: "/quan-ly/cau-hoi",
              cta: "Trả lời câu hỏi",
            }
          : null,
        counts.open_blocker_count > 0
          ? {
              key: "blocker",
              text: `${counts.open_blocker_count} mâu thuẫn cần phân xử`,
              to: "/quan-ly/cho-duyet",
              cta: "Phân xử mâu thuẫn",
            }
          : null,
        counts.review_count > 0
          ? {
              key: "review",
              text: `${counts.review_count} mục đang chờ duyệt`,
              to: "/quan-ly/cho-duyet",
              cta: "Mở hàng chờ duyệt",
            }
          : null,
      ].filter((action) => action !== null)
    : [];

  return (
    <section data-workarea="" aria-labelledby="overview-title" className="space-y-6">
      <div className="space-y-1">
        <h1 id="overview-title" className="text-3xl font-bold text-[#143980]">
          Tổng quan
        </h1>
        {liveDay ? <p className="text-base text-[#4a5568]">Bot đang phục vụ dữ liệu cập nhật ngày {liveDay}.</p> : null}
      </div>

      {noActiveRelease ? (
        <ExceptionBar
          severity="warning"
          title="Chưa có bản phát hành đang hoạt động"
          detail="Bot chưa phục vụ thông tin nào. Kiểm tra lại mục Lịch sử & khôi phục."
        />
      ) : null}
      {overviewFailure ? (
        <ExceptionBar title="Không tải được số liệu tổng quan" detail={overviewFailure.message} />
      ) : null}
      {overview.isPending ? <p className="text-base">Đang tải...</p> : null}

      {counts ? (
        <ul className="m-0 grid list-none gap-4 p-0 sm:grid-cols-2 xl:grid-cols-3">
          <li className="space-y-2 rounded-xl border border-[#c9d8ee] bg-[#eef3fb] p-5">
            <h2 className="text-lg font-semibold text-[#143980]">Chờ duyệt</h2>
            <p className="text-3xl font-bold">{counts.review_count}</p>
            <Link className="text-base font-medium text-[#143980] underline" to="/quan-ly/cho-duyet">
              Mở hàng chờ duyệt
            </Link>
          </li>
          <li className="space-y-2 rounded-xl border border-[#f0d9a6] bg-[#fff7e6] p-5">
            <h2 className="text-lg font-semibold text-[#8a5a00]">Câu hỏi bot chưa trả lời</h2>
            <p className="text-3xl font-bold">{counts.unanswered_open_count}</p>
            {counts.unanswered_overdue_count > 0 ? (
              <StatusPill tone="overdue" label={`${OVERDUE_LABEL}: ${counts.unanswered_overdue_count}`} />
            ) : null}
            <Link className="block text-base font-medium text-[#143980] underline" to="/quan-ly/cau-hoi">
              Xem câu hỏi
            </Link>
          </li>
          <li className="space-y-2 rounded-xl border border-[#f0d9a6] bg-[#fff7e6] p-5">
            <h2 className="text-lg font-semibold text-[#8a5a00]">Mâu thuẫn chưa phân xử</h2>
            <p className="text-3xl font-bold">{counts.open_blocker_count}</p>
            <Link className="text-base font-medium text-[#143980] underline" to="/quan-ly/cho-duyet">
              Xem mâu thuẫn
            </Link>
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
                  <p className="text-base font-medium">{action.text}</p>
                  <Link
                    to={action.to}
                    className="inline-flex min-h-11 items-center rounded-lg border border-[#d6deea] bg-white px-4 text-base font-semibold text-[#143980] hover:bg-[#eef3fb]"
                  >
                    {action.cta}
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </section>
      ) : null}
    </section>
  );
};

export default Overview;
