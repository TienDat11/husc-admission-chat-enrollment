// Tổng quan — the counts behind the three work cards, read from GET /admin/overview
// (Frame 5). The sidebar badge "Chờ duyệt (n)" reads the same review_count.
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { AdminApiError, fetchReleaseStatus } from "@/lib/adminApi";
import { fetchAdminOverview } from "@/lib/adminOpsApi";
import type { ApiFailure } from "@/lib/apiError";
import { ExceptionBar } from "@/components/ui-kit/ExceptionBar";
import { OVERDUE_LABEL, StatusPill } from "@/components/ui-kit/StatusPill";

function failureOf(error: unknown): ApiFailure | null {
  return error instanceof AdminApiError ? error.failure : null;
}

const Overview = () => {
  const overview = useQuery({
    queryKey: ["admin", "overview"],
    queryFn: () => fetchAdminOverview(null),
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

  return (
    <section data-workarea="" aria-labelledby="overview-title" className="space-y-6">
      <h1 id="overview-title" className="text-2xl font-bold text-[#143980]">
        Tổng quan
      </h1>

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
        <ul className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          <li className="space-y-2 rounded-md border border-zinc-400 p-4">
            <h2 className="text-lg font-semibold">Chờ duyệt</h2>
            <p className="text-3xl font-bold">{counts.review_count}</p>
            <Link className="text-base font-medium text-[#3730A3] underline" to="/quan-ly/cho-duyet">
              Mở hàng chờ duyệt
            </Link>
          </li>
          <li className="space-y-2 rounded-md border border-zinc-400 p-4">
            <h2 className="text-lg font-semibold">Câu hỏi bot chưa trả lời</h2>
            <p className="text-3xl font-bold">{counts.unanswered_open_count}</p>
            {counts.unanswered_overdue_count > 0 ? (
              <StatusPill tone="overdue" label={`${OVERDUE_LABEL}: ${counts.unanswered_overdue_count}`} />
            ) : null}
            <Link className="text-base font-medium text-[#3730A3] underline" to="/quan-ly/cau-hoi">
              Xem câu hỏi
            </Link>
          </li>
          <li className="space-y-2 rounded-md border border-zinc-400 p-4">
            <h2 className="text-lg font-semibold">Mâu thuẫn chưa phân xử</h2>
            <p className="text-3xl font-bold">{counts.open_blocker_count}</p>
            <Link className="text-base font-medium text-[#3730A3] underline" to="/quan-ly/cho-duyet">
              Xem mâu thuẫn
            </Link>
          </li>
        </ul>
      ) : null}

      {release.data && !noActiveRelease ? (
        <p className="text-base">
          Bản đang phục vụ: <span className="font-semibold">{release.data.active_release_id ?? "chưa có"}</span>
        </p>
      ) : null}
    </section>
  );
};

export default Overview;
