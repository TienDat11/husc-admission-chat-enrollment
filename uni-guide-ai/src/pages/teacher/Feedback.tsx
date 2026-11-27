import { useQuery } from "@tanstack/react-query";
import { fetchFeedback, fetchReleases } from "@/lib/adminApi";
import { listUnanswered } from "@/lib/teacherApi";
import { formatViDateTime } from "@/lib/utils";
import { adminKeys } from "@/lib/queryKeys";
import { ListSkeleton } from "@/components/ui-kit/PageState";

const FEEDBACK_SAMPLE = 50;

/**
 * "Đánh giá từ thí sinh & Bộ câu hỏi kiểm tra chất lượng" (frame gVQgC, sidebar
 * "Phản hồi & Golden"). Every figure comes from a real endpoint. There is no
 * endpoint for the golden question set, so that panel reports it as unavailable.
 */
const Feedback = () => {
  const feedback = useQuery({
    queryKey: adminKeys.feedback(FEEDBACK_SAMPLE),
    queryFn: () => fetchFeedback(null, FEEDBACK_SAMPLE, 0),
    retry: false,
  });
  const unanswered = useQuery({
    queryKey: adminKeys.unansweredAll(),
    queryFn: () => listUnanswered(null),
    retry: false,
  });
  const releases = useQuery({
    queryKey: adminKeys.releases(),
    queryFn: () => fetchReleases(null),
    retry: false,
  });

  const items = feedback.data?.items ?? [];
  const helpfulCount = items.filter((item) => item.helpful).length;
  const repeatedCount = (unanswered.data?.items ?? []).filter(
    (item) => !item.resolved && item.occurrence_count > 1,
  ).length;
  const latestRelease = [...(releases.data?.releases ?? [])].sort((a, b) =>
    (b.created_at ?? "").localeCompare(a.created_at ?? ""),
  )[0];
  const verdict = latestRelease?.golden_report?.verdict ?? null;

  return (
    <div className="space-y-5">
      <header className="space-y-1">
        <h1 className="!m-0 font-display text-[24px] font-bold leading-8 text-[#0F1B2D]">
          Đánh giá từ thí sinh & Bộ câu hỏi kiểm tra chất lượng
        </h1>
        <p className="text-[14px] text-[#474747]">
          Phản hồi thí sinh sau mỗi câu trả lời và kết quả kiểm tra bản phát hành gần nhất.
        </p>
      </header>

      <section aria-label="Chỉ số" className="grid gap-4 sm:grid-cols-3">
        <div className="rounded-[12px] border border-[#D6DEEA] bg-white p-5">
          <p className="text-[13px] text-[#6B7686]">Tổng số phản hồi</p>
          {feedback.isPending ? (
            <ListSkeleton rows={3} label="Đang tải…" />
          ) : feedback.isError ? (
            <p role="alert" className="note bad mt-2">Không tải được phản hồi.</p>
          ) : (
            <p className="mt-2 font-display text-[28px] font-bold text-[#0F1B2D]">
              {feedback.data.count}
            </p>
          )}
        </div>

        <div className="rounded-[12px] border border-[#D6DEEA] bg-white p-5">
          <p className="text-[13px] text-[#6B7686]">Hữu ích, {FEEDBACK_SAMPLE} phản hồi gần nhất</p>
          {feedback.isPending ? (
            <ListSkeleton rows={3} label="Đang tải…" />
          ) : feedback.isError ? (
            <p role="alert" className="note bad mt-2">Không tải được phản hồi.</p>
          ) : items.length === 0 ? (
            <p className="mt-2 text-[14px] text-[#474747]">Chưa có phản hồi nào.</p>
          ) : (
            <p className="mt-2 font-display text-[28px] font-bold text-[#0F1B2D]">
              {helpfulCount}/{items.length}
            </p>
          )}
        </div>

        <div className="rounded-[12px] border border-[#D6DEEA] bg-white p-5">
          <p className="text-[13px] text-[#6B7686]">Câu hỏi lặp lại chưa trả lời được</p>
          {unanswered.isPending ? (
            <ListSkeleton rows={3} label="Đang tải…" />
          ) : unanswered.isError ? (
            <p role="alert" className="note bad mt-2">Không tải được câu hỏi chưa trả lời.</p>
          ) : (
            <p className="mt-2 font-display text-[28px] font-bold text-[#0F1B2D]">
              {repeatedCount}
            </p>
          )}
          <p className="mt-1 text-[13px] text-[#6B7686]">Đã được hỏi từ 2 lần trở lên.</p>
        </div>
      </section>

      <div className="grid gap-4 lg:grid-cols-[7fr_5fr]">
        <section
          aria-labelledby="phan-hoi-gan-day"
          className="rounded-[12px] border border-[#D6DEEA] bg-white p-5"
        >
          <h2 id="phan-hoi-gan-day" className="!m-0 font-display text-[18px] font-semibold text-[#0F1B2D]">
            Phản hồi gần nhất
          </h2>
          {feedback.isPending ? (
            <ListSkeleton rows={3} label="Đang tải phản hồi…" />
          ) : feedback.isError ? (
            <p role="alert" className="note bad mt-3">Không tải được phản hồi.</p>
          ) : items.length === 0 ? (
            <p className="mt-3 text-[14px] text-[#474747]">Chưa có phản hồi nào được ghi nhận.</p>
          ) : (
            <ul className="mt-3 divide-y divide-[#E4E9F2]">
              {items.map((item) => (
                <li key={item.feedback_id} className="flex flex-wrap items-center gap-3 py-3">
                  <span className="font-mono text-[13px] text-[#6B7686]">
                    {formatViDateTime(item.created_at)}
                  </span>
                  <span className="inline-flex h-6 items-center rounded-full bg-[#E6ECF7] px-2.5 text-[12px] font-medium text-brand-navy">
                    {item.helpful ? "Hữu ích" : "Chưa hữu ích"}
                  </span>
                  <span className="min-w-0 flex-1 text-[14px] text-[#0F1B2D]">
                    {item.reason ?? "Không ghi lý do"}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section
          aria-labelledby="bo-cau-hoi-kiem-tra"
          className="rounded-[12px] border border-[#D6DEEA] bg-white p-5"
        >
          <h2 id="bo-cau-hoi-kiem-tra" className="!m-0 font-display text-[18px] font-semibold text-[#0F1B2D]">
            Bộ câu hỏi kiểm tra chất lượng
          </h2>
          <p className="mt-3 text-[14px] text-[#474747]">
            Hệ thống chưa cung cấp danh sách bộ câu hỏi kiểm tra, nên không hiển thị được từng câu.
          </p>
          <div className="mt-4 rounded-[8px] bg-[#F3F5F9] p-4">
            <p className="text-[13px] text-[#6B7686]">Kết quả kiểm tra bản phát hành gần nhất</p>
            {releases.isPending ? (
              <ListSkeleton rows={3} label="Đang tải…" />
            ) : releases.isError ? (
              <p role="alert" className="note bad mt-2">Không tải được bản phát hành.</p>
            ) : verdict ? (
              <p className="mt-2 text-[14px] text-[#0F1B2D]">
                Bản <span className="font-mono">{latestRelease.release_id}</span>: {verdict}
              </p>
            ) : (
              <p className="mt-2 text-[14px] text-[#474747]">Chưa có báo cáo kiểm tra.</p>
            )}
          </div>
        </section>
      </div>
    </div>
  );
};

export default Feedback;
