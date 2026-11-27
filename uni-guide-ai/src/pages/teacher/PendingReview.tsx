import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useAdminSession } from "@/hooks/useAdminSession";
import { useReviewActions } from "@/hooks/useReviewActions";
import { fetchReviewDiff, listRecords, runReviewPreview, type CanonicalRecord, type ReviewPreviewResult } from "@/lib/adminApi";
import { PAGE_SIZE } from "@/lib/constants";
import { adminKeys } from "@/lib/queryKeys";
import { EmptyState, ErrorState, ListSkeleton } from "@/components/ui-kit/PageState";
import { ReviewList } from "@/components/teacher/review/ReviewList";
import { ReviewDiffPanel, type Column } from "@/components/teacher/review/ReviewDiffPanel";
import { ReviewSandbox } from "@/components/teacher/review/ReviewSandbox";
import { ReviewDecisionBar } from "@/components/teacher/review/ReviewDecisionBar";

/**
 * "Chờ duyệt & phát hành" (frame Xcckq; mobile N8GHat). The list of records
 * waiting for review on the left; the selected record's comparison, sandbox and
 * decision on the right. Four-eyes: the publish button is disabled for the
 * record's author, who can still ask the Bot, compare, and return the record.
 */
const PendingReview = () => {
  const { identity } = useAdminSession();
  const queryClient = useQueryClient();
  const review = useReviewActions({ queryClient, identity });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [column, setColumn] = useState<Column>("proposed");
  const [preview, setPreview] = useState<ReviewPreviewResult | null>(null);
  const [previewQuery, setPreviewQuery] = useState("");
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

  const list = useQuery({
    queryKey: adminKeys.recordsByStatus("review"),
    queryFn: () => listRecords(null, { status: "review" }),
    retry: false,
  });

  const rows = list.data?.records ?? [];
  const selected = rows.find((record) => record.record_id === selectedId) ?? rows[0] ?? null;

  const diff = useQuery({
    queryKey: adminKeys.reviewDiff(selected?.record_id),
    queryFn: () => fetchReviewDiff(null, (selected as CanonicalRecord).record_id),
    enabled: Boolean(selected),
    retry: false,
  });

  const previewMutation = useMutation({
    mutationFn: (question: string) =>
      runReviewPreview(null, (selected as CanonicalRecord).record_id, { query: question }),
    onSuccess: (result) => setPreview(result),
    onError: () => setPreview(null),
  });

  // A new selection clears the preview and the pending confirm or reject reason.
  const selectRecord = (recordId: string) => {
    setSelectedId(recordId);
    setPreview(null);
    setColumn("proposed");
    review.resetForSelection();
  };

  const heading = (
    <h1 className="!m-0 font-display text-[24px] font-bold leading-8 text-[#0F1B2D]">Chờ duyệt &amp; phát hành</h1>
  );

  if (list.isPending) {
    return <ListSkeleton rows={3} label="Đang lấy danh sách chờ duyệt…" />;
  }

  if (list.isError) {
    return (
      <ErrorState
        message="Chưa lấy được danh sách chờ duyệt. Kiểm tra kết nối mạng rồi thử lại."
        onRetry={() => void list.refetch()}
      />
    );
  }

  if (!selected) {
    return (
      <div className="space-y-5">
        {heading}
        <EmptyState title="Không có thông tin nào chờ duyệt." hint="Chưa có mục nào để quyết định." />
        <div className="flex flex-wrap gap-2" aria-label="Quyết định">
          <button type="button" className="btn" disabled>
            Trả lại kèm lý do
          </button>
          <button type="button" className="btn btn-primary" disabled>
            Duyệt &amp; Phát hành
          </button>
        </div>
      </div>
    );
  }

  const changes = diff.data?.diff?.changes ?? [];
  const baselineVersion = diff.data?.diff?.baseline_version;
  const isOwnWork = selected.authored_by === identity?.actor_id;
  const canApprove = Boolean(identity?.actions.includes("records.approve"));
  const canReview = !isOwnWork && canApprove;

  return (
    <div className="space-y-5 pb-[var(--review-bar-h,6rem)] md:pb-0">
      {heading}

      {isOwnWork ? (
        <div className="rounded-[8px] border border-[#F2C77A] bg-[#FFF3DC] p-4 text-[#6B3E00]">
          <p className="font-display text-[15px] font-semibold">
            Bạn là người tạo mục này nên cần một quản trị viên khác duyệt.
          </p>
          <p className="mt-1 text-[14px] leading-6">
            {canApprove
              ? "Hệ thống khóa nút Duyệt với tài khoản tác giả. Bạn vẫn có thể hỏi thử, so sánh và trả lại kèm lý do."
              : "Vai trò hiện tại của bạn không có quyền duyệt nên nút Duyệt bị khóa. Bạn vẫn có thể hỏi thử và so sánh."}
          </p>
        </div>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-[minmax(260px,3fr)_7fr]">
        <ReviewList
          rows={rows}
          selectedId={selected.record_id}
          actorId={identity?.actor_id}
          visibleCount={visibleCount}
          onShowMore={() => setVisibleCount((n) => n + PAGE_SIZE)}
          onSelect={selectRecord}
          review={review}
        />
        <div className="min-w-0 space-y-4">
          <ReviewDiffPanel
            record={selected}
            column={column}
            onColumnChange={setColumn}
            loading={diff.isPending}
            failed={diff.isError}
            changes={changes}
            baselineVersion={baselineVersion}
          />
          <ReviewSandbox
            question={previewQuery}
            onQuestionChange={setPreviewQuery}
            onAsk={(question) => previewMutation.mutate(question)}
            pending={previewMutation.isPending}
            preview={preview}
          />
          <ReviewDecisionBar
            review={review}
            record={selected}
            canReview={canReview}
            isOwnWork={isOwnWork}
            diffPending={diff.isPending}
          />
        </div>
      </div>

      <p className="text-sm text-zinc-600">
        Về <Link to="/quan-ly">Tổng quan</Link>
      </p>
    </div>
  );
};

export default PendingReview;
