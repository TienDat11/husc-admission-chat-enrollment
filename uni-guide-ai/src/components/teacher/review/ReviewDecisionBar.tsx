import { useEffect, useRef, useState } from "react";
import type { CanonicalRecord } from "@/lib/adminApi";
import type { ReviewActions } from "@/hooks/useReviewActions";
import { cn } from "@/lib/utils";

type Props = {
  review: ReviewActions;
  record: CanonicalRecord;
  canReview: boolean;
  isOwnWork: boolean;
  diffPending: boolean;
};

/**
 * Decision section: approve (with confirm), and reject. Reject stays disabled
 * until a reason is typed. On mobile the reason is collapsed behind a toggle
 * (CSS only: the textarea is always mounted); on md+ it is always visible.
 * The fixed bottom bar publishes its height as --review-bar-h so the page can
 * pad its content above it on mobile.
 */
export function ReviewDecisionBar({ review, record, canReview, isOwnWork, diffPending }: Props) {
  const [reasonOpen, setReasonOpen] = useState(false);
  const barRef = useRef<HTMLDivElement>(null);
  const busy = review.pendingAction != null;
  const reason = review.rejectReason;
  const trimmedReason = reason.trim();

  useEffect(() => {
    const decisionBar = barRef.current;
    if (!decisionBar || typeof ResizeObserver === "undefined") return;
    const htmlEl = document.documentElement;
    const syncHeight = () => htmlEl.style.setProperty("--review-bar-h", `${decisionBar.offsetHeight}px`);
    syncHeight();
    const observer = new ResizeObserver(syncHeight);
    observer.observe(decisionBar);
    return () => {
      observer.disconnect();
      htmlEl.style.removeProperty("--review-bar-h");
    };
  }, []);

  const onReject = () => {
    if (!trimmedReason) return;
    review.commit(record, "reject", trimmedReason);
  };

  return (
    <section aria-labelledby="cho-duyet-quyet-dinh" className="rounded-[12px] border border-[#D6DEEA] bg-white p-5">
      <h2 id="cho-duyet-quyet-dinh" className="!m-0 font-display text-[15px] font-semibold text-[#0F1B2D]">
        Quyết định
      </h2>

      {review.actionError ? (
        <p role="alert" className="note bad mt-3">
          {review.actionError}
        </p>
      ) : null}

      {review.confirming && canReview ? (
        <div
          role="alertdialog"
          aria-labelledby="xac-nhan-xuat-ban-h"
          aria-describedby="xac-nhan-xuat-ban-d"
          className="dialog mt-3"
        >
          <h3 id="xac-nhan-xuat-ban-h">Xuất bản "{record.record_type} {record.admission_year}"?</h3>
          <p id="xac-nhan-xuat-ban-d">
            Bot sẽ dùng thông tin này sau khi xuất bản. Có thể khôi phục bản trước trong Lịch sử.
          </p>
          <div className="actions">
            <button type="button" className="btn" onClick={() => review.setConfirming(false)}>
              Chưa, quay lại
            </button>
            <button
              type="button"
              className="btn btn-primary"
              disabled={review.pendingAction != null}
              onClick={() => review.commit(record, "approve")}
            >
              {review.pendingAction === "approve" ? "Đang xuất bản…" : "Xuất bản"}
            </button>
          </div>
        </div>
      ) : null}

      <div
        ref={barRef}
        className="fixed inset-x-0 bottom-0 z-10 space-y-2 border-t border-[#D6DEEA] bg-white p-3 md:static md:mt-4 md:space-y-3 md:border-0 md:bg-transparent md:p-0"
      >
        <div>
          <button
            type="button"
            className="btn min-h-11 px-2 text-[13px] font-medium text-brand-navy md:hidden"
            aria-expanded={reasonOpen}
            aria-controls="duyet-ly-do-wrap"
            onClick={() => setReasonOpen((open) => !open)}
          >
            Ghi chú trả lại
          </button>
          {trimmedReason && !reasonOpen ? (
            <p className="mt-1 truncate text-[13px] text-[#474747] md:hidden">{trimmedReason}</p>
          ) : null}
          <div id="duyet-ly-do-wrap" className={cn("field mt-2 w-full min-w-0", reasonOpen ? "block" : "hidden md:block")}>
            <label htmlFor="duyet-ly-do">Lý do trả lại (bắt buộc khi trả lại)</label>
            <textarea
              id="duyet-ly-do"
              rows={2}
              value={reason}
              onChange={(event) => review.setRejectReason(event.target.value)}
            />
          </div>
        </div>
        {trimmedReason ? null : (
          <p id="duyet-ly-do-hint" className="text-[13px] text-[#474747]">Nhập lý do trả lại để bật nút</p>
        )}
        <div className="flex flex-col gap-2 md:flex-row md:flex-wrap md:items-end">
          <button
            type="button"
            className="btn flex-1 !border-brand-navy !bg-white !text-brand-navy md:flex-none"
            disabled={diffPending || busy || !trimmedReason}
            aria-describedby={trimmedReason ? undefined : "duyet-ly-do-hint"}
            onClick={onReject}
          >
            {review.pendingAction === "reject" ? "Đang trả lại…" : "Trả lại kèm lý do"}
          </button>
          <button
            type="button"
            className="btn btn-primary flex-1 md:flex-none"
            disabled={!canReview || diffPending || busy}
            onClick={() => review.setConfirming(true)}
          >
            {review.pendingAction === "approve" ? "Đang xuất bản…" : "Duyệt & Phát hành"}
            <span aria-hidden="true" className="md:hidden">
              {" "}→
            </span>
          </button>
        </div>
        {isOwnWork ? <p className="text-[13px] text-[#474747]">Không thể tự duyệt mục do bạn tạo.</p> : null}
      </div>
    </section>
  );
}
