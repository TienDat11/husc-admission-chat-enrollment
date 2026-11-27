import type { CanonicalRecord } from "@/lib/adminApi";
import type { ReviewActions } from "@/hooks/useReviewActions";
import { cn } from "@/lib/utils";
import { recordTitle } from "./recordTitle";
type Props = {
  rows: CanonicalRecord[];
  selectedId: string;
  actorId: string | undefined;
  visibleCount: number;
  onShowMore: () => void;
  onSelect: (recordId: string) => void;
  review: ReviewActions;
};

/** Left column: the records waiting for review, batch publish and draft claim. */
export function ReviewList({ rows, selectedId, actorId, visibleCount, onShowMore, onSelect, review }: Props) {
  const {
    selectedIds, toggleSelect, batchKey, batchConfirming, setBatchConfirming, runBatchApprove, batchResult,
    claimConfirmingId, setClaimConfirmingId, claimKey, runClaim, claimError,
  } = review;
  return (
    <section
      aria-labelledby="cho-duyet-danh-sach"
      className="self-start rounded-[12px] border border-[#D6DEEA] bg-white p-4"
    >
      <h2 id="cho-duyet-danh-sach" className="!m-0 font-display text-[15px] font-semibold text-[#0F1B2D]">
        Chờ duyệt · {rows.length} mục
      </h2>
      <ul className="mt-3 space-y-2">
        {rows.slice(0, visibleCount).map((record) => {
          const active = selectedId === record.record_id;
          const own = record.authored_by === actorId;
          return (
            <li
              key={record.record_id}
              className={cn("rounded-[8px] border p-3", active ? "border-brand-navy bg-[#E6ECF7]" : "border-[#E4E9F2] bg-[#F7F9FC]")}
            >
              <div className="flex items-start gap-2">
                {own ? null : (
                  <input
                    type="checkbox"
                    className="mt-1"
                    checked={selectedIds.includes(record.record_id)}
                    onChange={() => toggleSelect(record.record_id)}
                    aria-label={`Chọn ${record.record_type} (${record.admission_year}) để duyệt hàng loạt`}
                  />
                )}
                <button
                  type="button"
                  aria-pressed={active}
                  onClick={() => onSelect(record.record_id)}
                  className="min-w-0 flex-1 text-left"
                >
                  <span className="block font-display text-[14px] font-semibold text-[#0F1B2D]">{recordTitle(record)}</span>
                  <span className="block text-[13px] text-[#6B7686]">
                    Tác giả: {own ? "bạn" : (record.authored_by ?? "không rõ")} · phiên bản{" "}
                    {record.version}
                  </span>
                </button>
              </div>
              {own ? (
                <span className="mt-2 inline-flex h-6 items-center rounded-full bg-[#E6E9EF] px-2.5 text-[12px] font-medium text-[#474747]">
                  Cần người khác duyệt
                </span>
              ) : null}
              {record.authored_by == null ? (
                <div className="mt-2">
                  <button type="button" className="btn" onClick={() => setClaimConfirmingId(record.record_id)}>
                    Nhận nháp
                  </button>
                  {claimConfirmingId === record.record_id ? (
                    <div role="alertdialog" aria-label="Xác nhận nhận nháp" className="dialog">
                      <h3>Xác nhận nhận nháp?</h3>
                      <p>
                        Sau khi nhận, bạn sẽ trở thành tác giả của bản nháp này và{" "}
                        <strong>không thể tự duyệt</strong> — cần một người khác duyệt.
                      </p>
                      <div className="actions">
                        <button type="button" className="btn" onClick={() => setClaimConfirmingId(null)}>
                          Chưa, quay lại
                        </button>
                        <button
                          type="button"
                          className="btn btn-primary"
                          disabled={claimKey != null}
                          onClick={() => void runClaim(record)}
                        >
                          Nhận nháp
                        </button>
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : null}
            </li>
          );
        })}
      </ul>
      {rows.length > visibleCount ? (
        <div className="mt-3 flex items-center justify-between gap-3 text-[13px] text-[#474747]">
          <span>Hiển thị {visibleCount} / {rows.length} mục</span>
          <button type="button" className="btn" onClick={onShowMore}>
            Hiển thị thêm
          </button>
        </div>
      ) : null}
      {claimError ? (
        <p role="alert" className="note bad mt-3">
          {claimError}
        </p>
      ) : null}
      {selectedIds.length > 0 ? (
        <div className="mt-3 space-y-3" aria-label="Duyệt hàng loạt">
          <button type="button" className="btn" disabled={batchKey != null} onClick={() => setBatchConfirming(true)}>
            {batchKey != null ? "Đang xuất bản…" : `Duyệt và xuất bản ${selectedIds.length} mục`}
          </button>
          {batchConfirming ? (
            <div role="alertdialog" aria-labelledby="xac-nhan-hang-loat-h" aria-describedby="xac-nhan-hang-loat-d" className="dialog">
              <h3 id="xac-nhan-hang-loat-h">Duyệt và xuất bản {selectedIds.length} mục?</h3>
              <p id="xac-nhan-hang-loat-d">
                Bot sẽ dùng thông tin mới cho {selectedIds.length} mục này sau khi xuất bản. Có thể khôi phục bản trước trong Lịch sử.
              </p>
              <div className="actions">
                <button type="button" className="btn" onClick={() => setBatchConfirming(false)}>
                  Chưa, quay lại
                </button>
                <button type="button" className="btn btn-primary" onClick={() => void runBatchApprove(rows)}>
                  Duyệt và xuất bản {selectedIds.length} mục
                </button>
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
      {batchResult ? (
        <div role="status" aria-live="polite" className="note mt-3">
          <p>
            Đã xuất bản {batchResult.succeeded.length} mục
            {batchResult.failures.length > 0 ? `, ${batchResult.failures.length} mục thất bại:` : "."}
          </p>
          {batchResult.failures.length > 0 ? (
            <ul className="list-disc pl-5">
              {batchResult.failures.map((failure) => (
                <li key={failure.recordId}>
                  <code>{failure.recordId}</code>: {failure.error}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
