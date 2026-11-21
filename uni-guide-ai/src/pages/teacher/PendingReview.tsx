import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useAdminSession } from "@/hooks/useAdminSession";
import {
  fetchReviewDiff,
  listRecords,
  newIdempotencyKey,
  runReviewPreview,
  type CanonicalRecord,
  type ReviewPreviewResult,
} from "@/lib/adminApi";
import { RECORD_TYPE_LABELS } from "@/lib/adminValidation";
import {
  approveAndPublish,
  approveAndPublishMany,
  claimDraft,
  rejectWithReason,
} from "@/lib/teacherApi";
import { cn } from "@/lib/utils";

type Column = "current" | "proposed" | "pdf";

const COLUMNS: { id: Column; tab: string }[] = [
  { id: "current", tab: "Đang chạy" },
  { id: "proposed", tab: "Đề xuất" },
  { id: "pdf", tab: "PDF gốc" },
];

const recordTitle = (record: CanonicalRecord) =>
  `${RECORD_TYPE_LABELS[record.record_type] ?? record.record_type} ${record.admission_year}`;

/**
 * "Chờ duyệt & phát hành" (frame Xcckq; mobile N8GHat). The list of records
 * waiting for review on the left; the selected record's three comparison
 * columns, sandbox and decision on the right. Four-eyes: the publish button is
 * disabled for the record's author, who can still ask the Bot, compare, and
 * return the record with a reason.
 */
const PendingReview = () => {
  const { identity } = useAdminSession();
  const queryClient = useQueryClient();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [column, setColumn] = useState<Column>("proposed");
  const [preview, setPreview] = useState<ReviewPreviewResult | null>(null);
  const [previewQuery, setPreviewQuery] = useState("");
  const [rejectReason, setRejectReason] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  // One key per intention, minted when the teacher commits to an action,
  // kept across retries of that same click (the way RecordList does it).
  const [intent, setIntent] = useState<{
    key: string;
    recordId: string;
    action: "approve" | "reject";
  } | null>(null);
  // T15 batch review: one key per record per intention so a retry replays.
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [batchConfirming, setBatchConfirming] = useState(false);
  const [batchKey, setBatchKey] = useState<string | null>(null);
  const [batchResult, setBatchResult] = useState<{
    succeeded: string[];
    failures: Array<{ recordId: string; error: string }>;
  } | null>(null);
  // T15 claim: per-record confirm state so the 409 reason renders per card.
  const [claimConfirmingId, setClaimConfirmingId] = useState<string | null>(null);
  const [claimKey, setClaimKey] = useState<string | null>(null);
  const [claimError, setClaimError] = useState<string | null>(null);

  const list = useQuery({
    queryKey: ["admin", "records", { status: "review" }],
    queryFn: () => listRecords(null, { status: "review" }),
    retry: false,
  });

  const rows = list.data?.records ?? [];
  const selected = rows.find((record) => record.record_id === selectedId) ?? rows[0] ?? null;

  const diff = useQuery({
    queryKey: ["admin", "review-diff", selected?.record_id],
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

  const commit = (record: CanonicalRecord, action: "approve" | "reject", reason?: string) => {
    const reused =
      intent && intent.recordId === record.record_id && intent.action === action
        ? intent.key
        : null;
    const key = reused ?? newIdempotencyKey();
    setIntent({ key, recordId: record.record_id, action });
    setActionError(null);
    const run =
      action === "approve"
        ? approveAndPublish(null, record.record_id, record.version, key)
        : rejectWithReason(null, record.record_id, record.version, reason ?? "", key);
    run
      .then(() => {
        setConfirming(false);
        setRejectReason("");
        setIntent(null);
        void queryClient.invalidateQueries({ queryKey: ["admin", "records"] });
      })
      .catch((failure: unknown) => {
        setActionError(
          failure instanceof Error ? failure.message : "Hành động thất bại.",
        );
      });
  };

  const toggleSelect = (recordId: string) => {
    setSelectedIds((ids) =>
      ids.includes(recordId) ? ids.filter((id) => id !== recordId) : [...ids, recordId],
    );
    setBatchResult(null);
  };
  const runBatchApprove = async () => {
    const key = batchKey ?? newIdempotencyKey();
    setBatchKey(key);
    setBatchConfirming(false);
    const result = await approveAndPublishMany(
      null,
      rows
        .filter((r) => selectedIds.includes(r.record_id))
        .map((r) => ({ recordId: r.record_id, version: r.version })),
      (recordId) => `${key}:${recordId}`,
    );
    setBatchResult(result);
    setBatchKey(null);
    setSelectedIds([]);
    void queryClient.invalidateQueries({ queryKey: ["admin", "records"] });
  };
  const runClaim = async (record: CanonicalRecord) => {
    const key = claimKey ?? newIdempotencyKey();
    setClaimKey(key);
    setClaimConfirmingId(null);
    setClaimError(null);
    try {
      await claimDraft(null, record.record_id, record.version, key);
      setClaimKey(null);
      void queryClient.invalidateQueries({ queryKey: ["admin", "records"] });
    } catch (failure: unknown) {
      const message =
        failure instanceof Error
          ? failure.message
          : "Hành động thất bại.";
      setClaimError(message);
      setClaimKey(null);
    }
  };

  const heading = (
    <h1 className="!m-0 font-display text-[24px] font-bold leading-8 text-[#0F1B2D]">
      Chờ duyệt & phát hành
    </h1>
  );

  if (list.isPending) {
    return (
      <p role="status" aria-live="polite" className="text-zinc-600">
        Đang lấy danh sách chờ duyệt…
      </p>
    );
  }

  if (list.isError) {
    return (
      <div role="alert" className="note bad">
        Chưa lấy được danh sách chờ duyệt. Kiểm tra kết nối mạng rồi thử lại.
      </div>
    );
  }

  if (!selected) {
    return (
      <div className="space-y-5">
        {heading}
        <p role="status" className="note ok">
          Không có thông tin nào chờ duyệt.
        </p>
      </div>
    );
  }

  const changes = diff.data?.diff?.changes ?? [];
  const baselineVersion = diff.data?.diff?.baseline_version;
  const isOwnWork = selected.authored_by === identity?.actor_id;
  const canReview = !isOwnWork && Boolean(identity?.actions.includes("records.approve"));

  return (
    <div className="space-y-5 pb-24 md:pb-0">
      {heading}

      {isOwnWork ? (
        <div className="rounded-[8px] border border-[#F2C77A] bg-[#FFF3DC] p-4 text-[#6B3E00]">
          <p className="font-display text-[15px] font-semibold">
            Bạn là người tạo mục này nên cần một quản trị viên khác duyệt.
          </p>
          <p className="mt-1 text-[14px] leading-6">
            Hệ thống khóa nút Duyệt với tài khoản tác giả. Bạn vẫn có thể hỏi thử, so sánh và
            trả lại kèm lý do.
          </p>
        </div>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-[minmax(260px,3fr)_7fr]">
        <section
          aria-labelledby="cho-duyet-danh-sach"
          className="self-start rounded-[12px] border border-[#D6DEEA] bg-white p-4"
        >
          <h2
            id="cho-duyet-danh-sach"
            className="!m-0 font-display text-[15px] font-semibold text-[#0F1B2D]"
          >
            Chờ duyệt · {rows.length} mục
          </h2>
          <ul className="mt-3 space-y-2">
            {rows.map((record) => {
              const active = selected.record_id === record.record_id;
              const own = record.authored_by === identity?.actor_id;
              return (
                <li
                  key={record.record_id}
                  className={cn(
                    "rounded-[8px] border p-3",
                    active
                      ? "border-[#143980] bg-[#E6ECF7]"
                      : "border-[#E4E9F2] bg-[#F7F9FC]",
                  )}
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
                      onClick={() => {
                        setSelectedId(record.record_id);
                        setPreview(null);
                        setColumn("proposed");
                      }}
                      className="min-w-0 flex-1 text-left"
                    >
                      <span className="block font-display text-[14px] font-semibold text-[#0F1B2D]">
                        {recordTitle(record)}
                      </span>
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
                      <button
                        type="button"
                        className="btn"
                        onClick={() => setClaimConfirmingId(record.record_id)}
                      >
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
                            <button
                              type="button"
                              className="btn"
                              onClick={() => setClaimConfirmingId(null)}
                            >
                              Chưa, quay lại
                            </button>
                            <button
                              type="button"
                              className="btn btn-primary"
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

          {claimError ? (
            <p role="alert" className="note bad mt-3">
              {claimError}
            </p>
          ) : null}

          {selectedIds.length > 0 ? (
            <div className="mt-3 space-y-3" aria-label="Duyệt hàng loạt">
              <button
                type="button"
                className="btn"
                onClick={() => setBatchConfirming(true)}
              >
                Duyệt và xuất bản {selectedIds.length} mục
              </button>
              {batchConfirming ? (
                <div
                  role="alertdialog"
                  aria-labelledby="xac-nin-nhieu-h"
                  aria-describedby="xac-nin-nhieu-d"
                  className="dialog"
                >
                  <h3 id="xac-nin-nhieu-h">
                    Duyệt và xuất bản {selectedIds.length} mục?
                  </h3>
                  <p id="xac-nin-nhieu-d">
                    Bot sẽ trả lời theo thông tin mới cho {selectedIds.length} mục này cho
                    thí sinh, trong khoảng 5–10 phút sau khi xuất bản. Có thể khôi phục
                    bản trước bất cứ lúc nào.
                  </p>
                  <div className="actions">
                    <button
                      type="button"
                      className="btn"
                      onClick={() => setBatchConfirming(false)}
                    >
                      Chưa, quay lại
                    </button>
                    <button
                      type="button"
                      className="btn btn-primary"
                      onClick={() => {
                        void runBatchApprove();
                      }}
                    >
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
                {batchResult.failures.length > 0
                  ? `, ${batchResult.failures.length} mục thất bại:`
                  : "."}
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

        <div className="min-w-0 space-y-4">
          <section
            aria-label="So sánh bản đang chạy, đề xuất mới và bản gốc PDF"
            className="rounded-[12px] border border-[#D6DEEA] bg-white p-5"
          >
            <h2 className="!m-0 font-display text-[18px] font-semibold text-[#0F1B2D]">
              {recordTitle(selected)}
            </h2>

            <div role="tablist" aria-label="Chọn cột so sánh" className="mt-4 flex gap-2 md:hidden">
              {COLUMNS.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  role="tab"
                  aria-selected={column === item.id}
                  onClick={() => setColumn(item.id)}
                  className={cn(
                    "min-h-11 flex-1 rounded-[8px] border px-2 text-[13px] font-medium",
                    column === item.id
                      ? "border-[#143980] bg-[#143980] text-white"
                      : "border-[#D6DEEA] bg-white text-[#0F1B2D]",
                  )}
                >
                  {item.tab}
                </button>
              ))}
            </div>

            {diff.isPending ? (
              <p role="status" className="mt-4 text-zinc-600">
                Đang tải so sánh…
              </p>
            ) : diff.isError ? (
              <p role="alert" className="note bad mt-4">
                Không tải được bản so sánh.
              </p>
            ) : (
              <div className="mt-4 grid gap-4 md:grid-cols-3">
                <div
                  className={cn(
                    "rounded-[8px] bg-[#F3F5F9] p-4",
                    column === "current" ? "block" : "hidden",
                    "md:block",
                  )}
                >
                  <h3 className="!m-0 font-display text-[15px] font-semibold">Đang chạy trên Bot</h3>
                  {baselineVersion == null ? (
                    <p className="mt-2 text-[14px] text-[#474747]">
                      Chưa có bản đang chạy để so sánh.
                    </p>
                  ) : changes.length === 0 ? (
                    <p className="mt-2 text-[14px] text-[#474747]">Không có thay đổi.</p>
                  ) : (
                    <ul className="mt-2 list-none space-y-2 p-0 text-[14px]">
                      {changes.map((change) => (
                        <li key={change.field}>
                          <span className="font-medium">{change.field}:</span>{" "}
                          {change.old != null ? String(change.old) : "chưa có"}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                <div
                  className={cn(
                    "rounded-[8px] border border-[#143980] p-4",
                    column === "proposed" ? "block" : "hidden",
                    "md:block",
                  )}
                >
                  <h3 className="!m-0 font-display text-[15px] font-semibold">Đề xuất mới</h3>
                  <p className="mt-2 whitespace-pre-line text-[14px] leading-6 text-[#0F1B2D]">
                    {selected.text}
                  </p>
                  {changes.length > 0 ? (
                    <ul className="mt-3 list-none space-y-1 p-0 text-[13px]">
                      {changes.map((change) => (
                        <li key={change.field}>
                          {change.field}:{" "}
                          <span className="changed">{String(change.new ?? "")} (đã đổi)</span>
                        </li>
                      ))}
                    </ul>
                  ) : null}
                </div>

                <div
                  className={cn(
                    "rounded-[8px] bg-[#F3F5F9] p-4",
                    column === "pdf" ? "block" : "hidden",
                    "md:block",
                  )}
                >
                  <h3 className="!m-0 font-display text-[15px] font-semibold">Bản gốc PDF</h3>
                  <p className="mt-2 font-mono text-[13px] text-[#0F1B2D]">
                    {selected.source_document_id ?? "không ghi nhận"}
                    {selected.source_page ? ` · trang ${selected.source_page}` : ""}
                  </p>
                  <p className="mt-2 text-[13px] text-[#474747]">
                    Đối chiếu nội dung trực tiếp trên tệp gốc.
                  </p>
                  {selected.source_url ? (
                    <a
                      href={selected.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="mt-2 inline-flex min-h-11 items-center text-[14px] font-semibold text-[#143980] underline"
                    >
                      Mở tài liệu gốc
                    </a>
                  ) : null}
                </div>
              </div>
            )}
          </section>

          <section
            aria-labelledby="cho-duyet-sandbox"
            className="rounded-[12px] border border-[#D6DEEA] bg-white p-5"
          >
            <h2
              id="cho-duyet-sandbox"
              className="!m-0 font-display text-[15px] font-semibold text-[#0F1B2D]"
            >
              Hỏi thử Bot với dữ liệu này (Sandbox)
            </h2>
            <p className="mt-1 text-[13px] text-[#6B7686]">
              Chỉ tìm trong bản nháp và chỉ mục đang phục vụ. Không ghi vào phiên bản phát hành.
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              <label htmlFor="duyet-cau-hoi" className="sr-only">
                Câu hỏi thử của thí sinh
              </label>
              <input
                id="duyet-cau-hoi"
                type="text"
                className="min-w-[14rem] flex-1"
                value={previewQuery}
                onChange={(event) => setPreviewQuery(event.target.value)}
                placeholder="Câu hỏi của thí sinh"
              />
              <button
                type="button"
                className="btn"
                disabled={!previewQuery.trim() || previewMutation.isPending}
                onClick={() => previewMutation.mutate(previewQuery.trim())}
              >
                {previewMutation.isPending ? "Đang chạy…" : "Hỏi thử"}
              </button>
            </div>
            {preview ? (
              <div className="answer mt-3" role="status" aria-live="polite">
                <p>
                  <strong>Bot sẽ trả lời:</strong> {preview.answer ?? "(chưa có câu trả lời)"}
                </p>
                {preview.sources.length > 0 ? (
                  <div>
                    <p className="hint">Nguồn trích dẫn:</p>
                    {preview.sources.map((source) => (
                      <span
                        key={`${source.source_document_id ?? ""}|${source.section ?? ""}|${source.source}`}
                        className="cite"
                      >
                        {source.title ?? source.section ?? source.source}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : null}
          </section>

          <section
            aria-labelledby="cho-duyet-quyet-dinh"
            className="rounded-[12px] border border-[#D6DEEA] bg-white p-5"
          >
            <h2
              id="cho-duyet-quyet-dinh"
              className="!m-0 font-display text-[15px] font-semibold text-[#0F1B2D]"
            >
              Quyết định
            </h2>

            {actionError ? (
              <p role="alert" className="note bad mt-3">
                {actionError}
              </p>
            ) : null}

            {confirming ? (
              <div
                role="alertdialog"
                aria-labelledby="xac-nin-h"
                aria-describedby="xac-nin-d"
                className="dialog mt-3"
              >
                <h3 id="xac-nin-h">Xuất bản "{recordTitle(selected)}"?</h3>
                <p id="xac-nin-d">
                  Bot sẽ trả lời theo thông tin này cho thí sinh, trong khoảng 5–10 phút
                  sau khi xuất bản. Có thể khôi phục bản trước bất cứ lúc nào.
                </p>
                <div className="actions">
                  <button type="button" className="btn" onClick={() => setConfirming(false)}>
                    Chưa, quay lại
                  </button>
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={() => commit(selected, "approve")}
                  >
                    Xuất bản
                  </button>
                </div>
              </div>
            ) : null}

            <div className="fixed inset-x-0 bottom-0 z-10 space-y-2 border-t border-[#D6DEEA] bg-white p-3 md:static md:mt-4 md:space-y-3 md:border-0 md:bg-transparent md:p-0">
              <div className="flex flex-col gap-2 md:flex-row md:flex-wrap md:items-end">
                <div className="field w-full min-w-0 flex-1 md:min-w-[16rem]">
                  <label htmlFor="duyet-ly-do">Lý do trả lại (bắt buộc khi trả lại)</label>
                  <textarea
                    id="duyet-ly-do"
                    rows={1}
                    value={rejectReason}
                    onChange={(event) => setRejectReason(event.target.value)}
                  />
                </div>
                <button
                  type="button"
                  className="btn flex-1 !border-[#143980] !bg-white !text-[#143980] md:flex-none"
                  disabled={!rejectReason.trim() || diff.isPending}
                  onClick={() => commit(selected, "reject", rejectReason.trim())}
                >
                  Trả lại kèm lý do
                </button>
                <button
                  type="button"
                  className="btn btn-primary flex-1 md:flex-none"
                  disabled={!canReview || diff.isPending}
                  onClick={() => setConfirming(true)}
                >
                  Duyệt & Phát hành
                  <span aria-hidden="true" className="md:hidden">
                    {" "}→
                  </span>
                </button>
              </div>
              {isOwnWork ? (
                <p className="text-[13px] text-[#474747]">
                  Không thể tự duyệt mục do bạn tạo.
                </p>
              ) : null}
            </div>
          </section>
        </div>
      </div>

      <p className="text-sm text-zinc-600">
        Về <Link to="/quan-ly">Tổng quan</Link>
      </p>
    </div>
  );
};

export default PendingReview;
