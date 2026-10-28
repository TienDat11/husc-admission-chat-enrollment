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
import { approveAndPublish, approveAndPublishMany, claimDraft, rejectWithReason } from "@/lib/teacherApi";

const STATUS_LABELS: Record<string, string> = {
  draft: "Bản nháp",
  review: "Chờ duyệt",
  approved: "Đã duyệt",
  published: "Đang dùng",
  archived: "Đã lưu trữ",
};

/**
 * Ticket T09 — "Chờ duyệt" (mockup 04, 05 §4.4).
 *
 * The card list, the side-by-side diff, the two actions ("Trả lại để sửa"
 * with a required reason; "Duyệt và xuất bản" behind a confirmation box that
 * names the consequence). Four-eyes is enforced by hiding the publish button
 * when the record's author is the signed-in reviewer — the text reads
 * "Cần một người khác duyệt" instead.
 */
const PendingReview = () => {
  const { identity } = useAdminSession();
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState<CanonicalRecord | null>(null);
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

  const rows = list.data?.records ?? [];
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
      () => key,
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

  if (list.isPending) {
    return (
      <p role="status" aria-live="polite" className="text-zinc-600">
        <span aria-hidden="true">⏳</span> Đang lấy danh sách chờ duyệt…
      </p>
    );
  }

  if (list.isError) {
    return (
      <div role="alert" className="note bad">
        <span aria-hidden="true">⚠</span> Chưa lấy được danh sách chờ duyệt.
        Kiểm tra kết nối mạng rồi thử lại.
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <p role="status" className="note ok">
        <span aria-hidden="true">✓</span> Không có thông tin nào chờ duyệt.
      </p>
    );
  }

  const changes = diff.data?.diff?.changes ?? [];
  const isOwnWork =
    selected !== null && selected.authored_by === identity?.actor_id;
  const canReview = !isOwnWork && Boolean(identity?.actions.includes("records.approve"));

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Chờ duyệt ({rows.length})</h1>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {rows.map((record) => (
          <div key={record.record_id} className="card">
            <h3>
              {STATUS_LABELS[record.record_type] ?? record.record_type} (
              {record.admission_year})
            </h3>
            <p>
              {record.authored_by ?? "không rõ"} · {record.version} thay đổi
            </p>
            {record.authored_by === identity?.actor_id ? (
              <p>
                <span className="status s-muted">👤 Cần một người khác duyệt</span>
              </p>
            ) : (
              <div className="flex items-center gap-3">
                <label className="check">
                  <input
                    type="checkbox"
                    checked={selectedIds.includes(record.record_id)}
                    onChange={() => toggleSelect(record.record_id)}
                    aria-label={`Chọn ${record.record_type} (${record.admission_year}) để duyệt hàng loạt`}
                  />
                  Chọn
                </label>
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    setSelected(record);
                    setPreview(null);
                  }}
                >
                  Mở để duyệt
                </button>
                {record.authored_by == null ? (
                  <>
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
                    {claimError ? (
                      <p role="alert" className="note bad">
                        <span aria-hidden="true">⚠</span> {claimError}
                      </p>
                    ) : null}
                  </>
                ) : null}
              </div>
            )}

          </div>
        ))}
      </div>

      {/* T15 batch action bar */}
      {selectedIds.length > 0 || batchResult !== null ? (
        <div className="space-y-3" aria-label="Duyệt hàng loạt">
          {selectedIds.length > 0 ? (
            <div className="actions">
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => setBatchConfirming(true)}
              >
                Duyệt và xuất bản {selectedIds.length} mục
              </button>
            </div>
          ) : null}
          {batchResult ? (
            <div role="status" aria-live="polite" className="note">
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

      {selected ? (
        <section aria-label="So sánh bản đang dùng và bản mới" className="space-y-4">
          <h2 className="text-xl font-bold">
            {selected.record_type} — so sánh bản đang dùng và bản mới
          </h2>

          {diff.isPending ? (
            <p role="status" className="text-zinc-600">
              Đang tải so sánh…
            </p>
          ) : diff.isError ? (
            <p role="alert" className="note bad">
              <span aria-hidden="true">⚠</span> Không tải được bản so sánh.
            </p>
          ) : (
            <div className="diff">
              <div className="col">
                <h3>Bản đang dùng</h3>
                <p>{selected.text}</p>
              </div>
              <div className="col">
                <h3>Bản mới</h3>
                {changes.length === 0 ? (
                  <p>Không có thay đổi.</p>
                ) : (
                  <ul className="m-0 list-none space-y-1 p-0">
                    {changes.map((change) => (
                      <li key={change.field}>
                        {change.field}:{" "}
                        {change.old != null ? (
                          <span className="line-through opacity-60">{String(change.old)}</span>
                        ) : null}{" "}
                        <span className="changed">
                          {String(change.new ?? "")} (đã đổi)
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}

          {/* Test-ask before deciding */}
          <div className="space-y-2">
            <label className="font-semibold" htmlFor="duyet-cau-hoi">
              Hỏi thử trước khi quyết định
            </label>
            <div className="flex gap-2">
              <input
                id="duyet-cau-hoi"
                type="text"
                className="flex-1"
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
              <div className="answer" role="status" aria-live="polite">
                <p>
                  <strong>Bot sẽ trả lời:</strong> {preview.answer ?? "(chưa có câu trả lời)"}
                </p>
                {preview.sources.length > 0 ? (
                  <div>
                    <p className="hint">Nguồn (bấm để mở đúng chỗ):</p>
                    {preview.sources.map((source, index) => (
                      <a
                        key={`${source.source_document_id ?? source.source}-${index}`}
                        className="cite"
                        href="#"
                        onClick={(event) => event.preventDefault()}
                      >
                        📄 {source.title ?? source.section ?? source.source}
                      </a>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>

          {/* Actions */}
          <div className="space-y-3">
            <div className="field">
              <label htmlFor="duyet-ly-do">Lý do trả lại (bắt buộc nếu trả lại)</label>
              <textarea
                id="duyet-ly-do"
                value={rejectReason}
                onChange={(event) => setRejectReason(event.target.value)}
              />
            </div>

            {actionError ? (
              <p role="alert" className="note bad">
                <span aria-hidden="true">⚠</span> {actionError}
              </p>
            ) : null}

            {isOwnWork ? (
              <p className="text-zinc-600">
                <span className="status s-muted">👤 Cần một người khác duyệt</span> —
                người soạn không duyệt được bài của mình.
              </p>
            ) : (
              <div className="actions">
                <button
                  type="button"
                  className="btn"
                  disabled={!rejectReason.trim() || diff.isPending}
                  onClick={() => commit(selected, "reject", rejectReason.trim())}
                >
                  Trả lại để sửa
                </button>
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={!canReview || diff.isPending}
                  onClick={() => setConfirming(true)}
                >
                  Duyệt và xuất bản
                </button>
              </div>
            )}

            {confirming ? (
              <div
                role="alertdialog"
                aria-labelledby="xac-nin-h"
                aria-describedby="xac-nin-d"
                className="dialog"
              >
                <h3 id="xac-nin-h">Xuất bản "{selected.record_type} ({selected.admission_year})"?</h3>
                <p id="xac-nin-d">
                  Bot sẽ trả lời theo thông tin này cho thí sinh, trong khoảng 5–10 phút
                  sau khi xuất bản. Có thể khôi phục bản trước bất cứ lúc nào.
                </p>
                <div className="actions">
                  <button
                    type="button"
                    className="btn"
                    onClick={() => setConfirming(false)}
                  >
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
          </div>

          <button
            type="button"
            className="btn"
            onClick={() => {
              setSelected(null);
              setPreview(null);
              setConfirming(false);
            }}
          >
            Quay lại danh sách
          </button>
        </section>
      ) : null}

      <p className="text-sm text-zinc-600">
        Về <Link to="/quan-ly">Tổng quan</Link>
      </p>
    </div>
  );
};

export default PendingReview;
