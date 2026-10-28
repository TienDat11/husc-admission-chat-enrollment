// Review diff + preview — ticket 25.
//
// The admissions officer sees, before anything reaches students:
//
//   1. exactly what this draft changes against the currently published
//      version (server-computed diff; high-impact fields flagged server-side,
//      never re-derived here);
//   2. the answer a student would actually receive if it were published —
//      produced by the real retrieval+generation path, not a simulated copy;
//   3. validation errors that would block publication, rendered at the field
//      they name, which also *block the submit button* — a draft with a
//      provenance hole cannot leave this screen.
//
// Submitting for review happens here (criterion 4): the same lifecycle
// transition the list offers, sent from the preview screen with the signed-in
// actor's credential, so the reviewer never has to leave the flow to act on
// what they just saw.
import { useCallback, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  fetchRecord,
  fetchReviewDiff,
  newIdempotencyKey,
  runReviewPreview,
  openSourceDocument,
  transitionRecord,
  type ReviewPreviewResult,
} from "@/lib/adminApi";
import { useAdminSession } from "@/hooks/useAdminSession";
import { Label } from "@/components/ui/label";
import { STATUS_LABELS } from "@/lib/adminValidation";

/** Human labels for the fields the preview's validation can name. */
const FIELD_LABELS: Record<string, string> = {
  source_document_id: "Tài liệu nguồn",
  effective_from: "Hiệu lực từ",
  "": "Toàn bộ record",
};

function fieldLabel(field: string): string {
  return FIELD_LABELS[field] ?? field;
}

/** One clickable citation for a source document. On click, fetches the raw bytes
 * with the session cookie, opens the blob in a new tab, and reclaims the object URL. */
function SourceLink({ source }: { source: NonNullable<ReviewPreviewResult["sources"]>[number] }) {
  const [fetching, setFetching] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const open = useCallback(async () => {
    if (!source.source_document_id) return;
    setFetching(true);
    try {
      const url = await openSourceDocument(null, source.source_document_id);
      window.open(
        `${url}${source.page != null ? `#page=${source.page}` : ""}`,
        "_blank",
        "noopener,noreferrer",
      );
      URL.revokeObjectURL(url);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "lỗi tải nguồn");
    } finally {
      setFetching(false);
    }
  }, [source.source_document_id, source.page]);

  const label = source.title ?? source.section ?? source.source ?? source.source_document_id;

  if (err) {
    return <span title={err}>{label}</span>;
  }
  if (!source.source_document_id) {
    return <span>{label}</span>;
  }
  return (
    <button
      type="button"
      onClick={open}
      disabled={fetching}
      className="text-muted-foreground dark:text-zinc-400 underline"
    >
      {fetching ? "Đang tải…" : label}
    </button>
  );
}

const ReviewPreview = () => {
  const { recordId = "" } = useParams();
  const navigate = useNavigate();
  const { can } = useAdminSession();
  const [query, setQuery] = useState("");
  const [preview, setPreview] = useState<ReviewPreviewResult | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  // One key per submit intention, reused across retries of that click.
  const [submitKey, setSubmitKey] = useState<string | null>(null);

  const record = useQuery({
    queryKey: ["admin", "record", recordId],
    queryFn: () => fetchRecord(null, recordId),
    enabled: recordId.length > 0,
    retry: false,
  });

  const diff = useQuery({
    queryKey: ["admin", "review-diff", recordId],
    queryFn: () => fetchReviewDiff(null, recordId),
    enabled: recordId.length > 0,
    retry: false,
  });

  const previewMutation = useMutation({
    mutationFn: (question: string) =>
      runReviewPreview(null, recordId, { query: question }),
    onSuccess: (result) => setPreview(result),
  });

  const submit = useMutation({
    mutationFn: (input: { expectedVersion: number; idempotencyKey: string }) =>
      transitionRecord(
        null,
        recordId,
        { action: "submit", expected_version: input.expectedVersion },
        { idempotencyKey: input.idempotencyKey },
      ),
    onSuccess: () => {
      setSubmitError(null);
      void record.refetch();
      void diff.refetch();
    },
    onError: (failure: unknown) => {
      setSubmitError(
        failure instanceof Error ? failure.message : "Gửi duyệt thất bại.",
      );
    },
  });

  const validationErrors = diff.data?.validation_errors ?? [];
  const hasBlockingErrors = validationErrors.length > 0;
  const changes = diff.data?.diff.changes ?? [];
  const highImpact = changes.filter((change) => change.high_impact);

  const handleSubmit = () => {
    if (!record.data || hasBlockingErrors) return;
    const key = submitKey ?? newIdempotencyKey();
    setSubmitKey(key);
    submit.mutate({
      expectedVersion: record.data.version,
      idempotencyKey: key,
    });
  };

  if (record.isPending || diff.isPending) {
    return (
      <p role="status" className="text-xs text-muted-foreground dark:text-zinc-400">
        Đang tải bản so sánh…
      </p>
    );
  }

  if (record.isError || diff.isError) {
    return (
      <p role="alert" className="text-xs text-red-600 dark:text-red-400">
        Không tải được bản so sánh của {recordId}.
      </p>
    );
  }

  const current = record.data;
  const draftInfo = diff.data;

  return (
    <div className="max-w-3xl space-y-5">
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="text-sm font-semibold text-foreground dark:text-zinc-100">
          Xem trước trước khi gửi duyệt —{" "}
          <span className="font-mono">{recordId}</span>
        </h2>
        <Link
          to="/admin/records"
          className="text-xs text-muted-foreground dark:text-zinc-400 underline-offset-2 hover:underline"
        >
          Về danh sách
        </Link>
      </div>

      <p className="text-xs text-muted-foreground dark:text-zinc-400">
        Trạng thái hiện tại:{" "}
        <span data-status={current.status} className="text-foreground dark:text-zinc-200">
          {STATUS_LABELS[current.status] ?? current.status}
        </span>{" "}
        · bản đang soạn v{draftInfo?.draft_version}
        {draftInfo?.diff.baseline_version != null ? (
          <>
            {" "}
            · so với bản đã công bố v{draftInfo.diff.baseline_version}
          </>
        ) : (
          " · chưa có bản đã công bố để so sánh"
        )}
      </p>

      {/* ---- 1. Diff vs the published version (server-computed) ---- */}
      <section aria-labelledby="diff-heading" className="space-y-2">
        <h3 id="diff-heading" className="text-xs font-semibold text-foreground dark:text-zinc-200">
          Thay đổi so với bản đang công bố
        </h3>
        {changes.length === 0 ? (
          <p role="status" className="text-xs text-muted-foreground dark:text-zinc-400">
            Bản đang soạn không khác bản đã công bố.
          </p>
        ) : (
          <ul className="space-y-1">
            {changes.map((change) => (
              <li
                key={change.field}
                className="rounded-sm border border-border dark:border-zinc-800 px-3 py-2 text-xs"
              >
                <span className="font-mono text-foreground dark:text-zinc-300">{change.field}</span>
                {change.high_impact ? (
                  <span
                    data-high-impact="true"
                    className="ml-2 rounded-sm bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-700 dark:text-amber-300"
                  >
                    ảnh hưởng trực tiếp câu trả lời
                  </span>
                ) : null}
                <div className="mt-1 grid gap-1 text-muted-foreground dark:text-zinc-400">
                  <p>
                    <span className="text-muted-foreground dark:text-zinc-400">cũ: </span>
                    {JSON.stringify(change.old)}
                  </p>
                  <p className="text-foreground dark:text-zinc-200">
                    <span className="text-muted-foreground dark:text-zinc-400">mới: </span>
                    {JSON.stringify(change.new)}
                  </p>
                </div>
              </li>
            ))}
          </ul>
        )}
        {highImpact.length > 0 ? (
          <p className="text-xs text-amber-700 dark:text-amber-300">
            {highImpact.length} field ảnh hưởng trực tiếp đến câu trả lời học
            sinh sẽ nhận — hãy xem trước bên dưới trước khi gửi duyệt.
          </p>
        ) : null}
      </section>

      {/* ---- 2. Validation: errors at the field, blocking submit ---- */}
      {hasBlockingErrors ? (
        <section
          aria-labelledby="validation-heading"
          className="space-y-1 rounded-sm border border-red-900/60 bg-red-950/20 px-3 py-2"
        >
          <h3 id="validation-heading" className="text-xs font-semibold text-red-700 dark:text-red-300">
            Lỗi chặn công bố — sửa trước khi gửi duyệt
          </h3>
          <ul>
            {validationErrors.map((issue) => (
              <li
                key={`${issue.field}:${issue.message}`}
                data-testid={`validation-${issue.field || "record"}`}
                className="text-xs text-red-700 dark:text-red-300"
              >
                <span className="font-mono">{fieldLabel(issue.field)}</span>:{" "}
                {issue.message}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {/* ---- 3. Preview on the real answer path ---- */}
      <section aria-labelledby="preview-heading" className="space-y-2">
        <h3 id="preview-heading" className="text-xs font-semibold text-foreground dark:text-zinc-200">
          Học sinh sẽ nhận câu trả lời nào với dữ liệu này
        </h3>
        <form
          className="flex items-end gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            if (query.trim()) previewMutation.mutate(query.trim());
          }}
        >
          <div className="flex-1 space-y-1">
            <Label htmlFor="preview-query" className="text-xs text-foreground dark:text-zinc-300">
              Câu hỏi của học sinh
            </Label>
            <Input
              id="preview-query"
              className="h-8 !border-zinc-500 text-xs"
              placeholder="Điểm chuẩn ngành Công nghệ thông tin 2026 là bao nhiêu?"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>
          <Button
            type="submit"
            size="sm"
            variant="outline"
            className="!border-zinc-500"
            disabled={!query.trim() || previewMutation.isPending}
          >
            {previewMutation.isPending ? "Đang chạy…" : "Xem trước"}
          </Button>
        </form>
        {previewMutation.isError ? (
          <p role="alert" className="text-xs text-red-600 dark:text-red-400">
            Không chạy được bản xem trước:{" "}
            {previewMutation.error instanceof Error
              ? previewMutation.error.message
              : "lỗi không rõ."}
          </p>
        ) : null}
        {preview ? (
          <div className="space-y-2 rounded-sm border border-border dark:border-zinc-800 px-3 py-3">
            <p className="text-xs text-muted-foreground dark:text-zinc-400">
              <span data-preview-status={preview.preview_status}>
                {preview.preview_status === "ok"
                  ? "Câu trả lời thật từ đúng đường trả lời của học sinh."
                  : preview.preview_status === "auto_answer"
                    ? "Câu trả lời mẫu của bộ định tuyến — KHÔNG phải do mô hình kiểm chứng."
                    : preview.preview_status === "blocked_validation"
                      ? "Bản xem trước bị chặn: dữ liệu chưa hợp lệ."
                      : preview.preview_status === "blocked_pii"
                        ? "Bản xem trước bị chặn: bản nháp chứa thông tin cá nhân (CCCD/email/sĐT) nên không chạy thử được."
                        : "Không sinh được câu trả lời (thiếu dữ liệu hoặc lỗi nhà cung cấp) — phần căn cứ bên dưới vẫn hiển thị để soi."}
              </span>
              {preview.answer_status ? (
                <span className="ml-2 font-mono text-[10px] text-muted-foreground dark:text-zinc-400">
                  {preview.answer_status}
                </span>
              ) : null}
            </p>
            {preview.answer ? (
              <p className="rounded-sm bg-muted dark:bg-zinc-900/60 px-3 py-2 text-sm text-foreground dark:text-zinc-100">
                {preview.answer}
              </p>
            ) : null}
            {preview.preview_status !== "blocked_pii" ? (
              <>
                {preview.chunks.length > 0 ? (
                <details className="text-xs text-muted-foreground dark:text-zinc-400" open>
                    <summary className="cursor-pointer select-none">
                      Căn cứ ({preview.chunks.length} chunk · {preview.sources.length} nguồn)
                    </summary>
                    <ul className="mt-1 space-y-1">
                      {preview.chunks.map((chunk) => (
                        <li
                          key={chunk.chunk_id}
                          className="border-l border-border dark:border-zinc-700 pl-2"
                        >
                          <span className="text-muted-foreground dark:text-zinc-400">
                            {chunk.source} · {chunk.score.toFixed(3)}
                          </span>
                          <p className="text-foreground dark:text-zinc-300">{chunk.text}</p>
                        </li>
                      ))}
                    </ul>
                  </details>
                ) : null}
                {preview.sources.length > 0 && (
                  <p className="mt-2 text-xs text-muted-foreground dark:text-zinc-400">Nguồn</p>
                )}
                <ul className="space-y-1">
                  {preview.sources.map((source) => (
                    <li key={source.source_document_id ?? source.source}>
                      <SourceLink source={source} />
                    </li>
                  ))}
                </ul>
              </>
            ) : null}
          </div>
        ) : null}
      </section>

      {/* ---- 4. Submit for review, from this screen ---- */}
      <section className="flex items-center gap-3 border-t border-border dark:border-zinc-800 pt-4">
        {can("records.submit") ? (
          <Button
            size="sm"
            className="bg-blue-600 text-white"
            disabled={
              hasBlockingErrors ||
              current.status !== "draft" ||
              submit.isPending ||
              Boolean(submit.isSuccess)
            }
            onClick={handleSubmit}
          >
            {submit.isPending
              ? "Đang gửi…"
              : submit.isSuccess
                ? "Đã gửi duyệt"
                : "Gửi duyệt"}
          </Button>
        ) : null}
        {submit.isSuccess ? (
          <Button size="sm" variant="ghost" onClick={() => navigate("/admin/records")}>
            Về danh sách
          </Button>
        ) : null}
        {submitError ? (
          <p role="alert" className="text-xs text-red-600 dark:text-red-400">
            {submitError}
          </p>
        ) : null}
        {hasBlockingErrors ? (
          <p className="text-xs text-red-700 dark:text-red-300">
            Không thể gửi duyệt khi còn lỗi chặn ở trên.
          </p>
        ) : null}
      </section>
    </div>
  );
};

export default ReviewPreview;
