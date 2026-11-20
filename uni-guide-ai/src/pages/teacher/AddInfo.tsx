// Thêm thông tin — step 1 (Frame 1) and the mobile layout (Frame 10).
// Upload is real: POST /admin/source-documents/upload answers 202 with the
// parse job; GET /admin/jobs/{id} is polled until it is terminal. The job id is
// kept in sessionStorage, so a reload keeps following the same job.
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { z } from "zod";
import { useAdminSession } from "@/hooks/useAdminSession";
import { AdminApiError } from "@/lib/adminApi";
import type { ApiFailure } from "@/lib/apiError";
import { fetchJob, uploadSourceDocument, type UploadReport } from "@/lib/adminOpsApi";
import { forEachLimited, sendDraftForReview } from "@/lib/draftSubmit";
import { ExceptionBar } from "@/components/ui-kit/ExceptionBar";
import { PrimaryButton } from "@/components/ui-kit/PrimaryButton";
import { SecondaryButton } from "@/components/ui-kit/SecondaryButton";
import { StatusPill, type PillTone } from "@/components/ui-kit/StatusPill";

const STEPS = ["1 Chọn tệp", "2 Kiểm tra", "3 Hỏi thử", "4 Gửi duyệt"];
const STORAGE_KEY = "admin.addinfo.upload";
const TERMINAL_STATES: Record<string, true> = { completed: true, ready_for_review: true, failed_terminal: true, cancelled: true };
const POLL_MS = 2000;
const NETWORK_MESSAGE = "Không kết nối được máy chủ. Kiểm tra mạng rồi thử lại.";

const StoredUploadSchema = z.object({
  parseJobId: z.string(),
  fileName: z.string(),
  pageCount: z.number(),
  draftRecordIds: z.array(z.string()).default([]),
  warnings: z.array(z.string()),
});
type StoredUpload = z.infer<typeof StoredUploadSchema>;

function readStored(): StoredUpload | null {
  const raw = sessionStorage.getItem(STORAGE_KEY);
  if (raw === null) return null;
  const parsed = StoredUploadSchema.safeParse(JSON.parse(raw));
  return parsed.success ? parsed.data : null;
}

function toStored(report: UploadReport, fileName: string): StoredUpload {
  return {
    parseJobId: report.parse_job_id,
    fileName,
    pageCount: report.report.page_count,
    draftRecordIds: report.draft_record_ids ?? [],
    warnings: report.warnings ?? [],
  };
}

const JOB_LABEL: Record<string, { label: string; tone: PillTone }> = {
  completed: { label: "Hoàn tất", tone: "ok" },
  ready_for_review: { label: "Sẵn sàng duyệt", tone: "ok" },
  failed_terminal: { label: "Thất bại", tone: "blocked" },
  cancelled: { label: "Đã huỷ", tone: "muted" },
};

function failureOf(error: unknown): ApiFailure {
  if (error instanceof AdminApiError) return error.failure;
  return { kind: "error", status: 0, code: null, message: NETWORK_MESSAGE };
}

type DraftSend = { status: "idle" } | { status: "sending" } | { status: "sent" } | { status: "error"; message: string };

const SEND_CONCURRENCY = 3;

function DraftReview({ ids }: { ids: string[] }) {
  const [sends, setSends] = useState<Record<string, DraftSend>>({});
  const busy = Object.values(sends).some((send) => send.status === "sending");
  useEffect(() => {
    if (!busy) return;
    const warnBeforeLeaving = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warnBeforeLeaving);
    return () => window.removeEventListener("beforeunload", warnBeforeLeaving);
  }, [busy]);
  const sentCount = ids.filter((id) => sends[id]?.status === "sent").length;
  const pending = ids.filter((id) => sends[id]?.status !== "sent");

  const sendOne = async (id: string) => {
    setSends((prev) => ({ ...prev, [id]: { status: "sending" } }));
    try {
      await sendDraftForReview(id);
      setSends((prev) => ({ ...prev, [id]: { status: "sent" } }));
    } catch (error: unknown) {
      setSends((prev) => ({ ...prev, [id]: { status: "error", message: failureOf(error).message } }));
    }
  };

  return (
    <div className="space-y-3">
      <p className="font-semibold">Bản nháp đã tạo: {ids.length}</p>
      <div role="status" aria-live="polite" className="text-base">
        {busy ? "Đang gửi duyệt…" : sentCount > 0 ? `Đã gửi duyệt ${sentCount}/${ids.length} bản nháp.` : null}
      </div>
      <ul className="space-y-3">
        {ids.map((id) => {
          const send: DraftSend = sends[id] ?? { status: "idle" };
          return (
            <li key={id} className="space-y-2 rounded-md border border-zinc-400 bg-white p-3">
              <p className="break-all text-sm">{id}</p>
              {send.status === "sent" ? (
                <p className="font-semibold text-emerald-800">Đã gửi duyệt</p>
              ) : (
                <PrimaryButton
                  onClick={() => void sendOne(id)}
                  disabled={busy}
                  aria-busy={send.status === "sending"}
                  aria-label={`Gửi duyệt bản nháp ${id}`}
                >
                  {send.status === "sending" ? "Đang gửi…" : "Gửi duyệt"}
                </PrimaryButton>
              )}
              {send.status === "error" ? (
                <p role="alert" className="text-base text-red-800">
                  {send.message}
                </p>
              ) : null}
              <Link to={`/quan-ly/them/ngoai-le/${encodeURIComponent(id)}`} className="block underline">
                Xử lý ngoại lệ bảng
              </Link>
            </li>
          );
        })}
      </ul>
      {ids.length > 1 && pending.length > 0 ? (
        <PrimaryButton onClick={() => void forEachLimited(pending, SEND_CONCURRENCY, sendOne)} disabled={busy}>
          {busy ? "Đang gửi…" : `Gửi duyệt tất cả (${pending.length})`}
        </PrimaryButton>
      ) : null}
      {!busy && sentCount > 0 ? (
        <Link to="/quan-ly/cho-duyet" className="block underline">
          Mở danh sách chờ duyệt
        </Link>
      ) : null}
    </div>
  );
}

const AddInfo = () => {
  const { can } = useAdminSession();
  const canUpload = can("records.create");
  const [stored, setStored] = useState<StoredUpload | null>(readStored);
  const [chosen, setChosen] = useState<File | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const [lastFile, setLastFile] = useState<File | null>(null);

  const upload = useMutation({
    mutationFn: (file: File) => uploadSourceDocument(null, file),
    onMutate: (file) => {
      setLastFile(file);
      setFailure(null);
    },
    onSuccess: (report, file) => {
      const next = toStored(report, file.name);
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      setStored(next);
      setChosen(null);
    },
    onError: (error) => setFailure(failureOf(error)),
  });

  const job = useQuery({
    queryKey: ["admin", "job", stored?.parseJobId ?? ""],
    queryFn: () => fetchJob(null, stored?.parseJobId ?? ""),
    enabled: stored !== null,
    retry: false,
    refetchInterval: (query) => {
      const state = query.state.data?.state;
      return state !== undefined && TERMINAL_STATES[state] === true ? false : POLL_MS;
    },
  });

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (chosen !== null) upload.mutate(chosen);
  };

  const jobState = job.data?.state;
  const terminal = jobState !== undefined && TERMINAL_STATES[jobState] === true;
  const jobView = jobState !== undefined && JOB_LABEL[jobState] !== undefined ? JOB_LABEL[jobState] : null;
  const duplicate = failure?.kind === "duplicate-document" ? failure : null;

  return (
    <section data-workarea="" aria-labelledby="addinfo-title" className="space-y-6">
      <h1 id="addinfo-title" className="text-2xl font-bold text-[#143980]">
        Thêm thông tin
      </h1>

      <ol aria-label="Các bước" className="flex flex-wrap gap-2">
        {STEPS.map((label, index) => (
          <li key={label}>
            <StatusPill tone={index === 0 ? "pending" : "muted"} label={label} />
          </li>
        ))}
      </ol>

      {!canUpload ? (
        <ExceptionBar severity="info" title="Bạn chỉ có quyền xem" detail="Liên hệ người soạn để tải tài liệu lên." />
      ) : null}

      {canUpload && stored === null ? (
        <form onSubmit={onSubmit} className="space-y-4" noValidate>
          <div className="space-y-2">
            <label htmlFor="upload-file" className="block text-base font-semibold">
              Chọn tệp PDF, DOCX hoặc XLSX (tối đa 20 MB)
            </label>
            <input
              id="upload-file"
              type="file"
              accept=".pdf,.docx,.xlsx"
              disabled={upload.isPending}
              onChange={(event) => setChosen(event.target.files?.[0] ?? null)}
              className="block min-h-11 w-full rounded-md border border-zinc-500 px-2 py-2 text-base text-zinc-900 file:mr-3 file:min-h-11 file:rounded-md file:border file:border-zinc-500 file:bg-white file:px-3 file:font-semibold file:text-zinc-900"
            />
            {chosen ? <p className="text-base">Đã chọn: {chosen.name}</p> : null}
          </div>

          {failure !== null && failure.kind !== "duplicate-document" ? (
            <ExceptionBar title={failure.message}>
              {lastFile !== null && failure.kind === "error" ? (
                <SecondaryButton onClick={() => upload.mutate(lastFile)} disabled={upload.isPending}>
                  Thử lại
                </SecondaryButton>
              ) : null}
            </ExceptionBar>
          ) : null}

          {duplicate !== null ? (
            <ExceptionBar title={duplicate.message} detail="Bạn có thể mở bản đã đăng ký hoặc chọn tệp khác.">
              <div className="flex flex-wrap gap-2">
                <Link
                  to={`/quan-ly/kho?source_document_id=${encodeURIComponent(duplicate.sourceDocumentId)}`}
                  className="inline-flex min-h-11 items-center rounded-md border border-zinc-400 bg-white px-4 font-semibold text-zinc-900"
                >
                  Mở bản cũ
                </Link>
                <SecondaryButton onClick={() => setFailure(null)}>Chọn tệp khác</SecondaryButton>
              </div>
            </ExceptionBar>
          ) : null}

          <div className="sticky bottom-0 bg-white py-3 md:static md:p-0">
            <PrimaryButton type="submit" disabled={chosen === null || upload.isPending} className="w-full md:w-auto">
              {upload.isPending ? "Đang tải lên..." : "Tải lên"}
            </PrimaryButton>
          </div>
        </form>
      ) : null}

      {stored !== null ? (
        <div className="space-y-4">
          <p className="text-base">
            Tệp: <span className="font-semibold">{stored.fileName}</span>
          </p>

          {job.isError ? (
            <ExceptionBar title={failureOf(job.error).message}>
              <SecondaryButton onClick={() => job.refetch()}>Thử lại</SecondaryButton>
            </ExceptionBar>
          ) : null}

          <div role="status" aria-live="polite" className="space-y-2">
            {jobView !== null ? (
              <StatusPill tone={jobView.tone} label={jobView.label} />
            ) : (
              <StatusPill tone="pending" label="Đang xử lý" />
            )}
            {job.data ? (
              <p className="text-base">
                Giai đoạn: {String(job.data.progress.stage ?? "chưa rõ")}
                {job.data.progress.total !== undefined
                  ? ` — ${String(job.data.progress.done ?? 0)}/${String(job.data.progress.total)}`
                  : ""}
              </p>
            ) : null}
          </div>

          {terminal ? (
            <div className="space-y-2 text-base">
              <p>Số trang đã đọc: {stored.pageCount}</p>
              {stored.draftRecordIds.length === 0 ? <p>Tệp này không tạo bản nháp nào.</p> : null}
              {stored.warnings.map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
              {stored.draftRecordIds.length > 0 ? (
                <DraftReview key={stored.parseJobId} ids={stored.draftRecordIds} />
              ) : null}
            </div>
          ) : null}

          {terminal ? (
            <SecondaryButton
              onClick={() => {
                sessionStorage.removeItem(STORAGE_KEY);
                setStored(null);
              }}
            >
              Tải tệp khác
            </SecondaryButton>
          ) : null}
        </div>
      ) : null}

      <div className="flex flex-wrap gap-2">
        {STEPS.slice(1).map((label) => (
          <button
            key={label}
            type="button"
            disabled
            className="min-h-11 rounded-md border border-zinc-400 px-4 text-base text-zinc-600 disabled:cursor-not-allowed"
          >
            {label}
          </button>
        ))}
      </div>
    </section>
  );
};

export default AddInfo;
