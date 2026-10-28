// Release + index-job screen — ticket 26.
//
// The question this screen answers, in the words the ticket uses: "cán bộ
// tuyển sinh biết release nào đang chạy, release trước là gì, và publish đã
// thực sự lên chưa". Three rules it holds to, because each of them is a way the
// screen could lie:
//
// 1. **Đã lưu không phải đã publish.** The backend sends `saved` and `serving`
//    as separate fields; this screen renders both, and the "đang phục vụ" badge
//    appears only where `serving` is true — which the backend sets for `active`
//    and nothing else. A build in the staging area is labelled "đã build, chưa
//    phục vụ" from the server, not re-worded here.
// 2. **Rollback phải có lý do.** The route requires a non-blank reason, so the
//    button is disabled until one is typed — the refusal happens before the
//    request, and the test asserts nothing is sent. The key is minted per
//    intention and reused on retry, the way RecordList does it.
// 3. **Trạng thái job tự cập nhật.** The jobs query polls (`refetchInterval`),
//    so a job that moves from running to active shows up without a reload.
//
// The credential comes from the same session the records screen uses; the
// release screen is mounted inside RecordsGate, so it never renders without one.
import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAdminSession } from "@/hooks/useAdminSession";
import {
  AdminApiError,
  INDEX_JOBS_REFETCH_MS,
  fetchIndexJobs,
  fetchReleaseStatus,
  newIdempotencyKey,
  rollbackRelease,
  type IndexJob,
  type JobStateName,
  type ReleaseView,
} from "@/lib/adminApi";

/**
 * Badge colour per state. The *text* is always the server's own `label`: the
 * backend already names each state in Vietnamese, and re-wording it here would
 * be a second source for the one string this screen must not get wrong.
 */
const STATE_STYLES: Record<JobStateName, string> = {
  pending: "border-border dark:border-zinc-600 text-muted-foreground dark:text-zinc-300",
  running: "border-sky-600/60 text-sky-700 dark:border-sky-500/60 dark:text-sky-400",
  failed: "border-red-600/60 text-red-700 dark:border-red-500/60 dark:text-red-400",
  active: "border-emerald-600/60 text-emerald-700 dark:border-emerald-500/60 dark:text-emerald-400",
};

function formatWhen(value: string | null): string {
  if (!value) return "không ghi nhận";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString("vi-VN");
}

function ReleaseCard({ title, release }: { title: string; release: ReleaseView | null }) {
  if (release === null) {
    return (
      <div className="rounded-sm border border-border dark:border-zinc-700 p-3">
        <h3 className="text-xs font-medium text-foreground dark:text-zinc-300">{title}</h3>
        <p className="mt-1 text-xs text-muted-foreground dark:text-zinc-400">Không có release nào.</p>
      </div>
    );
  }
  return (
    <div className="rounded-sm border border-border dark:border-zinc-700 p-3">
      <h3 className="text-xs font-medium text-foreground dark:text-zinc-300">{title}</h3>
      <p className="mt-1 font-mono text-sm text-foreground dark:text-zinc-100">{release.release_id}</p>
      <dl className="mt-2 space-y-0.5 text-xs text-muted-foreground dark:text-zinc-400">
        <div className="flex gap-2">
          <dt className="w-28 shrink-0">Tạo lúc</dt>
          <dd>{formatWhen(release.created_at)}</dd>
        </div>
        <div className="flex gap-2">
          <dt className="w-28 shrink-0">Release trước</dt>
          <dd className="font-mono">{release.parent_release_id ?? "không có"}</dd>
        </div>
        <div className="flex gap-2">
          <dt className="w-28 shrink-0">Tag LanceDB</dt>
          <dd className="font-mono">{release.lancedb_tag ?? "không ghi nhận"}</dd>
        </div>
        <div className="flex gap-2">
          <dt className="w-28 shrink-0">Tài liệu / chunk</dt>
          <dd>
            {release.document_count} / {release.chunk_count}
          </dd>
        </div>
      </dl>
    </div>
  );
}

function JobRow({ job }: { job: IndexJob }) {
  // `serving` alone earns the serving wording, and the wording itself is the
  // server's `label` — a build the server marks `saved: true, serving: false`
  // therefore cannot reach the serving wording, because the server never puts
  // it in that label.
  const style = job.serving
    ? STATE_STYLES.active
    : job.state
      ? STATE_STYLES[job.state]
      : "border-border dark:border-zinc-600 text-muted-foreground dark:text-zinc-400";

  return (
    <div className="rounded-sm border border-border dark:border-zinc-700 p-3">
      <div className="flex flex-wrap items-center gap-2">
        {job.state !== null ? (
          <span className={`rounded-sm border px-2 py-0.5 text-xs ${style}`}>{job.label}</span>
        ) : null}
        <span className="font-mono text-xs text-foreground dark:text-zinc-300">{job.job_id}</span>
        {job.saved ? (
          <span className="rounded-sm border border-amber-600/60 px-2 py-0.5 text-xs text-amber-700 dark:text-amber-400">
            Đã build
          </span>
        ) : null}
      </div>
      {/* The server's label again, where the badge cannot carry it: a row with
          no state (a build on disk) has no badge, so its label is the only
          wording on the row — and it is the server's, not ours. */}
      {job.state === null ? <p className="mt-1 text-xs text-muted-foreground dark:text-zinc-400">{job.label}</p> : null}
      {job.staged_dir ? (
        <p className="mt-1 break-all font-mono text-xs text-muted-foreground dark:text-zinc-400">{job.staged_dir}</p>
      ) : null}
      {job.reason ? (
        <p role="alert" className="mt-2 text-xs text-red-600 dark:text-red-400">
          Lý do: {job.reason}
        </p>
      ) : null}
      {job.retry.length > 0 ? (
        <div className="mt-2 space-y-1">
          {job.retry.map((step) => (
            <div key={step.step} className="text-xs text-muted-foreground dark:text-zinc-400">
              <span className="font-medium">Thử lại — {step.step}: </span>
              {step.runnable && step.command ? (
                <span className="font-mono text-emerald-700 dark:text-emerald-400">{step.command}</span>
              ) : (
                <span className="text-muted-foreground dark:text-zinc-400">{step.reason}</span>
              )}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

const ReleaseStatusPage = () => {
  const { can } = useAdminSession();
  const queryClient = useQueryClient();
  const [reason, setReason] = useState("");
  const [target, setTarget] = useState("");
  const [intent, setIntent] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  const releases = useQuery({
    queryKey: ["admin", "releases"],
    queryFn: () => fetchReleaseStatus(null),
  });

  // Polling, not a reload: a job that finishes while the operator watches moves
  // from running to active on its own.
  const jobs = useQuery({
    queryKey: ["admin", "index-jobs"],
    queryFn: () => fetchIndexJobs(null),
    refetchInterval: INDEX_JOBS_REFETCH_MS,
  });

  const canRollback = can("records.rollback");
  const rollback = useMutation({
    mutationFn: (input: { reason: string; target?: string; idempotencyKey: string }) =>
      rollbackRelease(
        null,
        {
          reason: input.reason,
          ...(input.target ? { target_release_id: input.target } : {}),
        },
        { idempotencyKey: input.idempotencyKey },
      ),
    onSuccess: (result) => {
      setIntent(null);
      setReason("");
      setOutcome(
        `Đã rollback về ${result.release_id} (trước đó: ${result.previous_release_id}).`,
      );
      setFailure(null);
      void queryClient.invalidateQueries({ queryKey: ["admin", "releases"] });
    },
    onError: (error: unknown) => {
      // The key survives a failure on purpose: the retry must reuse it.
      setFailure(
        error instanceof AdminApiError ? error.message : "Rollback thất bại.",
      );
    },
  });

  const problems = useMemo(
    () => [...(releases.data?.problems ?? []), ...(jobs.data?.problems ?? [])],
    [releases.data, jobs.data],
  );

  const submitRollback = () => {
    if (!reason.trim()) return;
    const key = intent ?? newIdempotencyKey();
    setIntent(key);
    setOutcome(null);
    rollback.mutate({ reason: reason.trim(), target: target.trim(), idempotencyKey: key });
  };

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-sm font-medium text-foreground dark:text-zinc-100">Release &amp; index job</h2>
        <p className="text-xs text-muted-foreground dark:text-zinc-400">
          Release nào đang chạy, release trước là gì, và job re-index đang ở trạng thái nào.
          “Đã build” không có nghĩa là đã phục vụ.
        </p>
      </div>

      {problems.length > 0 ? (
        <div role="alert" className="rounded-sm border border-amber-600/60 p-3 text-xs text-amber-700 dark:text-amber-400">
          <p className="font-medium">Không đọc được hết trạng thái:</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4">
            {problems.map((problem) => (
              <li key={problem}>{problem}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <section className="space-y-2">
        <h3 className="text-xs font-medium uppercase tracking-wide text-muted-foreground dark:text-zinc-400">
          Release đang chạy
        </h3>
        {releases.isLoading ? (
          <p role="status" className="text-xs text-muted-foreground dark:text-zinc-400">
            Đang tải release…
          </p>
        ) : releases.isError ? (
          <p role="alert" className="text-xs text-red-600 dark:text-red-400">
            {releases.error instanceof AdminApiError
              ? releases.error.message
              : "Không đọc được trạng thái release."}
          </p>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            <ReleaseCard title="Đang active" release={releases.data?.active ?? null} />
            <ReleaseCard title="Release trước (rollback về đây)" release={releases.data?.previous ?? null} />
          </div>
        )}
      </section>

      <section className="space-y-2">
        <h3 className="text-xs font-medium uppercase tracking-wide text-muted-foreground dark:text-zinc-400">
          Rollback
        </h3>
        {canRollback ? (
          <form
            className="max-w-xl space-y-2 rounded-sm border border-border dark:border-zinc-700 p-3"
            onSubmit={(event) => {
              event.preventDefault();
              submitRollback();
            }}
          >
            <div className="space-y-1">
              <Label htmlFor="rollback-reason" className="text-xs text-foreground dark:text-zinc-300">
                Lý do rollback (bắt buộc)
              </Label>
              <Input
                id="rollback-reason"
                className="!border-zinc-500"
                value={reason}
                onChange={(event) => setReason(event.target.value)}
                placeholder="Ví dụ: khôi phục sau sự cố dữ liệu 2026"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="rollback-target" className="text-xs text-foreground dark:text-zinc-300">
                Release đích (bỏ trống để lùi một bước)
              </Label>
              <Input
                id="rollback-target"
                className="!border-zinc-500"
                value={target}
                onChange={(event) => setTarget(event.target.value)}
                placeholder={releases.data?.previous_release_id ?? "release trước"}
              />
            </div>
            {failure ? (
              <p role="alert" className="text-xs text-red-600 dark:text-red-400">
                {failure}
              </p>
            ) : null}
            {outcome ? (
              <p role="status" className="text-xs text-emerald-700 dark:text-emerald-400">
                {outcome}
              </p>
            ) : null}
            <Button type="submit" size="sm" className="bg-blue-600 text-white" disabled={!reason.trim() || rollback.isPending}>
              {rollback.isPending ? "Đang rollback…" : "Rollback"}
            </Button>
          </form>
        ) : (
          <p className="text-xs text-muted-foreground dark:text-zinc-400">
            Vai trò của bạn không có quyền <span className="font-mono">records.rollback</span>.
          </p>
        )}
      </section>

      <section className="space-y-2">
        <h3 className="text-xs font-medium uppercase tracking-wide text-muted-foreground dark:text-zinc-400">
          Index job
        </h3>
        {jobs.data ? (
          <p className="text-xs text-muted-foreground dark:text-zinc-400">
            {jobs.data.reindex_needed ? "Cần re-index" : "Không cần re-index"}:{" "}
            {jobs.data.reindex_reason}
            {jobs.data.pending_outbox_entries > 0 ? (
              <>
                {" "}
                — {jobs.data.pending_outbox_entries} thay đổi đang chờ job kế tiếp
                {jobs.data.pending_affected_record_ids.length > 0
                  ? ` (${jobs.data.pending_affected_record_ids.length} record)`
                  : ""}
                .
              </>
            ) : null}
          </p>
        ) : null}
        {jobs.isLoading ? (
          <p role="status" className="text-xs text-muted-foreground dark:text-zinc-400">
            Đang tải trạng thái job…
          </p>
        ) : jobs.isError ? (
          <p role="alert" className="text-xs text-red-600 dark:text-red-400">
            {jobs.error instanceof AdminApiError
              ? jobs.error.message
              : "Không đọc được trạng thái job."}
          </p>
        ) : (jobs.data?.jobs.length ?? 0) === 0 ? (
          <p className="text-xs text-muted-foreground dark:text-zinc-400">
            Không có job re-index nào. Corpus đang phục vụ không do job nào trong phiên này
            build.
          </p>
        ) : (
          <div className="space-y-2">
            {jobs.data?.jobs.map((job) => <JobRow key={job.job_id} job={job} />)}
          </div>
        )}
      </section>
    </div>
  );
};

export default ReleaseStatusPage;
