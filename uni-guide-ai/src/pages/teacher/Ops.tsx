// Vận hành — API and worker health, parse jobs, warmup. Admin and ops only.
// There is deliberately no rollback control here (rollback lives in Lịch sử).
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useAdminSession } from "@/hooks/useAdminSession";
import { AdminApiError } from "@/lib/adminApi";
import type { ApiFailure } from "@/lib/apiError";
import {
  cancelJob,
  fetchApiHealth,
  fetchJobs,
  fetchWorkers,
  retryJob,
  warmup,
  type WorkerStatus,
} from "@/lib/adminOpsApi";
import { ExceptionBar } from "@/components/ui-kit/ExceptionBar";
import { DangerButton } from "@/components/ui-kit/DangerButton";
import { PrimaryButton } from "@/components/ui-kit/PrimaryButton";
import { SecondaryButton } from "@/components/ui-kit/SecondaryButton";
import { StatusPill } from "@/components/ui-kit/StatusPill";

/** A worker whose heartbeat is older than this is treated as disconnected. */
const WORKER_STALE_MS = 60_000;

function isWorkerDisconnected(worker: WorkerStatus, now: number): boolean {
  if (!worker.last_heartbeat_at) return true;
  return now - Date.parse(worker.last_heartbeat_at) > WORKER_STALE_MS;
}

const OPS_ROLES: Record<string, true> = { admin: true, ops: true };
const REFRESH_MS = 15_000;

function failureOf(error: unknown): ApiFailure | null {
  return error instanceof AdminApiError ? error.failure : null;
}

const Ops = () => {
  const { status, identity, can } = useAdminSession();
  const queryClient = useQueryClient();
  const allowed = status === "ready" && identity !== null && OPS_ROLES[identity.role] === true;
  const enabled = allowed && can("jobs.read");

  const health = useQuery({
    queryKey: ["health"],
    queryFn: fetchApiHealth,
    refetchInterval: REFRESH_MS,
    enabled: allowed,
  });
  const workers = useQuery({
    queryKey: ["admin", "workers"],
    queryFn: () => fetchWorkers(null),
    refetchInterval: REFRESH_MS,
    retry: false,
    enabled,
  });
  const jobs = useQuery({
    queryKey: ["admin", "jobs"],
    queryFn: () => fetchJobs(null),
    refetchInterval: REFRESH_MS,
    retry: false,
    enabled,
  });

  const refresh = () => queryClient.invalidateQueries({ queryKey: ["admin"] });
  const retry = useMutation({ mutationFn: (jobId: string) => retryJob(null, jobId), onSuccess: refresh });
  const cancel = useMutation({ mutationFn: (jobId: string) => cancelJob(null, jobId), onSuccess: refresh });
  const warm = useMutation({ mutationFn: () => warmup(null) });

  if (status !== "ready" || identity === null) {
    return <p className="text-base text-zinc-700">Đang tải...</p>;
  }
  if (OPS_ROLES[identity.role] !== true) {
    return <ExceptionBar title="Chỉ tài khoản kỹ thuật hoặc quản trị được xem trang này." />;
  }

  const apiUp = health.data === true;
  const now = Date.now();
  const actionFailure = failureOf(retry.error) ?? failureOf(cancel.error) ?? failureOf(warm.error);

  return (
    <section data-workarea="" aria-labelledby="ops-title" className="space-y-6">
      <h1 id="ops-title" className="text-2xl font-bold text-[#143980]">
        Vận hành
      </h1>

      {!apiUp && health.isFetched ? (
        <ExceptionBar title="Máy chủ API không phản hồi" detail="Kiểm tra lại sau ít phút." />
      ) : null}

      <div className="space-y-2">
        <h2 className="text-xl font-semibold">Trạng thái hệ thống</h2>
        <ul className="space-y-2">
          <li>
            <StatusPill
              tone={apiUp ? "ok" : "blocked"}
              label={apiUp ? "API hoạt động" : "API không phản hồi"}
            />
          </li>
          {(workers.data ?? []).map((worker) => {
            const down = isWorkerDisconnected(worker, now);
            return (
              <li key={worker.worker_id} className="flex flex-wrap items-center gap-2">
                <StatusPill
                  tone={down ? "blocked" : "ok"}
                  label={down ? "Mất kết nối worker" : "Worker hoạt động"}
                />
                <span className="text-base">{worker.worker_id}</span>
              </li>
            );
          })}
        </ul>
        {workers.isError ? (
          <ExceptionBar title="Không đọc được trạng thái worker" detail={failureOf(workers.error)?.message} />
        ) : null}
      </div>

      {can("ops.warmup") ? (
        <div className="space-y-2">
          <PrimaryButton onClick={() => warm.mutate()} disabled={warm.isPending}>
            Làm nóng bộ nhớ
          </PrimaryButton>
          {warm.isSuccess ? <p role="status" className="text-base">Đã gửi yêu cầu làm nóng.</p> : null}
        </div>
      ) : null}

      <div className="space-y-2">
        <h2 className="text-xl font-semibold">Công việc xử lý tài liệu</h2>
        {jobs.isError ? (
          <ExceptionBar title="Không đọc được danh sách công việc" detail={failureOf(jobs.error)?.message} />
        ) : null}
        {actionFailure ? <ExceptionBar title={actionFailure.message} /> : null}
        {jobs.data && jobs.data.length === 0 ? <p className="text-base">Chưa có công việc nào.</p> : null}
        {jobs.data && jobs.data.length > 0 ? (
          <table className="w-full text-left text-base">
            <thead>
              <tr>
                <th scope="col" className="py-2">Công việc</th>
                <th scope="col" className="py-2">Trạng thái</th>
                <th scope="col" className="py-2">Thao tác</th>
              </tr>
            </thead>
            <tbody>
              {jobs.data.map((job) => (
                <tr key={job.job_id} className="border-t border-zinc-300">
                  <td className="py-2">
                    {job.kind} <span className="text-sm text-zinc-700">{job.job_id.slice(0, 8)}</span>
                  </td>
                  <td className="py-2">{job.state}</td>
                  <td className="flex flex-wrap gap-2 py-2">
                    {can("jobs.manage") && job.state.startsWith("failed") ? (
                      <SecondaryButton onClick={() => retry.mutate(job.job_id)} disabled={retry.isPending}>
                        Thử lại
                      </SecondaryButton>
                    ) : null}
                    {can("jobs.manage") && (job.state === "queued" || job.state === "running") ? (
                      <DangerButton onClick={() => cancel.mutate(job.job_id)} disabled={cancel.isPending}>
                        Huỷ
                      </DangerButton>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </div>
    </section>
  );
};

export default Ops;
