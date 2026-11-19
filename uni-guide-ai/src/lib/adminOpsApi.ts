// Admin calls for the Tài liệu/Vận hành screens: overview counts, upload, parse
// jobs, workers and warmup. Each goes through adminRequest, so every failure
// reaches the UI as an ApiFailure (apiError.ts).
import { adminRequest } from "./adminApi";
import { getApiBaseUrl } from "./api";
import type { components } from "./api-types";

export type OverviewView = components["schemas"]["OverviewResponse"];
export type UploadReport = components["schemas"]["UploadReport"];
export type JobSummary = components["schemas"]["JobSummary"];
export type JobTransition = components["schemas"]["JobTransition"];
export type WorkerStatus = components["schemas"]["WorkerStatus"];

/** Counts behind the sidebar badges and the Frame 5 cards. */
export function fetchAdminOverview(token: string | null): Promise<OverviewView> {
  return adminRequest<OverviewView>(token, "/admin/overview");
}

/** Upload one PDF/DOCX/XLSX. Answers 202 with the parse job to follow. */
export function uploadSourceDocument(token: string | null, file: File): Promise<UploadReport> {
  const body = new FormData();
  body.append("file", file);
  return adminRequest<UploadReport>(token, "/admin/source-documents/upload", {
    method: "POST",
    body,
  });
}

export function fetchJob(token: string | null, jobId: string): Promise<JobSummary> {
  return adminRequest<JobSummary>(token, `/admin/jobs/${encodeURIComponent(jobId)}`);
}

export function fetchJobs(token: string | null): Promise<JobSummary[]> {
  return adminRequest<JobSummary[]>(token, "/admin/jobs");
}

export function fetchWorkers(token: string | null): Promise<WorkerStatus[]> {
  return adminRequest<WorkerStatus[]>(token, "/admin/ops/workers");
}

export function retryJob(token: string | null, jobId: string): Promise<JobTransition> {
  return adminRequest<JobTransition>(token, `/admin/jobs/${encodeURIComponent(jobId)}/retry`, {
    method: "POST",
  });
}

export function cancelJob(token: string | null, jobId: string): Promise<JobTransition> {
  return adminRequest<JobTransition>(token, `/admin/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
  });
}

export function warmup(token: string | null): Promise<unknown> {
  return adminRequest<unknown>(token, "/admin/warmup", { method: "POST" });
}

/** API liveness (`/health`). Public, so it is not an admin call. */
export async function fetchApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${getApiBaseUrl()}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
