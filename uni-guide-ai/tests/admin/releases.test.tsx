// Ticket 26 acceptance at the UI seam: the release + index-job screen.
//
// Each test covers a checklist item that would otherwise be a claim about a
// screenshot:
//
//   * the active release and the release behind it are on screen;
//   * a job's pending/running/failed/active state renders, and a failed job
//     shows its reason;
//   * **đã lưu không phải đã publish** — a build the server marks
//     `saved: true, serving: false` is NOT rendered with the serving/active
//     wording, and the payload's own label survives to the screen;
//   * rollback refuses to submit without a reason (nothing is sent), and sends
//     the reason plus an `Idempotency-Key` when one is typed;
//   * the job state updates without a page reload — the query is configured to
//     poll (`refetchInterval`) and a refetch re-renders the new state.
//
// The route table mirrors App.tsx's releases route; App.tsx's own wiring is
// asserted separately by rendering the real app (admin.test.tsx).
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import RecordsGate from "@/pages/admin/RecordsGate";
import ReleaseStatus from "@/pages/admin/ReleaseStatus";
import { AdminSessionProvider } from "@/hooks/useAdminSession";
import {
  INDEX_JOBS_REFETCH_MS,
  type AdminIdentity,
  type IndexJob,
  type IndexJobsStatus,
  type ReleaseStatus as ReleaseStatusPayload,
} from "@/lib/adminApi";

const ADMIN: AdminIdentity = {
  actor_id: "cb.quan-tri",
  role: "admin",
  actions: [
    "records.read",
    "records.create",
    "records.edit",
    "records.submit",
    "records.approve",
    "records.reject",
    "records.publish",
    "records.supersede",
    "records.rollback",
    "records.override",
  ],
};

const VIEWER: AdminIdentity = {
  actor_id: "cb.chi-xem",
  role: "viewer",
  actions: ["records.read"],
};

const ACTIVE_ID = "R-2026-09-01";
const PREVIOUS_ID = "R-2026-08-01";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

interface Call {
  url: string;
  method: string;
  body: Record<string, unknown> | null;
  headers: Record<string, string>;
  credentials: RequestCredentials | undefined;
}

function mockFetch(
  handler: (url: string, method: string, body: Record<string, unknown> | null) => Response,
): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = (init?.method ?? "GET").toUpperCase();
      const body = typeof init?.body === "string" ? JSON.parse(init.body) : null;
      const headers = (init?.headers as Record<string, string> | undefined) ?? {};
      calls.push({ url, method, body, headers, credentials: init?.credentials });
      return handler(url, method, body);
    }),
  );
  return calls;
}

function releasesPayload(overrides: Partial<ReleaseStatusPayload> = {}): ReleaseStatusPayload {
  return {
    active_release_id: ACTIVE_ID,
    previous_release_id: PREVIOUS_ID,
    active: {
      release_id: ACTIVE_ID,
      created_at: "2026-09-01T02:00:00Z",
      parent_release_id: PREVIOUS_ID,
      lancedb_tag: `release-${ACTIVE_ID}`,
      lancedb_version: 12,
      document_count: 7,
      chunk_count: 1679,
    },
    previous: {
      release_id: PREVIOUS_ID,
      created_at: "2026-08-01T02:00:00Z",
      parent_release_id: null,
      lancedb_tag: `release-${PREVIOUS_ID}`,
      lancedb_version: 11,
      document_count: 6,
      chunk_count: 1500,
    },
    problems: [],
    ...overrides,
  };
}

function job(overrides: Partial<IndexJob> = {}): IndexJob {
  return {
    job_id: "live:running",
    state: "running",
    saved: false,
    serving: false,
    label: "đang chạy (running)",
    release_id: null,
    staged_dir: null,
    reason: null,
    retry: [],
    ...overrides,
  };
}

function jobsPayload(overrides: Partial<IndexJobsStatus> = {}): IndexJobsStatus {
  return {
    jobs: [job()],
    reindex_needed: false,
    reindex_reason: "corpus đang phục vụ khớp stamp của code đang chạy",
    pending_outbox_entries: 0,
    pending_affected_record_ids: [],
    problems: [],
    ...overrides,
  };
}

function renderReleases(
  identity: AdminIdentity,
  handler: (url: string, method: string, body: Record<string, unknown> | null) => Response,
) {
  const calls = mockFetch((url, method, body) => {
    if (url.includes("/admin/identity")) return json(identity);
    return handler(url, method, body);
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <AdminSessionProvider>
        <MemoryRouter initialEntries={["/admin/releases"]}>
          <Routes>
            <Route path="/admin/releases" element={<RecordsGate />}>
              <Route index element={<ReleaseStatus />} />
            </Route>
          </Routes>
        </MemoryRouter>
      </AdminSessionProvider>
    </QueryClientProvider>,
  );
  return { calls, client };
}

beforeEach(() => {
  window.sessionStorage.clear();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("Release screen", () => {
  it("shows the active release and the release behind it", async () => {
    renderReleases(ADMIN, (url) =>
      url.includes("/admin/index-jobs") ? json(jobsPayload()) : json(releasesPayload()),
    );

    // R-2026-08-01 appears twice on purpose: as the previous release's own id
    // and as the active release's parent. Assert the pair rather than one node.
    expect(await screen.findByText(ACTIVE_ID)).toBeInTheDocument();
    expect(screen.getAllByText(PREVIOUS_ID).length).toBeGreaterThan(0);
    expect(screen.getByText("Đang active")).toBeInTheDocument();
    expect(screen.getByText("Release trước (rollback về đây)")).toBeInTheDocument();
    // Document/chunk counts of the active release, rendered as one pair.
    expect(screen.getByText("7 / 1679")).toBeInTheDocument();
    expect(screen.getByText(`release-${ACTIVE_ID}`)).toBeInTheDocument();
  });

  it("surfaces an unreadable pointer as a problem, not as 'chưa publish'", async () => {
    renderReleases(ADMIN, (url) =>
      url.includes("/admin/index-jobs")
        ? json(jobsPayload())
        : json(
            releasesPayload({
              active_release_id: null,
              previous_release_id: null,
              active: null,
              previous: null,
              problems: ["active-release pointer ... is not valid JSON"],
            }),
          ),
    );

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Không đọc được hết trạng thái");
    expect(alert).toHaveTextContent("is not valid JSON");
    // Both cards are empty — two cards, two "no release" lines, and crucially
    // the screen says the pointer is unreadable rather than that nothing was
    // ever published.
    expect(screen.getAllByText("Không có release nào.")).toHaveLength(2);
  });
});

describe("Index job screen", () => {
  it("renders each of the four job states", async () => {
    const states = ["pending", "running", "failed", "active"] as const;
    renderReleases(ADMIN, (url) =>
      url.includes("/admin/index-jobs")
        ? json(
            jobsPayload({
              jobs: [
                job({ job_id: "j-pending", state: "pending", label: "chờ chạy (pending)" }),
                job({ job_id: "j-running", state: "running", label: "đang chạy (running)" }),
                job({
                  job_id: "j-failed",
                  state: "failed",
                  label: "thất bại (failed)",
                  reason: "provider hết quota",
                }),
                job({
                  job_id: "j-active",
                  state: "active",
                  serving: true,
                  saved: true,
                  label: "đang phục vụ (active)",
                }),
              ],
            }),
          )
        : json(releasesPayload()),
    );

    await screen.findByText("j-pending");
    for (const state of states) {
      expect(screen.getByText(new RegExp(`\\(${state}\\)`))).toBeInTheDocument();
    }
  });

  it("shows a failed job's reason", async () => {
    renderReleases(ADMIN, (url) =>
      url.includes("/admin/index-jobs")
        ? json(
            jobsPayload({
              jobs: [
                job({
                  job_id: "j-failed",
                  state: "failed",
                  label: "thất bại (failed)",
                  reason: "embedder trả 403 insufficient quota",
                  retry: [
                    { step: "rebuild", runnable: false, command: null, reason: "chưa có runner" },
                  ],
                }),
              ],
            }),
          )
        : json(releasesPayload()),
    );

    expect(await screen.findByText(/403 insufficient quota/)).toBeInTheDocument();
    expect(screen.getByText(/chưa có runner/)).toBeInTheDocument();
  });

  /**
   * Ticket 26 criterion 3, at the render seam.
   *
   * The payload is the one the real route produces for a build in staging:
   * `saved: true`, `serving: false`, `state: null`, label "đã build, chưa phục
   * vụ". The screen must show it as saved-and-not-serving — no "Đang phục vụ",
   * no "active", no "đã publish" anywhere for that row.
   */
  it("never renders a saved-but-not-serving build as published or active", async () => {
    renderReleases(ADMIN, (url) =>
      url.includes("/admin/index-jobs")
        ? json(
            jobsPayload({
              jobs: [
                job({
                  job_id: "job-20260913T000000000000",
                  state: null,
                  saved: true,
                  serving: false,
                  label: "đã build, chưa phục vụ",
                  staged_dir: "D:/.../canonical_corpus.staging/job-20260913T000000000000",
                }),
              ],
            }),
          )
        : json(releasesPayload()),
    );

    await screen.findByText("job-20260913T000000000000");
    // The saved marker is there...
    expect(screen.getByText("Đã build")).toBeInTheDocument();
    // ...the server's own label is rendered verbatim, and there is exactly one
    // of it (no badge repeating it as a state)...
    expect(screen.getAllByText("đã build, chưa phục vụ")).toHaveLength(1);
    // ...and the serving/published wording is nowhere on the screen.
    expect(screen.queryByText("Đang phục vụ")).not.toBeInTheDocument();
    expect(screen.queryByText("(active)")).not.toBeInTheDocument();
    expect(screen.queryByText(/đã publish/i)).not.toBeInTheDocument();
  });

  it("does not claim a job serves when the payload says saved-but-not-serving", async () => {
    renderReleases(ADMIN, (url) =>
      url.includes("/admin/index-jobs")
        ? json(
            jobsPayload({
              jobs: [
                job({ job_id: "j1", state: null, saved: true, serving: false, label: "đã build, chưa phục vụ" }),
                job({ job_id: "j2", state: "failed", saved: true, serving: false, label: "thất bại (failed)" }),
              ],
            }),
          )
        : json(releasesPayload()),
    );

    await screen.findByText("j1");
    // Two saved rows, zero serving rows: the badge count has to match.
    expect(screen.getAllByText("Đã build")).toHaveLength(2);
    expect(screen.queryAllByText("Đang phục vụ")).toHaveLength(0);
  });

  /**
   * Ticket 26 criterion 6: the state updates without a reload.
   *
   * Asserted two ways, because either alone is weak. First the query's
   * configuration is the poll interval (a refactor that drops `refetchInterval`
   * fails here); then a real refetch through the query client re-renders the new
   * state, which is what polling buys — the DOM moves from "đang chạy" to
   * "đang phục vụ" with no navigation and no reload.
   */
  it("polls the jobs query and re-renders on refetch without a reload", async () => {
    let state: IndexJob = job({ job_id: "j1", state: "running", label: "đang chạy (running)" });
    const { calls, client } = renderReleases(ADMIN, (url) =>
      url.includes("/admin/index-jobs") ? json(jobsPayload({ jobs: [state] })) : json(releasesPayload()),
    );

    await screen.findByText("đang chạy (running)");

    // 1. The query is configured to poll — that is what makes the state move on
    //    its own in the browser.
    const jobsQuery = client
      .getQueryCache()
      .findAll({ queryKey: ["admin", "index-jobs"] })
      .at(0);
    expect(jobsQuery?.options.refetchInterval).toBe(INDEX_JOBS_REFETCH_MS);

    // 2. The next poll returns a new state, and the screen renders it.
    state = job({
      job_id: "j1",
      state: "active",
      serving: true,
      saved: true,
      label: "đang phục vụ (active)",
    });
    const before = calls.filter((c) => c.url.includes("/admin/index-jobs")).length;
    await client.refetchQueries({ queryKey: ["admin", "index-jobs"] });

    expect(await screen.findByText("đang phục vụ (active)")).toBeInTheDocument();
    expect(screen.queryByText("đang chạy (running)")).not.toBeInTheDocument();
    expect(calls.filter((c) => c.url.includes("/admin/index-jobs")).length).toBeGreaterThan(before);
  });
});

describe("Rollback from the release screen", () => {
  it("refuses to submit without a reason, and sends nothing", async () => {
    const { calls } = renderReleases(ADMIN, (url) =>
      url.includes("/admin/index-jobs") ? json(jobsPayload()) : json(releasesPayload()),
    );

    const button = await screen.findByRole("button", { name: "Rollback" });
    expect(button).toBeDisabled();

    // Clicking a disabled button is not a submit; assert the request count too,
    // so a button that merely *looks* disabled still fails this test.
    await userEvent.click(button);
    await waitFor(() => expect(button).toBeDisabled());
    expect(calls.filter((c) => c.url.includes("/admin/rollback"))).toHaveLength(0);

    // A blank-but-typed reason is still no reason.
    await userEvent.type(screen.getByLabelText(/Lý do rollback/), "   ");
    expect(button).toBeDisabled();
    await userEvent.click(button);
    expect(calls.filter((c) => c.url.includes("/admin/rollback"))).toHaveLength(0);
  });

  it("sends the reason and an Idempotency-Key once a reason is typed", async () => {
    const { calls } = renderReleases(ADMIN, (url, method) => {
      if (url.includes("/admin/index-jobs")) return json(jobsPayload());
      if (url.includes("/admin/rollback")) {
        return json({ release_id: PREVIOUS_ID, previous_release_id: ACTIVE_ID });
      }
      return json(releasesPayload());
    });

    const reason = await screen.findByLabelText(/Lý do rollback/);
    await userEvent.type(reason, "khôi phục sau sự cố dữ liệu");
    await userEvent.click(screen.getByRole("button", { name: "Rollback" }));

    await waitFor(() =>
      expect(calls.some((c) => c.url.includes("/admin/rollback"))).toBe(true),
    );
    const call = calls.find((c) => c.url.includes("/admin/rollback"));
    expect(call?.method).toBe("POST");
    expect(call?.body).toEqual({ reason: "khôi phục sau sự cố dữ liệu" });
    expect(call?.headers["Idempotency-Key"]).toBeTruthy();
    expect(call?.headers["X-Admin-Token"]).toBeUndefined();
    expect(call?.credentials).toBe("same-origin");

    expect(await screen.findByText(/Đã rollback về R-2026-08-01/)).toBeInTheDocument();
  });

  it("reuses the same key when the rollback is retried after a failure", async () => {
    let attempt = 0;
    const { calls } = renderReleases(ADMIN, (url) => {
      if (url.includes("/admin/index-jobs")) return json(jobsPayload());
      if (url.includes("/admin/rollback")) {
        attempt += 1;
        return attempt === 1
          ? json({ detail: "retrieval chưa sẵn sàng" }, 503)
          : json({ release_id: PREVIOUS_ID, previous_release_id: ACTIVE_ID });
      }
      return json(releasesPayload());
    });

    await userEvent.type(await screen.findByLabelText(/Lý do rollback/), "khôi phục");
    await userEvent.click(screen.getByRole("button", { name: "Rollback" }));
    expect(await screen.findByText("retrieval chưa sẵn sàng")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Rollback" }));
    await screen.findByText(/Đã rollback về/);

    const keys = calls
      .filter((c) => c.url.includes("/admin/rollback"))
      .map((c) => c.headers["Idempotency-Key"]);
    expect(keys).toHaveLength(2);
    expect(keys[0]).toBe(keys[1]);
  });

  it("hides the rollback form from a role without the grant", async () => {
    renderReleases(VIEWER, (url) =>
      url.includes("/admin/index-jobs") ? json(jobsPayload()) : json(releasesPayload()),
    );

    expect(await screen.findByText(ACTIVE_ID)).toBeInTheDocument();
    expect(screen.queryByLabelText(/Lý do rollback/)).not.toBeInTheDocument();
    expect(screen.getByText(/records\.rollback/)).toBeInTheDocument();
  });
});
