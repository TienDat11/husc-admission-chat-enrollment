import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Feedback from "@/pages/teacher/Feedback";

type Routes = Record<string, { status?: number; body: unknown }>;

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });

const stubApi = (routes: Routes) => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string | URL | Request) => {
      const raw = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      const { pathname } = new URL(raw);
      const hit = routes[pathname];
      if (!hit) return json({ detail: "not mocked" }, 404);
      return json(hit.body, hit.status ?? 200);
    }),
  );
};

const renderFeedback = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <Feedback />
    </QueryClientProvider>,
  );
};

const FEEDBACK_ROW = {
  feedback_id: "f1",
  trace_id: "t1",
  helpful: true,
  reason: null,
  answered_at: "2026-04-01T08:00:00Z",
  linked_item: true,
  created_at: "2026-04-01T08:00:00Z",
};

const RELEASES = {
  releases: [
    {
      release_id: "rel-2026-04",
      status: "active",
      created_at: "2026-04-02T00:00:00Z",
      golden_report: { verdict: "PASS" },
    },
  ],
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Phản hồi & Golden (Feedback)", () => {
  it("shows the real feedback total, helpful share, repeated questions and latest verdict", async () => {
    stubApi({
      "/admin/unanswered/feedback": {
        body: { count: 137, items: [FEEDBACK_ROW, { ...FEEDBACK_ROW, feedback_id: "f2", helpful: false, reason: "Sai học phí" }] },
      },
      "/admin/unanswered/": {
        body: {
          count: 3,
          items: [
            { trace_id: "a", question: "Q1", resolved: false, occurrence_count: 4 },
            { trace_id: "b", question: "Q2", resolved: false, occurrence_count: 1 },
            { trace_id: "c", question: "Q3", resolved: true, occurrence_count: 5 },
          ],
        },
      },
      "/admin/releases": { body: RELEASES },
    });

    renderFeedback();

    expect(await screen.findByText("137")).toBeInTheDocument();
    expect(await screen.findByText("1/2")).toBeInTheDocument();
    expect(await screen.findByText("1")).toBeInTheDocument();
    expect(await screen.findByText(/Sai học phí/)).toBeInTheDocument();
    expect(await screen.findByText("PASS", { exact: false })).toBeInTheDocument();
    expect(screen.getByText(/Hệ thống chưa cung cấp danh sách bộ câu hỏi kiểm tra/)).toBeInTheDocument();
  });

  it("reports an empty feedback table honestly instead of inventing rows", async () => {
    stubApi({
      "/admin/unanswered/feedback": { body: { count: 0, items: [] } },
      "/admin/unanswered/": { body: { count: 0, items: [] } },
      "/admin/releases": { body: { releases: [] } },
    });

    renderFeedback();

    expect(await screen.findByText("Chưa có phản hồi nào được ghi nhận.")).toBeInTheDocument();
    expect(await screen.findByText("Chưa có báo cáo kiểm tra.")).toBeInTheDocument();
  });

  it("shows an error state when the feedback endpoint fails", async () => {
    stubApi({
      "/admin/unanswered/feedback": { status: 500, body: { detail: "boom" } },
      "/admin/unanswered/": { body: { count: 0, items: [] } },
      "/admin/releases": { body: { releases: [] } },
    });

    renderFeedback();

    expect((await screen.findAllByText("Không tải được phản hồi.")).length).toBeGreaterThan(0);
  });
});
