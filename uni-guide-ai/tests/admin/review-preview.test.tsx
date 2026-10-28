// Ticket 25 acceptance at the UI seam: the review diff + preview screen.
//
// Each test pins one checklist item the ticket lists:
//
// * before submitting, the officer sees the diff against the PUBLISHED
//   version, with the baseline version named (criterion 1);
// * high-impact changes are flagged — from the server's classification, the
//   screen only renders the `high_impact` bit (criterion 2);
// * the preview runs the real answer path by calling the review-preview
//   endpoint and renders the answer, the chunks, and the provenance the
//   reviewer needs even when generation was blocked (criteria 3, 6 — the FE
//   cannot prove "real path" itself, only that it asks the one endpoint the
//   BE proved real; the BE test does the rest);
// * submitting for review happens FROM this screen, through the existing
//   lifecycle transitions endpoint with the reviewer's expected_version and an
//   idempotency key — no bespoke submit route (criterion 4);
// * validation errors render at the field they name and the submit button is
//   disabled while any remain (criterion 5).
//
// The route table mirrors App.tsx's records gate.
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import ReviewPreview from "@/pages/admin/ReviewPreview";
import RecordsGate from "@/pages/admin/RecordsGate";
import { AdminSessionProvider } from "@/hooks/useAdminSession";
import {
  IDEMPOTENCY_KEY_HEADER,
  type AdminIdentity,
  type CanonicalRecord,
} from "@/lib/adminApi";

const EDITOR: AdminIdentity = {
  actor_id: "cb.bien-tap",
  role: "editor",
  actions: ["records.read", "records.create", "records.edit", "records.submit"],
};

const RECORD_ID = "DIEM_CHUAN:nganh:7480201:2026";

function record(overrides: Partial<CanonicalRecord> = {}): CanonicalRecord {
  return {
    schema_version: "1.0",
    record_id: RECORD_ID,
    record_type: "DIEM_CHUAN",
    version: 4,
    admission_year: 2026,
    effective_from: "2026-01-01",
    effective_to: "2026-12-31",
    status: "draft",
    authority_level: "husc_official",
    source_document_id: "de-an-tuyen-sinh-2026",
    source_url: "https://tuyensinh.husc.edu.vn/de-an-2026.pdf",
    source_page: 24,
    published_at: null,
    authored_by: "cb.bien-tap",
    supersedes: null,
    locale: "vi",
    text: "Điểm chuẩn ngành Công nghệ thông tin 2026: 25.",
    values: { ma_nganh: "7480201", diem_chuan: 25 },
    ...overrides,
  };
}

const diffResult = (overrides: Record<string, unknown> = {}) => ({
  record_id: RECORD_ID,
  draft_version: 4,
  high_impact_version: "2026.09",
  diff: {
    record_id: RECORD_ID,
    baseline_version: 3,
    draft_version: 4,
    high_impact_version: "2026.09",
    has_high_impact: true,
    changes: [
      {
        field: "text",
        old: "Điểm chuẩn ngành Công nghệ thông tin 2026: 24.",
        new: "Điểm chuẩn ngành Công nghệ thông tin 2026: 25.",
        high_impact: true,
      },
      { field: "source_page", old: 23, new: 24, high_impact: false },
    ],
  },
  validation_errors: [],
  ...overrides,
});

const previewResult = (overrides: Record<string, unknown> = {}) => ({
  record_id: RECORD_ID,
  draft_version: 4,
  baseline_version: 3,
  high_impact_version: "2026.09",
  diff: diffResult().diff,
  validation_errors: [],
  preview_status: "ok",
  answer: "Điểm chuẩn năm 2026 của ngành là 25.",
  answer_status: "generated",
  model_answer: true,
  chunks: [
    {
      chunk_id: "c-1",
      text: "Điểm chuẩn ngành Công nghệ thông tin 2026: 25.",
      score: 0.87,
      source: "de-an-2026.pdf",
      metadata: { record_id: RECORD_ID },
    },
  ],
  sources: [
    {
      source_document_id: "de-an-tuyen-sinh-2026",
      source: "de-an-2026.pdf",
      page: 24,
      section: null,
      title: null,
      record_id: RECORD_ID,
      record_type: "DIEM_CHUAN",
      effective_from: "2026-01-01",
      effective_to: "2026-12-31",
      authority_level: "husc_official",
    },
  ],
  ...overrides,
});

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

const sourceDoc = (type = "application/pdf") =>
  new Response(new Blob(["source bytes"], { type }), {
    status: 200,
    headers: { "Content-Type": type },
  });


interface Call {
  url: string;
  method: string;
  body: Record<string, unknown> | null;
  headers: Record<string, string>;
  credentials: RequestCredentials | undefined;
}

function mockFetch(
  handler: (
    url: string,
    method: string,
    body: Record<string, unknown> | null,
  ) => Response,
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

function renderReview(
  handler: (
    url: string,
    method: string,
    body: Record<string, unknown> | null,
  ) => Response,
  identity: AdminIdentity = EDITOR,
): Call[] {
  const calls = mockFetch((url, method, body) => {
    if (url.includes("/admin/identity")) return json(identity);
    if (url.includes("/source-documents/content")) return sourceDoc();
    return handler(url, method, body);
  });
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <AdminSessionProvider>
        <MemoryRouter initialEntries={[`/admin/records/${encodeURIComponent(RECORD_ID)}/review`]}>
          <Routes>
            <Route path="/admin/records" element={<RecordsGate />}>
              <Route path=":recordId/review" element={<ReviewPreview />} />
            </Route>
          </Routes>
        </MemoryRouter>
      </AdminSessionProvider>
    </QueryClientProvider>,
  );
  return calls;
}

/** The default handler: a clean draft with one high-impact change. */
const clean = () => json(diffResult());

beforeEach(() => {
  window.sessionStorage.clear();
  vi.unstubAllGlobals();
});

describe("Review diff before submitting", () => {
  it("shows what changed against the published version, naming the baseline", async () => {
    renderReview((url) => {
      if (url.includes("/review-diff")) return clean();
      if (url.includes("/admin/records/")) return json(record());
      throw new Error(`unmocked ${url}`);
    });

    expect(await screen.findByText(/so với bản đã công bố v3/i)).toBeInTheDocument();
    // The old/new values render through JSON.stringify, so match the value,
    // not the quotes around it.
    expect(screen.getByText(/2026: 24\./)).toBeInTheDocument();
    expect(screen.getByText(/2026: 25\./)).toBeInTheDocument();
  });

  it("flags exactly the fields the server marked high-impact", async () => {
    renderReview((url) => {
      if (url.includes("/review-diff")) return clean();
      if (url.includes("/admin/records/")) return json(record());
      throw new Error(`unmocked ${url}`);
    });
    await screen.findByText(/so với bản đã công bố v3/i);

    // Count via the data attribute so the assertion is about the server's bit.
    const badges = document.querySelectorAll('[data-high-impact="true"]');
    expect(badges).toHaveLength(1);
    const row = badges[0].closest("li") as HTMLElement;
    expect(within(row).getByText("text")).toBeInTheDocument();
    // The non-high-impact change is listed but not badged.
    const sourcePageRow = screen
      .getByText("source_page")
      .closest("li") as HTMLElement;
    expect(
      sourcePageRow.querySelector('[data-high-impact="true"]'),
    ).toBeNull();
  });
});

describe("Preview on the real answer path", () => {
  it("asks the review-preview endpoint and renders the answer with its chunks", async () => {
    const calls = renderReview((url, method) => {
      if (url.includes("/review-diff")) return clean();
      if (url.includes("/review-preview")) return json(previewResult());
      if (url.includes("/admin/records/")) return json(record());
      throw new Error(`unmocked ${url}`);
    });
    await screen.findByText(/so với bản đã công bố v3/i);

    await userEvent.type(
      screen.getByLabelText("Câu hỏi của học sinh"),
      "Điểm chuẩn ngành Công nghệ thông tin 2026 là bao nhiêu?",
    );
    await userEvent.click(screen.getByRole("button", { name: "Xem trước" }));

    await waitFor(() => {
      const post = calls.find(
        (call) => call.method === "POST" && call.url.includes("/review-preview"),
      );
      expect(post).toBeDefined();
      expect(post?.body).toMatchObject({
        query: "Điểm chuẩn ngành Công nghệ thông tin 2026 là bao nhiêu?",
      });
    });
    expect(
      await screen.findByText("Điểm chuẩn năm 2026 của ngành là 25."),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Câu trả lời thật từ đúng đường trả lời/),
    ).toBeInTheDocument();
    // Grounding is visible: the chunk behind the answer.
    expect(
      screen.getByText("Điểm chuẩn ngành Công nghệ thông tin 2026: 25.", {
        selector: "li p",
      }),
    ).toBeInTheDocument();
  });

  it("keeps showing the grounding when the provider blocked generation", async () => {
    renderReview((url, method) => {
      if (url.includes("/review-diff")) return clean();
      if (url.includes("/review-preview"))
        return json(
          previewResult({
            preview_status: "insufficient_evidence_provider",
            answer: null,
            answer_status: "generation_blocked",
            model_answer: false,
          }),
        );
      if (url.includes("/admin/records/")) return json(record());
      throw new Error(`unmocked ${url}`);
    });
    await screen.findByText(/so với bản đã công bố v3/i);

    await userEvent.type(screen.getByLabelText("Câu hỏi của học sinh"), "Điểm chuẩn?");
    await userEvent.click(screen.getByRole("button", { name: "Xem trước" }));

    const status = await screen.findByText(/Không sinh được câu trả lời/);
    expect(status).toHaveAttribute("data-preview-status", "insufficient_evidence_provider");
    // The chunks are still on screen for the reviewer to inspect grounding.
    expect(screen.getByText(/Căn cứ \(1 chunk · 1 nguồn\)/)).toBeInTheDocument();
  });
});

describe("Submitting for review from the preview screen", () => {
  it("sends the lifecycle submit with expected_version and an idempotency key", async () => {
    const calls = renderReview((url, method) => {
      if (url.includes("/review-diff")) return clean();
      if (url.includes("/transitions"))
        return json({
          record: record({ status: "review", version: 4 }),
          event: { action: "submit", actor_id: "cb.bien-tap" },
        });
      if (url.includes("/admin/records/")) return json(record());
      throw new Error(`unmocked ${url}`);
    });
    await screen.findByText(/so với bản đã công bố v3/i);

    await userEvent.click(screen.getByRole("button", { name: "Gửi duyệt" }));

    await waitFor(() => {
      const submit = calls.find(
        (call) => call.method === "POST" && call.url.includes("/transitions"),
      );
      expect(submit).toBeDefined();
      expect(submit?.body).toMatchObject({ action: "submit", expected_version: 4 });
      expect(submit?.headers[IDEMPOTENCY_KEY_HEADER]).toMatch(/[0-9a-f-]{36}/i);
    });
    // No bespoke submit endpoint was invented: the only writes are transitions.
    expect(
      calls.some(
        (call) => call.method === "POST" && call.url.includes("/review-submit"),
      ),
    ).toBe(false);
    expect(await screen.findByText("Đã gửi duyệt")).toBeInTheDocument();
  });

  it("disables submit while the server reports blocking validation errors", async () => {
    const calls = renderReview((url) => {
      if (url.includes("/review-diff"))
        return json(
          diffResult({
            validation_errors: [
              {
                field: "source_document_id",
                message: "Bản công bố bắt buộc phải có source_document_id.",
              },
            ],
          }),
        );
      if (url.includes("/admin/records/"))
        return json(record({ source_document_id: null }));
      throw new Error(`unmocked ${url}`);
    });

    // The error is rendered at the field it names.
    const atField = await screen.findByTestId("validation-source_document_id");
    expect(atField).toHaveTextContent(/source_document_id/i);
    expect(
      screen.getByText(/Bản công bố bắt buộc phải có source_document_id/),
    ).toBeInTheDocument();

    const submit = screen.getByRole("button", { name: "Gửi duyệt" });
    expect(submit).toBeDisabled();
    await userEvent.click(submit).catch(() => undefined);
    expect(
      calls.some((call) => call.url.includes("/transitions")),
    ).toBe(false);
  });
});
/**
 * Ticket T05 — frontend acceptance at the preview seam: clickable source
 * citations with token-backed fetch and page/section labels.
 */
describe("Source document citations — ticket T05 (preview provenance)", () => {
  beforeEach(() => {
    Object.defineProperty(URL, "createObjectURL", {
      value: vi.fn(() => "blob:mock-url"),
      writable: true,
      configurable: true,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      value: vi.fn(() => undefined),
      writable: true,
      configurable: true,
    });
  });

  /** One PDF citation carrying page 12. */
  const pdfPage12 = (overrides: Record<string, unknown> = {}) =>
    previewResult({
      preview_status: "ok",
      answer: null,
      answer_status: null,
      model_answer: null,
      chunks: [],
      sources: [
        {
          source_document_id: "de-an-tuyen-sinh-2026",
          source: "de-an-2026.pdf",
          page: 12,
          record_id: RECORD_ID,
          record_type: "DIEM_CHUAN",
          effective_from: "2026-01-01",
          effective_to: "2026-12-31",
          authority_level: "husc_official",
          ...overrides,
        },
      ],
      ...overrides,
    });

  /** One DOCX citation carrying a section, no page. */
  const docxSection = (overrides: Record<string, unknown> = {}) =>
    previewResult({
      preview_status: "ok",
      answer: null,
      answer_status: null,
      model_answer: null,
      chunks: [],
      sources: [
        {
          source_document_id: "don-phu-diem-chuan",
          source: "don-phu-2026.docx",
          section: "Điều 4. Học phí",
          page: null,
          title: null,
          record_id: RECORD_ID,
          record_type: "DIEM_CHUAN",
          effective_from: "2026-01-01",
          effective_to: "2026-12-31",
          authority_level: "husc_official",
          ...overrides,
        },
      ],
      ...overrides,
    });

  /** A citation missing source_document_id entirely. */
  const orphan = (overrides: Record<string, unknown> = {}) =>
    previewResult({
      preview_status: "ok",
      answer: null,
      answer_status: null,
      model_answer: null,
      chunks: [],
      sources: [
        {
          source_document_id: null,
          source: "tai-lieu-tham-khao.pdf",
          page: null,
          section: null,
          title: null,
          record_id: RECORD_ID,
          record_type: "DIEM_CHUAN",
          effective_from: "2026-01-01",
          effective_to: "2026-12-31",
          authority_level: "husc_official",
          ...overrides,
        },
      ],
      ...overrides,
    });

  it("PDF with page: 12 opens a citation that carries #page=12 and the right document_id", async () => {
    const open = vi.fn(() => ({ close: vi.fn(), addEventListener: vi.fn() }));
    vi.spyOn(window, "open").mockImplementation(open);
    const calls = renderReview((url, method) => {
      if (url.includes("/review-diff")) return clean();
      if (url.includes("/review-preview")) return json(pdfPage12());
      if (url.includes("/admin/records/")) return json(record());
      throw new Error(`unmocked ${url}`);
    });
    await screen.findByText(/so với bản đã công bố v3/i);
    await userEvent.type(
      screen.getByLabelText("Câu hỏi của học sinh"),
      "Điểm chuẩn?",
    );
    await userEvent.click(screen.getByRole("button", { name: "Xem trước" }));
    await waitFor(() => {
      const post = calls.find(
        (call) => call.method === "POST" && call.url.includes("/review-preview"),
      );
      expect(post).toBeDefined();
    });
    await screen.findByText("de-an-2026.pdf");
    const link = screen.getByRole("button", { name: /de-an-2026\.pdf/i });
    await userEvent.click(link);
    expect(link).toBeInTheDocument();
    await waitFor(() => {
      expect(open).toHaveBeenCalledOnce();
    });
    const opened = open.mock.calls[0][0] as string;
    expect(opened).toContain("#page=12");
    const contentCall = calls.find(
      (call) => call.url.includes("/source-documents/content"),
    );
    expect(contentCall).toBeDefined();
    expect(contentCall?.url).toContain("document_id=de-an-tuyen-sinh-2026");
    expect(contentCall?.headers).not.toHaveProperty("X-Admin-Token");
    expect(contentCall?.credentials).toBe("same-origin");
    expect(contentCall?.method).toBe("GET");
  });

  it("DOCX with a section and page: null renders the section name without #page", async () => {
    const open = vi.fn(() => ({ close: vi.fn(), addEventListener: vi.fn() }));
    vi.spyOn(window, "open").mockImplementation(open);
    const calls = renderReview((url, method) => {
      if (url.includes("/review-diff")) return clean();
      if (url.includes("/review-preview")) return json(docxSection());
      if (url.includes("/admin/records/")) return json(record());
      throw new Error(`unmocked ${url}`);
    });
    await screen.findByText(/so với bản đã công bố v3/i);
    await userEvent.type(
      screen.getByLabelText("Câu hỏi của học sinh"),
      "Điểm chuẩn?",
    );
    await userEvent.click(screen.getByRole("button", { name: "Xem trước" }));
    await waitFor(() => {
      const post = calls.find(
        (call) => call.method === "POST" && call.url.includes("/review-preview"),
      );
      expect(post).toBeDefined();
    });
    await screen.findByText("Điều 4. Học phí");
    const link = screen.getByRole("button", { name: /Điều 4\. Học phí/i });
    await userEvent.click(link);
    expect(link).toBeInTheDocument();
    await waitFor(() => {
      expect(open).toHaveBeenCalledOnce();
    });
    const opened = open.mock.calls[0][0] as string;
    expect(opened).not.toContain("#page=");
    const contentCall = calls.find(
      (call) => call.url.includes("/source-documents/content"),
    );
    expect(contentCall?.url).toContain("document_id=don-phu-diem-chuan");
  });

  it("citation without source_document_id renders as plain text, never a dead link", async () => {
    const calls = renderReview((url, method) => {
      if (url.includes("/review-diff")) return clean();
      if (url.includes("/review-preview")) return json(orphan());
      if (url.includes("/admin/records/")) return json(record());
      throw new Error(`unmocked ${url}`);
    });
    await screen.findByText(/so với bản đã công bố v3/i);
    await userEvent.type(
      screen.getByLabelText("Câu hỏi của học sinh"),
      "Điểm chuẩn?",
    );
    await userEvent.click(screen.getByRole("button", { name: "Xem trước" }));
    await waitFor(() => {
      const post = calls.find(
        (call) => call.method === "POST" && call.url.includes("/review-preview"),
      );
      expect(post).toBeDefined();
    });
    await screen.findByText("tai-lieu-tham-khao.pdf");
    const link = screen.queryByRole("button", { name: /tai-lieu-tham-khao\.pdf/i });
    expect(link).toBeNull();
    const contentCall = calls.find(
      (call) => call.url.includes("/source-documents/content"),
    );
    expect(contentCall).toBeUndefined();
  });

  it("blocked_pii shows the PII notice and hides the grounding block", async () => {
    renderReview((url, method) => {
      if (url.includes("/review-diff")) return clean();
      if (url.includes("/review-preview"))
        return json(
          previewResult({
            preview_status: "blocked_pii",
            answer: null,
            answer_status: "blocked",
            chunks: [
              {
                chunk_id: "c-1",
                text: "nội dung bị chặn",
                score: 0.1,
                source: "x.pdf",
                metadata: {},
              },
            ],
            sources: [
              {
                source_document_id: "de-an-tuyen-sinh-2026",
                source: "de-an-2026.pdf",
                page: 24,
                record_id: RECORD_ID,
                record_type: "DIEM_CHUAN",
                effective_from: "2026-01-01",
                effective_to: "2026-12-31",
                authority_level: "husc_official",
              },
            ],
          }),
        );
      if (url.includes("/admin/records/")) return json(record());
      throw new Error(`unmocked ${url}`);
    });
    await screen.findByText(/so với bản đã công bố v3/i);
    await userEvent.type(
      screen.getByLabelText("Câu hỏi của học sinh"),
      "Điểm chuẩn?",
    );
    await userEvent.click(screen.getByRole("button", { name: "Xem trước" }));
    await screen.findByText(/thông tin cá nhân/i);
    expect(screen.queryByText(/Căn cứ/i)).toBeNull();
  });

});
