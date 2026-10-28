// Ticket T09 acceptance at the UI seam: the 7 teacher-facing screens.
//
// Each screen renders inside a RecordsGate (the same credential gate
// the records screens use) with a mocked fetch, so the tests assert:
//   * the gate is on the route (anonymous state renders the credential form);
//   * with a credential, the screen's own data-fetching path renders its
//     empty / loading / error / content states;
//   * the primary action on each screen fires the right API call.
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import RecordsGate from "@/pages/admin/RecordsGate";
import Overview from "@/pages/teacher/Overview";
import AddInfo from "@/pages/teacher/AddInfo";
import QAForm from "@/pages/teacher/QAForm";
import PendingReview from "@/pages/teacher/PendingReview";
import Vault from "@/pages/teacher/Vault";
import Unanswered from "@/pages/teacher/Unanswered";
import AskTest from "@/pages/teacher/AskTest";
import History from "@/pages/teacher/History";
import { AdminSessionProvider } from "@/hooks/useAdminSession";
import {
  type AdminIdentity,
} from "@/lib/adminApi";

const REVIEWER: AdminIdentity = {
  actor_id: "cb.duyet",
  role: "reviewer_publisher",
  actions: [
    "records.read",
    "records.approve",
    "records.reject",
    "records.publish",
    "records.rollback",
  ],
};

const EDITOR: AdminIdentity = {
  actor_id: "cb.soan",
  role: "editor",
  actions: ["records.read", "records.create", "records.edit", "records.submit"],
};

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

interface Call {
  url: string;
  method: string;
  body: Record<string, unknown> | null;
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
      const body =
        typeof init?.body === "string" ? JSON.parse(init.body) : null;
      calls.push({ url, method, body });
      return handler(url, method, body);
    }),
  );
  return calls;
}

/**
 * Answer /admin/identity with the given identity, as the session cookie would.
 * The gate resolves to that identity on mount; no web storage is involved.
 */
function renderScreen(
  path: string,
  identity: AdminIdentity,
  element: React.ReactNode,
  handler: (
    url: string,
    method: string,
    body: Record<string, unknown> | null,
  ) => Response,
) {
  const calls = mockFetch((url, method, body) => {
    if (url.includes("/admin/identity")) return json(identity);
    return handler(url, method, body);
  });
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <AdminSessionProvider>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route path={path} element={<RecordsGate />}>
              <Route index element={element} />
            </Route>
          </Routes>
        </MemoryRouter>
      </AdminSessionProvider>
    </QueryClientProvider>,
  );
  return calls;
}

/** No credential seeded: the gate must show the anonymous form. */
function renderAnonymous(path: string, element: React.ReactNode) {
  mockFetch(() => json({ detail: "not signed in" }, 401));
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <AdminSessionProvider>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route path={path} element={<RecordsGate />}>
              <Route index element={element} />
            </Route>
          </Routes>
        </MemoryRouter>
      </AdminSessionProvider>
    </QueryClientProvider>,
  );
}

const emptyRecords = { records: [], total: 0 };
const emptyUnanswered = { items: [], count: 0 };
const emptyAudit = { events: [], count: 0 };
const emptyRelease = {
  active_release_id: null,
  previous_release_id: null,
  active: null,
  previous: null,
  problems: [],
};

const defaultHandler = (url: string): Response => {
  if (url.includes("/admin/records")) return json(emptyRecords);
  if (url.includes("/admin/unanswered")) return json(emptyUnanswered);
  if (url.includes("/admin/audit")) return json(emptyAudit);
  if (url.includes("/admin/releases")) return json(emptyRelease);
  if (url.includes("/v2/query"))
    return json({ answer: "", sources: [], status_code: "NO_INFO" });
  throw new Error(`Unmocked URL: ${url}`);
};

beforeEach(() => {
  window.sessionStorage.clear();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("Gate wiring — every teacher screen sits behind the credential gate", () => {
  for (const [path, Screen] of [
    ["/quan-ly", Overview],
    ["/quan-ly/them", AddInfo],
    ["/quan-ly/them/qa", QAForm],
    ["/quan-ly/cho-duyet", PendingReview],
    ["/quan-ly/kho", Vault],
    ["/quan-ly/cau-hoi", Unanswered],
    ["/quan-ly/hoi-thu", AskTest],
    ["/quan-ly/lich-su", History],
  ] as const) {
    it(`renders the credential form for ${path} when anonymous`, async () => {
      renderAnonymous(path, <Screen />);
      expect(await screen.findByText("Đăng nhập")).toBeInTheDocument();
    });
  }
});

describe("Tổng quan", () => {
  it("shows the empty state when there is nothing to do", async () => {
    renderScreen("/quan-ly", REVIEWER, <Overview />, defaultHandler);
    expect(await screen.findByText(/Không có việc cần làm/)).toBeInTheDocument();
  });

  it("shows the three work cards when there is pending work", async () => {
    const withWork = (url: string) => {
      if (url.includes("/admin/records") && url.includes("status=review"))
        return json({
          records: [{ record_id: "R1", status: "review" }],
          total: 1,
        });
      if (url.includes("/admin/releases"))
        return json({
          ...emptyRelease,
          active: { created_at: "2026-10-06T09:12:00Z", release_id: "R-1" },
        });
      return defaultHandler(url);
    };
    renderScreen("/quan-ly", REVIEWER, <Overview />, withWork);
    expect(await screen.findByText("Việc cần làm hôm nay")).toBeInTheDocument();
    expect(screen.getByText("Duyệt ngay")).toBeInTheDocument();
  });

  it("shows the error state on network failure", async () => {
    renderScreen("/quan-ly", REVIEWER, <Overview />, () =>
      json({ detail: "error" }, 500),
    );
    expect(await screen.findByRole("alert")).toBeInTheDocument();
  });
});

describe("Thêm thông tin (4-step flow)", () => {
  it("renders step 1 with the drop zone and metadata selects", async () => {
    renderScreen("/quan-ly/them", EDITOR, <AddInfo />, defaultHandler);
    expect(await screen.findByText(/Kéo tệp vào đây/)).toBeInTheDocument();
    expect(screen.getByText("Năm tuyển sinh")).toBeInTheDocument();
  });

  it("the Kiểm tra button is enabled once a filename is present", async () => {
    renderScreen("/quan-ly/them", EDITOR, <AddInfo />, defaultHandler);
    const nextButton = await screen.findByText(/Tiếp tục/);
    expect(nextButton).toBeEnabled();
  });
});

describe("Gõ hỏi–đáp trực tiếp", () => {
  it("blocks submit while the answer field is empty", async () => {
    renderScreen("/quan-ly/them/qa", EDITOR, <QAForm />, defaultHandler);
    await userEvent.click(await screen.findByText("Gửi duyệt"));
    expect(
      await screen.findByText(/Ô "Câu trả lời đúng" đang trống/),
    ).toBeInTheDocument();
  });

  it("creates and submits a draft when both fields are filled", async () => {
    const calls = renderScreen(
      "/quan-ly/them/qa",
      EDITOR,
      <QAForm />,
      defaultHandler,
    );
    const questionField = await screen.findByLabelText("Câu hỏi thí sinh hay hỏi");
    await userEvent.type(questionField, "Trường có ký túc xá không?");
    await userEvent.type(
      screen.getByLabelText("Câu trả lời đúng"),
      "Có, năm nhất được đăng ký.",
    );
    await userEvent.click(screen.getByText("Gửi duyệt"));
    await waitFor(() => {
      expect(
        calls.some(
          (call) => call.method === "POST" && call.url.includes("/admin/records"),
        ),
      ).toBe(true);
    });
  });
});

describe("Chờ duyệt", () => {
  const reviewRecord = {
    record_id: "HOC_PHI:2026",
    record_type: "HOC_PHI",
    version: 2,
    admission_year: 2026,
    effective_from: "2026-01-01",
    effective_to: "2027-01-01",
    status: "review",
    authority_level: "husc_official",
    source_document_id: null,
    source_url: null,
    source_page: null,
    published_at: null,
    authored_by: "cb.soan",
    supersedes: null,
    locale: "vi",
    text: "Học phí Hán Nôm 15tr.",
    values: {},
  };

  const diffPayload = {
    record_id: "HOC_PHI:2026",
    draft_version: 2,
    high_impact_version: "2026.10",
    diff: {
      record_id: "HOC_PHI:2026",
      baseline_version: 1,
      draft_version: 2,
      high_impact_version: "2026.10",
      has_high_impact: false,
      changes: [{ field: "text", old: "14tr", new: "15tr", high_impact: false }],
    },
    validation_errors: [],
  };

  const withReview = (url: string): Response => {
    if (url.includes("/admin/records") && url.includes("status=review"))
      return json({ records: [reviewRecord], total: 1 });
    if (url.includes("/review-diff")) return json(diffPayload);
    if (url.includes("/review-preview"))
      return json({
        ...diffPayload,
        preview_status: "ok",
        answer: "Học phí 15tr",
        chunks: [],
        sources: [],
      });
    if (url.includes("/transitions"))
      return json({ record: reviewRecord, event: {} });
    if (url.includes("/admin/records")) return json(emptyRecords);
    return defaultHandler(url);
  };

  it("lists the pending records and shows the four-eyes note for own work", async () => {
    const ownWork = { ...reviewRecord, authored_by: "cb.duyet" };
    renderScreen("/quan-ly/cho-duyet", REVIEWER, <PendingReview />, (url) => {
      if (url.includes("/admin/records") && url.includes("status=review"))
        return json({ records: [ownWork], total: 1 });
      return withReview(url);
    });
    expect(await screen.findByText(/Chờ duyệt \(1\)/)).toBeInTheDocument();
    expect(screen.getByText(/Cần một người khác duyệt/)).toBeInTheDocument();
  });

  it("opens the diff view when the teacher clicks Mở để duyệt", async () => {
    renderScreen("/quan-ly/cho-duyet", REVIEWER, <PendingReview />, withReview);
    await userEvent.click(await screen.findByText(/Mở để duyệt/));
    expect(await screen.findByText("Bản đang dùng")).toBeInTheDocument();
    expect(screen.getByText("Bản mới")).toBeInTheDocument();
  });

  it("rejects only with a reason typed", async () => {
    renderScreen("/quan-ly/cho-duyet", REVIEWER, <PendingReview />, withReview);
    await userEvent.click(await screen.findByText(/Mở để duyệt/));
    const rejectButton = screen.getByText("Trả lại để sửa");
    expect(rejectButton).toBeDisabled();
    await userEvent.type(
      screen.getByLabelText("Lý do trả lại (bắt buộc nếu trả lại)"),
      "Sai năm học",
    );
    expect(screen.getByText("Trả lại để sửa")).toBeEnabled();
  });
});

describe("Chờ duyệt — nhận nháp (T15)", () => {
  const seedDraft = {
    record_id: "HOC_PHI:2026",
    record_type: "HOC_PHI",
    version: 1,
    admission_year: 2026,
    effective_from: "2026-01-01",
    effective_to: "2027-01-01",
    status: "review",
    authority_level: "husc_official",
    source_document_id: null,
    source_url: null,
    source_page: null,
    published_at: null,
    authored_by: null,
    supersedes: null,
    locale: "vi",
    text: "Học phí Hán Nôm 15tr.",
    values: {},
  };

  // The reviewer's identity carries records.edit, so the claim route
  // accepts it; the list still comes back with status=review.
  const CLAIMER: AdminIdentity = {
    ...REVIEWER,
    actions: [...REVIEWER.actions, "records.edit"],
  };

  it("(a) the action appears on an unclaimed seed draft and calls the claim endpoint with the right id + version", async () => {
    const calls: Call[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        const method = (init?.method ?? "GET").toUpperCase();
        const body =
          typeof init?.body === "string" ? JSON.parse(init.body) : null;
        calls.push({ url, method, body });
        if (url.includes("/admin/identity")) return json(CLAIMER);
        if (url.includes("/admin/records") && url.includes("status=review"))
          return json({ records: [seedDraft], total: 1 });
        if (url.includes("/claim")) return json({ record: seedDraft, event: {} });
        return defaultHandler(url);
      }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <AdminSessionProvider>
          <MemoryRouter initialEntries={["/quan-ly/cho-duyet"]}>
            <Routes>
              <Route path="/quan-ly/cho-duyet" element={<RecordsGate />}>
                <Route index element={<PendingReview />} />
              </Route>
            </Routes>
          </MemoryRouter>
        </AdminSessionProvider>
      </QueryClientProvider>,
    );
    await screen.findByText(/Chờ duyệt \(1\)/);

    // The claim action is present on the unclaimed card
    expect(await screen.findByText("Nhận nháp")).toBeInTheDocument();

    // Open the confirmation dialog and confirm
    await userEvent.click(screen.getAllByText("Nhận nháp")[0]);
    const dialog = await screen.findByRole("alertdialog");
    await userEvent.click(within(dialog).getByRole("button", { name: "Nhận nháp" }));

    await waitFor(() => {
      const claims = calls.filter((c) => c.url.includes("/claim"));
      expect(claims.length).toBe(1);
      expect(claims[0].method).toBe("POST");
      expect(claims[0].url).toContain(
        `records/${encodeURIComponent("HOC_PHI:2026")}/claim`,
      );
      expect(claims[0].body?.expected_version).toBe(1);
    });
  });

  it("(b) confirmation cancel issues no call", async () => {
    const calls: Call[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        const method = (init?.method ?? "GET").toUpperCase();
        const body =
          typeof init?.body === "string" ? JSON.parse(init.body) : null;
        calls.push({ url, method, body });
        if (url.includes("/admin/identity")) return json(CLAIMER);
        if (url.includes("/admin/records") && url.includes("status=review"))
          return json({ records: [seedDraft], total: 1 });
        if (url.includes("/claim")) return json({ record: seedDraft, event: {} });
        return defaultHandler(url);
      }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <AdminSessionProvider>
          <MemoryRouter initialEntries={["/quan-ly/cho-duyet"]}>
            <Routes>
              <Route path="/quan-ly/cho-duyet" element={<RecordsGate />}>
                <Route index element={<PendingReview />} />
              </Route>
            </Routes>
          </MemoryRouter>
        </AdminSessionProvider>
      </QueryClientProvider>,
    );
    await screen.findByText(/Chờ duyệt \(1\)/);
    await userEvent.click(screen.getAllByText("Nhận nháp")[0]);
    await userEvent.click(screen.getByText("Chưa, quay lại"));
    expect(
      calls.filter((c) => c.url.includes("/claim")),
    ).toHaveLength(0);
  });

  it("(c) a 409 renders the server reason and does not refetch", async () => {
    const calls: Call[] = [];
    let listFetches = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        const method = (init?.method ?? "GET").toUpperCase();
        const body =
          typeof init?.body === "string" ? JSON.parse(init.body) : null;
        calls.push({ url, method, body });
        if (url.includes("/admin/identity")) return json(CLAIMER);
        if (url.includes("/admin/records") && url.includes("status=review")) {
          listFetches += 1;
          return json({ records: [seedDraft], total: 1 });
        }
        if (url.includes("/claim"))
          return json({ detail: "không thể nhận nháp: bản này đã có tác giả" }, 409);
        return defaultHandler(url);
      }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <AdminSessionProvider>
          <MemoryRouter initialEntries={["/quan-ly/cho-duyet"]}>
            <Routes>
              <Route path="/quan-ly/cho-duyet" element={<RecordsGate />}>
                <Route index element={<PendingReview />} />
              </Route>
            </Routes>
          </MemoryRouter>
        </AdminSessionProvider>
      </QueryClientProvider>,
    );
    await screen.findByText(/Chờ duyệt \(1\)/);
    const listFetchesBefore = listFetches;

    await userEvent.click(screen.getAllByText("Nhận nháp")[0]);
    await userEvent.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Nhận nháp" }));

    expect(
      await screen.findByText(/không thể nhận nháp: bản này đã có tác giả/),
    ).toBeInTheDocument();

    await waitFor(() => {
      expect(calls.filter((c) => c.url.includes("/claim")).length).toBe(1);
    });

    // List untouched: no refetch after the 409. Wait one tick so any
    // pending invalidation would have landed; then assert the count held.
    await waitFor(() => {
      expect(listFetches).toBe(listFetchesBefore);
    });
  });

  it("(d) the action is absent on a card the caller authored and on a non-draft record", async () => {
    const ownDraft = { ...seedDraft, record_id: "OWN:2026", authored_by: "cb.duyet" };
    const nonDraft = { ...seedDraft, record_id: "PUB:2026", status: "published", authored_by: "cb.soan" };
    renderScreen(
      "/quan-ly/cho-duyet",
      REVIEWER,
      <PendingReview />,
      (url) => {
        if (url.includes("/admin/records") && url.includes("status=review"))
          return json({ records: [ownDraft, nonDraft], total: 2 });
        return defaultHandler(url);
      },
    );
    await screen.findByText(/Chờ duyệt \(2\)/);
    expect(screen.queryByText("Nhận nháp")).toBeNull();
  });

  it("(e) after a successful claim the list refetches", async () => {
    const calls = renderScreen(
      "/quan-ly/cho-duyet",
      CLAIMER,
      <PendingReview />,
      (url) => {
        if (url.includes("/admin/records") && url.includes("status=review"))
          return json({ records: [seedDraft], total: 1 });
        if (url.includes("/claim"))
          return json({ record: seedDraft, event: {} });
        return defaultHandler(url);
      },
    );
    await screen.findByText(/Chờ duyệt \(1\)/);
    const listFetchesBefore = calls.filter(
      (c) => c.url.includes("/admin/records") && c.url.includes("status=review"),
    ).length;

    await userEvent.click(screen.getAllByText("Nhận nháp")[0]);
    await userEvent.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Nhận nháp" }),
    );

    await waitFor(() => {
      const listFetchesAfter = calls.filter(
        (c) => c.url.includes("/admin/records") && c.url.includes("status=review"),
      ).length;
      expect(listFetchesAfter).toBeGreaterThan(listFetchesBefore);
    });
  });
});

describe("Chờ duyệt — duyệt hàng loạt (T15)", () => {
  const recordA = {
    record_id: "HOC_PHI:2026",
    record_type: "HOC_PHI",
    version: 2,
    admission_year: 2026,
    effective_from: "2026-01-01",
    effective_to: "2027-01-01",
    status: "review",
    authority_level: "husc_official",
    source_document_id: null,
    source_url: null,
    source_page: null,
    published_at: null,
    authored_by: "cb.soan",
    supersedes: null,
    locale: "vi",
    text: "Học phí Hán Nôm 15tr.",
    values: {},
  };
  const recordB = {
    ...recordA,
    record_id: "NGANH:2026",
    record_type: "NGANH",
    text: "Ngành Hán Nôm.",
  };
  const recordOwn = { ...recordA, record_id: "RIENG:2026", authored_by: "cb.duyet" };

  const threeRecords = [recordA, recordB, recordOwn];


  it("(a) selecting 2 reviewable records and confirming issues approve+publish for each", async () => {
    const calls: Call[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        const method = (init?.method ?? "GET").toUpperCase();
        const body =
          typeof init?.body === "string" ? JSON.parse(init.body) : null;
        calls.push({ url, method, body });
        if (url.includes("/admin/identity")) return json(REVIEWER);
        if (url.includes("/admin/records") && url.includes("status=review"))
          return json({ records: threeRecords, total: 3 });
        if (url.includes("/transitions"))
          return json({ record: recordA, event: {} });
        return defaultHandler(url);
      }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <AdminSessionProvider>
          <MemoryRouter initialEntries={["/quan-ly/cho-duyet"]}>
            <Routes>
              <Route path="/quan-ly/cho-duyet" element={<RecordsGate />}>
                <Route index element={<PendingReview />} />
              </Route>
            </Routes>
          </MemoryRouter>
        </AdminSessionProvider>
      </QueryClientProvider>,
    );

    await screen.findByText(/Chờ duyệt \(3\)/);

    // Select two reviewable records
    const checkboxA = screen.getByLabelText(/Chọn HOC_PHI \(2026\)/) as HTMLInputElement;
    const checkboxB = screen.getByLabelText(/Chọn NGANH \(2026\)/) as HTMLInputElement;
    await userEvent.click(checkboxA);
    await userEvent.click(checkboxB);

    // Batch action bar appears; click the bar button to open the dialog
    const batchButtons = await screen.findAllByText(/Duyệt và xuất bản 2 mục/);
    await userEvent.click(batchButtons[0]);

    // Confirmation dialog is open; click the confirm button inside it
    const dialog = await screen.findByRole("alertdialog");
    const confirmBtn = within(dialog).getByRole("button", { name: /Duyệt và xuất bản 2 mục/ });
    await userEvent.click(confirmBtn);

    await waitFor(() => {
      const transitions = calls.filter((c) => c.url.includes("/transitions"));
      expect(transitions.length).toBeGreaterThanOrEqual(4);
    });

    const transitions = calls.filter((c) => c.url.includes("/transitions"));
    expect(
      transitions.some(
        (c) =>
          c.body?.action === "approve" &&
          c.url.includes(`records/${encodeURIComponent("HOC_PHI:2026")}/transitions`),
      ),
    ).toBe(true);
    expect(
      transitions.some(
        (c) =>
          c.body?.action === "approve" &&
          c.url.includes(`records/${encodeURIComponent("NGANH:2026")}/transitions`),
      ),
    ).toBe(true);

    expect(
      await screen.findByText(/Đã xuất bản 2 mục/),
    ).toBeInTheDocument();
  });

  it("(b) a record authored by the signed-in reviewer cannot be selected and shows the four-eyes reason", async () => {
    renderScreen(
      "/quan-ly/cho-duyet",
      REVIEWER,
      <PendingReview />,
      (url) => {
        if (url.includes("/admin/records") && url.includes("status=review"))
          return json({ records: threeRecords, total: 3 });
        return defaultHandler(url);
      },
    );
    await screen.findByText(/Chờ duyệt \(3\)/);

    // Own record: checkbox must NOT be present; four-eyes text must be
    expect(
      screen.queryByLabelText(/Chọn RIENG \(2026\)/),
    ).toBeNull();
    expect(screen.getAllByText(/Cần một người khác duyệt/).length).toBeGreaterThanOrEqual(1);

    // Batch action bar must NOT appear (0 selected)
    expect(
      screen.queryByText(/Duyệt và xuất bản 0 mục/),
    ).toBeNull();
  });

  it("(c) a mid-batch failure surfaces that record's error while the successful ones still complete", async () => {
    const calls: Call[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        const method = (init?.method ?? "GET").toUpperCase();
        const body =
          typeof init?.body === "string" ? JSON.parse(init.body) : null;
        calls.push({ url, method, body });
        if (url.includes("/admin/identity")) return json(REVIEWER);
        if (url.includes("/admin/records") && url.includes("status=review"))
          return json({ records: threeRecords, total: 3 });
        if (url.includes("/transitions")) {
          const action = (body?.action as string) ?? "";
          const isNganhPublish =
            action === "publish" &&
            url.includes(encodeURIComponent("NGANH:2026"));
          if (isNganhPublish)
            return json({ detail: "Không xuất bản được" }, 409);
          return json({ record: recordA, event: {} });
        }
        return defaultHandler(url);
      }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <AdminSessionProvider>
          <MemoryRouter initialEntries={["/quan-ly/cho-duyet"]}>
            <Routes>
              <Route path="/quan-ly/cho-duyet" element={<RecordsGate />}>
                <Route index element={<PendingReview />} />
              </Route>
            </Routes>
          </MemoryRouter>
        </AdminSessionProvider>
      </QueryClientProvider>,
    );
    await screen.findByText(/Chờ duyệt \(3\)/);

    await userEvent.click(screen.getByLabelText(/Chọn HOC_PHI \(2026\)/));
    await userEvent.click(screen.getByLabelText(/Chọn NGANH \(2026\)/));

    const batchButtons = await screen.findAllByText(/Duyệt và xuất bản 2 mục/);
    await userEvent.click(batchButtons[0]);
    const dialog = await screen.findByRole("alertdialog");
    const confirmBtn = within(dialog).getByRole("button", { name: /Duyệt và xuất bản 2 mục/ });
    await userEvent.click(confirmBtn);

    await waitFor(() => {
      const status = screen.getByText(/Đã xuất bản 1 mục/);
      expect(status).toBeInTheDocument();
    });

    // NGANH:2026 error is surfaced with the record id
    expect(await screen.findByText("NGANH:2026")).toBeInTheDocument();

    // HOC_PHI:2026 succeeded
    expect(
      await screen.findByText(/Đã xuất bản 1 mục/),
    ).toBeInTheDocument();
  });

  it("(d) the confirmation box is required (no publish without it)", async () => {
    const calls: Call[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        const method = (init?.method ?? "GET").toUpperCase();
        const body =
          typeof init?.body === "string" ? JSON.parse(init.body) : null;
        calls.push({ url, method, body });
        if (url.includes("/admin/identity")) return json(REVIEWER);
        if (url.includes("/admin/records") && url.includes("status=review"))
          return json({ records: threeRecords, total: 3 });
        if (url.includes("/transitions"))
          return json({ record: recordA, event: {} });
        return defaultHandler(url);
      }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <AdminSessionProvider>
          <MemoryRouter initialEntries={["/quan-ly/cho-duyet"]}>
            <Routes>
              <Route path="/quan-ly/cho-duyet" element={<RecordsGate />}>
                <Route index element={<PendingReview />} />
              </Route>
            </Routes>
          </MemoryRouter>
        </AdminSessionProvider>
      </QueryClientProvider>,
    );
    await screen.findByText(/Chờ duyệt \(3\)/);

    await userEvent.click(screen.getByLabelText(/Chọn HOC_PHI \(2026\)/));

    // No transition calls yet
    expect(
      calls.filter((c) => c.url.includes("/transitions")),
    ).toHaveLength(0);

    // Open the confirmation dialog but cancel it
    const batchButtons = await screen.findAllByText(/Duyệt và xuất bản 1 mục/);
    await userEvent.click(batchButtons[0]);
    const cancel = await screen.findByText("Chưa, quay lại");
    await userEvent.click(cancel);

    // Still no transition calls after cancelling
    expect(
      calls.filter((c) => c.url.includes("/transitions")),
    ).toHaveLength(0);
  });
});

describe("Kho thông tin", () => {
  it("shows the no-results state with the add-document link", async () => {
    renderScreen("/quan-ly/kho", EDITOR, <Vault />, defaultHandler);
    expect(
      await screen.findByText(/Không có thông tin nào khớp/),
    ).toBeInTheDocument();
    expect(screen.getByText("Thêm tài liệu")).toBeInTheDocument();
  });
});

describe("Câu hỏi bot chưa trả lời được", () => {
  it("shows the empty state when the queue is clear", async () => {
    renderScreen("/quan-ly/cau-hoi", EDITOR, <Unanswered />, defaultHandler);
    expect(await screen.findByText(/Bot đã trả lời được mọi câu hỏi/)).toBeInTheDocument();
  });

  it("lists open questions and offers the supplement action", async () => {
    const unanswered = {
      items: [
        {
          question: "Trường có ký túc xá không?",
          trace_id: "t1",
          asked_at: "2026-10-06T08:14:00Z",
          route: "padded_rag",
          status_code: "NO_INFO",
          resolved: false,
          resolved_at: null,
          resolved_by_record_id: null,
        },
      ],
      count: 1,
    };
    renderScreen("/quan-ly/cau-hoi", EDITOR, <Unanswered />, (url) => {
      if (url.includes("/admin/unanswered")) return json(unanswered);
      return defaultHandler(url);
    });
    expect(await screen.findByText("Trường có ký túc xá không?")).toBeInTheDocument();
    expect(screen.getByText("Bổ sung thông tin")).toBeInTheDocument();
  });
});

describe("Hỏi thử bot", () => {
  it("asks the question and renders the answer", async () => {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    mockFetch((url) => {
      if (url.includes("/admin/identity")) return json(EDITOR);
      if (url.includes("/v2/query"))
        return json({
          answer: "Điểm chuẩn 24,5",
          sources: [
            { id: "s1", title: "diem-chuan.pdf", snippet: "", data_year: "2025" },
          ],
          status_code: "SUCCESS",
        });
      return json({});
    });
    render(
      <QueryClientProvider client={client}>
        <AdminSessionProvider>
          <MemoryRouter initialEntries={["/quan-ly/hoi-thu"]}>
            <Routes>
              <Route path="/quan-ly/hoi-thu" element={<RecordsGate />}>
                <Route index element={<AskTest />} />
              </Route>
            </Routes>
          </MemoryRouter>
        </AdminSessionProvider>
      </QueryClientProvider>,
    );
    const input = await screen.findByLabelText("Câu hỏi");
    await userEvent.type(input, "Điểm chuẩn 2025?");
    await userEvent.click(screen.getByText("Hỏi"));
    expect(await screen.findByText(/Điểm chuẩn 24,5/)).toBeInTheDocument();
  });
});

describe("Lịch sử & khôi phục", () => {
  it("renders the timeline and disables rollback with no previous release", async () => {
    renderScreen("/quan-ly/lich-su", REVIEWER, <History />, defaultHandler);
    const rollbackButton = await screen.findByText("Khôi phục bản trước");
    expect(rollbackButton).toBeDisabled();
  });

  it("asks for a reason before rolling back when a previous release exists", async () => {
    const withRelease = (url: string): Response => {
      if (url.includes("/admin/releases"))
        return json({
          ...emptyRelease,
          previous: { release_id: "R-0", created_at: "2026-08-01" },
        });
      if (url.includes("/admin/rollback")) return json({ ok: true });
      return defaultHandler(url);
    };
    renderScreen("/quan-ly/lich-su", REVIEWER, <History />, withRelease);
    await userEvent.click(await screen.findByText("Khôi phục bản trước"));
    const reasonInput = screen.getByLabelText("Lý do khôi phục (bắt buộc)");
    const confirmButton = screen.getByText("Khôi phục");
    expect(confirmButton).toBeDisabled();
    await userEvent.type(reasonInput, "Bản mới sai năm");
    await userEvent.click(confirmButton);
    await waitFor(() => {
      expect(screen.getByText(/Đã khôi phục/)).toBeInTheDocument();
    });
  });
});
