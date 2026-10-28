// Ticket 19 acceptance at the UI seam: the admissions CRUD screens.
//
// What these tests are for — each one covers a checklist item that would
// otherwise be a claim about a screenshot:
//
//   * create end-to-end (form -> POST -> confirmation);
//   * field-level validation blocks the save and renders at the offending
//     input (asserted through aria-describedby, not by looking for text
//     anywhere on the page);
//   * a server-side 422 is mapped onto the field FastAPI names;
//   * edit is presented as a NEW version and never sends a write that could
//     overwrite the published one (no `version`, no `status`, expected_version
//     carried);
//   * an action a role lacks is absent from the screen (viewer sees no create,
//     edit or transition; editor sees no publish);
//   * the filters reach the API as query parameters;
//   * archiving goes through the lifecycle transition, not a delete.
//
// The route table here mirrors App.tsx's records routes; App.tsx's own wiring
// is asserted separately by rendering the real app.
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { ThemeProvider } from "@/hooks/useTheme";
import App from "@/App";
import AdminLayout from "@/pages/admin/AdminLayout";
import RecordsGate from "@/pages/admin/RecordsGate";
import RecordList from "@/pages/admin/RecordList";
import RecordForm from "@/pages/admin/RecordForm";
import { AdminSessionProvider } from "@/hooks/useAdminSession";
import type { AdminIdentity, CanonicalRecord } from "@/lib/adminApi";

const EDITOR: AdminIdentity = {
  actor_id: "cb.bien-tap",
  role: "editor",
  actions: ["records.read", "records.create", "records.edit", "records.submit"],
};

const VIEWER: AdminIdentity = {
  actor_id: "cb.chi-xem",
  role: "viewer",
  actions: ["records.read"],
};

const PUBLISHER: AdminIdentity = {
  actor_id: "cb.xuat-ban",
  role: "publisher",
  actions: ["records.read", "records.publish", "records.supersede"],
};

const RECORD_ID = "CHI_TIEU:nganh:7480201:2026:ttts";

function record(overrides: Partial<CanonicalRecord> = {}): CanonicalRecord {
  return {
    schema_version: "1.0",
    record_id: RECORD_ID,
    record_type: "CHI_TIEU",
    version: 3,
    admission_year: 2026,
    effective_from: "2026-01-01",
    effective_to: "2026-12-31",
    status: "published",
    authority_level: "husc_official",
    source_document_id: "de-an-tuyen-sinh-2026",
    source_url: "https://tuyensinh.husc.edu.vn/de-an-2026.pdf",
    source_page: 24,
    published_at: "2026-06-02T00:00:00Z",
    authored_by: "cb.bien-tap",
    supersedes: null,
    locale: "vi",
    text: "Chỉ tiêu tuyển sinh 2026 ngành Công nghệ thông tin: 120.",
    values: { ma_nganh: "7480201", ten_nganh: "Công nghệ thông tin", chi_tieu: 120 },
    ...overrides,
  };
}

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
}

/**
 * Stub `fetch` with a per-test handler and record every call.
 *
 * The handler returns the response for a URL; an unmocked URL throws, so a
 * screen that calls an endpoint the test did not intend fails loudly instead of
 * silently rendering an empty state.
 */
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
      calls.push({ url, method, body, headers });
      return handler(url, method, body);
    }),
  );
  return calls;
}

function identityFor(identity: AdminIdentity) {
  return (url: string) => (url.includes("/admin/identity") ? json(identity) : null);
}

function renderRecords(initialPath: string, identity: AdminIdentity, handler: (url: string, method: string, body: Record<string, unknown> | null) => Response) {
  const identityResponse = identityFor(identity);
  const calls = mockFetch((url, method, body) => {
    const fromIdentity = identityResponse(url);
    if (fromIdentity) return fromIdentity;
    return handler(url, method, body);
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <AdminSessionProvider>
      <MemoryRouter initialEntries={[initialPath]}>
        <Routes>
          <Route path="/admin/records" element={<RecordsGate />}>
            <Route index element={<RecordList />} />
            <Route path="new" element={<RecordForm mode="create" />} />
            <Route path=":recordId/edit" element={<RecordForm mode="edit" />} />
          </Route>
        </Routes>
      </MemoryRouter>
      </AdminSessionProvider>
    </QueryClientProvider>,
  );
  return calls;
}

/** The message rendered at a field, via the input's aria-describedby. */
function errorAt(fieldId: string): string | null {
  const field = document.getElementById(fieldId);
  expect(field).not.toBeNull();
  expect(field).toHaveAttribute("aria-invalid", "true");
  const describedBy = field?.getAttribute("aria-describedby");
  if (!describedBy) return null;
  return document.getElementById(describedBy)?.textContent ?? null;
}

beforeEach(() => {
  window.sessionStorage.clear();
  vi.unstubAllGlobals();
});

describe("Record list", () => {
  it("lists records with their ngành, status and version", async () => {
    renderRecords("/admin/records", VIEWER, () =>
      json({ records: [record()], total: 1 }),
    );

    const row = (await screen.findByText(RECORD_ID)).closest("tr");
    expect(row).not.toBeNull();
    expect(within(row as HTMLElement).getByText("Công nghệ thông tin (7480201)")).toBeInTheDocument();
    expect(within(row as HTMLElement).getByText("Đã công bố")).toBeInTheDocument();
    expect(within(row as HTMLElement).getByText("v3")).toBeInTheDocument();
  });

  it("sends the ngành / năm / trạng thái filters as query parameters", async () => {
    const calls = renderRecords("/admin/records", VIEWER, () =>
      json({ records: [record()], total: 1 }),
    );
    await screen.findByText(RECORD_ID);

    await userEvent.type(screen.getByLabelText("Ngành"), "cong nghe");
    await userEvent.type(screen.getByLabelText("Năm"), "2026");
    await userEvent.selectOptions(screen.getByLabelText("Trạng thái"), "published");
    await userEvent.click(screen.getByRole("button", { name: "Lọc" }));

    await waitFor(() => {
      const urls = calls.filter((call) => call.method === "GET").map((call) => call.url);
      expect(urls.some((url) => url.includes("nganh=cong+nghe") && url.includes("year=2026") && url.includes("status=published"))).toBe(true);
    });
  });

  it("hides every action a viewer is not granted, and offers none it may not call", async () => {
    renderRecords("/admin/records", VIEWER, () => json({ records: [record()], total: 1 }));
    await screen.findByText(RECORD_ID);

    expect(screen.queryByRole("link", { name: "Tạo record mới" })).toBeNull();
    expect(screen.queryByRole("link", { name: "Sửa" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Gửi duyệt" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Công bố" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Lưu trữ" })).toBeNull();
  });

  it("offers an editor the draft actions and the create/edit links, but not publish", async () => {
    renderRecords("/admin/records", EDITOR, () =>
      json({ records: [record({ status: "draft", version: 1 })], total: 1 }),
    );
    await screen.findByText(RECORD_ID);

    expect(screen.getByRole("link", { name: "Tạo record mới" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Sửa" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Gửi duyệt" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Công bố" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Lưu trữ" })).toBeNull();
  });

  it("archives a published record through the lifecycle, never by deleting it", async () => {
    const calls = renderRecords("/admin/records", PUBLISHER, (url) =>
      url.includes("/transitions")
        ? json({
            record: record({ status: "archived", version: 4 }),
            event: { action: "supersede", actor_id: "cb.xuat-ban" },
          })
        : json({ records: [record()], total: 1 }),
    );
    await screen.findByText(RECORD_ID);

    await userEvent.click(screen.getByRole("button", { name: "Thay thế" }));
    // The confirmation explains what archiving does to the published version.
    expect(await screen.findByText(/bản published chuyển sang archived/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Xác nhận" }));

    await waitFor(() => {
      const transition = calls.find((call) => call.url.includes("/transitions"));
      expect(transition?.body).toMatchObject({ action: "supersede", expected_version: 3 });
    });
    // No delete verb anywhere: the lifecycle transition is the whole mechanism.
    expect(calls.every((call) => call.method !== "DELETE")).toBe(true);
  });

  it("blocks publish on a record with no provenance, and says why", async () => {
    renderRecords("/admin/records", PUBLISHER, () =>
      json({
        records: [
          record({ status: "approved", source_document_id: null, source_url: null }),
        ],
        total: 1,
      }),
    );
    await screen.findByText(RECORD_ID);

    const publish = screen.getByRole("button", { name: "Công bố" });
    expect(publish).toBeDisabled();
    expect(publish).toHaveAttribute(
      "title",
      expect.stringContaining("source_document_id"),
    );
  });

  it("asks for a reason before rejecting, and will not send a reject without one", async () => {
    const calls = renderRecords(
      "/admin/records",
      { actor_id: "cb.phan-bien", role: "reviewer", actions: ["records.read", "records.reject"] },
      () => json({ records: [record({ status: "review" })], total: 1 }),
    );
    await screen.findByText(RECORD_ID);

    await userEvent.click(screen.getByRole("button", { name: "Từ chối" }));
    const confirm = await screen.findByRole("button", { name: "Xác nhận" });
    expect(confirm).toBeDisabled();
    expect(calls.some((call) => call.url.includes("/transitions"))).toBe(false);

    await userEvent.type(screen.getByLabelText(/Lý do/), "Số liệu chưa khớp đề án.");
    await userEvent.click(confirm);

    await waitFor(() => {
      const transition = calls.find((call) => call.url.includes("/transitions"));
      expect(transition?.body).toMatchObject({
        action: "reject",
        reason: "Số liệu chưa khớp đề án.",
      });
    });
  });

  it("sends an idempotency key with a transition, and reuses it on retry", async () => {
    // The point of the key is the retry: a response that never arrived must be
    // retried under the *same* key or the backend runs the transition twice.
    // A key minted inside the retry would look correct and dedupe nothing.
    let failFirst = true;
    const calls = renderRecords("/admin/records", PUBLISHER, (url, method) => {
      if (url.includes("/transitions")) {
        if (failFirst) {
          failFirst = false;
          // A dropped response: the server may well have applied it.
          return json({ detail: "Bad Gateway" }, 502);
        }
        return json({
          record: record({ status: "archived", version: 4 }),
          event: { action: "supersede", actor_id: "cb.xuat-ban" },
        });
      }
      return json({ records: [record()], total: 1 });
    });
    await screen.findByText(RECORD_ID);

    await userEvent.click(screen.getByRole("button", { name: "Thay thế" }));
    await userEvent.click(await screen.findByRole("button", { name: "Xác nhận" }));

    // The failure surfaces, and the confirm dialog stays open for a retry.
    expect(await screen.findByRole("alert")).toHaveTextContent("Bad Gateway");
    await userEvent.click(screen.getByRole("button", { name: "Xác nhận" }));

    await waitFor(() => {
      const attempts = calls.filter((call) => call.url.includes("/transitions"));
      expect(attempts).toHaveLength(2);
      const first = attempts[0].headers["Idempotency-Key"];
      const retry = attempts[1].headers["Idempotency-Key"];
      expect(first).toBeTruthy();
      // Same key: the retry replays instead of archiving twice.
      expect(retry).toBe(first);
    });
  });

  it("mints a fresh key for a different action rather than reusing the last one", async () => {
    // The backend refuses a key reused for a different request (422), so a
    // stale key leaking into a new intention would break the second action.
    // The list re-reads after the submit, so it must report the new status or
    // the next legal action is not offered.
    let status: CanonicalRecord["status"] = "draft";
    const calls = renderRecords(
      "/admin/records",
      {
        actor_id: "cb.quan-tri",
        role: "admin",
        actions: ["records.read", "records.submit", "records.approve"],
      },
      (url) => {
        if (url.includes("/transitions")) {
          status = "review";
          return json({
            record: record({ status: "review", version: 2 }),
            event: { action: "submit", actor_id: "cb.quan-tri" },
          });
        }
        return json({ records: [record({ status, version: status === "draft" ? 1 : 2 })], total: 1 });
      },
    );
    await screen.findByText(RECORD_ID);

    await userEvent.click(screen.getByRole("button", { name: "Gửi duyệt" }));
    await waitFor(() => {
      expect(calls.filter((call) => call.url.includes("/transitions"))).toHaveLength(1);
    });

    const submit = calls.find((call) => call.url.includes("/transitions"));
    await userEvent.click(screen.getByRole("button", { name: "Duyệt" }));
    await waitFor(() => {
      const attempts = calls.filter((call) => call.url.includes("/transitions"));
      expect(attempts).toHaveLength(2);
      const approve = attempts[1];
      expect(approve.body).toMatchObject({ action: "approve" });
      expect(approve.headers["Idempotency-Key"]).not.toBe(submit?.headers["Idempotency-Key"]);
    });
  });
});

describe("Record form — create", () => {
  it("creates a record end-to-end as a draft version 1", async () => {
    const created = record({ status: "draft", version: 1, published_at: null });
    const calls = renderRecords("/admin/records/new", EDITOR, (url, method) =>
      url.endsWith("/admin/records") && method === "POST"
        ? json({ record: created, event: { action: "create", actor_id: "cb.bien-tap" } })
        : json({ records: [], total: 0 }),
    );

    await userEvent.type(await screen.findByLabelText("Mã record"), RECORD_ID);
    await userEvent.clear(screen.getByLabelText("Năm tuyển sinh"));
    await userEvent.type(screen.getByLabelText("Năm tuyển sinh"), "2026");
    await userEvent.type(screen.getByLabelText("Hiệu lực từ"), "2026-01-01");
    await userEvent.type(screen.getByLabelText("Tên ngành"), "Công nghệ thông tin");
    await userEvent.type(screen.getByLabelText("Mã ngành"), "7480201");
    await userEvent.type(screen.getByLabelText("Nội dung"), "Chỉ tiêu 2026: 120.");
    await userEvent.click(screen.getByRole("button", { name: "Lưu record mới" }));

    await waitFor(() => expect(calls.some((call) => call.method === "POST")).toBe(true));
    const post = calls.find((call) => call.method === "POST");
    expect(post?.url).toContain("/admin/records");
    expect(post?.body).toMatchObject({
      record: {
        record_id: RECORD_ID,
        record_type: "CHI_TIEU",
        version: 1,
        status: "draft",
        admission_year: 2026,
        effective_from: "2026-01-01",
        text: "Chỉ tiêu 2026: 120.",
        values: { ma_nganh: "7480201", ten_nganh: "Công nghệ thông tin" },
      },
    });

    expect(await screen.findByRole("status")).toHaveTextContent("Đã lưu");
    expect(screen.getByRole("status")).toHaveTextContent("v1");
  });

  it("blocks the save and renders each error at its own field", async () => {
    const calls = renderRecords("/admin/records/new", EDITOR, () =>
      json({ records: [], total: 0 }),
    );

    await userEvent.type(await screen.findByLabelText("Hiệu lực từ"), "2026-12-31");
    await userEvent.type(screen.getByLabelText("Hiệu lực đến"), "2026-06-01");
    await userEvent.clear(screen.getByLabelText("Năm tuyển sinh"));
    await userEvent.type(screen.getByLabelText("Năm tuyển sinh"), "1999");
    await userEvent.click(screen.getByRole("button", { name: "Lưu record mới" }));

    // Nothing was sent: invalid data never reaches the store.
    expect(calls.some((call) => call.method === "POST")).toBe(false);

    // Each message is attached to its own input, not dropped in a banner.
    expect(errorAt("record_id")).toContain("Mã record không được để trống");
    expect(errorAt("admission_year")).toContain("2000 đến 2100");
    expect(errorAt("effective_to")).toContain("sau hoặc bằng ngày hiệu lực");
    expect(errorAt("text")).toContain("Nội dung không được để trống");
  });

  it("renders a server-side validation error at the field the server names", async () => {
    renderRecords("/admin/records/new", EDITOR, (url, method) =>
      method === "POST"
        ? json(
            {
              detail: [
                {
                  loc: ["body", "record", "admission_year"],
                  msg: "Input should be less than or equal to 2100",
                  type: "less_than_equal",
                },
              ],
            },
            422,
          )
        : json({ records: [], total: 0 }),
    );

    await userEvent.type(await screen.findByLabelText("Mã record"), RECORD_ID);
    await userEvent.type(screen.getByLabelText("Hiệu lực từ"), "2026-01-01");
    await userEvent.type(screen.getByLabelText("Nội dung"), "Chỉ tiêu 2026: 120.");
    await userEvent.click(screen.getByRole("button", { name: "Lưu record mới" }));

    await waitFor(() => {
      expect(errorAt("admission_year")).toContain("less than or equal to 2100");
    });
  });
});

describe("Record form — edit", () => {
  it("saves a NEW version and says the published version is not overwritten", async () => {
    const published = record();
    const calls = renderRecords(
      `/admin/records/${encodeURIComponent(RECORD_ID)}/edit`,
      EDITOR,
      (url, method) =>
        method === "POST"
          ? json({
              record: record({ status: "draft", version: 4, published_at: null }),
              event: { action: "edit", actor_id: "cb.bien-tap" },
            })
          : json(published),
    );

    // The banner states the consequence before anything is typed.
    const banner = await screen.findByText(
      (_, element) => element?.tagName === "P" && /Lưu sẽ tạo/.test(element.textContent ?? ""),
    );
    expect(banner).toHaveTextContent(/version v4 \(nháp\)/i);
    expect(banner).toHaveTextContent(/bản đã công bố vẫn nguyên vẹn/i);

    const text = screen.getByLabelText("Nội dung");
    await userEvent.clear(text);
    await userEvent.type(text, "Chỉ tiêu 2026: 130 (đã rà soát).");
    await userEvent.click(screen.getByRole("button", { name: "Lưu thành version v4" }));

    await waitFor(() => expect(calls.some((call) => call.method === "POST")).toBe(true));
    const post = calls.find((call) => call.method === "POST");
    expect(post?.url).toContain(`/admin/records/${encodeURIComponent(RECORD_ID)}/edits`);
    expect(post?.body).toMatchObject({
      // The optimistic-concurrency token of the version being edited…
      expected_version: 3,
      text: "Chỉ tiêu 2026: 130 (đã rà soát).",
      // …and the extractor's other structured keys survive the hand edit.
      values: { ma_nganh: "7480201", ten_nganh: "Công nghệ thông tin", chi_tieu: 120 },
    });
    // The edit request never names the version or the status: the store owns
    // both, so there is no field a caller could use to rewrite history.
    expect(post?.body).not.toHaveProperty("version");
    expect(post?.body).not.toHaveProperty("status");

    expect(await screen.findByRole("status")).toHaveTextContent("v4");
    expect(
      screen.getByText(/version mới\. Version đã công bố trước đó vẫn nguyên vẹn/i),
    ).toBeInTheDocument();
  });

  it("shows no edit form to a role without records.edit", async () => {
    renderRecords(`/admin/records/${encodeURIComponent(RECORD_ID)}/edit`, VIEWER, () =>
      json(record()),
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "không có quyền sửa dữ liệu tuyển sinh",
    );
    expect(screen.queryByLabelText("Nội dung")).toBeNull();
  });
});

describe("The credential gate", () => {
  /**
   * Identity answers 401 until /admin/login has set the session cookie, then
   * answers as VIEWER. The fake server is the only thing that knows the cookie
   * exists, so the test exercises the real sign-in path, not a stored token.
   */
  function renderGate(wrongPassword = false) {
    let signedIn = false;
    const calls = mockFetch((url) => {
      if (url.includes("/admin/login")) {
        if (wrongPassword) return json({ detail: "bad credentials" }, 401);
        signedIn = true;
        return json({ actor_id: VIEWER.actor_id, role: VIEWER.role, session_seconds: 28800 });
      }
      if (url.includes("/admin/identity")) {
        return signedIn ? json(VIEWER) : json({ detail: "not signed in" }, 401);
      }
      return json({ records: [record()], total: 1 });
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <AdminSessionProvider>
          <MemoryRouter initialEntries={["/admin/records"]}>
            <Routes>
              <Route path="/admin/records" element={<RecordsGate />}>
                <Route index element={<RecordList />} />
              </Route>
            </Routes>
          </MemoryRouter>
        </AdminSessionProvider>
      </QueryClientProvider>,
    );
    return calls;
  }

  it("asks for a credential before showing any admissions data", async () => {
    const calls = renderGate();

    expect(await screen.findByLabelText("Tài khoản")).toBeInTheDocument();
    expect(screen.getByLabelText("Mật khẩu")).toHaveAttribute("type", "password");
    expect(screen.queryByText(RECORD_ID)).toBeNull();
    expect(calls.some((call) => call.url.includes("/admin/records"))).toBe(false);

    await userEvent.type(screen.getByLabelText("Tài khoản"), VIEWER.actor_id);
    await userEvent.type(screen.getByLabelText("Mật khẩu"), "matkhau-thu");
    await userEvent.click(screen.getByRole("button", { name: "Đăng nhập" }));

    expect(await screen.findByText(RECORD_ID)).toBeInTheDocument();
    const login = calls.find((call) => call.url.includes("/admin/login"));
    expect(login?.method).toBe("POST");
    expect(login?.body).toEqual({ actor_id: VIEWER.actor_id, password: "matkhau-thu" });
    // The password and the session live only in the HttpOnly cookie, never in web storage.
    expect(window.localStorage.length).toBe(0);
    expect(window.sessionStorage.length).toBe(0);
  });

  it("refuses a wrong password with the Vietnamese 401 message and shows no data", async () => {
    const calls = renderGate(true);

    await userEvent.type(await screen.findByLabelText("Tài khoản"), VIEWER.actor_id);
    await userEvent.type(screen.getByLabelText("Mật khẩu"), "sai");
    await userEvent.click(screen.getByRole("button", { name: "Đăng nhập" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Sai tài khoản hoặc mật khẩu.");
    expect(screen.queryByRole("table")).toBeNull();
    expect(calls.some((call) => call.url.includes("/admin/login"))).toBe(true);
  });
});

describe("The admin shell", () => {
  it("mounts the admissions surface at /admin/records through the real app", async () => {
    mockFetch(() => json({ detail: "not signed in" }, 401));
    window.history.pushState({}, "", "/admin/records");
    render(
      <ThemeProvider>
        <App />
      </ThemeProvider>,
    );

    // The real route table reaches the gate, not a test-local one.
    expect(await screen.findByLabelText("Tài khoản")).toBeInTheDocument();
  });

  it("exposes the admissions screen from the sidebar", async () => {
    render(
      <ThemeProvider>
        <MemoryRouter initialEntries={["/admin/records"]}>
          <Routes>
            <Route path="/admin" element={<AdminLayout />}>
              <Route path="records" element={<p>records surface</p>} />
            </Route>
          </Routes>
        </MemoryRouter>
      </ThemeProvider>,
    );

    const nav = screen.getByRole("navigation");
    const link = within(nav).getByRole("link", { name: "Dữ liệu" });
    expect(link).toHaveAttribute("href", "/admin/records");
  });
});
