import { expect, test, type Page, type Route } from "@playwright/test";

// U10: the account screen, with every /admin call answered by a route mock.
const ADMIN_IDENTITY = { actor_id: "cb.quan-tri", role: "admin", actions: ["roles.manage"] };
const EDITOR_IDENTITY = { actor_id: "cb.bien-tap", role: "editor", actions: [] };

const ROWS = [
  {
    subject_id: "11111111-1111-4111-8111-111111111111",
    actor_id: "cb.quan-tri",
    role: "admin",
    disabled: false,
    last_login_at: null,
  },
  {
    subject_id: "22222222-2222-4222-8222-222222222222",
    actor_id: "cb.bien-tap",
    role: "editor",
    disabled: false,
    last_login_at: null,
  },
];

async function mockAdmin(page: Page, identity: object, onWrite?: (route: Route) => Promise<void>) {
  await page.route(/\/admin\/identity$/, (route) => route.fulfill({ json: identity }));
  await page.route(/\/admin\/identities$/, async (route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({ json: ROWS });
      return;
    }
    if (onWrite) await onWrite(route);
  });
  await page.route(/\/admin\/identities\/[^/]+$/, async (route) => {
    if (onWrite) await onWrite(route);
  });
}

test("admin creates an account and sees the header token once", async ({ page }) => {
  await mockAdmin(page, ADMIN_IDENTITY, async (route) => {
    if (route.request().method() === "POST") {
      await route.fulfill({
        status: 200,
        json: { ...ROWS[1], actor_id: "cb.moi", subject_id: "33333333-3333-4333-8333-333333333333", header_token: "tok-once-123" },
      });
    }
  });

  await page.goto("/quan-ly/tai-khoan");
  await expect(page.getByRole("heading", { name: "Tài khoản", exact: true })).toBeVisible();

  await page.getByLabel("Lý do thay đổi (bắt buộc)").fill("Cấp quyền mới");
  await page.getByLabel("Tên đăng nhập").fill("cb.moi");
  await page.getByLabel("Cấp mã truy cập một lần").check();
  await page.getByRole("button", { name: "Tạo tài khoản" }).click();

  await expect(page.getByText("tok-once-123")).toBeVisible();
  await expect(page.getByText("chỉ hiển thị lúc này")).toBeVisible();
});

test("a non-admin sees the notice and no account list", async ({ page }) => {
  let listed = false;
  await mockAdmin(page, EDITOR_IDENTITY);
  await page.route(/\/admin\/identities$/, async (route) => {
    listed = true;
    await route.fulfill({ json: ROWS });
  });

  await page.goto("/quan-ly/tai-khoan");
  await expect(page.getByText("Chỉ quản trị viên mới quản lý tài khoản.")).toBeVisible();
  expect(listed).toBe(false);
});

test("demoting the last admin shows the LAST_ADMIN refusal", async ({ page }) => {
  await mockAdmin(page, ADMIN_IDENTITY, async (route) => {
    if (route.request().method() === "PATCH") {
      await route.fulfill({
        status: 409,
        json: { detail: { code: "LAST_ADMIN", message: "last active admin" } },
      });
    }
  });

  await page.goto("/quan-ly/tai-khoan");
  await page.getByLabel("Lý do thay đổi (bắt buộc)").fill("Đổi vai trò");
  await page.getByLabel("Vai trò của cb.quan-tri").selectOption("editor");

  await expect(page.getByText("Không thể bỏ quyền hoặc khóa quản trị viên cuối cùng.")).toBeVisible();
});
