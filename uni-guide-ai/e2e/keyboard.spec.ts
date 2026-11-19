import { expect, test, type Page } from "@playwright/test";
import { E2E_PASSWORD } from "./admin-helpers";

const LOCKED_TITLE = "Tài khoản bị khoá tạm thời";

/** The lockout banner is the one role=alert/role=status element holding the
 * title; matching the element avoids the nested text nodes getByText also hits. */
const lockBanner = (page: Page) =>
  page.locator('[role="alert"], [role="status"]').filter({ hasText: LOCKED_TITLE });

test.describe("Đăng nhập — bàn phím và khoá tạm thời", () => {
  test("thứ tự Tab: tên đăng nhập → mật khẩu → nút Đăng nhập", async ({ page }) => {
    await page.goto("/quan-ly/dang-nhap");
    await page.locator("#login-actor").focus();

    await page.keyboard.press("Tab");
    await expect(page.locator("#login-password")).toBeFocused();

    await page.keyboard.press("Tab");
    await expect(page.getByRole("button", { name: "Đăng nhập" })).toBeFocused();
  });

  test("sai mật khẩu liên tiếp: lần thứ 6 báo khoá và khoá các ô nhập", async ({ page }) => {
    // The backend refuses the first five wrong passwords with 401 and answers
    // the next request with 423 (see test_admin_session_sql.py).
    await page.goto("/quan-ly/dang-nhap");
    for (let attempt = 0; attempt < 6; attempt += 1) {
      if (await lockBanner(page).isVisible()) break;
      await page.locator("#login-actor").fill("e2e-lockout-probe");
      await page.locator("#login-password").fill(`wrong-${attempt}`);
      await Promise.all([
        page.waitForResponse((response) => response.url().includes("/admin/login")),
        page.getByRole("button", { name: "Đăng nhập" }).click(),
      ]);
      await expect(
        lockBanner(page).or(page.locator("#login-password:enabled")),
      ).toBeVisible();
    }
    await expect(lockBanner(page)).toBeVisible();
    await expect(lockBanner(page).getByText(/Thử lại sau \d+ giây/)).toBeVisible();
    await expect(page.locator("#login-password")).toBeDisabled();
  });

  test("đăng nhập đúng quay về trang được yêu cầu (?next=)", async ({ page }) => {
    await page.goto("/quan-ly/dang-nhap?next=%2Fquan-ly%2Fkho");
    await page.locator("#login-actor").fill("e2e-viewer");
    await page.locator("#login-password").fill(E2E_PASSWORD);
    await page.getByRole("button", { name: "Đăng nhập" }).click();
    await expect(page).toHaveURL(/\/quan-ly\/kho$/);
  });
});
