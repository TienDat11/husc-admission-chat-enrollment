import { expect, test } from "@playwright/test";
import { loginAs } from "./admin-helpers";

test.describe("Vận hành", () => {
  test("worker mất heartbeat: báo đỏ worker, API vẫn xanh", async ({ page }) => {
    await loginAs(page, "e2e-ops");
    const stale = new Date(Date.now() - 5 * 60_000).toISOString();
    await page.route("**/admin/ops/workers", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([{ worker_id: "worker-e2e", last_heartbeat_at: stale }]),
      }),
    );

    await page.goto("/quan-ly/van-hanh");
    await expect(page.getByText("Mất kết nối worker")).toBeVisible();
    await expect(page.getByText("API hoạt động")).toBeVisible();
  });

  test("không có nút khôi phục (rollback)", async ({ page }) => {
    await loginAs(page, "e2e-admin");
    await page.goto("/quan-ly/van-hanh");
    await expect(page.getByText("API hoạt động")).toBeVisible();
    await expect(page.getByRole("button", { name: /khôi phục|rollback|hoàn tác/i })).toHaveCount(0);
  });

  test("viewer không vào được trang vận hành", async ({ page }) => {
    await loginAs(page, "e2e-viewer");
    await page.goto("/quan-ly/van-hanh");
    await expect(page.getByText("Chỉ tài khoản kỹ thuật hoặc quản trị được xem trang này.")).toBeVisible();
  });
});
