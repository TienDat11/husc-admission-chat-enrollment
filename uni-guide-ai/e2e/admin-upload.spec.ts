import { expect, test } from "@playwright/test";
import { API_URL, loginAs, uniquePdf } from "./admin-helpers";

test.describe("Thêm thông tin — tải tài liệu", () => {
  test("viewer không thấy nút tải lên", async ({ page }) => {
    await loginAs(page, "e2e-viewer");
    await page.goto("/quan-ly/them");
    await expect(page.getByText("Bạn chỉ có quyền xem")).toBeVisible();
    await expect(page.locator('input[type="file"]')).toHaveCount(0);
  });

  test("tải PDF, theo dõi tiến trình, và giữ job sau khi tải lại", async ({ page }) => {
    await loginAs(page, "e2e-editor");
    await page.goto("/quan-ly/them");
    const pdf = uniquePdf();
    await page.locator('input[type="file"]').setInputFiles(pdf);
    await page.getByRole("button", { name: "Đọc và kiểm tra tài liệu →" }).click();

    await expect(page.getByRole("status")).toContainText(/Hoàn tất|Đang xử lý/, { timeout: 60_000 });
    await expect(page.getByText("Số trang đã đọc:")).toBeVisible({ timeout: 60_000 });

    await page.reload();
    await expect(page.getByText(`Tệp: ${pdf.name}`)).toBeVisible();
    await expect(page.getByText("Số trang đã đọc:")).toBeVisible({ timeout: 60_000 });
  });

  test("mất mạng: báo lỗi và cho thử lại", async ({ page }) => {
    await loginAs(page, "e2e-editor");
    await page.goto("/quan-ly/them");
    const pdf = uniquePdf();
    await page.route("**/admin/source-documents/upload", (route) => route.abort());
    await page.locator('input[type="file"]').setInputFiles(pdf);
    await page.getByRole("button", { name: "Đọc và kiểm tra tài liệu →" }).click();

    await expect(page.getByText("Không kết nối được máy chủ")).toBeVisible();
    await page.unroute("**/admin/source-documents/upload");
    await page.getByRole("button", { name: "Thử lại" }).click();
    await expect(page.getByText("Số trang đã đọc:")).toBeVisible({ timeout: 60_000 });
  });

  test("một nút chính trong khu làm việc, và không tràn ngang ở 320px", async ({ page }) => {
    await loginAs(page, "e2e-editor");
    await page.goto("/quan-ly/them");
    await expect(page.locator("#upload-file")).toBeVisible();
    await expect(page.locator('[data-workarea] [data-variant="primary"]')).toHaveCount(1);

    const overflows = await page.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
    );
    expect(overflows).toBe(false);
  });

  test("sidebar hiển thị Chờ duyệt (n) đúng bằng review_count", async ({ page }) => {
    await loginAs(page, "e2e-reviewer");
    const response = await page.request.get(`${API_URL}/admin/overview`);
    expect(response.ok()).toBe(true);
    const { review_count: reviewCount } = (await response.json()) as { review_count: number };

    const link = page.getByRole("link", { name: /^Chờ duyệt/ });
    await expect(link).toBeVisible();
    if (reviewCount > 0) {
      await expect(link).toHaveAccessibleName(`Chờ duyệt (${reviewCount})`);
    } else {
      await expect(link).toHaveAccessibleName("Chờ duyệt");
    }
  });
});
