import { test, expect } from "@playwright/test";
import { loginAs } from "./admin-helpers";

test("admin review page: batch approve and retrieval preview", async ({ page }) => {
  await loginAs(page, "reviewer_e2e");

  await page.goto("/quan-ly/review");
  await expect(page.locator("h1")).toContainText(/Duyệt|review/i);

  // Batch select first row
  const firstCheckbox = page.locator('input[type="checkbox"]').first();
  await expect(firstCheckbox).toBeVisible();
  await firstCheckbox.check();

  // Click batch approve
  await page.getByRole("button", { name: /Duyệt|Approve/i }).click();

  // Expect feedback
  await expect(page.locator("text=/áp dụng|applied|thành công/i")).toBeVisible();

  // Open retrieval preview
  await page.getByRole("button", { name: /Xem trước|Preview|Sandbox/i }).click();
  await expect(page.locator("text=/trước|before|sau|after/i")).toBeVisible();
});
