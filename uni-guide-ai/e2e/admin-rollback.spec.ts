import { test, expect } from "@playwright/test";
import { loginAs } from "./admin-helpers";

test("admin releases: activate ready release and rollback", async ({ page }) => {
  await loginAs(page, "admin_e2e");

  await page.goto("/quan-ly/releases");
  await expect(page.locator("h1")).toContainText(/Release|Phát hành/i);

  // Find a ready release and activate it
  const activateBtn = page.getByRole("button", { name: /Kích hoạt|Activate/i }).first();
  if (await activateBtn.isVisible()) {
    await activateBtn.click();
    await expect(page.locator("text=/active|hoạt động/i")).toBeVisible();
  }

  // Rollback requires reason >= 10 chars
  const rollbackBtn = page.getByRole("button", { name: /Hoàn tác|Rollback/i }).first();
  if (await rollbackBtn.isVisible()) {
    await rollbackBtn.click();
    const reasonInput = page.locator('input[name="reason"], textarea[name="reason"]').first();
    await reasonInput.fill("short");
    await page.getByRole("button", { name: /Xác nhận|Confirm/i }).click();
    await expect(page.locator("text=/10|ky hieu|characters/i")).toBeVisible();

    // Fill valid reason
    await reasonInput.fill("Rollback due to golden gate regression");
    await page.getByRole("button", { name: /Xác nhận|Confirm/i }).click();
    await expect(page.locator("text=/rolled_back|hoàn tác/i")).toBeVisible();
  }
});
