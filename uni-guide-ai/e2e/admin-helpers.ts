import { expect, type Page } from "@playwright/test";

export const E2E_PASSWORD = process.env.E2E_PASSWORD ?? "e2e-password-123";
export const API_URL = process.env.E2E_API_URL ?? "http://localhost:8000";

/** Signs in through the real login form and waits until the admin area loads. */
export async function loginAs(page: Page, actorId: string, password = E2E_PASSWORD): Promise<void> {
  await page.goto("/quan-ly/dang-nhap");
  await page.locator("#login-actor").fill(actorId);
  await page.locator("#login-password").fill(password);
  await page.getByRole("button", { name: "Đăng nhập" }).click();
  await expect(page).not.toHaveURL(/dang-nhap/);
}

/** A PDF whose content differs per call, so the duplicate check never fires. */
export function uniquePdf(): { name: string; buffer: Buffer } {
  const stamp = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const body = [
    "%PDF-1.4",
    "1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj",
    "2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj",
    "3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 144]/Contents 4 0 R/Resources<<>>>>endobj",
    `% e2e ${stamp}`,
    "4 0 obj<</Length 44>>stream",
    "BT /F1 12 Tf 72 712 Td (Hello) Tj ET",
    "endstream endobj",
    "trailer<</Root 1 0 R>>",
    "%%EOF",
  ].join("\n");
  return { name: `e2e-${stamp}.pdf`, buffer: Buffer.from(body, "latin1") };
}
