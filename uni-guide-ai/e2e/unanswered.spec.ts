import { expect, test } from "@playwright/test";
import { loginAs } from "./admin-helpers";

const E2E_PASSWORD = process.env.E2E_PASSWORD ?? "e2e-password-123";

test("Frame 7: lists unanswered questions, shows Quá han SLA badge, out-of-scope and promote", async ({
  page,
}) => {
  await loginAs(page, "e2e-editor", E2E_PASSWORD);

  // The Frame 7 list is seeded with two deterministic questions, one overdue.
  await page.goto("/quan-ly/cau-hoi");
  await expect(
    page.getByRole("heading", { name: /Câu hỏi bot chưa trả lời được/ })
  ).toBeVisible();

  const openQuestion = page.getByText(
    "Học phí ngành Hán Nôm năm 2026 bao nhiêu?"
  );
  const overdueQuestion = page.getByText(
    "Ngày thi tuyển sinh ngành Công nghệ thông tin 2026"
  );

  await expect(openQuestion).toBeVisible();
  await expect(overdueQuestion).toBeVisible();

  // SLA badge: overdue row shows "Quá han" as text (never color alone).
  await expect(page.getByText("Quá han", { exact: true })).toBeVisible();
  await expect(page.getByText("Còn hạn", { exact: true })).toBeVisible();

  // Occurrence badge is present on rows.
  const openRow = page.getByRole("row").filter({ has: openQuestion });
  await expect(openRow.getByText("1 lần")).toBeVisible();

  // --- "Ngoài phạm vi" (out-of-scope): marks the row, it leaves the default view ---
  await openRow.getByRole("button", { name: /Ngoài phạm vi/ }).click();
  await expect(openQuestion).not.toBeVisible();

  // --- Promote-golden: answer + submit creates a draft record + golden query ---
  const overdueRow = page.getByRole("row").filter({ has: overdueQuestion });
  await overdueRow
    .getByRole("link", { name: /Trả lời & Đưa vào bộ câu hỏi chuẩn/ })
    .click();

  await expect(
    page.getByRole("heading", { name: /Trả lời & Đưa vào bộ câu hỏi chuẩn/ })
  ).toBeVisible();

  await page.getByLabel("Câu trả lời đúng").fill("Học phí Hán Nôm 2026 là …");
  await page.getByRole("button", { name: /Đưa vào bộ chuẩn/ }).click();

  await expect(
    page.getByText(/Đã đưa câu hỏi vào bộ câu hỏi chuẩn/)
  ).toBeVisible();

  // The promoted question is resolved, so it no longer appears in the open list.
  await page.getByRole("button", { name: /Về danh sách câu hỏi/ }).click();
  await expect(overdueQuestion).not.toBeVisible();
});
