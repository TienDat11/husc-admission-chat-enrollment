import { expect, test, type Page } from "@playwright/test";

// U7 e2e: exception handling view at /quan-ly/them/ngoai-le.
// Every /admin call is route-mocked; only the frontend dev server is needed.

const DOC_ID = "THONG_TIN_TUYEN_SINH_2026.pdf";

async function mockExceptions(page: Page) {
  await page.route(/\/admin\/source-documents\/[^/]+\/exceptions$/, (route) =>
    route.fulfill({
      json: {
        document_id: DOC_ID,
        blocks: [
          {
            block_id: "block:table-5",
            document_id: DOC_ID,
            page: 5,
            ordinal: 2,
            block_type: "TABLE",
            text: "Hoc phi nganh Cong nghe thong tin 2026-2027",
            state: "open",
            warnings: ["TABLE_CELL_DROP_SUSPECTED"],
            missing_pypdf_rows: [3, 7],
            coordinate_key: null,
            version: 1,
          },
          {
            block_id: "block:ocr-9",
            document_id: DOC_ID,
            page: 9,
            ordinal: 1,
            block_type: "TEXT",
            text: "Trang anh chup khong doc duoc",
            state: "open",
            warnings: ["OCR_LOW_CONFIDENCE"],
            missing_pypdf_rows: [],
            coordinate_key: null,
            version: 1,
          },
        ],
        unanchored: [
          {
            record_id: "unanchored:X",
            version: 1,
            coordinate_key: null,
            record_type: "THONG_BAO",
            status: "archived",
            archive_reason: "withdrawn",
            text: "Ban ghi chua gan toa do",
            structured_payload: {},
            authored_by: "cb.quan-tri",
          },
        ],
      },
    }),
  );

  await page.route(/\/admin\/source-documents\/[^/]+\/blocks\/[^/]+$/, async (route) => {
    if (route.request().method() === "PATCH") {
      await route.fulfill({
        json: {
          block_id: route.request().url().split("/blocks/")[1],
          state: "fixed",
          cleared_blockers: ["TABLE_CELL_DROP_SUSPECTED", "UNANCHORED_DRAFT"],
          record: { record_id: "unanchored:X", version: 1 },
        },
      });
      return;
    }
    await route.fulfill({ status: 404 });
  });

  await page.route(/\/admin\/source-documents\/[^/]+\/blocks\/[^/]+\/insert-missing-rows$/, (route) =>
    route.fulfill({
      json: { block_id: "block:table-5", inserted: 2, state: "fixed", cleared_blockers: ["TABLE_CELL_DROP_SUSPECTED"] },
    }),
  );

  await page.route(/\/admin\/source-documents\/[^/]+\/reocr$/, (route) =>
    route.fulfill({ status: 202, json: { job_id: "job-reocr-1", state: "queued", page_numbers: [5, 9] } }),
  );

  await page.route(/\/admin\/previews$/, (route) =>
    route.fulfill({
      json: {
        preview_id: "preview-1",
        record_id: "CHI_TIEU:nganh:cntt:2026",
        record_version: 1,
        total_chunks: 3,
        total_tokens: 300,
        histogram: { "<50": 1, "50-249": 1, "250-420": 1, "421-512": 0, ">512": 0 },
        items: [
          { chunk_id: "c1", record_id: "r", record_version: 1, ordinal: 0, text: "Chunk mot", token_count: 45, metadata: {} },
          { chunk_id: "c2", record_id: "r", record_version: 1, ordinal: 1, text: "Chunk hai", token_count: 120, metadata: {} },
          { chunk_id: "c3", record_id: "r", record_version: 1, ordinal: 2, text: "Chunk ba", token_count: 135, metadata: {} },
        ],
        warnings: [],
      },
    }),
  );

  await page.route(/\/admin\/records\/[^/]+\/transitions$/, async (route) => {
    if (route.request().method() === "POST") {
      const body = await route.request().postBodyJSON();
      if (body.action === "submit") {
        await route.fulfill({ json: { record_id: DOC_ID, version: 1, status: "review" } });
        return;
      }
    }
    await route.fulfill({ status: 404 });
  });

  await page.route(/\/admin\/identity$/, (route) =>
    route.fulfill({ json: { actor_id: "cb.quan-tri", role: "admin", actions: ["records.edit", "records.read"] } }),
  );
}

test("exception list shows blockers and submit is blocked", async ({ page }) => {
  await mockExceptions(page);
  await page.goto(`/quan-ly/them/ngoai-le?documentId=${DOC_ID}`);

  await expect(page.getByRole("heading", { name: /Xử lý ngoại lệ tài liệu/ })).toBeVisible();
  await expect(page.getByText("Bảng thiếu dòng")).toBeVisible();
  await expect(page.getByText("OCR thấp")).toBeVisible();
  await expect(page.getByRole("button", { name: /Chưa thể gửi duyệt/ })).toBeDisabled();
});

test("1-click insert missing rows clears TABLE_CELL_DROP_SUSPECTED", async ({ page }) => {
  await mockExceptions(page);
  await page.goto(`/quan-ly/them/ngoai-le?documentId=${DOC_ID}`);

  await page.getByRole("button", { name: /Chèn dòng thiếu/ }).click();
  await expect(page.getByText("Bảng thiếu dòng")).not.toBeVisible();
});

test("Excel paste modal opens, parses TSV, and submits", async ({ page }) => {
  await mockExceptions(page);
  await page.goto(`/quan-ly/them/ngoai-le?documentId=${DOC_ID}`);

  // Open the paste modal via the block row action
  await page.getByRole("button", { name: /Dán từ Excel/ }).click();
  await expect(page.getByRole("dialog")).toBeVisible();

  // Paste TSV
  const textarea = page.getByLabel("Dán dữ liệu Excel (TSV)");
  await textarea.click();
  await textarea.press("Control+a");
  await textarea.type("Ngành\tHọc phí 2026\tTrạng thái\nCông nghệ\t18.500.000\tCập nhật\n");

  await page.getByRole("button", { name: /Luu bang/ }).click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
});

test("Excel paste modal: Esc restores focus to trigger", async ({ page }) => {
  await mockExceptions(page);
  await page.goto(`/quan-ly/them/ngoai-le?documentId=${DOC_ID}`);

  const trigger = page.getByRole("button", { name: /Dán từ Excel/ });
  await trigger.click();
  await expect(page.getByRole("dialog")).toBeVisible();

  await page.press("body", "Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await expect(trigger).toBeFocused();
});

test("chunk preview histogram renders with warnings", async ({ page }) => {
  await mockExceptions(page);
  await page.goto(`/quan-ly/them/ngoai-le?documentId=${DOC_ID}`);

  await page.getByRole("button", { name: /Chạy xem trước/ }).click();
  await expect(page.getByText(/tổng chunk/)).toBeVisible();
  await expect(page.getByText(/300 token/)).toBeVisible();
});

test("submit unblocked when all blockers resolved", async ({ page }) => {
  await mockExceptions(page);
  await page.goto(`/quan-ly/them/ngoai-le?documentId=${DOC_ID}`);

  // Resolve all blockers first
  await page.getByRole("button", { name: /Chèn dòng thiếu/ }).click();
  await page.getByRole("button", { name: /Gán tọa độ/ }).click();

  await expect(page.getByRole("button", { name: /Gửi duyệt/ })).toBeEnabled();
  await page.getByRole("button", { name: /Gửi duyệt/ }).click();
  await expect(page.getByText("Đã gửi duyệt")).toBeVisible();
});
