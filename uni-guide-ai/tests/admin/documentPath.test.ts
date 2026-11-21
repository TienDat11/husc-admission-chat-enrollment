import { afterEach, describe, expect, it, vi } from "vitest";
import { listExceptions, patchBlock } from "@/lib/adminApi";

// Upload ids look like "uploads/<sha256>.pdf". The backend routes take the id as
// path segments, so an encoded "%2F" is never routed and answers 404.
const UPLOAD_ID = "uploads/89bf979001d659eb382c8ea1240329585940032892cefb095ad614b7e59e3bd5.pdf";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("document ids that contain a slash", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("requests the exceptions list with the slash kept as a path separator", async () => {
    const fetchMock = vi.fn(async () => jsonResponse({ document_id: UPLOAD_ID, blocks: [], unanchored: [] }));
    vi.stubGlobal("fetch", fetchMock);

    await listExceptions(null, UPLOAD_ID);

    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toMatch(/\/admin\/source-documents\/uploads\/89bf[0-9a-f]+\.pdf\/exceptions$/);
    expect(url).not.toContain("%2F");
  });

  it("PATCHes a block with the slash kept as a path separator", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({ block_id: "b-1", state: "fixed", cleared_blockers: [], record: null }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await patchBlock(null, UPLOAD_ID, "b-1", { content: "x", expected_version: 1 });

    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toMatch(/\/admin\/source-documents\/uploads\/89bf[0-9a-f]+\.pdf\/blocks\/b-1$/);
    expect(url).not.toContain("%2F");
  });
});
