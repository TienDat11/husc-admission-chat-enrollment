import { afterEach, describe, expect, it, vi } from "vitest";
import { AdminApiError, IDEMPOTENCY_KEY_HEADER } from "@/lib/adminApi";
import { forEachLimited, sendDraftForReview } from "@/lib/draftSubmit";

const RECORD_ID = "DIEU_KIEN:06/2026/TT-BGDĐT:dieu_1:2026:none:all";
const ENCODED = encodeURIComponent(RECORD_ID);

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("sendDraftForReview (upload-result submit flow)", () => {
  it("reads the current version, then submits with it, an idempotency key, and an encoded id", async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) =>
      init?.method === "POST"
        ? json(200, { record_id: RECORD_ID, status: "review" })
        : json(200, { record_id: RECORD_ID, version: 3, status: "draft" }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await sendDraftForReview(RECORD_ID);

    expect(fetchMock).toHaveBeenCalledTimes(2);
    const [getUrl] = fetchMock.mock.calls[0] as [string, RequestInit?];
    expect(getUrl).toContain(`/admin/records/${ENCODED}`);
    const [postUrl, postInit] = fetchMock.mock.calls[1] as [string, RequestInit];
    expect(postUrl).toContain(`/admin/records/${ENCODED}/transitions`);
    expect(JSON.parse(String(postInit.body))).toEqual({ action: "submit", expected_version: 3 });
    expect(new Headers(postInit.headers).get(IDEMPOTENCY_KEY_HEADER)).toBeTruthy();
  });

  it("surfaces a server refusal (blocked draft) as AdminApiError instead of swallowing it", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init?: RequestInit) =>
        init?.method === "POST"
          ? json(422, { detail: "bản nháp chưa neo vào Điều hoặc mã ngành (UNANCHORED_DRAFT)" })
          : json(200, { record_id: RECORD_ID, version: 1, status: "draft" }),
      ),
    );

    const error = await sendDraftForReview(RECORD_ID).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(AdminApiError);
    expect((error as AdminApiError).message).toContain("UNANCHORED_DRAFT");
  });
});

describe("forEachLimited (batch send concurrency)", () => {
  it("never has more than the limit in flight, and visits every id once", async () => {
    const ids = ["a", "b", "c", "d", "e", "f", "g"];
    const visited: string[] = [];
    let inFlight = 0;
    let peak = 0;
    await forEachLimited(ids, 3, async (id) => {
      inFlight += 1;
      peak = Math.max(peak, inFlight);
      await Promise.resolve();
      visited.push(id);
      inFlight -= 1;
    });
    expect(peak).toBe(3);
    expect([...visited].sort()).toEqual(ids);
  });

  it("starts fewer lanes than the limit when there are fewer ids, and resolves on empty input", async () => {
    let peak = 0;
    let inFlight = 0;
    const task = async () => {
      inFlight += 1;
      peak = Math.max(peak, inFlight);
      await Promise.resolve();
      inFlight -= 1;
    };
    await forEachLimited(["x", "y"], 3, task);
    expect(peak).toBe(2);
    await expect(forEachLimited([], 3, task)).resolves.toBeUndefined();
  });
});
