import { afterEach, describe, expect, it, vi } from "vitest";
import { IDEMPOTENCY_KEY_HEADER } from "@/lib/adminApi";
import { approveAndPublish, approveAndPublishMany } from "@/lib/teacherApi";

const RECORD_A = "DIEU_KIEN:06/2026/TT-BGDĐT:dieu_1:2026:none:all";
const RECORD_B = "DIEU_KIEN:06/2026/TT-BGDĐT:dieu_2:2026:none:all";

interface Call {
  url: string;
  action: string;
  expectedVersion: number;
  key: string | null;
}

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

/** Records every transition call; `respond` decides the server answer. */
function mockTransitions(respond: (call: Call) => Response = () => json(200, { ok: true })): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as { action: string; expected_version: number };
      const call: Call = {
        url,
        action: body.action,
        expectedVersion: body.expected_version,
        key: new Headers(init?.headers).get(IDEMPOTENCY_KEY_HEADER),
      };
      calls.push(call);
      return respond(call);
    }),
  );
  return calls;
}

const publishRefused = (call: Call): Response =>
  call.action === "publish"
    ? json(422, { detail: "idempotency key already used for a different request" })
    : json(200, { ok: true });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("approveAndPublish idempotency keys", () => {
  it("sends distinct keys for approve and publish, derived from the base key", async () => {
    const calls = mockTransitions();

    await approveAndPublish(null, RECORD_A, 2, "base");

    expect(calls.map((c) => c.action)).toEqual(["approve", "publish"]);
    expect(calls[0].key).toBe("base:approve");
    expect(calls[1].key).toBe("base:publish");
    expect(calls[0].key).not.toBe(calls[1].key);
    expect(calls.map((c) => c.expectedVersion)).toEqual([2, 3]);
  });

  it("a retry with the same base key derives the same keys, so each step replays", async () => {
    const first = mockTransitions();
    await approveAndPublish(null, RECORD_A, 2, "base");
    vi.unstubAllGlobals();

    const retry = mockTransitions();
    await approveAndPublish(null, RECORD_A, 2, "base");

    expect(retry.map((c) => c.key)).toEqual(first.map((c) => c.key));
  });

  it("reports a failed publish honestly and does not claim success", async () => {
    const calls = mockTransitions(publishRefused);

    const error = await approveAndPublish(null, RECORD_A, 2, "base").catch((e: unknown) => e);

    expect(calls.map((c) => c.action)).toEqual(["approve", "publish"]);
    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toMatch(/^Đã duyệt nhưng chưa xuất bản: /);
    expect((error as Error).message).toContain("idempotency key already used");
  });

  it("does not publish when the approve is refused, and surfaces the approve error", async () => {
    const calls = mockTransitions((call) =>
      call.action === "approve"
        ? json(409, { detail: "VERSION_CONFLICT: bản ghi đã thay đổi" })
        : json(200, { ok: true }),
    );

    const error = await approveAndPublish(null, RECORD_A, 2, "base").catch((e: unknown) => e);

    expect(calls.map((c) => c.action)).toEqual(["approve"]);
    expect((error as Error).message).toContain("VERSION_CONFLICT");
    expect((error as Error).message).not.toContain("Đã duyệt nhưng chưa xuất bản");
  });
});

describe("approveAndPublishMany idempotency keys", () => {
  it("gives each record its own keys and reports the failing record without stopping the rest", async () => {
    const calls = mockTransitions((call) =>
      call.url.includes(encodeURIComponent(RECORD_A)) ? publishRefused(call) : json(200, { ok: true }),
    );

    const result = await approveAndPublishMany(
      null,
      [
        { recordId: RECORD_A, version: 2 },
        { recordId: RECORD_B, version: 2 },
      ],
      (recordId) => `batch:${recordId}`,
    );

    const keys = calls.map((c) => c.key);
    expect(new Set(keys).size).toBe(keys.length);
    expect(keys).toContain(`${encodeURIComponent(`batch:${RECORD_A}`)}:approve`);
    expect(keys).toContain(`${encodeURIComponent(`batch:${RECORD_B}`)}:publish`);
    expect(result.succeeded).toEqual([RECORD_B]);
    expect(result.failures).toHaveLength(1);
    expect(result.failures[0].recordId).toBe(RECORD_A);
    expect(result.failures[0].error).toMatch(/^Đã duyệt nhưng chưa xuất bản: /);
  });
});
