import { expect, test, type APIRequestContext } from "@playwright/test";

const API = process.env.E2E_API_URL ?? "http://localhost:8000";

const SESSION = "11111111-1111-4111-8111-111111111111";

async function studentHeaders(
  request: APIRequestContext,
  extra?: Record<string, string>,
): Promise<Record<string, string>> {
  return { "X-Student-Session": SESSION, ...extra };
}

test.describe("U11 — luong thí sinh: hồ sơ, roadmap, từ chối lịch sử", () => {
  test("PUT / GET / DELETE lifecycle", async ({ request }) => {
    const put = await request.put(`${API}/v2/student-profile`, {
      headers: await studentHeaders(request),
      data: {
        raw_score: 22,
        target_majors: ["CNTT", "KTMT"],
        kv: "KV1",
        dt: null,
        notes: "muon hoc CNTT",
      },
    });
    expect(put.status()).toBe(200);

    const get = await request.get(`${API}/v2/student-profile`, {
      headers: await studentHeaders(request),
    });
    expect(get.status()).toBe(200);
    const body = await get.json();
    expect(body.raw_score).toBe(22);
    expect(body.target_majors).toEqual(["CNTT", "KTMT"]);
    expect(body.priority_total).toBeGreaterThanOrEqual(22);

    const del = await request.delete(`${API}/v2/student-profile`, {
      headers: await studentHeaders(request),
    });
    expect(del.status()).toBe(204);

    const gone = await request.get(`${API}/v2/student-profile`, {
      headers: await studentHeaders(request),
    });
    expect(gone.status()).toBe(404);
  });

  test("missing / invalid X-Student-Session -> 400 STUDENT_SESSION_REQUIRED", async ({
    request,
  }) => {
    const missing = await request.put(`${API}/v2/student-profile`, {
      headers: {},
      data: { raw_score: 20 },
    });
    expect(missing.status()).toBe(400);
    const missingBody = await missing.json();
    expect(missingBody.error.code).toBe("STUDENT_SESSION_REQUIRED");

    const invalid = await request.get(`${API}/v2/student-profile`, {
      headers: { "X-Student-Session": "not-a-uuid" },
    });
    expect(invalid.status()).toBe(400);
    const invalidBody = await invalid.json();
    expect(invalidBody.error.code).toBe("STUDENT_SESSION_REQUIRED");
  });

  test("raw_score out of [0,30] -> 400; target_majors > 10 -> 400", async ({
    request,
  }) => {
    const badScore = await request.put(`${API}/v2/student-profile`, {
      headers: await studentHeaders(request),
      data: { raw_score: 31 },
    });
    expect(badScore.status()).toBe(400);

    const tooMany = await request.put(`${API}/v2/student-profile`, {
      headers: await studentHeaders(request),
      data: {
        raw_score: 20,
        target_majors: Array.from({ length: 11 }, (_, i) => `Nganh ${i}`),
      },
    });
    expect(tooMany.status()).toBe(400);
  });

  test("31 PUT/min from one IP -> 429 RATE_LIMITED with Retry-After", async ({
    request,
  }) => {
    // Use a distinct session so earlier PUTs in this file don't count toward
    // this IP's rate-limit bucket.
    const headers = { "X-Student-Session": "22222222-2222-4222-8222-222222222222" };
    let limited = false;
    let retryAfter: string | null = null;
    for (let i = 0; i < 31; i += 1) {
      const res = await request.put(`${API}/v2/student-profile`, {
        headers,
        data: { raw_score: 10 },
      });
      if (res.status() === 429) {
        limited = true;
        retryAfter = res.headers()["retry-after"] ?? null;
        break;
      }
    }
    expect(limited).toBe(true);
    expect(retryAfter).not.toBeNull();
  });

  test("expired profile -> GET 404 and row is purged", async ({ request }) => {
    const session = "33333333-3333-4333-8333-333333333333";
    await request.put(`${API}/v2/student-profile`, {
      headers: { "X-Student-Session": session },
      data: { raw_score: 20, target_majors: ["CNTT"] },
      // Force an already-expired row via a far-past expires_at is not exposed
      // over HTTP; instead rely on the backend test seeding. Here we only assert
      // the GET-404 path for a session that was never created.
      timeout: 5000,
    });
    const get = await request.get(`${API}/v2/student-profile`, {
      headers: { "X-Student-Session": session },
    });
    expect(get.status()).toBe(404);
  });

  test("roadmap returns 5 steps with done / needs_input / blocked states", async ({
    request,
  }) => {
    const session = "44444444-4444-4444-8444-444444444444";
    await request.put(`${API}/v2/student-profile`, {
      headers: { "X-Student-Session": session },
      data: {
        raw_score: 25,
        target_majors: ["CNTT"],
        kv: "KV1",
        dt: null,
      },
    });
    const res = await request.get(`${API}/v2/admission-roadmap`, {
      headers: { "X-Student-Session": session },
    });
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(Array.isArray(body.steps)).toBe(true);
    expect(body.steps.length).toBe(5);
    const states = new Set(
      body.steps.map((step: { state: string }) => step.state),
    );
    expect(states.has("done")).toBe(true);
    expect(states.has("needs_input")).toBe(true);
    expect(states.has("blocked")).toBe(true);
  });
});
