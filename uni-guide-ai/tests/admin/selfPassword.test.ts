import { afterEach, describe, expect, it, vi } from "vitest";
import { patchIdentity } from "@/lib/adminApi";

// The backend body for PATCH /admin/identities/{subject_id} is IdentityPatchRequest:
// role, disabled, password (all optional) and reason (required). The self password
// change in Accounts.tsx must send only keys from that set, with reason and password.
const IDENTITY_PATCH_KEYS = ["disabled", "password", "reason", "role"];

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("self password change request shape", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("PATCHes the own subject with reason and password only", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({ subject_id: "s-1", actor_id: "biensoan", role: "editor", disabled: false }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await patchIdentity(null, "s-1", { reason: "Đổi mật khẩu cá nhân", password: "Mat-khau-moi-1" });

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(String(url)).toMatch(/\/admin\/identities\/s-1$/);
    expect(init.method).toBe("PATCH");
    const body = JSON.parse(String(init.body));
    expect(body).toEqual({ reason: "Đổi mật khẩu cá nhân", password: "Mat-khau-moi-1" });
    for (const key of Object.keys(body)) {
      expect(IDENTITY_PATCH_KEYS).toContain(key);
    }
  });
});
