import { defineConfig } from "@playwright/test";

// The spec mocks every /admin request, so it needs only the frontend dev server.
export default defineConfig({
  testDir: "./e2e",
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:5173",
  },
});
