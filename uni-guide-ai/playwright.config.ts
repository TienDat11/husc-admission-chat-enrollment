import { defineConfig, devices } from "@playwright/test";

const WEB_PORT = 8080;
const API_PORT = 8000;
const baseURL = process.env.E2E_BASE_URL ?? `http://localhost:${WEB_PORT}`;
const apiURL = `http://localhost:${API_PORT}`;
const databaseURL =
  process.env.E2E_DATABASE_URL ?? "postgresql+asyncpg://rag:rag@127.0.0.1:5432/husc";
const pythonBin = process.env.E2E_PYTHON ?? "python";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: "list",
  use: {
    baseURL,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "desktop",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 800 } },
    },
    {
      name: "mobile",
      use: { ...devices["Desktop Chrome"], viewport: { width: 390, height: 844 } },
    },
    {
      name: "mobile-320",
      use: { ...devices["Desktop Chrome"], viewport: { width: 320, height: 640 } },
    },
  ],
  webServer: [
    {
      command: `${pythonBin} scripts/seed_e2e.py && ${pythonBin} -m uvicorn src.main:app --port ${API_PORT}`,
      cwd: "../rag2025",
      url: `${apiURL}/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 240_000,
      env: {
        PYTHONPATH: ".",
        STORE_BACKEND: "postgres",
        DATABASE_URL: databaseURL,
      },
    },
    {
      command: `npm run build && npm run preview -- --port ${WEB_PORT} --strictPort`,
      url: baseURL,
      reuseExistingServer: !process.env.CI,
      timeout: 300_000,
      env: { VITE_API_BASE_URL: apiURL },
    },
  ],
});
