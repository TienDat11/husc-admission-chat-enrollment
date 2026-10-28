import { defineConfig } from 'vitest/config';

// Ticket 42 cross-stack smoke: the client under test is the plain Node one
// (fetch + import.meta.env), no DOM is involved, and tests/setup.ts is
// jsdom-oriented, so this suite deliberately runs bare in `node`.
//
// The default vitest.config.ts includes only `tests/**` - this directory is
// OUTSIDE it on purpose: `npm test` must not require a Python process. The
// smoke runs via `npm run test:smoke`, which boots the real FastAPI server
// first (scripts/admin-crud-smoke.mjs).
export default defineConfig({
  test: {
    environment: 'node',
    include: ['tests-smoke/**/*.test.ts'],
    testTimeout: 30000,
    hookTimeout: 30000,
  },
});
