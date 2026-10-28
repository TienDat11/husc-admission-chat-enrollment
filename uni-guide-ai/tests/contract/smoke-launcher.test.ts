/**
 * Ticket 42 guard: the smoke orchestrator must FAIL LOUDLY, never skip.
 *
 * The whole point of the cross-stack smoke is that a missing/broken backend
 * turns it RED. That only holds if scripts/admin-crud-smoke.mjs exits non-zero
 * when the server cannot start - not "skip because no server". This test runs
 * the orchestrator with a python binary that does not exist and asserts it
 * comes back failed with a reason, without ever booting a real server (fast).
 *
 * It lives in the DEFAULT suite (tests/**), so `npm test` catches a regression
 * where someone makes the orchestrator tolerant of a missing backend.
 */
import { describe, expect, it } from 'vitest';
import { execFile } from 'node:child_process';
import { resolve } from 'node:path';
import { promisify } from 'node:util';

const run = promisify(execFile);

const feRoot = resolve(__dirname, '..', '..');
const orchestrator = resolve(feRoot, 'scripts', 'admin-crud-smoke.mjs');

describe('admin-crud-smoke orchestrator fail-loud contract', () => {
  it('exits non-zero with a reason when the server cannot start', async () => {
    // A python path that cannot be spawned: the orchestrator must not swallow
    // the failure or report success.
    const { code, stderr } = await run(
      process.execPath,
      [orchestrator],
      {
        cwd: feRoot,
        env: {
          ...process.env,
          ADMIN_SMOKE_PYTHON: resolve(feRoot, '__no_such_python__.exe'),
        },
        // execFile rejects on non-zero; capture it and assert instead.
      },
    ).then(
      () => ({ code: 0, stderr: '' }),
      (err: { code?: number; stderr?: string }) => ({
        code: err.code ?? 1,
        stderr: err.stderr ?? '',
      }),
    );

    expect(code, 'orchestrator exited 0 despite a server that could not start').not.toBe(0);
    expect(stderr).toMatch(/FAIL/i);
    expect(stderr.toLowerCase()).toMatch(/python|spawn|ENOENT/);
  }, 30000);
});
