#!/usr/bin/env node
/* Orchestrator for the ticket-42 cross-stack CRUD smoke.
 *
 * Starts the REAL FastAPI admin app as a subprocess (uvicorn on an ephemeral
 * port, via scripts/admin_contract_server.py), mints random admin
 * credentials, points the REAL FE client at the socket, and runs the smoke
 * test with vitest. Nothing here mocks: the client's fetch hits a real HTTP
 * server with auth enforced.
 *
 * Fail-loud contract (the whole point of ticket 42): every step that cannot
 * complete - no python, server dies, no readiness line within the timeout,
 * vitest fails - exits NON-ZERO with the reason on stderr. There is no path
 * through this script that skips the smoke and reports success.
 *
 * Env overrides (used by the CI workflow and by the guard test that proves
 * the fail-loud behaviour):
 *   ADMIN_SMOKE_PYTHON   python binary (default: repo .venv on win32, else python)
 *   ADMIN_SMOKE_SERVER   launcher script (default: scripts/admin_contract_server.py)
 *   ADMIN_SMOKE_ARGS     extra args appended to the launcher command
 *   RAG_SRC_ROOT         passed through to the launcher (alternate app source)
 */
import { spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { tmpdir } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const feRoot = resolve(here, '..');
const repoRoot = resolve(feRoot, '..');

function fail(reason) {
  console.error(`[admin-crud-smoke] FAIL: ${reason}`);
  process.exit(1);
}

const python =
  process.env.ADMIN_SMOKE_PYTHON ||
  (process.platform === 'win32'
    ? resolve(repoRoot, '.venv', 'Scripts', 'python.exe')
    : 'python');
const serverScript =
  process.env.ADMIN_SMOKE_SERVER ||
  resolve(feRoot, 'scripts', 'admin_contract_server.py');

// Roles the smoke walks: editor creates/edits/submits, reviewer approves,
// publisher publishes. Tokens are random per run; only the launcher's
// identity file knows them, so a leaked run cannot be replayed.
const identities = ['editor', 'reviewer', 'publisher'].map((role) => ({
  token: randomUUID(),
  actor_id: `smoke-${role}`,
  role,
}));

const workDir = mkdtempSync(resolve(tmpdir(), 'husc-smoke-'));
const identityFile = resolve(workDir, 'identities.json');
const storeRoot = resolve(workDir, 'store');
writeFileSync(identityFile, JSON.stringify(identities));

const server = spawn(
  python,
  [serverScript, '--port', '0', ...(process.env.ADMIN_SMOKE_ARGS ?? '').split(' ').filter(Boolean)],
  {
    cwd: repoRoot,
    env: {
      ...process.env,
      ADMIN_SMOKE_IDENTITIES: identityFile,
      ADMIN_SMOKE_STORE_ROOT: storeRoot,
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  },
);
server.on('error', (err) => {
  rmSync(workDir, { recursive: true, force: true });
  fail(`cannot spawn python (${python}): ${err.message}`);
});

let stderrTail = '';
server.stderr.on('data', (chunk) => {
  stderrTail = (stderrTail + chunk.toString()).slice(-4000);
});

const READY_TIMEOUT_MS = 60_000;
let readyResolve;
let readyReject;
const ready = new Promise((res, rej) => {
  readyResolve = res;
  readyReject = rej;
});
let stdoutBuf = '';
server.stdout.on('data', (chunk) => {
  stdoutBuf += chunk.toString();
  const m = stdoutBuf.match(/^ADMIN_SMOKE_READY (\d+)\s*$/m);
  if (m) readyResolve(Number(m[1]));
});
server.on('exit', (code) => {
  readyReject(
    new Error(
      `server exited (code ${code}) before announcing readiness. stderr:\n${stderrTail}`,
    ),
  );
});

const timer = setTimeout(
  () => readyReject(new Error(`no ADMIN_SMOKE_READY line within ${READY_TIMEOUT_MS / 1000}s`)),
  READY_TIMEOUT_MS,
);

let port;
try {
  port = await ready;
} catch (err) {
  clearTimeout(timer);
  server.kill();
  rmSync(workDir, { recursive: true, force: true });
  fail(err.message);
}
clearTimeout(timer);

const vitestEntry = fileURLToPath(
  pathToFileURL(resolve(feRoot, 'node_modules', 'vitest', 'vitest.mjs')),
);
const smokeConfig = resolve(feRoot, 'vitest.smoke.config.ts');

const child = spawn(
  process.execPath,
  [vitestEntry, 'run', '--config', smokeConfig],
  {
    cwd: feRoot,
    stdio: 'inherit',
    env: {
      ...process.env,
      VITE_API_BASE_URL: `http://127.0.0.1:${port}`,
      ADMIN_SMOKE_TOKENS: JSON.stringify(
        Object.fromEntries(identities.map((i) => [i.role, i.token])),
      ),
      ADMIN_SMOKE_STORE_ROOT: storeRoot,
    },
  },
);

child.on('error', (err) => {
  server.kill();
  rmSync(workDir, { recursive: true, force: true });
  fail(`cannot spawn vitest: ${err.message}`);
});

child.on('exit', (code) => {
  server.kill();
  rmSync(workDir, { recursive: true, force: true });
  if (code !== 0) {
    fail(`smoke vitest exited ${code} against the live server on port ${port}`);
  }
  console.log(`[admin-crud-smoke] PASS (server port ${port})`);
  process.exit(0);
});
