/**
 * Ticket 42 acceptance: the cross-stack CRUD smoke.
 *
 * Everything the FE admin surface claims about the contract is driven here
 * through the REAL client functions in src/lib/adminApi.ts against a REAL
 * FastAPI server (scripts/admin_contract_server.py under uvicorn,
 * booted by scripts/admin-crud-smoke.mjs). No fetch is stubbed anywhere in
 * this file: the request goes over a real socket, the auth layer runs, the
 * state machine decides, and the bytes land on disk.
 *
 * If the orchestrator did not set VITE_API_BASE_URL / ADMIN_SMOKE_TOKENS the
 * server is not there - that is a hard failure, never a skip. (The guard test
 * in tests/contract/smoke-launcher.test.ts proves that path exits non-zero.)
 */
import { beforeAll, describe, expect, it } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import {
  ADMIN_TOKEN_HEADER,
  IDEMPOTENCY_KEY_HEADER,
  createRecord,
  editRecord,
  fetchAdminIdentity,
  fetchRecord,
  listRecords,
  transitionRecord,
  type LifecycleWriteResult,
} from '../src/lib/adminApi';

const BASE_URL = process.env.VITE_API_BASE_URL;
const TOKENS: Record<string, string> = (() => {
  try {
    return JSON.parse(process.env.ADMIN_SMOKE_TOKENS ?? '');
  } catch {
    return {};
  }
})();
const STORE_ROOT = process.env.ADMIN_SMOKE_STORE_ROOT;

function requireEnv(): void {
  const missing = [
    !BASE_URL && 'VITE_API_BASE_URL',
    !['editor', 'reviewer', 'publisher'].find((r) => TOKENS[r]) &&
      'ADMIN_SMOKE_TOKENS (editor/reviewer/publisher)',
    !STORE_ROOT && 'ADMIN_SMOKE_STORE_ROOT',
  ].filter(Boolean);
  if (missing.length) {
    throw new Error(
      `smoke prerequisites missing: ${missing.join(', ')} - run via ` +
        '`npm run test:smoke`, which boots the real FastAPI server first',
    );
  }
}

// Same identity the BE fixture (tests/routers/test_admin_records_api.py) uses
// across the HTTP boundary: a CHI_TIEU record with full provenance, so the
// publish gate has what it needs.
const REC_ID = 'CHI_TIEU:nganh:7220104:2026:smoke';
function recordBody(text: string): Record<string, unknown> {
  return {
    record_id: REC_ID,
    record_type: 'CHI_TIEU',
    version: 1,
    admission_year: 2026,
    effective_from: '2026-01-01',
    effective_to: '2026-12-31',
    status: 'draft',
    authority_level: 'husc_official',
    source_document_id: 'de-an-tuyen-sinh-2026',
    source_url: 'https://tuyensinh.husc.edu.vn/de-an-2026.pdf',
    source_page: 24,
    text,
  };
}

beforeAll(requireEnv);

describe('admin CRUD over the real wire (FE client -> FastAPI)', () => {
  it('sends the real X-Admin-Token header: unknown credential is refused by the server', async () => {
    // The client attaches the header itself (adminRequest); an unregistered
    // token must fail with the server's own status code - src/routers/
    // admin_auth.py answers 403 "unknown admin credential" - not a mocked one.
    await expect(fetchAdminIdentity('tok-not-registered-anywhere')).rejects.toMatchObject({
      status: 403,
    });
  });

  it('GET /admin/identity returns the server-decided role and grants', async () => {
    const who = await fetchAdminIdentity(TOKENS.editor);
    expect(who.actor_id).toBe('smoke-editor');
    expect(who.role).toBe('editor');
    expect(who.actions).toContain('records.create');
    expect(who.actions).not.toContain('records.publish');
  });

  it('createRecord -> the id the client parses comes from the real response', async () => {
    const created = await createRecord(TOKENS.editor, recordBody('Chỉ tiêu tuyển sinh 2026.'));
    // The mutation this ticket exists for: if the backend renames the field
    // the client parses (`result.record.record_id`), this assertion is RED -
    // the fetch-mocked suite could never see it.
    expect(created.record.record_id).toBe(REC_ID);
    expect(created.record.version).toBe(1);
    expect(created.record.status).toBe('draft');
    expect(created.event.action).toBe('create');
    expect(created.event.actor_id).toBe('smoke-editor');
  });

  it('editRecord appends a version (v1 bytes stay on disk untouched)', async () => {
    const edited = await editRecord(TOKENS.editor, REC_ID, {
      expected_version: 1,
      text: 'Chỉ tiêu tuyển sinh 2026 (da ra soat).',
    });
    expect(edited.record.version).toBe(2);
    expect(edited.record.status).toBe('draft');
    expect(edited.event.action).toBe('edit');

    // Read the real store files the server wrote: the edit appended, it did
    // not rewrite (CanonicalRecordStore layout: records/rec-<id>_r/vNNNN.json).
    const files = readdirSync(storeRecordDir());
    expect(files).toContain('v0001.json');
    expect(files).toContain('v0002.json');
    const v1 = JSON.parse(readFileSync(resolve(storeRecordDir(), 'v0001.json'), 'utf-8'));
    expect(v1.text).toBe('Chỉ tiêu tuyển sinh 2026.');
  });

  /** Directory the server keeps this record's version files in. */
  function storeRecordDir(): string {
    const dir = resolve(STORE_ROOT!, 'records');
    const recordDirs = readdirSync(dir).filter((name) =>
      name.includes(encodeURIComponent(REC_ID)),
    );
    expect(recordDirs).toHaveLength(1);
    return resolve(dir, recordDirs[0]);
  }

  // Bytes of the first published version, captured at publish time; the later
  // edit must not touch them (see the lifecycle + RBAC tests below).
  let publishedBytes: string;

  it('a stale expected_version is refused 409 by the state machine', async () => {
    await expect(
      editRecord(TOKENS.editor, REC_ID, { expected_version: 1, text: 'ban cu' }),
    ).rejects.toMatchObject({ status: 409 });
  });

  it('walks the lifecycle: submit -> approve -> publish', async () => {
    const submitted = await transitionRecord(TOKENS.editor, REC_ID, {
      action: 'submit',
      expected_version: 2,
    });
    expect(submitted.record.status).toBe('review');
    expect(submitted.record.version).toBe(3);

    const approved = await transitionRecord(TOKENS.reviewer, REC_ID, {
      action: 'approve',
      expected_version: 3,
    });
    expect(approved.record.status).toBe('approved');
    expect(approved.record.version).toBe(4);

    const published = await transitionRecord(TOKENS.publisher, REC_ID, {
      action: 'publish',
      expected_version: 4,
      reason: 'du dieu kien cong bo',
    });
    expect(published.record.status).toBe('published');
    expect(published.record.version).toBe(5);

    // The published bytes are on disk NOW; the RBAC test edits this record
    // next, and that edit must append v6 without rewriting v5.
    publishedBytes = readFileSync(resolve(storeRecordDir(), 'v0005.json'), 'utf-8');
    expect(JSON.parse(publishedBytes).status).toBe('published');
  });

  it('RBAC is enforced server-side: roles without the grant are refused 403', async () => {
    // Editor lacks records.publish; reviewer lacks records.publish.
    await expect(
      transitionRecord(TOKENS.editor, REC_ID, { action: 'publish', expected_version: 5 }),
    ).rejects.toMatchObject({ status: 403 });
    await expect(
      transitionRecord(TOKENS.reviewer, REC_ID, { action: 'publish', expected_version: 5 }),
    ).rejects.toMatchObject({ status: 403 });
    // The editor also cannot approve (no records.approve grant): after its own
    // new edit+submit, an approve attempt is still refused by the server.
    await editRecord(TOKENS.editor, REC_ID, { expected_version: 5, text: 'ban khac' });
    // Publishing is immutable: the v5 bytes are untouched and v6 is a draft.
    expect(readFileSync(resolve(storeRecordDir(), 'v0005.json'), 'utf-8')).toBe(
      publishedBytes,
    );
    const v6 = JSON.parse(readFileSync(resolve(storeRecordDir(), 'v0006.json'), 'utf-8'));
    expect(v6.status).toBe('draft');
    await transitionRecord(TOKENS.editor, REC_ID, { action: 'submit', expected_version: 6 });
    await expect(
      transitionRecord(TOKENS.editor, REC_ID, { action: 'approve', expected_version: 7 }),
    ).rejects.toMatchObject({ status: 403 });
  });

  it('sends the real Idempotency-Key header: the server replays, does not re-apply', async () => {
    const key = `smoke-${crypto.randomUUID()}`;
    // Reviewer approves the editor's v7 -> v8 approved; publisher then publishes.
    await transitionRecord(TOKENS.reviewer, REC_ID, { action: 'approve', expected_version: 7 });
    const first = await transitionRecord(
      TOKENS.publisher,
      REC_ID,
      { action: 'publish', expected_version: 8, reason: 'cong bo' },
      { idempotencyKey: key },
    );
    expect(first.record.status).toBe('published');
    expect(first.record.version).toBe(9);

    // Raw fetch with the SAME header names the client builds, to prove they
    // are the ones the server reads (Idempotency-Key in, Idempotent-Replay
    // back) and that the retry replays the first answer instead of re-applying.
    const raw = await fetch(
      `${BASE_URL}/admin/records/${encodeURIComponent(REC_ID)}/transitions`,
      {
        method: 'POST',
        headers: {
          [ADMIN_TOKEN_HEADER]: TOKENS.publisher,
          [IDEMPOTENCY_KEY_HEADER]: key,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'publish',
          expected_version: 8,
          reason: 'cong bo',
        }),
      },
    );
    expect(raw.status).toBe(200);
    expect(raw.headers.get('idempotent-replay')).toBe('true');
    const replay = (await raw.json()) as LifecycleWriteResult;
    expect(replay).toEqual(first);
  });

  it('list + read back through the client see the published record', async () => {
    const fresh = await fetchRecord(TOKENS.editor, REC_ID);
    expect(fresh.status).toBe('published');
    expect(fresh.version).toBe(9);

    const page = await listRecords(TOKENS.reviewer, { status: 'published' });
    expect(page.records.map((r) => r.record_id)).toContain(REC_ID);
    expect(page.total).toBeGreaterThanOrEqual(1);
  });
});
