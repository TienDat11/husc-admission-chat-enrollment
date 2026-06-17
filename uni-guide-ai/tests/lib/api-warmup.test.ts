/**
 * SLICE-A.LATENCY — fire-and-forget warmup contract.
 * The FE calls warmup() on chat mount to force the BE to preload the
 * Qwen3 embedding model. The contract: it must always call fetch on
 * `${API_BASE_URL}/warmup` and never throw, even on network error.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { warmup } from '@/lib/api';

describe('api.warmup', () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it('calls fetch on /warmup with GET', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200 });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await warmup();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/warmup$/);
    // Must be a GET (or undefined default) — never POST/PUT.
    const method = (init && init.method) || 'GET';
    expect(String(method).toUpperCase()).toBe('GET');
  });

  it('never throws when fetch rejects', async () => {
    globalThis.fetch = vi
      .fn()
      .mockRejectedValue(new Error('network down')) as unknown as typeof fetch;

    // The whole point: this must NOT throw.
    await expect(warmup()).resolves.toBeUndefined();
  });

  it('never throws on non-2xx responses (still 200 expected, but resilient)', async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValue({ ok: false, status: 503 }) as unknown as typeof fetch;
    await expect(warmup()).resolves.toBeUndefined();
  });
});
