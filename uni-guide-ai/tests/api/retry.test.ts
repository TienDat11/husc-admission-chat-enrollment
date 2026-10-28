// @spec(plan-warmup-ux-20260705-111031 §3.2)
// api.ts — retry wrapper + classifyFetchError contract.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  queryStream,
  sendChatMessage,
  sendChatMessageWithRetry,
  classifyFetchError,
  type QueryError,
} from '@/lib/api';

interface FetchCallArgs {
  url: string;
  init?: RequestInit;
}

function jsonResponse(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', ...headers },
  });
}

function sseFrame(event: string, data: unknown): Uint8Array {
  const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  return new TextEncoder().encode(payload);
}

function sseStream(frames: Uint8Array[]): ReadableStream<Uint8Array> {
  let i = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i < frames.length) {
        controller.enqueue(frames[i++]);
      } else {
        controller.close();
      }
    },
  });
}

const originalFetch = globalThis.fetch;

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('classifyFetchError', () => {
  it('maps TypeError("Failed to fetch") to kind=network, retryable=true', () => {
    const err = new TypeError('Failed to fetch');
    const cls = classifyFetchError(err);
    expect(cls.kind).toBe('network');
    expect(cls.retryable).toBe(true);
  });

  it('maps AbortError to kind=network, retryable=true', () => {
    const err = new DOMException('aborted', 'AbortError');
    const cls = classifyFetchError(err);
    expect(cls.kind).toBe('network');
    expect(cls.retryable).toBe(true);
  });

  it('maps HTTP 503 with Retry-After to kind=warming, retryable=true, retryAfterMs', () => {
    const tagged = new Error('warming') as Error & { status: number; headers: Headers };
    tagged.status = 503;
    tagged.headers = new Headers({ 'Retry-After': '5' });
    const cls = classifyFetchError(tagged);
    expect(cls.kind).toBe('warming');
    expect(cls.retryable).toBe(true);
    expect(cls.retryAfterMs).toBe(5000);
  });

  it('maps HTTP 401 / 403 to kind=unauthorized, retryable=false', () => {
    for (const s of [401, 403]) {
      const tagged = new Error('forbidden') as Error & { status: number };
      tagged.status = s;
      const cls = classifyFetchError(tagged);
      expect(cls.kind).toBe('unauthorized');
      expect(cls.retryable).toBe(false);
    }
  });

  it('maps HTTP 5xx (not 503) to kind=server, retryable=false (Risk 3 mitigation)', () => {
    for (const s of [500, 502, 504]) {
      const tagged = new Error('oops') as Error & { status: number };
      tagged.status = s;
      const cls = classifyFetchError(tagged);
      expect(cls.kind).toBe('server');
      expect(cls.retryable).toBe(false);
    }
  });
});

describe('queryStream — bounded retry wrapper', () => {
  it('recovers after one TypeError network failure (1 fail → 200)', async () => {
    const fetchMock = vi
      .fn<Promise<Response>, [FetchCallArgs]>()
      .mockImplementationOnce(() => Promise.reject(new TypeError('Failed to fetch')))
      .mockImplementationOnce(() =>
        Promise.resolve(
          new Response(sseStream([sseFrame('done', { answer: 'OK' })]), {
            status: 200,
            headers: { 'content-type': 'text/event-stream' },
          }),
        ),
      );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const onDelta = vi.fn();
    const onDone = vi.fn();
    await queryStream('hi?', { onDelta, onDone });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(onDone).toHaveBeenCalledWith('OK');
  });

  it('throws QueryError{kind=network} after two consecutive network failures', async () => {
    const fetchMock = vi
      .fn<Promise<Response>, [FetchCallArgs]>()
      .mockImplementation(() => Promise.reject(new TypeError('Failed to fetch')));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    let thrown: unknown = null;
    try {
      await queryStream('hi?', { onDone: vi.fn() });
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toMatchObject({ kind: 'network', retryable: false });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('retries on 503 (warming) and recovers on second attempt', async () => {
    const fetchMock = vi
      .fn<Promise<Response>, [FetchCallArgs]>()
      .mockImplementationOnce(() =>
        Promise.resolve(
          new Response(JSON.stringify({ detail: 'warming' }), {
            status: 503,
            headers: { 'Retry-After': '2', 'content-type': 'application/json' },
          }),
        ),
      )
      .mockImplementationOnce(() =>
        Promise.resolve(
          new Response(sseStream([sseFrame('done', { answer: 'pong' })]), {
            status: 200,
            headers: { 'content-type': 'text/event-stream' },
          }),
        ),
      );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const onDone = vi.fn();
    await queryStream('ping', { onDone });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(onDone).toHaveBeenCalledWith('pong');
  });

  it('does NOT retry on 401 — throws kind=unauthorized immediately', async () => {
    const fetchMock = vi
      .fn<Promise<Response>, [FetchCallArgs]>()
      .mockImplementation(() =>
        Promise.resolve(jsonResponse({ detail: 'auth required' }, 401)),
      );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    let thrown: unknown = null;
    try {
      await queryStream('hi', { onDone: vi.fn() });
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toMatchObject({ kind: 'unauthorized' });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('does NOT retry on 500 — throws kind=server immediately (Risk 3 mitigation)', async () => {
    const fetchMock = vi
      .fn<Promise<Response>, [FetchCallArgs]>()
      .mockImplementation(() => Promise.resolve(jsonResponse({ detail: 'boom' }, 500)));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    let thrown: unknown = null;
    try {
      await queryStream('hi', { onDone: vi.fn() });
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toMatchObject({ kind: 'server' });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('backoff: elapsed time between attempts is at least ~1000ms', async () => {
    const fetchMock = vi
      .fn<Promise<Response>, [FetchCallArgs]>()
      .mockImplementationOnce(() => Promise.reject(new TypeError('Failed to fetch')))
      .mockImplementationOnce(() =>
        Promise.resolve(
          new Response(sseStream([sseFrame('done', { answer: 'OK' })]), {
            status: 200,
            headers: { 'content-type': 'text/event-stream' },
          }),
        ),
      );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const start = Date.now();
    await queryStream('hi', { onDone: vi.fn() });
    const elapsed = Date.now() - start;
    // Real timers here — we don't fake the fetch resolution, only sleep.
    // Allow a generous lower bound since the real sleep is 1000ms + setup.
    expect(elapsed).toBeGreaterThanOrEqual(950);
  });
});

describe('sendChatMessage — retry wrapper (non-stream)', () => {
  it('recovers after one network failure', async () => {
    const fetchMock = vi
      .fn<Promise<Response>, [FetchCallArgs]>()
      .mockImplementationOnce(() => Promise.reject(new TypeError('Failed to fetch')))
      .mockImplementationOnce(() =>
        Promise.resolve(jsonResponse({ answer: 'recovered', status_code: 'SUCCESS' })),
      );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const out = await sendChatMessage('hello');
    expect(out.answer).toBe('recovered');
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('throws classified QueryError after both attempts fail', async () => {
    const fetchMock = vi
      .fn<Promise<Response>, [FetchCallArgs]>()
      .mockImplementation(() => Promise.reject(new TypeError('Failed to fetch')));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    let thrown: unknown = null;
    try {
      await sendChatMessageWithRetry('hello');
    } catch (e) {
      thrown = e;
    }
    const cls = thrown as QueryError;
    expect(cls.kind).toBe('network');
    expect(cls.retryable).toBe(false);
  });
});
