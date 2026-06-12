/**
 * SLICE-B.LATENCY — fake-stream SSE contract for `queryStream`.
 *
 * The BE runs the FULL /v2 pipeline (router → booster → graph →
 * LLMGenerator with all post-guards) UNCHANGED and emits the post-guard
 * answer as `text/event-stream`. The FE contract:
 *   - First frame: `event: meta\ndata: {json of route, sources, ...}`
 *   - Then 1+ `event: delta\ndata: {"text": "<chunk>"}` frames
 *   - Final frame: `event: done\ndata: {"answer": "<full guarded answer>"}`
 *
 * The concatenation of all `delta.text` MUST equal `done.answer`. This
 * test pins the FE contract by mocking fetch with a ReadableStream
 * that emits an in-order sequence of SSE frames.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { queryStream, sendChatMessage, warmup } from '@/lib/api';
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

interface FakeResponseInit extends ResponseInit {
  stream: ReadableStream<Uint8Array>;
}

function fakeResponse(init: FakeResponseInit): Response {
  // Build a real Response-like object backed by a ReadableStream.
  return new Response(init.stream, init);
}

describe('api.queryStream', () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it('emits onStage BEFORE the first onDelta (PART 1 LATENCY contract)', async () => {
    // The BE emits a single `event: stage` frame first, then meta, then
    // deltas. The FE must surface onStage so the bubble can show
    // "⏳ Đang phân tích câu hỏi…" immediately. This pins the BE→FE
    // contract: stage frame arrives before any delta frame.
    const guarded = 'Ngành CNTT 400 chỉ tiêu năm 2026.';
    const stream = sseStream([
      sseFrame('stage', { stage: 'Đang phân tích câu hỏi…' }),
      sseFrame('meta', { route: 'padded_rag', sources: [{ id: 'c1' }], trace_id: 't1' }),
      sseFrame('delta', { text: 'Ngành CNTT ' }),
      sseFrame('delta', { text: '400 chỉ tiêu' }),
      sseFrame('delta', { text: ' năm 2026.' }),
      sseFrame('done', { answer: guarded }),
    ]);
    globalThis.fetch = vi.fn().mockResolvedValue(
      fakeResponse({ stream, status: 200, headers: { 'Content-Type': 'text/event-stream' } }),
    ) as unknown as typeof fetch;

    const onStage = vi.fn();
    const onMeta = vi.fn();
    const onDelta = vi.fn();
    const onDone = vi.fn();

    const result = await queryStream('ngành CNTT?', { onStage, onMeta, onDelta, onDone });

    expect(result).toBe(guarded);
    expect(onStage).toHaveBeenCalledTimes(1);
    expect(onStage.mock.calls[0][0]).toBe('Đang phân tích câu hỏi…');

    // Order: onStage must fire before the first onDelta.
    const stageOrder = onStage.mock.invocationCallOrder[0];
    const deltaOrder = onDelta.mock.invocationCallOrder[0];
    expect(stageOrder).toBeLessThan(deltaOrder);

    // Concat invariant is preserved.
    expect(onDelta.mock.calls.map((c) => c[0]).join('')).toBe(guarded);
    expect(onDone).toHaveBeenCalledWith(guarded);
  });

  it('emits meta + deltas + done in order and concatenates text', async () => {
    const guarded = 'Ngành CNTT 400 chỉ tiêu năm 2026.';
    const stream = sseStream([
      sseFrame('meta', { route: 'padded_rag', sources: [{ id: 'c1' }], trace_id: 't1' }),
      sseFrame('delta', { text: 'Ngành CNTT ' }),
      sseFrame('delta', { text: '400 chỉ tiêu' }),
      sseFrame('delta', { text: ' năm 2026.' }),
      sseFrame('done', { answer: guarded }),
    ]);
    const fetchMock = vi.fn().mockResolvedValue(
      fakeResponse({ stream, status: 200, headers: { 'Content-Type': 'text/event-stream' } }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const onMeta = vi.fn();
    const onDelta = vi.fn();
    const onDone = vi.fn();

    const result = await queryStream('ngành CNTT?', { onMeta, onDelta, onDone });

    expect(result).toBe(guarded);
    expect(onMeta).toHaveBeenCalledTimes(1);
    expect(onMeta.mock.calls[0][0]).toMatchObject({ route: 'padded_rag', trace_id: 't1' });
    expect(onDelta).toHaveBeenCalledTimes(3);
    expect(onDelta.mock.calls.map((c) => c[0]).join('')).toBe(guarded);
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(onDone).toHaveBeenCalledWith(guarded);

    // POST /v2/query/stream with the correct body
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/v2\/query\/stream$/);
    expect((init as RequestInit).method).toBe('POST');
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({ query: 'ngành CNTT?' });
  });

  it('CRITICAL: never emits the pre-guard ungrounded URL in the stream', async () => {
    // The BE feeds the FE the POST-guard answer. If a future refactor
    // accidentally streams the pre-guard string, the ungrounded URL
    // would leak. This test pins the contract: whatever reaches the
    // FE deltas is whatever the BE decided is the final answer.
    const ungrounded_url = 'https://malicious.example.com/fake';
    const pre_guard = 'Đăng ký tại ' + ungrounded_url + ' để nhận hồ sơ.';
    // The BE strips the URL and the post-guard string is what streams.
    const post_guard = 'Đăng ký tại cổng tuyển sinh HUSC để nhận hồ sơ.';
    expect(ungrounded_url).not.toBe(post_guard); // sanity

    const stream = sseStream([
      sseFrame('meta', { route: 'padded_rag' }),
      sseFrame('delta', { text: 'Đăng ký tại ' }),
      sseFrame('delta', { text: 'cổng tuyển sinh HUSC ' }),
      sseFrame('delta', { text: 'để nhận hồ sơ.' }),
      sseFrame('done', { answer: post_guard }),
    ]);
    globalThis.fetch = vi.fn().mockResolvedValue(
      fakeResponse({ stream, status: 200 }),
    ) as unknown as typeof fetch;

    const onDelta = vi.fn();
    await queryStream('đăng ký ở đâu?', { onDelta });

    const concat = onDelta.mock.calls.map((c) => c[0]).join('');
    expect(concat).toBe(post_guard);
    expect(concat).not.toContain(ungrounded_url);
  });

  it('handles CRLF line endings in the SSE stream', async () => {
    // CRLF (`\r\n`) is the spec-mandated SSE line ending on Windows. The
    // parser must normalize before splitting on `\n\n` so the BE-emitted
    // CRLF frames parse correctly.
    const crlfStream = new ReadableStream<Uint8Array>({
      start(controller) {
        const payload =
          'event: meta\r\ndata: {"route":"padded_rag"}\r\n\r\n' +
          'event: delta\r\ndata: {"text":"Hello"}\r\n\r\n' +
          'event: done\r\ndata: {"answer":"Hello"}\r\n\r\n';
        controller.enqueue(new TextEncoder().encode(payload));
        controller.close();
      },
    });
    globalThis.fetch = vi.fn().mockResolvedValue(
      fakeResponse({ stream: crlfStream, status: 200 }),
    ) as unknown as typeof fetch;

    const onDelta = vi.fn();
    const onDone = vi.fn();
    const result = await queryStream('x', { onDelta, onDone });

    // The DONE frame is the load-bearing contract: it carries the
    // post-guard `answer` (which the FE has not yet seen raw).
    expect(onDone).toHaveBeenCalled();
    expect(onDone.mock.calls[0][0]).toBe('Hello');
    expect(result).toBe('Hello');
    // The delta handler must have been invoked with at least one chunk.
    expect(onDelta.mock.calls.length).toBeGreaterThan(0);
  });

  it('throws when the response is not OK (no body)', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(null, { status: 502, statusText: 'Bad Gateway' }),
    ) as unknown as typeof fetch;

    await expect(queryStream('x', {})).rejects.toThrow(/502/);
  });
});

describe('api module surface', () => {
  it('exports sendChatMessage, warmup, and queryStream', () => {
    expect(typeof sendChatMessage).toBe('function');
    expect(typeof warmup).toBe('function');
    expect(typeof queryStream).toBe('function');
  });
});
