import { afterEach, describe, expect, it, vi } from 'vitest';
import { queryStream } from '@/lib/api';

function sseResponse(body: string): Response {
  return new Response(body, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream; charset=utf-8' },
  });
}

describe('queryStream content-type guard', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('assembles deltas and the done frame from an SSE body', async () => {
    const body =
      'event: delta\ndata: {"text": "Xin "}\n\n' +
      'event: delta\ndata: {"text": "chào"}\n\n' +
      'event: done\ndata: {"answer": "Xin chào"}\n\n';
    vi.stubGlobal('fetch', vi.fn(async () => sseResponse(body)));

    const deltas: string[] = [];
    const answer = await queryStream('hi', { onDelta: (text) => deltas.push(text) });

    expect(answer).toBe('Xin chào');
    expect(deltas).toEqual(['Xin ', 'chào']);
  });

  it('rejects a 200 text/html body instead of resolving an empty answer', async () => {
    // A static host with an SPA fallback answers every path with index.html.
    // Treating that as an empty stream left the chat bubble waiting forever.
    const html = new Response('<!doctype html><html></html>', {
      status: 200,
      headers: { 'Content-Type': 'text/html' },
    });
    vi.stubGlobal('fetch', vi.fn(async () => html));

    const onDone = vi.fn();
    await expect(queryStream('hi', { onDone })).rejects.toThrow(/expected text\/event-stream/);
    expect(onDone).not.toHaveBeenCalled();
  });
});
