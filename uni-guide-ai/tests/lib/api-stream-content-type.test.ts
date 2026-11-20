import { afterEach, describe, expect, it, vi } from 'vitest';
import { queryStream } from '@/lib/api';

function sseResponse(body: string): Response {
  return new Response(body, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream; charset=utf-8' },
  });
}

// Real body captured from prod POST /v2/query/stream (no NS24 key, so the
// deterministic fallback answer). Sources trimmed to two; frames unchanged.
const REAL_CAPTURE = String.raw`event: meta
data: {"route": "hybrid", "sources": [{"id": "diem_chuan_2025_full", "title": "Xếp hạng điểm chuẩn 2025. Cao nhất THPT: Lịch sử (22.0đ). Cao nhất học bạ: Lịch", "url": null, "snippet": "Xếp hạng điểm chuẩn 2025. Cao nhất THPT: Lịch sử (22.0đ). Cao nhất học bạ: Lịch sử (23.5đ).", "data_year": "2025"}, {"id": "6f916b6830cc71bc", "title": "(CTĐT Cử nhân)\n17,50\nTHPT\n18,25*\nCông nghệ thông tin", "url": null, "snippet": "(CTĐT Cử nhân)\n17,50\nTHPT\n18,25*\nCông nghệ thông tin", "data_year": "2026"}], "status_code": "GENERATION_FAILED", "status_reason": "generation_failed", "internal_status_code": "GENERATION_FAILED", "answer_confidence": null, "confidence": 1.0, "citations": [], "trace_id": "6531a785-613f-4b22-b6e1-f3fc7bf9e460", "provider": "Fallback(NoProvider)", "rerank_skipped_reason": "disabled_by_config"}

event: delta
data: {"text": "Mình chưa đủ dữ liệu để trả lời chính "}

event: delta
data: {"text": "xác câu này trong cấu hình hiện tại."}

event: done
data: {"answer": "Mình chưa đủ dữ liệu để trả lời chính xác câu này trong cấu hình hiện tại."}

`;

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

  it('renders the fallback answer and surfaces meta sources from the real prod capture', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => sseResponse(REAL_CAPTURE)));

    const deltas: string[] = [];
    const onMeta = vi.fn();
    const answer = await queryStream('Điểm chuẩn ngành Công nghệ thông tin năm 2025?', {
      onMeta,
      onDelta: (text) => deltas.push(text),
    });

    expect(answer).toBe('Mình chưa đủ dữ liệu để trả lời chính xác câu này trong cấu hình hiện tại.');
    expect(deltas.join('')).toBe(answer);
    expect(onMeta).toHaveBeenCalledTimes(1);
    const meta = onMeta.mock.calls[0][0];
    expect(meta.status_code).toBe('GENERATION_FAILED');
    expect(meta.sources).toHaveLength(2);
    expect(meta.sources[1].title).toContain('\n');
  });
});
