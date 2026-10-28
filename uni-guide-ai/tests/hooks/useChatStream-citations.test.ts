/**
 * useChatStream — citation propagation on the STREAMING path.
 *
 * Regression guard: the BE ships the citation set in the `meta` SSE frame
 * (rag2025/src/main.py:1888-1894 sets `sources`), but the hook only read
 * `status_code` and `data_gap_hints` from it, and the final placeholder
 * update omitted `sources` entirely. Net effect: a successful STREAMED
 * answer rendered with no source chips, while the non-stream fallback
 * rendered them — the user sees citations vanish intermittently.
 *
 * The assertion is deliberately on the observable message list (what
 * MessageBubble reads), not on hook internals: the contract is "a streamed
 * answer carries the citations the backend sent".
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import type { Message } from '@/lib/chat-types';

const mockQueryStream = vi.fn();
const mockSendChatMessage = vi.fn();

vi.mock('@/lib/api', () => ({
  queryStream: (...args: unknown[]) => mockQueryStream(...args),
  sendChatMessage: (...args: unknown[]) => mockSendChatMessage(...args),
  classifyFetchError: vi.fn((err: unknown) => ({
    kind: 'unknown',
    retryable: false,
    message: err instanceof Error ? err.message : String(err),
  })),
}));

import { useChatStream } from '@/hooks/useChatStream';

beforeEach(() => {
  mockQueryStream.mockReset();
  mockSendChatMessage.mockReset();
});

/** Drive the hook through one send() and return the final message list. */
async function runSend(queryStreamImpl: (handlers: Record<string, Function>) => Promise<void>) {
  mockQueryStream.mockImplementationOnce(
    async (_content: string, handlers: Record<string, Function>) => queryStreamImpl(handlers),
  );

  let messages: Message[] = [];
  const { result } = renderHook(() =>
    useChatStream({
      messages,
      onMessagesChange: (next: Message[]) => {
        messages = next;
      },
      generateId: (() => {
        let n = 0;
        return () => `id-${++n}`;
      })(),
    }),
  );

  await act(async () => {
    await result.current.send('Điểm chuẩn CNTT 2024?');
  });

  return messages;
}

function assistantMessage(messages: Message[]): Message {
  const msg = messages.find((m) => m.role === 'assistant');
  if (!msg) throw new Error(`no assistant message in ${JSON.stringify(messages)}`);
  return msg;
}

describe('useChatStream — streamed citations', () => {
  it('streams the backend meta sources onto the final assistant message', async () => {
    const messages = await runSend(async (handlers) => {
      handlers.onMeta({
        route: 'padded_rag',
        status_code: 'SUCCESS',
        data_gap_hints: [],
        sources: [
          {
            id: 's1',
            title: 'Thông báo tuyển sinh 2024',
            url: 'https://husc.edu.vn/tb2024',
            data_year: '2024',
          },
        ],
      });
      handlers.onDelta('Điểm chuẩn CNTT là ');
      handlers.onDelta('22.5 điểm.');
      handlers.onDone('Điểm chuẩn CNTT là 22.5 điểm.');
    });

    const answer = assistantMessage(messages);
    expect(answer.content).toContain('22.5');

    // The observable contract: the citations the BE sent are present for
    // MessageBubble to render (it renders chips only from message.sources).
    expect(answer.sources).toBeDefined();
    expect(answer.sources).toHaveLength(1);
    expect(answer.sources![0].title).toBe('Thông báo tuyển sinh 2024');
    expect(answer.sources![0].url).toBe('https://husc.edu.vn/tb2024');
    expect(answer.sources![0].data_year).toBe('2024');
  });

  it('carries multiple sources and does not fall back to the non-stream path', async () => {
    const messages = await runSend(async (handlers) => {
      handlers.onMeta({
        route: 'graph_rag',
        status_code: 'SUCCESS',
        sources: [
          { id: 'a', title: 'Quy chế tuyển sinh' },
          { id: 'b', title: 'Thông báo 2025' },
          { id: 'c', title: 'Đề án tuyển sinh' },
        ],
      });
      handlers.onDelta('Trả lời.');
      handlers.onDone('Trả lời.');
    });

    const answer = assistantMessage(messages);
    expect(answer.sources).toHaveLength(3);
    expect(answer.sources!.map((s) => s.title)).toEqual([
      'Quy chế tuyển sinh',
      'Thông báo 2025',
      'Đề án tuyển sinh',
    ]);
    // Stream succeeded — the fallback must not have run.
    expect(mockSendChatMessage).not.toHaveBeenCalled();
  });

  it('omits sources cleanly when the backend sends none', async () => {
    const messages = await runSend(async (handlers) => {
      handlers.onMeta({ route: 'padded_rag', status_code: 'SUCCESS', sources: [] });
      handlers.onDelta('Không có nguồn.');
      handlers.onDone('Không có nguồn.');
    });

    const answer = assistantMessage(messages);
    expect(answer.sources ?? []).toHaveLength(0);
  });

  it('still maps string[] sources from meta (legacy /query shape)', async () => {
    const messages = await runSend(async (handlers) => {
      handlers.onMeta({ status_code: 'SUCCESS', sources: ['Thông báo cũ 2023'] });
      handlers.onDelta('Trả lời.');
      handlers.onDone('Trả lời.');
    });

    const answer = assistantMessage(messages);
    expect(answer.sources).toHaveLength(1);
    expect(answer.sources![0].title).toBe('Thông báo cũ 2023');
  });

  /**
   * Ticket 17: retrieval confidence rides the meta frame. The BE sends one
   * number per response and the FE renders it onto every chip. Before this,
   * the meta frame carried no `confidence`, so the hook's 0.5 default was
   * mapped onto every streamed chip — a fabricated number that disagreed
   * with the same answer served over the non-stream path.
   */
  it('puts the backend retrieval confidence on the chips, not the 0.5 default', async () => {
    const messages = await runSend(async (handlers) => {
      handlers.onMeta({
        route: 'padded_rag',
        status_code: 'SUCCESS',
        confidence: 0.83,
        sources: [
          { id: 's1', title: 'Thông báo tuyển sinh 2026' },
          { id: 's2', title: 'Đề án tuyển sinh 2026' },
        ],
      });
      handlers.onDelta('Trả lời.');
      handlers.onDone('Trả lời.');
    });

    const answer = assistantMessage(messages);
    expect(answer.confidence).toBe(0.83);
    // The chips are the observable surface — every one carries the real value.
    expect(answer.sources!.map((s) => s.confidence)).toEqual([0.83, 0.83]);
  });

  /**
   * FIX-05 (ticket 44): the BE emits the degraded status surface on the meta
   * frame, but the hook only read status_code/data_gap_hints — so a failed
   * generation rendered as a normal answer on the streamed path only.
   */
  it('carries the degraded status surface from the meta frame', async () => {
    const messages = await runSend(async (handlers) => {
      handlers.onMeta({
        route: 'padded_rag',
        status_code: 'GENERATION_FAILED',
        status_reason: 'generation_failed',
        internal_status_code: 'GENERATION_FAILED',
        answer_confidence: null,
        confidence: 0.83,
        sources: [{ id: 's1', title: 'Thông báo tuyển sinh 2026' }],
      });
      handlers.onDelta('Hệ thống đang gặp sự cố.');
      handlers.onDone('Hệ thống đang gặp sự cố.');
    });

    const answer = assistantMessage(messages);
    expect(answer.statusCode).toBe('GENERATION_FAILED');
    expect(answer.statusReason).toBe('generation_failed');
    expect(answer.internalStatusCode).toBe('GENERATION_FAILED');
  });

  it('keeps the 0.5 default only when the backend omits confidence', async () => {
    const messages = await runSend(async (handlers) => {
      handlers.onMeta({
        status_code: 'SUCCESS',
        sources: [{ id: 's1', title: 'Nguồn cũ' }],
      });
      handlers.onDelta('Trả lời.');
      handlers.onDone('Trả lời.');
    });

    const answer = assistantMessage(messages);
    expect(answer.confidence).toBe(0.5);
  });
});
