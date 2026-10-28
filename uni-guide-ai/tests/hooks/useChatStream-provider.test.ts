/**
 * Ticket 17: useChatStream must carry `provider` — the ONLY field that
 * separates a policy abstain/conflict from a model answer on the wire
 * (llm_generator.py:873 ships refusals with status_code SUCCESS).
 *
 * Before this, provider was dropped on both paths, so the answer surface
 * could not render abstain/conflict states at all — they were invisible
 * states, and per the ticket, "a state that can never be produced by the
 * real payload is not a state".
 *
 * Also criterion 5 (transport errors): the error bot must carry the original
 * question in `retryQuery` so the retry affordance can re-send it verbatim —
 * the same contract ticket 43 established for GENERATION_FAILED.
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
    kind: 'network',
    retryable: true,
    message: err instanceof Error ? err.message : String(err),
  })),
}));

import { useChatStream } from '@/hooks/useChatStream';

beforeEach(() => {
  mockQueryStream.mockReset();
  mockSendChatMessage.mockReset();
});

async function runSend(
  queryStreamImpl: (handlers: Record<string, Function>) => Promise<void>,
) {
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
    await result.current.send('Điểm chuẩn ngành CNTT năm nay?');
  });
  return messages;
}

function assistantMessage(messages: Message[]): Message {
  const msg = messages.find((m) => m.role === 'assistant');
  if (!msg) throw new Error(`no assistant message in ${JSON.stringify(messages)}`);
  return msg;
}

describe('useChatStream — provider propagation (stream path)', () => {
  it('carries an abstain provider from the meta frame onto the message', async () => {
    const messages = await runSend(async (handlers) => {
      handlers.onMeta({
        status_code: 'SUCCESS',
        provider: 'evidence-policy:year-mismatch',
        answer_confidence: null,
        sources: [{ id: 's1', title: 'TB59', data_year: '2025' }],
      });
      handlers.onDelta('Từ chối theo chính sách.');
      handlers.onDone('Từ chối theo chính sách.');
    });
    const answer = assistantMessage(messages);
    expect(answer.provider).toBe('evidence-policy:year-mismatch');
    expect(answer.statusCode).toBe('SUCCESS');
  });

  it('carries a conflict provider from the meta frame', async () => {
    const messages = await runSend(async (handlers) => {
      handlers.onMeta({
        status_code: 'SUCCESS',
        provider: 'evidence-policy:unresolved-conflict',
        sources: [],
      });
      handlers.onDelta('Hai nguồn mâu thuẫn.');
      handlers.onDone('Hai nguồn mâu thuẫn.');
    });
    expect(assistantMessage(messages).provider).toBe('evidence-policy:unresolved-conflict');
  });

  it('carries a model provider verbatim (no rewriting, no defaulting)', async () => {
    const messages = await runSend(async (handlers) => {
      handlers.onMeta({ status_code: 'SUCCESS', provider: 'llama-3.3-70b', sources: [] });
      handlers.onDelta('Trả lời.');
      handlers.onDone('Trả lời.');
    });
    expect(assistantMessage(messages).provider).toBe('llama-3.3-70b');
  });

  it('keeps provider null when the BE sends null (scope block — Main measured provider null)', async () => {
    const messages = await runSend(async (handlers) => {
      handlers.onMeta({ status_code: 'NOT_IN_HUSC_SCOPE', provider: null, sources: [] });
      handlers.onDelta('Ngoài phạm vi.');
      handlers.onDone('Ngoài phạm vi.');
    });
    expect(assistantMessage(messages).provider).toBeNull();
  });
});

describe('useChatStream — provider propagation (non-stream fallback)', () => {
  async function runFallback(response: Record<string, unknown>) {
    mockQueryStream.mockRejectedValueOnce(new Error('stream down'));
    mockSendChatMessage.mockResolvedValueOnce(response);
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
      await result.current.send('Học phí ngành CNTT?');
    });
    return messages;
  }

  it('maps provider from the JSON response onto the message', async () => {
    const messages = await runFallback({
      answer: 'OK',
      status_code: 'SUCCESS',
      provider: 'evidence-policy:no-evidence',
      sources: [],
    });
    expect(assistantMessage(messages).provider).toBe('evidence-policy:no-evidence');
  });

  it('parity: the same abstain payload yields the same provider on both paths', async () => {
    const streamed = await runSend(async (handlers) => {
      handlers.onMeta({ status_code: 'SUCCESS', provider: 'evidence-policy:lineage-missing', sources: [] });
      handlers.onDelta('A');
      handlers.onDone('A');
    });
    const fallback = await runFallback({
      answer: 'A',
      status_code: 'SUCCESS',
      provider: 'evidence-policy:lineage-missing',
      sources: [],
    });
    expect(assistantMessage(streamed).provider).toBe(assistantMessage(fallback).provider);
  });
});

describe('useChatStream — transport error bot (criterion 5 data plumbing)', () => {
  it('carries the original question in retryQuery so the bubble can offer retry', async () => {
    mockQueryStream.mockRejectedValueOnce(new Error('stream down'));
    mockSendChatMessage.mockRejectedValueOnce(new Error('Failed to fetch'));
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
      await result.current.send('Ngành nào xét học bạ?');
    });
    const bot = assistantMessage(messages);
    expect(bot.errorKind).toBe('network');
    expect(bot.retryQuery).toBe('Ngành nào xét học bạ?');
    // No fabricated answer content on the error path.
    expect(bot.content).toBe('');
  });
});
