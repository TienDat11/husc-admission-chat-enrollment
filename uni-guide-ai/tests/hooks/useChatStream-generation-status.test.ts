/**
 * Ticket 43 — useChatStream surfaces the degraded-generation contract.
 *
 * Three things are guarded here, all on the observable message list (what
 * MessageBubble actually reads) rather than hook internals:
 *
 *  1. GENERATION_FAILED from /v2 reaches the message on BOTH the streamed and
 *     the non-stream path, with identical decoration — a user must not get a
 *     normal-looking answer just because the SSE connection happened to work.
 *  2. `answer_confidence` is copied verbatim. `null` (the backend refuses to
 *     vouch for its own text) must stay `null`; a defaulted number would claim
 *     trust the system does not have.
 *  3. `retryQuery` carries the question that produced the answer, so the retry
 *     affordance can re-send it without ChatLayout re-deriving anything.
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
import { GENERATION_FAILED_STATUS } from '@/lib/chat-helpers';

const QUESTION = 'Điểm chuẩn CNTT 2024?';

/**
 * The literal sentence llm_generator.py:852-855 puts in `answer` when every
 * provider raised. Kept verbatim so the test breaks if the FE starts inventing
 * its own copy instead of displaying what the backend sent.
 */
const BE_FALLBACK_ANSWER =
  'Hệ thống đã tìm thấy tài liệu liên quan nhưng dịch vụ tạo câu trả lời ' +
  'đang tạm thời không khả dụng. Vui lòng thử lại sau.';

beforeEach(() => {
  mockQueryStream.mockReset();
  mockSendChatMessage.mockReset();
});

/** Drive one send() over the STREAMED path (meta + delta + done). */
async function runStreamed(meta: Record<string, unknown>, answer: string) {
  mockQueryStream.mockImplementationOnce(
    async (_content: string, handlers: Record<string, Function>) => {
      handlers.onMeta(meta);
      if (answer) handlers.onDelta(answer);
      handlers.onDone(answer);
    },
  );
  return runCapture();
}

/**
 * Drive one send() over the NON-STREAM path: queryStream throws immediately,
 * so the hook falls back to sendChatMessage.
 */
async function runNonStreamed(payload: Record<string, unknown>) {
  mockQueryStream.mockImplementationOnce(async () => {
    throw new Error('stream unavailable');
  });
  mockSendChatMessage.mockResolvedValueOnce(payload);
  return runCapture();
}

async function runCapture(): Promise<Message[]> {
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
    await result.current.send(QUESTION);
  });
  return messages;
}

function assistantMessage(messages: Message[]): Message {
  const msg = messages.find((m) => m.role === 'assistant');
  if (!msg) throw new Error(`no assistant message in ${JSON.stringify(messages)}`);
  return msg;
}

describe('useChatStream — GENERATION_FAILED reaches the UI (streamed)', () => {
  it('carries status, decorated content and retryQuery off the meta frame', async () => {
    const messages = await runStreamed(
      {
        route: 'padded_rag',
        status_code: GENERATION_FAILED_STATUS,
        status_reason: 'generation_failed',
        internal_status_code: GENERATION_FAILED_STATUS,
        answer_confidence: null,
        confidence: 0.83,
        data_gap_hints: [],
        sources: [],
      },
      BE_FALLBACK_ANSWER,
    );

    const answer = assistantMessage(messages);
    expect(answer.statusCode).toBe(GENERATION_FAILED_STATUS);
    expect(answer.internalStatusCode).toBe(GENERATION_FAILED_STATUS);
    expect(answer.statusReason).toBe('generation_failed');
    expect(answer.retryQuery).toBe(QUESTION);
    // Decorated, not raw: the warning header is what distinguishes the block.
    expect(answer.content).not.toBe(BE_FALLBACK_ANSWER);
    expect(answer.content).toContain('Máy chủ gặp sự cố khi tạo câu trả lời');
    expect(answer.content).toContain(BE_FALLBACK_ANSWER);
  });
});

describe('useChatStream — GENERATION_FAILED reaches the UI (non-streamed)', () => {
  it('carries the same fields when the stream never opened', async () => {
    const messages = await runNonStreamed({
      answer: BE_FALLBACK_ANSWER,
      sources: [],
      confidence: 0.83,
      answer_confidence: null,
      status_code: GENERATION_FAILED_STATUS,
      status_reason: 'generation_failed',
      internal_status_code: GENERATION_FAILED_STATUS,
      data_gap_hints: [],
    });

    const answer = assistantMessage(messages);
    expect(answer.statusCode).toBe(GENERATION_FAILED_STATUS);
    expect(answer.internalStatusCode).toBe(GENERATION_FAILED_STATUS);
    expect(answer.retryQuery).toBe(QUESTION);
    expect(answer.content).toContain('Máy chủ gặp sự cố khi tạo câu trả lời');
  });
});

describe('useChatStream — both paths agree (criterion 5)', () => {
  it('produces identical content, status surface and retryQuery', async () => {
    const streamed = assistantMessage(
      await runStreamed(
        {
          status_code: GENERATION_FAILED_STATUS,
          status_reason: 'generation_failed',
          internal_status_code: GENERATION_FAILED_STATUS,
          answer_confidence: null,
          confidence: 0.83,
          data_gap_hints: [],
          sources: [],
        },
        BE_FALLBACK_ANSWER,
      ),
    );
    const nonStreamed = assistantMessage(
      await runNonStreamed({
        answer: BE_FALLBACK_ANSWER,
        sources: [],
        confidence: 0.83,
        answer_confidence: null,
        status_code: GENERATION_FAILED_STATUS,
        status_reason: 'generation_failed',
        internal_status_code: GENERATION_FAILED_STATUS,
        data_gap_hints: [],
      }),
    );

    expect(streamed.content).toBe(nonStreamed.content);
    expect(streamed.statusCode).toBe(nonStreamed.statusCode);
    expect(streamed.internalStatusCode).toBe(nonStreamed.internalStatusCode);
    expect(streamed.statusReason).toBe(nonStreamed.statusReason);
    expect(streamed.answerConfidence).toBe(nonStreamed.answerConfidence);
    expect(streamed.retryQuery).toBe(nonStreamed.retryQuery);
  });
});

describe('useChatStream — answer_confidence is copied verbatim (criterion 3)', () => {
  it('keeps null on a failed generation instead of defaulting it', async () => {
    const messages = await runStreamed(
      {
        status_code: GENERATION_FAILED_STATUS,
        internal_status_code: GENERATION_FAILED_STATUS,
        answer_confidence: null,
        confidence: 0.83,
        sources: [],
      },
      BE_FALLBACK_ANSWER,
    );
    const answer = assistantMessage(messages);
    expect(answer.answerConfidence).toBeNull();
    // Retrieval confidence keeps its own meaning and value.
    expect(answer.confidence).toBe(0.83);
  });

  it('carries a real number on a successful generation', async () => {
    const messages = await runStreamed(
      {
        status_code: 'SUCCESS',
        answer_confidence: 0.92,
        confidence: 0.83,
        sources: [],
      },
      'Điểm chuẩn 22.5.',
    );
    const answer = assistantMessage(messages);
    expect(answer.answerConfidence).toBe(0.92);
    expect(answer.content).toBe('Điểm chuẩn 22.5.');
  });

  it('leaves it undefined when the backend omits the field entirely', async () => {
    const messages = await runStreamed(
      { status_code: 'SUCCESS', confidence: 0.83, sources: [] },
      'Điểm chuẩn 22.5.',
    );
    expect(assistantMessage(messages).answerConfidence).toBeUndefined();
  });

  it('reads it on the non-stream path too', async () => {
    const messages = await runNonStreamed({
      answer: BE_FALLBACK_ANSWER,
      sources: [],
      confidence: 0.83,
      answer_confidence: null,
      status_code: GENERATION_FAILED_STATUS,
      internal_status_code: GENERATION_FAILED_STATUS,
    });
    expect(assistantMessage(messages).answerConfidence).toBeNull();
  });
});

describe('useChatStream — SUCCESS passthrough is untouched (criterion 8)', () => {
  it('does not decorate, does not set a failure status', async () => {
    const messages = await runStreamed(
      { status_code: 'SUCCESS', answer_confidence: 0.9, sources: [] },
      'Điểm chuẩn CNTT 2024 là 22.5.',
    );
    const answer = assistantMessage(messages);
    expect(answer.content).toBe('Điểm chuẩn CNTT 2024 là 22.5.');
    expect(answer.statusCode).toBe('SUCCESS');
    expect(answer.internalStatusCode).toBeUndefined();
  });
});

describe('useChatStream — empty failed answer stays readable (criterion 6)', () => {
  it('decorates an empty answer with the failure label', async () => {
    const messages = await runStreamed(
      {
        status_code: GENERATION_FAILED_STATUS,
        internal_status_code: GENERATION_FAILED_STATUS,
        status_reason: 'generation_failed',
        answer_confidence: null,
        sources: [],
      },
      '',
    );
    const answer = assistantMessage(messages);
    expect(answer.content).not.toBe('');
    expect(answer.content).toContain('Máy chủ gặp sự cố khi tạo câu trả lời');
  });
});

describe('useChatStream — data_gap_hints are not shown as missing data (criterion 2)', () => {
  it('suppresses the hint block when generation failed', async () => {
    const messages = await runStreamed(
      {
        status_code: GENERATION_FAILED_STATUS,
        internal_status_code: GENERATION_FAILED_STATUS,
        status_reason: 'generation_failed',
        answer_confidence: null,
        // Stale hints from an earlier retrieval stage.
        data_gap_hints: ['Cập nhật đề án tuyển sinh 2025'],
        sources: [],
      },
      BE_FALLBACK_ANSWER,
    );
    const answer = assistantMessage(messages);
    expect(answer.content).not.toContain('Gợi ý bổ sung dữ liệu');
    expect(answer.content).not.toContain('Cập nhật đề án tuyển sinh 2025');
    // The hints are still on the message for diagnostics — only the copy hides them.
    expect(answer.dataGapHints).toEqual(['Cập nhật đề án tuyển sinh 2025']);
  });

  it('still shows the hint block for a genuine data gap', async () => {
    const messages = await runStreamed(
      {
        status_code: 'INSUFFICIENT_DATA',
        internal_status_code: 'INSUFFICIENT_DATA',
        status_reason: 'thiếu dữ liệu',
        data_gap_hints: ['Cập nhật đề án tuyển sinh 2025'],
        sources: [],
      },
      'Tôi chưa có dữ liệu.',
    );
    expect(assistantMessage(messages).content).toContain('Gợi ý bổ sung dữ liệu');
  });
});

describe('useChatStream — unknown status keeps the existing fallback label (criterion 7)', () => {
  it('falls back to the data-check label without throwing', async () => {
    const messages = await runStreamed(
      {
        status_code: 'SOMETHING_NEW_FROM_BE',
        internal_status_code: null,
        status_reason: '',
        sources: [],
      },
      'Nội dung mới từ BE.',
    );
    const answer = assistantMessage(messages);
    expect(answer.content).toContain('Cần kiểm tra dữ liệu');
    expect(answer.content).toContain('Nội dung mới từ BE.');
  });
});
