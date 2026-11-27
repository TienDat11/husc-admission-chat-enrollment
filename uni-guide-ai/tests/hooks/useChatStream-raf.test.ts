import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import type { Message } from '@/lib/chat-types';

const mockQueryStream = vi.hoisted(() => vi.fn());
const mockSendChatMessage = vi.hoisted(() => vi.fn());

vi.mock('@/lib/api', () => ({
  queryStream: (...args: unknown[]) => mockQueryStream(...args),
  sendChatMessage: (...args: unknown[]) => mockSendChatMessage(...args),
  classifyFetchError: vi.fn((err: unknown) => ({ kind: 'unknown', retryable: false, message: String(err) })),
}));

import { useChatStream } from '@/hooks/useChatStream';

type Handlers = { onDelta: (text: string) => void; onDone?: (answer: string) => void };

let frames: Map<number, FrameRequestCallback>;
let nextFrameId: number;

beforeEach(() => {
  frames = new Map();
  nextFrameId = 1;
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
    const id = nextFrameId++;
    frames.set(id, cb);
    return id;
  });
  vi.stubGlobal('cancelAnimationFrame', (id: number) => {
    frames.delete(id);
  });
  mockQueryStream.mockReset();
  mockSendChatMessage.mockReset();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function assistantContents(onChange: ReturnType<typeof vi.fn>): string[] {
  return onChange.mock.calls.flatMap(([msgs]) =>
    (msgs as Message[]).filter((m) => m.role === 'assistant').map((m) => m.content),
  );
}

describe('useChatStream delta batching', () => {
  it('applies deltas once per frame and keeps all text when the stream ends', async () => {
    const onChange = vi.fn();
    let updatesDuringDeltas = -1;
    let framesDuringDeltas = -1;
    mockQueryStream.mockImplementationOnce(async (_q: string, handlers: Handlers) => {
      const before = onChange.mock.calls.length;
      handlers.onDelta('Xin ');
      handlers.onDelta('chào');
      updatesDuringDeltas = onChange.mock.calls.length - before;
      framesDuringDeltas = frames.size;
      handlers.onDone?.('Xin chào');
    });
    const { result } = renderHook(() => useChatStream({ messages: [], onMessagesChange: onChange }));
    await act(async () => {
      await result.current.send('câu hỏi');
    });
    expect(updatesDuringDeltas).toBe(0);
    expect(framesDuringDeltas).toBe(1);
    expect(frames.size).toBe(0);
    const last = onChange.mock.calls.at(-1)?.[0] as Message[];
    expect(last.find((m) => m.role === 'assistant')?.content).toContain('Xin chào');
  });

  it('flushes buffered text before an aborted stream falls back', async () => {
    const onChange = vi.fn();
    mockQueryStream.mockImplementationOnce(async (_q: string, handlers: Handlers) => {
      handlers.onDelta('Dở dang');
      throw new DOMException('aborted', 'AbortError');
    });
    mockSendChatMessage.mockResolvedValueOnce({ answer: 'Dự phòng', sources: [], status_code: 'SUCCESS', confidence: 0.9 });
    const { result } = renderHook(() => useChatStream({ messages: [], onMessagesChange: onChange }));
    await act(async () => {
      await result.current.send('câu hỏi');
    });
    expect(assistantContents(onChange)).toContain('Dở dang');
    expect(frames.size).toBe(0);
  });
});
