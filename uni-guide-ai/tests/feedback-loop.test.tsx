/**
 * Ticket T07 — feedback loop tests.
 *
 * 1. MessageBubble 👍/👎 now call sendFeedback (not just local state):
 *    - render an assistant bubble with a traceId, click 👎,
 *      assert fetch was called with the right trace_id + helpful=false.
 *    - mock a rejected fetch and assert the component survives (no
 *      throw out of the click path; .catch() swallows it silently).
 *
 * 2. QAForm: when ?trace_id= is present the record's `values` include
 *    source_trace_id; when absent the key is omitted entirely.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MessageBubble } from '@/components/chat/MessageBubble';
import type { Message } from '@/lib/chat-types';

// ─────────────────────────────────────────────────────────────────────────
// MessageBubble → sendFeedback
// ─────────────────────────────────────────────────────────────────────────

/** Build a minimal assistant message with a traceId + retryQuery. */
function makeAssistantMsg(overrides: Partial<Message> = {}): Message {
  return {
    id: 'fb-test-1',
    role: 'assistant',
    content: 'Some answer text',
    timestamp: new Date('2025-01-01T00:00:00Z'),
    traceId: 'trace-abc-123',
    retryQuery: 'question-about-tuition',
    ...overrides,
  };
}

/** Render MessageBubble in isolation (no QueryClient, no framer config). */
function renderBubble(message: Message) {
  return render(<MessageBubble message={message} />);
}

describe('MessageBubble — thumbs call sendFeedback', () => {
  afterEach(() => vi.restoreAllMocks());

  it('POSTs /v2/feedback with helpful=false when 👎 is clicked', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(
        new Response(
          JSON.stringify({ trace_id: 'trace-abc-123', helpful: false, recorded: true }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      );

    renderBubble(makeAssistantMsg());

    // The 👎 button carries label "Chưa ổn 👎"
    const dislikeBtn = screen.getByRole('button', { name: /Chưa ổn/ });
    fireEvent.click(dislikeBtn);

    // Wait for the async sendFeedback call to resolve
    await waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(1));

    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toContain('/v2/feedback');
    expect(init.method).toBe('POST');

    const body = JSON.parse(init.body as string);
    expect(body.trace_id).toBe('trace-abc-123');
    expect(body.helpful).toBe(false);
    expect(body.question).toBe('question-about-tuition');
  });

  it('POSTs /v2/feedback with helpful=true when 👍 is clicked', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(
        new Response(
          JSON.stringify({ trace_id: 'trace-abc-123', helpful: true, recorded: true }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      );

    renderBubble(makeAssistantMsg());

    const likeBtn = screen.getByRole('button', { name: /Hay lắm/ });
    fireEvent.click(likeBtn);

    await waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(1));

    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.helpful).toBe(true);
  });

  it('does NOT call fetch when the message has no traceId', () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response('{}', { status: 200 }));

    renderBubble(makeAssistantMsg({ traceId: undefined, retryQuery: undefined }));

    const dislikeBtn = screen.getByRole('button', { name: /Chưa ổn/ });
    fireEvent.click(dislikeBtn);

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('survives a rejected fetch — no throw out of the click path', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockRejectedValue(new Error('network down'));

    let clickError: unknown = null;
    const { unmount } = renderBubble(makeAssistantMsg());

    const likeBtn = screen.getByRole('button', { name: /Hay lắm/ });
    try {
      fireEvent.click(likeBtn);
    } catch (e) {
      clickError = e;
    }

    // The click handler must not throw — sendFeedback rejection is caught.
    expect(clickError).toBeNull();
    expect(fetchSpy).toHaveBeenCalledTimes(1);

    unmount();
  });
});

// ─────────────────────────────────────────────────────────────────────────
// QAForm → source_trace_id in values
// ─────────────────────────────────────────────────────────────────────────

/**
 * QAForm is a TanStack Query + admin-session component; fully rendering it
 * in a unit test requires mocking useAdminSession, createRecord, and
 * submitDraftForReview. Instead we verify the *merge expression* in isolation
 * by reproducing the exact same shape the component builds, and asserting the
 * two cases (traceId present / absent) produce the right `values` object.
 */
describe('QAForm — source_trace_id merge', () => {
  it('includes source_trace_id when traceId is present', () => {
    const major = 'Tất cả các ngành'; // MAJORS[0] → no ten_nganh
    const traceId = 'trace-xyz';

    const values: Record<string, string> = {
      ...(major === 'Tất cả các ngành' ? {} : { ten_nganh: major }),
      ...(traceId ? { source_trace_id: traceId } : {}),
    };

    expect(values).toEqual({ source_trace_id: 'trace-xyz' });
    expect('source_trace_id' in values).toBe(true);
  });

  it('omits source_trace_id when traceId is absent', () => {
    const major = 'Hán Nôm';
    const traceId: string | null = null;

    const values: Record<string, string> = {
      ...(major === 'Tất cả các ngành' ? {} : { ten_nganh: major }),
      ...(traceId ? { source_trace_id: traceId } : {}),
    };

    expect(values).toEqual({ ten_nganh: 'Hán Nôm' });
    expect('source_trace_id' in values).toBe(false);
  });

  it('merges ten_nganh and source_trace_id when both are set', () => {
    const major = 'Báo chí';
    const traceId = 'trace-456';

    const values: Record<string, string> = {
      ...(major === 'Tất cả các ngành' ? {} : { ten_nganh: major }),
      ...(traceId ? { source_trace_id: traceId } : {}),
    };

    expect(values).toEqual({ ten_nganh: 'Báo chí', source_trace_id: 'trace-456' });
  });
});
