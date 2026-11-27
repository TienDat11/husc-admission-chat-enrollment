/**
 * ChatLayout — the sr-only answer announcer must re-announce every answer,
 * including identical consecutive answers (a live region only reads a change,
 * so the announcement text node is re-created per answer).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const mockSendChatMessage = vi.fn();

vi.mock('@/lib/api', () => ({
  setStudentSessionId: vi.fn(),
  sendChatMessage: (...args: unknown[]) => mockSendChatMessage(...args),
  checkHealth: vi.fn(),
  getApiBaseUrl: () => 'http://localhost:8000',
  classifyFetchError: vi.fn((err: unknown) => ({
    kind: 'unknown',
    retryable: false,
    message: err instanceof Error ? err.message : String(err),
  })),
}));

vi.mock('@/hooks/useBackendReadiness', () => ({
  useBackendReadiness: vi.fn(() => ({
    ready: true,
    stage: 'ready' as const,
    services: {},
    lastChecked: Date.now(),
    error: null,
  })),
}));

import { ChatLayout } from '@/components/chat/ChatLayout';
import { ThemeProvider } from '@/hooks/useTheme';
import { StudentProfileProvider } from '@/hooks/useStudentProfile';

const ANNOUNCE = 'Đã nhận được câu trả lời.';
const ANSWER = 'Đáp án giống nhau.';

function successResponse() {
  return {
    answer: ANSWER,
    sources: [],
    confidence: 0.9,
    status_code: 'SUCCESS',
    status_reason: '',
    data_gap_hints: [],
    internal_status_code: null,
  };
}

beforeEach(() => {
  mockSendChatMessage.mockReset();
});

describe('ChatLayout answer announcer', () => {
  it('re-announces an identical second answer by re-creating the live text node', async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockResolvedValueOnce(successResponse()).mockResolvedValueOnce(successResponse());

    render(
      <StudentProfileProvider>
        <ThemeProvider>
          <ChatLayout />
        </ThemeProvider>
      </StudentProfileProvider>
    );

    const region = document.querySelector('.sr-only[role="status"]') as HTMLElement;
    expect(region).toBeTruthy();

    const textarea = document.querySelector('textarea') as HTMLTextAreaElement;
    const sendBtn = Array.from(document.querySelectorAll('button')).find(
      (b) => b.getAttribute('aria-label') === 'Gửi tin nhắn'
    );

    await user.type(textarea, 'câu hỏi 1');
    fireEvent.click(sendBtn!);
    await waitFor(() => expect(region).toHaveTextContent(ANNOUNCE));
    const firstNode = region.firstChild;

    await user.type(textarea, 'câu hỏi 2');
    fireEvent.click(sendBtn!);
    await waitFor(() => expect(region.firstChild).not.toBe(firstNode));
    expect(region).toHaveTextContent(ANNOUNCE);
    expect(screen.getAllByText(ANSWER).length).toBeGreaterThan(0);
  });
});
