/**
 * Ticket 31 — real session history + answer-derived follow-up suggestions.
 *
 * The sidebar must show the CURRENT session's questions (no mock data), and
 * the suggestion strip must be derived from the answer that just arrived.
 * Both defects were fake data: `mockConversations` in the sidebar and a
 * static 6-question list in SuggestedQuestions.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const mockSendChatMessage = vi.fn();

vi.mock('@/lib/api', () => ({
  sendChatMessage: (...args: unknown[]) => mockSendChatMessage(...args),
  checkHealth: vi.fn(),
  getApiBaseUrl: () => 'http://localhost:8000',
  classifyFetchError: vi.fn((err: unknown) => {
    if (err instanceof Error) return { kind: 'unknown', retryable: false, message: err.message };
    return { kind: 'unknown', retryable: false, message: String(err) };
  }),
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
import { buildSessionHistory, removeExchange } from '@/lib/session-history';
import { deriveFollowUpQuestions } from '@/lib/follow-up-questions';
import type { Message } from '@/lib/chat-types';

function renderChatLayout() {
  return render(
    <ThemeProvider>
      <ChatLayout />
    </ThemeProvider>
  );
}

function textareaAndSend() {
  const textarea = document.querySelector('textarea') as HTMLTextAreaElement;
  const buttons = Array.from(document.querySelectorAll('button'));
  const sendBtn = buttons.find((b) => b.getAttribute('aria-label') === 'Gửi tin nhắn');
  return { textarea, sendBtn };
}

/** Ask one question and let the mocked BE answer with a source-bearing reply. */
async function ask(question: string, answer: string, dataYear: string) {
  mockSendChatMessage.mockResolvedValueOnce({
    answer,
    sources: [{ id: 's1', title: 'TB tuyển sinh', data_year: dataYear }],
    confidence: 0.9,
    status_code: 'SUCCESS',
    status_reason: '',
    data_gap_hints: [],
    internal_status_code: null,
  });
  const user = userEvent.setup();
  const { textarea, sendBtn } = textareaAndSend();
  await user.type(textarea, question);
  fireEvent.click(sendBtn!);
  await waitFor(() => expect(screen.getByText(answer)).toBeInTheDocument());
}

beforeEach(() => {
  mockSendChatMessage.mockReset();
});

describe('ChatSidebar — session history is real state', () => {
  it('shows the question actually asked, and never mock conversation data', async () => {
    renderChatLayout();

    // Before any turn: an explicit empty state, not invented history.
    expect(screen.getByTestId('session-history-empty')).toBeInTheDocument();

    await ask(
      'Điểm chuẩn ngành Công nghệ thông tin năm 2026?',
      'Điểm chuẩn ngành CNTT năm 2026 là 24.5 điểm.',
      '2026',
    );

    const items = screen.getAllByTestId('session-history-item');
    expect(items).toHaveLength(1);
    expect(items[0].textContent).toContain('Điểm chuẩn ngành Công nghệ thông tin năm 2026?');

    // The mock titles that used to be rendered as if they were real history.
    expect(screen.queryByText('Hỏi về ngành CNTT')).not.toBeInTheDocument();
    expect(screen.queryByText('Thông tin học phí')).not.toBeInTheDocument();
    expect(screen.queryByTestId('session-history-empty')).not.toBeInTheDocument();
  });

  it('a new chat clears the session history', async () => {
    renderChatLayout();
    await ask('Học phí ngành Luật năm 2026?', 'Học phí ngành Luật năm 2026 là 15 triệu.', '2026');
    expect(screen.getAllByTestId('session-history-item')).toHaveLength(1);

    const newChat = Array.from(document.querySelectorAll('button')).find((b) =>
      b.textContent?.includes('Chat mới'),
    );
    fireEvent.click(newChat!);

    await waitFor(() => {
      expect(screen.queryAllByTestId('session-history-item')).toHaveLength(0);
    });
    expect(screen.getByTestId('session-history-empty')).toBeInTheDocument();
  });

  it('deleting an entry removes that exchange from the transcript', async () => {
    renderChatLayout();
    await ask('Học phí ngành Luật năm 2026?', 'Học phí ngành Luật năm 2026 là 15 triệu.', '2026');

    const entry = screen.getAllByTestId('session-history-item')[0];
    const deleteBtn = entry.querySelector('button[aria-label="Xóa cuộc trò chuyện"]')!;
    fireEvent.click(deleteBtn);

    await waitFor(() => {
      expect(screen.queryByText('Học phí ngành Luật năm 2026 là 15 triệu.')).not.toBeInTheDocument();
    });
    expect(screen.queryAllByTestId('session-history-item')).toHaveLength(0);
  });
});

describe('SuggestedQuestions — derived from the last answer', () => {
  it('suggests follow-ups about the major and year just asked about', async () => {
    renderChatLayout();
    await ask(
      'Điểm chuẩn ngành Công nghệ thông tin năm 2026?',
      'Điểm chuẩn ngành CNTT năm 2026 là 24.5 điểm.',
      '2026',
    );

    const strip = screen.getByText('✨ Câu hỏi tiếp theo').parentElement!;
    const labels = Array.from(strip.querySelectorAll('button')).map((b) => b.textContent ?? '');

    expect(labels.length).toBeGreaterThan(0);
    for (const label of labels) {
      expect(label.toLowerCase()).toContain('công nghệ thông tin');
      expect(label.trim().endsWith('?')).toBe(true);
    }
    // The year the student asked about carries into the suggestions that are
    // year-scoped (tổ hợp xét tuyển is not).
    expect(labels.some((l) => l.includes('2026'))).toBe(true);
    expect(labels.some((l) => /Học phí ngành Công nghệ thông tin năm 2026\?/i.test(l))).toBe(true);
  });

  it('different questions produce different suggestions (not a static list)', async () => {
    renderChatLayout();
    await ask('Điểm chuẩn ngành Luật năm 2026?', 'Điểm chuẩn ngành Luật là 20 điểm.', '2026');

    const strip = screen.getByText('✨ Câu hỏi tiếp theo').parentElement!;
    const labels = Array.from(strip.querySelectorAll('button')).map((b) => b.textContent ?? '');
    expect(labels.some((l) => l.toLowerCase().includes('luật'))).toBe(true);
    // The old static list's first entry can never appear, because it is
    // derived from whatever the student actually asked about.
    expect(labels.some((l) => l.includes('Thư viện trường có những gì?'))).toBe(false);
  });
});

describe('session-history module', () => {
  const messages: Message[] = [
    { id: 'u1', role: 'user', content: 'Câu một?', timestamp: new Date() },
    { id: 'a1', role: 'assistant', content: 'Đáp một.', timestamp: new Date() },
    { id: 'u2', role: 'user', content: 'Câu hai?', timestamp: new Date() },
    { id: 'a2', role: 'assistant', content: 'Đáp hai.', timestamp: new Date() },
  ];

  it('builds one entry per user turn', () => {
    expect(buildSessionHistory(messages).map((e) => e.messageId)).toEqual(['u1', 'u2']);
  });

  it('removes the question and its answer, keeping later turns', () => {
    expect(removeExchange(messages, 'u1').map((m) => m.id)).toEqual(['u2', 'a2']);
  });
});

describe('follow-up-questions module', () => {
  it('returns nothing when the last turn failed', () => {
    const failed: Message[] = [
      { id: 'u1', role: 'user', content: 'Điểm chuẩn ngành Luật?', timestamp: new Date() },
      {
        id: 'a1',
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        errorKind: 'network',
      },
    ];
    expect(deriveFollowUpQuestions(failed)).toEqual([]);
  });

  it('returns nothing when no subject can be read from the turn', () => {
    const vague: Message[] = [
      { id: 'u1', role: 'user', content: 'Cho tôi hỏi chút?', timestamp: new Date() },
      { id: 'a1', role: 'assistant', content: 'Bạn cần hỏi gì ạ?', timestamp: new Date() },
    ];
    expect(deriveFollowUpQuestions(vague)).toEqual([]);
  });

  it('never repeats the question just asked', () => {
    const asked: Message[] = [
      { id: 'u1', role: 'user', content: 'Học phí ngành Luật năm 2026?', timestamp: new Date() },
      {
        id: 'a1',
        role: 'assistant',
        content: 'Học phí ngành Luật năm 2026 là 15 triệu.',
        timestamp: new Date(),
        sources: [{ id: 's1', title: 'TB', data_year: '2026', confidence: 0.9 }],
      },
    ];
    expect(deriveFollowUpQuestions(asked)).not.toContain('Học phí ngành Luật năm 2026?');
  });
});
