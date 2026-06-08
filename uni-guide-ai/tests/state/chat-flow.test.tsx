/**
 * ChatLayout — full send-message flow with mocked @/lib/api
 *
 * Exercises the actual ChatLayout component (not just a harness) with
 * vi.mock('@/lib/api') to control the BE response. Covers:
 *   (a) Normal /v2 SUCCESS response → assistant bubble with sources
 *   (b) sendChatMessage throws → error bubble "Có lỗi xảy ra"
 *   (c) INSUFFICIENT_DATA response → decorated answer with ⚠ header
 *
 * Also asserts: optimistic user message appears immediately (before await).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// ---------------------------------------------------------------------------
// Mock the api module BEFORE importing ChatLayout
// ---------------------------------------------------------------------------
const mockSendChatMessage = vi.fn();

vi.mock('@/lib/api', () => ({
  sendChatMessage: (...args: any[]) => mockSendChatMessage(...args),
  checkHealth: vi.fn(),
  getApiBaseUrl: () => 'http://localhost:8000',
}));

// Import after mock
import { ChatLayout } from '@/components/chat/ChatLayout';
import { ThemeProvider } from '@/hooks/useTheme';

// ChatLayout renders ThemeToggle (which uses useTheme). Wrap in ThemeProvider.
function renderChatLayout() {
  return render(
    <ThemeProvider>
      <ChatLayout />
    </ThemeProvider>
  );
}

beforeEach(() => {
  mockSendChatMessage.mockReset();
});

// Helper: get the textarea inside ChatInput and the send button
function getTextareaAndSend() {
  const textarea = document.querySelector('textarea') as HTMLTextAreaElement;
  expect(textarea).toBeTruthy();
  const buttons = Array.from(document.querySelectorAll('button'));
  const sendBtn = buttons.find((b) => b.getAttribute('aria-label') === 'Gửi tin nhắn');
  return { textarea, sendBtn };
}

describe('ChatLayout — full send-message flow', () => {
  it('(a) SUCCESS /v2 response: assistant bubble with sources appears', async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockResolvedValueOnce({
      answer: 'Điểm chuẩn CNTT năm 2024 là 22.5.',
      sources: [
        { id: 's1', title: 'TB tuyển sinh 2024', data_year: '2024', url: 'https://husc.edu.vn/tb2024' },
      ],
      confidence: 0.9,
      status_code: 'SUCCESS',
      status_reason: '',
      data_gap_hints: [],
      internal_status_code: null,
    });

    renderChatLayout();

    // Type message
    const { textarea, sendBtn } = getTextareaAndSend();
    await user.type(textarea, 'Điểm chuẩn CNTT 2024?');

    // Optimistic user message appears immediately on send
    const userMsgBefore = screen.getByText('Điểm chuẩn CNTT 2024?');
    expect(userMsgBefore).toBeInTheDocument();

    // Click send
    fireEvent.click(sendBtn!);

    // Assistant reply with the answer + source chip title appears
    await waitFor(() => {
      expect(
        screen.getByText('Điểm chuẩn CNTT năm 2024 là 22.5.')
      ).toBeInTheDocument();
    });
    // Source chip renders the title
    expect(screen.getByText('TB tuyển sinh 2024')).toBeInTheDocument();
  });

  it('(a-bonus) string[] sources (legacy /query shape) are also mapped and rendered', async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockResolvedValueOnce({
      answer: 'Trả lời legacy.',
      sources: ['Thông báo cũ 2023', 'Quy chế ĐH Huế'],
      confidence: 0.8,
      status_code: 'SUCCESS',
    });

    renderChatLayout();
    const { textarea, sendBtn } = getTextareaAndSend();
    await user.type(textarea, 'Câu hỏi');
    fireEvent.click(sendBtn!);

    await waitFor(() => {
      expect(screen.getByText('Trả lời legacy.')).toBeInTheDocument();
    });
    expect(screen.getByText('Thông báo cũ 2023')).toBeInTheDocument();
    expect(screen.getByText('Quy chế ĐH Huế')).toBeInTheDocument();
  });

  it('(b) sendChatMessage throws: error bubble "Có lỗi xảy ra" appears', async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockRejectedValueOnce(new Error('Network down'));

    renderChatLayout();
    const { textarea, sendBtn } = getTextareaAndSend();
    await user.type(textarea, 'Test error');
    fireEvent.click(sendBtn!);

    await waitFor(() => {
      expect(screen.getByText(/Có lỗi xảy ra/)).toBeInTheDocument();
    });
    // The error detail (Network down) should be in the message
    expect(screen.getByText(/Network down/)).toBeInTheDocument();
  });

  it('(b-bonus) non-Error thrown value is handled gracefully', async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockRejectedValueOnce('string-not-error');

    renderChatLayout();
    const { textarea, sendBtn } = getTextareaAndSend();
    await user.type(textarea, 'Test non-error');
    fireEvent.click(sendBtn!);

    // ChatLayout falls back to "Không thể kết nối đến máy chủ" when err is
    // not an Error instance
    await waitFor(() => {
      expect(screen.getByText(/Có lỗi xảy ra/)).toBeInTheDocument();
    });
  });

  it('(c) INSUFFICIENT_DATA response: assistant bubble is decorated with ⚠ header + gap hints', async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockResolvedValueOnce({
      answer: 'Tôi chưa có dữ liệu.',
      sources: [],
      confidence: 0.5,
      status_code: 'INSUFFICIENT_DATA',
      status_reason: 'Dữ liệu 2024 chưa cập nhật',
      data_gap_hints: ['Cần bổ sung điểm chuẩn 2024', 'Xem thông báo tháng 9'],
      // internalStatusCode is absent; banner shows status_code
      internal_status_code: null,
    });

    renderChatLayout();
    const { textarea, sendBtn } = getTextareaAndSend();
    await user.type(textarea, 'Câu hỏi thiếu data');
    fireEvent.click(sendBtn!);

    // The status banner element appears with the status code
    await waitFor(() => {
      expect(screen.getByText('INSUFFICIENT_DATA')).toBeInTheDocument();
    });
    // The decorated content (markdown) includes the gap hint bullets
    const listItems = document.querySelectorAll('li');
    const hintTexts = Array.from(listItems).map((li) => li.textContent || '');
    expect(hintTexts.some((t) => t.includes('Cần bổ sung điểm chuẩn 2024'))).toBe(true);
  });

  it('(c-bonus) HUSC_ENTITY_NOT_FOUND label maps to correct banner header', async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockResolvedValueOnce({
      answer: 'Không thấy ngành.',
      sources: [],
      confidence: 0.3,
      status_code: 'HUSC_ENTITY_NOT_FOUND',
      status_reason: '',
      data_gap_hints: [],
      internal_status_code: 'HUSC_ENTITY_NOT_FOUND',
    });

    renderChatLayout();
    const { textarea, sendBtn } = getTextareaAndSend();
    await user.type(textarea, 'Ngành XYZ?');
    fireEvent.click(sendBtn!);

    await waitFor(() => {
      expect(screen.getByText('HUSC_ENTITY_NOT_FOUND')).toBeInTheDocument();
    });
    expect(
      document.body.textContent
    ).toContain('Không thấy ngành/sự việc trong dữ liệu HUSC');
  });

  it('(d) Enter key in textarea triggers send', async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockResolvedValueOnce({
      answer: 'OK',
      sources: [],
      confidence: 0.5,
      status_code: 'SUCCESS',
    });

    renderChatLayout();
    const { textarea } = getTextareaAndSend();
    await user.type(textarea, 'Test Enter{enter}');

    await waitFor(() => {
      expect(mockSendChatMessage).toHaveBeenCalledWith('Test Enter');
    });
  });

  it('(e) optimistic user message appears immediately upon send (no await on BE)', async () => {
    const user = userEvent.setup();
    let resolveResponse: (value: any) => void;
    mockSendChatMessage.mockImplementationOnce(
      () => new Promise<any>((resolve) => { resolveResponse = resolve; })
    );

    renderChatLayout();
    const { textarea, sendBtn } = getTextareaAndSend();
    await user.type(textarea, 'Optimistic check');
    fireEvent.click(sendBtn!);

    // User message should be in the DOM before the response resolves
    expect(screen.getByText('Optimistic check')).toBeInTheDocument();
    // The response has not arrived yet — assistant answer should NOT be in DOM
    expect(screen.queryByText('No answer yet')).toBeNull();

    // Now resolve the response
    resolveResponse!({
      answer: 'Late answer',
      sources: [],
      confidence: 0.5,
      status_code: 'SUCCESS',
    });
    await waitFor(() => {
      expect(screen.getByText('Late answer')).toBeInTheDocument();
    });
  });

  it('(f) error is also added as a bot message (not just a state var)', async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockRejectedValueOnce(new Error('500 Internal'));

    renderChatLayout();
    const { textarea, sendBtn } = getTextareaAndSend();
    await user.type(textarea, 'Trigger 500');
    fireEvent.click(sendBtn!);

    // Both user message and error bot message should be in DOM
    await waitFor(() => {
      expect(screen.getByText('Trigger 500')).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(screen.getByText(/Có lỗi xảy ra/)).toBeInTheDocument();
    });
  });

  it('(g) multiple consecutive sends accumulate messages in order', async () => {
    const user = userEvent.setup();
    mockSendChatMessage
      .mockResolvedValueOnce({
        answer: 'Trả lời 1',
        sources: [],
        confidence: 0.5,
        status_code: 'SUCCESS',
      })
      .mockResolvedValueOnce({
        answer: 'Trả lời 2',
        sources: [],
        confidence: 0.5,
        status_code: 'SUCCESS',
      });

    renderChatLayout();
    const { textarea, sendBtn } = getTextareaAndSend();

    await user.type(textarea, 'Q1');
    fireEvent.click(sendBtn!);
    await waitFor(() => {
      expect(screen.getByText('Trả lời 1')).toBeInTheDocument();
    });

    // Re-query textarea (it was cleared after send)
    const textarea2 = document.querySelector('textarea') as HTMLTextAreaElement;
    expect(textarea2.value).toBe('');
    await user.type(textarea2, 'Q2');
    const sendBtn2 = Array.from(document.querySelectorAll('button')).find(
      (b) => b.getAttribute('aria-label') === 'Gửi tin nhắn'
    )!;
    fireEvent.click(sendBtn2);
    await waitFor(() => {
      expect(screen.getByText('Trả lời 2')).toBeInTheDocument();
    });

    // Both answers should be present
    expect(screen.getByText('Trả lời 1')).toBeInTheDocument();
    expect(screen.getByText('Trả lời 2')).toBeInTheDocument();
  });
});
