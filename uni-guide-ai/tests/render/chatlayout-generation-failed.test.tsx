/**
 * Ticket 43 — the degraded-generation block on the actual chat screen.
 *
 * Renders the real ChatLayout (not a copy of its mapping logic) with
 * @/lib/api mocked, and asserts what the student sees when /v2 answers with
 * GENERATION_FAILED:
 *
 *  - a block distinct from a normal answer: its own warning label, and NOT
 *    the 'Cần kiểm tra dữ liệu' fallback (that label is about missing data;
 *    here the data was found and the generator died).
 *  - the backend's fallback sentence shown inside that block, never as a
 *    plain answer, with no "Gợi ý bổ sung dữ liệu" dressing it up as a gap.
 *  - a retry button that re-sends the immediately preceding question verbatim
 *    through ChatLayout's own handleSendMessage (never empty, never another
 *    question).
 *  - answer_confidence null → no confidence readout at all; a real number on
 *    SUCCESS → the readout appears.
 *  - SUCCESS passthrough untouched.
 *
 * Both transport shapes are exercised: the streamed meta frame and the
 * non-stream sendChatMessage payload must produce the same screen.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const mockSendChatMessage = vi.fn();
const mockQueryStream = vi.fn();

vi.mock('@/lib/api', () => ({
  sendChatMessage: (...args: unknown[]) => mockSendChatMessage(...args),
  queryStream: (...args: unknown[]) => mockQueryStream(...args),
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
import { GENERATION_FAILED_STATUS } from '@/lib/chat-helpers';

const QUESTION = 'Điểm chuẩn CNTT 2024?';
const FAILURE_LABEL = 'Máy chủ gặp sự cố khi tạo câu trả lời';
const RETRY_ARIA = 'Máy chủ gặp sự cố. Vui lòng thử lại sau ít phút.';
const BE_FALLBACK_ANSWER =
  'Hệ thống đã tìm thấy tài liệu liên quan nhưng dịch vụ tạo câu trả lời ' +
  'đang tạm thời không khả dụng. Vui lòng thử lại sau.';

function renderChatLayout() {
  return render(
    <ThemeProvider>
      <ChatLayout />
    </ThemeProvider>,
  );
}

beforeEach(() => {
  mockSendChatMessage.mockReset();
  mockQueryStream.mockReset();
  // Default: the stream never opens, so the hook falls back to sendChatMessage.
  mockQueryStream.mockImplementation(async () => {
    throw new Error('stream unavailable');
  });
});

/** Type into ChatInput and press the send button. */
async function typeAndSend(text: string) {
  const user = userEvent.setup();
  const textarea = document.querySelector('textarea') as HTMLTextAreaElement;
  expect(textarea).toBeTruthy();
  const buttons = Array.from(document.querySelectorAll('button'));
  const sendBtn = buttons.find((b) => b.getAttribute('aria-label') === 'Gửi tin nhắn');
  expect(sendBtn).toBeTruthy();
  await user.type(textarea, text);
  fireEvent.click(sendBtn!);
}

function failedPayload(overrides: Record<string, unknown> = {}) {
  return {
    answer: BE_FALLBACK_ANSWER,
    sources: [],
    confidence: 0.83,
    answer_confidence: null,
    status_code: GENERATION_FAILED_STATUS,
    status_reason: 'generation_failed',
    internal_status_code: GENERATION_FAILED_STATUS,
    data_gap_hints: [],
    ...overrides,
  };
}

describe('ChatLayout — GENERATION_FAILED renders a distinct failure block (non-stream)', () => {
  it('shows its own warning label, not SUCCESS and not the data-check fallback', async () => {
    mockSendChatMessage.mockResolvedValueOnce(failedPayload());
    renderChatLayout();
    await typeAndSend(QUESTION);

    await waitFor(() => {
      expect(screen.getByText(/Máy chủ gặp sự cố khi tạo câu trả lời/)).toBeInTheDocument();
    });
    // Distinct from a normal answer AND from the missing-data fallback.
    expect(screen.queryByText(/Cần kiểm tra dữ liệu/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Thiếu dữ liệu để trả lời chắc chắn/)).not.toBeInTheDocument();
  });

  it('shows the backend sentence inside the block, without a data-gap suggestion list', async () => {
    mockSendChatMessage.mockResolvedValueOnce(
      failedPayload({ data_gap_hints: ['Cập nhật đề án tuyển sinh 2025'] }),
    );
    renderChatLayout();
    await typeAndSend(QUESTION);

    await waitFor(() => {
      expect(screen.getByText(BE_FALLBACK_ANSWER)).toBeInTheDocument();
    });
    // The stale hints must not dress the outage up as missing data.
    expect(screen.queryByText(/Gợi ý bổ sung dữ liệu/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Cập nhật đề án tuyển sinh 2025/)).not.toBeInTheDocument();
  });

  it('null answer_confidence renders no confidence readout', async () => {
    mockSendChatMessage.mockResolvedValueOnce(failedPayload());
    renderChatLayout();
    await typeAndSend(QUESTION);

    await waitFor(() => {
      expect(screen.getByText(/Máy chủ gặp sự cố khi tạo câu trả lời/)).toBeInTheDocument();
    });
    // Mutation target: a `?? 0.5` default would make this query find a node.
    expect(screen.queryByTestId('answer-confidence')).not.toBeInTheDocument();
  });
});

describe('ChatLayout — streamed failure renders the same screen (criterion 5)', () => {
  it('meta frame produces the same label, retry button and readout absence', async () => {
    mockQueryStream.mockImplementationOnce(
      async (_c: string, handlers: Record<string, Function>) => {
        handlers.onMeta({
          route: 'padded_rag',
          status_code: GENERATION_FAILED_STATUS,
          status_reason: 'generation_failed',
          internal_status_code: GENERATION_FAILED_STATUS,
          answer_confidence: null,
          confidence: 0.83,
          data_gap_hints: [],
          sources: [],
        });
        handlers.onDelta(BE_FALLBACK_ANSWER);
        handlers.onDone(BE_FALLBACK_ANSWER);
      },
    );
    renderChatLayout();
    await typeAndSend(QUESTION);

    await waitFor(() => {
      expect(screen.getByText(/Máy chủ gặp sự cố khi tạo câu trả lời/)).toBeInTheDocument();
    });
    expect(screen.getByText(BE_FALLBACK_ANSWER)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: RETRY_ARIA })).toBeInTheDocument();
    expect(screen.queryByTestId('answer-confidence')).not.toBeInTheDocument();
    // The non-stream fallback must not have fired for a working stream.
    expect(mockSendChatMessage).not.toHaveBeenCalled();
  });
});

describe('ChatLayout — retry affordance (criterion 4)', () => {
  it('clicking re-sends the immediately preceding question, not an empty or other string', async () => {
    mockSendChatMessage.mockResolvedValueOnce(failedPayload());
    mockSendChatMessage.mockResolvedValueOnce({
      answer: 'Điểm chuẩn CNTT 2024 là 22.5.',
      sources: [],
      confidence: 0.9,
      answer_confidence: 0.9,
      status_code: 'SUCCESS',
    });
    renderChatLayout();
    await typeAndSend(QUESTION);

    await waitFor(() => {
      expect(screen.getByText(/Máy chủ gặp sự cố khi tạo câu trả lời/)).toBeInTheDocument();
    });
    expect(mockSendChatMessage).toHaveBeenCalledTimes(1);
    expect(mockSendChatMessage).toHaveBeenLastCalledWith(QUESTION, expect.anything());

    const retryBtn = screen.getByRole('button', { name: RETRY_ARIA });
    fireEvent.click(retryBtn);

    // Second send carries the ORIGINAL question.
    await waitFor(() => {
      expect(mockSendChatMessage).toHaveBeenCalledTimes(2);
    });
    expect(mockSendChatMessage).toHaveBeenLastCalledWith(QUESTION, expect.anything());
    // And the retried answer replaces the failure block with a normal one.
    await waitFor(() => {
      expect(screen.getByText('Điểm chuẩn CNTT 2024 là 22.5.')).toBeInTheDocument();
    });
  });

  it('SUCCESS answers have no retry button', async () => {
    mockSendChatMessage.mockResolvedValueOnce({
      answer: 'Điểm chuẩn CNTT 2024 là 22.5.',
      sources: [],
      confidence: 0.9,
      answer_confidence: 0.9,
      status_code: 'SUCCESS',
    });
    renderChatLayout();
    await typeAndSend(QUESTION);

    await waitFor(() => {
      expect(screen.getByText('Điểm chuẩn CNTT 2024 là 22.5.')).toBeInTheDocument();
    });
    expect(screen.queryByRole('button', { name: RETRY_ARIA })).not.toBeInTheDocument();
    // No banner on a passthrough answer (criterion 8).
    expect(screen.queryByText(/Máy chủ gặp sự cố khi tạo câu trả lời/)).not.toBeInTheDocument();
    // A real number DOES drive the readout — it is data, not a default.
    expect(screen.getByTestId('answer-confidence')).toHaveTextContent('90%');
  });

  it('a leftover internal token on a SUCCESS response does not raise the failure block', async () => {
    // Dev-mode payloads can carry internal_status_code alongside any public
    // code (rag2025 ERROR_EXPOSURE_MODE). The affordance gates on the PUBLIC
    // status only — criterion 1's mutation (flip status_code to SUCCESS) must
    // render a normal answer even when the internal token is present.
    mockSendChatMessage.mockResolvedValueOnce({
      answer: 'Điểm chuẩn CNTT 2024 là 22.5.',
      sources: [],
      confidence: 0.9,
      answer_confidence: 0.9,
      status_code: 'SUCCESS',
      status_reason: '',
      internal_status_code: GENERATION_FAILED_STATUS,
      data_gap_hints: [],
    });
    renderChatLayout();
    await typeAndSend(QUESTION);

    await waitFor(() => {
      expect(screen.getByText('Điểm chuẩn CNTT 2024 là 22.5.')).toBeInTheDocument();
    });
    expect(screen.queryByRole('button', { name: RETRY_ARIA })).not.toBeInTheDocument();
    expect(screen.queryByText(/Máy chủ gặp sự cố khi tạo câu trả lời/)).not.toBeInTheDocument();
  });
});

describe('ChatLayout — empty failed answer is still readable (criterion 6)', () => {
  it('decorates an empty answer with the failure label', async () => {
    mockSendChatMessage.mockResolvedValueOnce(failedPayload({ answer: '' }));
    renderChatLayout();
    await typeAndSend(QUESTION);

    await waitFor(() => {
      expect(screen.getByText(/Máy chủ gặp sự cố khi tạo câu trả lời/)).toBeInTheDocument();
    });
    expect(screen.getByRole('button', { name: RETRY_ARIA })).toBeInTheDocument();
  });
});

describe('ChatLayout — unknown status keeps the old fallback label (criterion 7)', () => {
  it('renders the data-check label without throwing', async () => {
    mockSendChatMessage.mockResolvedValueOnce({
      answer: 'Nội dung mang trạng thái mới từ BE',
      sources: [],
      confidence: 0.5,
      status_code: 'SOMETHING_NEW_FROM_BE',
      status_reason: '',
      internal_status_code: null,
      data_gap_hints: [],
    });
    renderChatLayout();
    await typeAndSend(QUESTION);

    await waitFor(() => {
      expect(screen.getByText(/Cần kiểm tra dữ liệu/)).toBeInTheDocument();
    });
    expect(screen.getByText('Nội dung mang trạng thái mới từ BE')).toBeInTheDocument();
    // The retry affordance is scoped to generation failures only.
    expect(screen.queryByRole('button', { name: RETRY_ARIA })).not.toBeInTheDocument();
  });
});
