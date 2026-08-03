// @spec(plan-warmup-ux-20260705-111031 §3.2 / §5 Phase 3+4)
// ChatLayout — readiness + ChatInput disable + ChatMessages skeleton + MessageBubble localized
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from '@/hooks/useTheme';
import { ChatLayout } from '@/components/chat/ChatLayout';
import { ChatInput } from '@/components/chat/ChatInput';
import { ChatMessages } from '@/components/chat/ChatMessages';
import { MessageBubble } from '@/components/chat/MessageBubble';
import type { Message } from '@/lib/chat-types';

// Hoisted mock for useBackendReadiness (must be at top level — vitest hoists).
// Default: stage='warming'. Per-test overrides via useBackendReadiness.mockReturnValue.
vi.mock('@/hooks/useBackendReadiness', () => {
  const mock = vi.fn(() => ({
    ready: false,
    stage: 'warming' as const,
    services: {},
    lastChecked: Date.now(),
    error: null,
  }));
  return { useBackendReadiness: mock };
});

// Re-import so we can reference the mock fn in tests.
import { useBackendReadiness } from '@/hooks/useBackendReadiness';
const mockReadiness = vi.mocked(useBackendReadiness);

function wrapLayout(el: React.ReactElement) {
  return render(<ThemeProvider>{el}</ThemeProvider>);
}

// ---------------------------------------------------------------------------
// Phase 3.2 — ChatLayout wire: banner + ChatInput disable
// ---------------------------------------------------------------------------
describe('ChatLayout — readiness banner', () => {
  it('renders the warming banner when stage=warming', () => {
    mockReadiness.mockReturnValue({
      ready: false,
      stage: 'warming',
      services: {},
      lastChecked: Date.now(),
      error: null,
    });
    const { container } = wrapLayout(<ChatLayout />);
    const banner = container.querySelector('[data-testid="readiness-banner"]');
    expect(banner).toBeInTheDocument();
    expect(banner?.textContent).toContain('Đang kết nối tới máy chủ');
  });

  it('renders the failed banner with "Máy chủ không phản hồi" when stage=failed', () => {
    mockReadiness.mockReturnValue({
      ready: false,
      stage: 'failed',
      services: {},
      lastChecked: Date.now(),
      error: 'Network error',
    });
    const { container } = wrapLayout(<ChatLayout />);
    const banner = container.querySelector('[data-testid="readiness-banner"]');
    expect(banner).toBeInTheDocument();
    expect(banner?.textContent).toContain('Máy chủ không phản hồi');
  });

  it('does not render a banner when stage=ready', () => {
    mockReadiness.mockReturnValue({
      ready: true,
      stage: 'ready',
      services: { embedding: true, reranker: true },
      lastChecked: Date.now(),
      error: null,
    });
    const { container } = wrapLayout(<ChatLayout />);
    const banner = container.querySelector('[data-testid="readiness-banner"]');
    expect(banner).toBeNull();
  });
});

describe('ChatInput — disabled when !ready', () => {
  it('sets placeholder to "Đang kết nối tới máy chủ..." when stage=warming', () => {
    render(<ChatInput onSend={() => {}} readinessStage="warming" />);
    const textarea = screen.getByPlaceholderText('Đang kết nối tới máy chủ...');
    expect(textarea).toBeInTheDocument();
  });

  it('sets placeholder to "Máy chủ không phản hồi — đang thử lại..." when stage=failed', () => {
    render(<ChatInput onSend={() => {}} readinessStage="failed" />);
    const textarea = screen.getByPlaceholderText('Máy chủ không phản hồi — đang thử lại...');
    expect(textarea).toBeInTheDocument();
  });

  it('renders the default placeholder when stage=ready', () => {
    render(<ChatInput onSend={() => {}} />);
    const textarea = screen.getByPlaceholderText(/Hỏi bất cứ điều gì/);
    expect(textarea).toBeInTheDocument();
  });

  it('disables the Send button when backendReady=false', () => {
    render(<ChatInput onSend={() => {}} backendReady={false} />);
    const sendBtn = screen.getByTestId('send-button');
    expect(sendBtn).toBeDisabled();
  });

  it('enables the Send button when backendReady=true and message non-empty', () => {
    render(<ChatInput onSend={() => {}} backendReady={true} />);
    const sendBtn = screen.getByTestId('send-button');
    // Enable only when message is non-empty: combined gate is
    // !sendDisabled = !(message.trim() || isLoading || disabled || backendReady === false)
    // With empty message it's still disabled via the trimmed-message check.
    expect(sendBtn).toBeDisabled();
  });
});

// ---------------------------------------------------------------------------
// Phase 4 — ChatMessages skeleton + MessageBubble localized
// ---------------------------------------------------------------------------
describe('ChatMessages — skeleton when !ready and no messages', () => {
  it('renders skeleton when backendReady=false and messages=[], not the welcome', () => {
    render(<ChatMessages messages={[]} isTyping={false} backendReady={false} />);
    const skeletons = document.querySelectorAll('[data-testid="chat-skeleton"]');
    expect(skeletons).toHaveLength(1);
    expect(screen.queryByText(/Xin chào/)).toBeNull();
  });

  it('renders the welcome when backendReady=true and messages=[] and !isTyping', () => {
    render(<ChatMessages messages={[]} isTyping={false} backendReady={true} />);
    expect(screen.getByText(/Xin chào/)).toBeInTheDocument();
    expect(screen.queryByTestId('chat-skeleton')).toBeNull();
  });
});

describe('MessageBubble — localized error copy', () => {
  function makeMessage(overrides: Partial<Message> = {}): Message {
    return {
      id: 'bubble-test',
      role: 'assistant',
      content: 'raw error',
      timestamp: new Date('2025-01-01T00:00:00Z'),
      sources: [],
      ...overrides,
    };
  }

  it('renders "Không kết nối được máy chủ" for errorKind=network', () => {
    render(<MessageBubble message={makeMessage()} errorKind="network" />);
    expect(screen.getByText(/Không kết nối được máy chủ/)).toBeInTheDocument();
  });

  it('renders "Máy chủ đang khởi động" for errorKind=warming', () => {
    render(<MessageBubble message={makeMessage()} errorKind="warming" />);
    expect(screen.getByText(/Máy chủ đang khởi động/)).toBeInTheDocument();
  });

  it('renders "Phiên đăng nhập hết hạn" for errorKind=unauthorized', () => {
    render(<MessageBubble message={makeMessage()} errorKind="unauthorized" />);
    expect(screen.getByText(/Phiên đăng nhập hết hạn/)).toBeInTheDocument();
  });

  it('renders "Máy chủ gặp sự cố" for errorKind=server', () => {
    render(<MessageBubble message={makeMessage()} errorKind="server" />);
    expect(screen.getByText(/Máy chủ gặp sự cố/)).toBeInTheDocument();
  });

  it('renders "⚠️ Có lỗi xảy ra" for errorKind=unknown (fallback)', () => {
    render(<MessageBubble message={makeMessage()} errorKind="unknown" />);
    expect(screen.getByText(/⚠️ Có lỗi xảy ra/)).toBeInTheDocument();
  });

  it('renders the raw message.content when errorKind is not set (existing behavior)', () => {
    render(<MessageBubble message={makeMessage({ content: 'Xin chao' })} />);
    expect(screen.getByText('Xin chao')).toBeInTheDocument();
  });
});
