/**
 * Ticket 23 — accessibility + responsive smoke tests for the ask/answer
 * surface (no axe dependency: these pin the REAL invariants directly).
 *
 * Covered:
 *  1. keyboard — the citation affordance is reached in tab order and is
 *     operable, while a URL-less citation creates no dead tab stop;
 *  2. screen reader — abstain / out-of-scope / generation-failed / transport
 *     error / processing states are announced via the alert + status live
 *     regions, and streamed tokens are NOT (anti per-token spam);
 *  3. status is never colour-only — the warning banner carries an icon plus
 *     the status-code text, and like/dislike expose aria-pressed + a fill
 *     change rather than a colour swap alone;
 *  4. narrow screens — the citation list wraps instead of dropping chips.
 *
 * Rendered through the real components (MessageBubble / ChatMessages) so the
 * tests fail if the shipped markup regresses, not if a helper does.
 */
import { describe, it, expect } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MessageBubble } from '@/components/chat/MessageBubble';
import { ChatMessages } from '@/components/chat/ChatMessages';
import { ChatSidebar } from '@/components/chat/ChatSidebar';
import type { Message, Source } from '@/lib/chat-types';
import type { SessionHistoryEntry } from '@/lib/session-history';

const T0 = new Date('2026-01-01T00:00:00Z');

function userMsg(content = 'Học phí ngành CNTT năm nay là bao nhiêu?'): Message {
  return { id: 'user-1', role: 'user', content, timestamp: T0 };
}

function botMsg(overrides: Partial<Message> = {}): Message {
  return {
    id: 'bot-1',
    role: 'assistant',
    content: 'Học phí ngành CNTT là 15.000.000đ/năm.',
    timestamp: T0,
    ...overrides,
  };
}

const LINKED_SOURCE: Source = {
  id: 's-linked',
  title: 'Thông báo tuyển sinh 2026 — học phí và học bổng chi tiết',
  url: 'https://husc.edu.vn/thong-bao-2026',
  data_year: '2026',
  confidence: 0.9,
};

describe('ticket 23 criterion 1 — keyboard navigation', () => {
  it('reaches the citation link first in tab order and exposes its full title despite visual truncation', async () => {
    const user = userEvent.setup();
    render(<MessageBubble message={botMsg({ sources: [LINKED_SOURCE] })} />);

    await user.tab();
    const first = document.activeElement as HTMLElement;
    expect(first.tagName).toBe('A');
    expect(first).toHaveAttribute('href', LINKED_SOURCE.url);

    // The visible label may shrink to 120px, but AT and the tooltip keep the
    // whole title (truncation is CSS-only).
    const titleSpan = first.querySelector('span.truncate') as HTMLElement;
    expect(titleSpan.textContent).toBe(LINKED_SOURCE.title);
    expect(titleSpan).toHaveAttribute('title', LINKED_SOURCE.title);

    // …then the answer actions, in order, all reachable from the keyboard.
    await user.tab();
    expect(document.activeElement).toHaveAccessibleName('Sao chép');
    await user.tab();
    expect(document.activeElement).toHaveAccessibleName('Hay lắm! 👍');
    await user.tab();
    expect(document.activeElement).toHaveAccessibleName('Chưa ổn 👎');
  });

  it('renders a citation without a URL as text, not as a dead tab stop', () => {
    render(
      <MessageBubble
        message={botMsg({
          sources: [{ id: 's-plain', title: 'Nguồn không có liên kết', confidence: 0.7 }],
        })}
      />
    );

    const list = screen.getByRole('list', { name: 'Nguồn tham khảo' });
    expect(within(list).queryAllByRole('button')).toHaveLength(0);
    expect(within(list).queryAllByRole('link')).toHaveLength(0);
    expect(within(list).getByText('Nguồn không có liên kết')).toHaveAttribute(
      'title',
      'Nguồn không có liên kết'
    );
  });

  it('keeps the answer actions operable when the citation has no URL (no focus swallowed)', async () => {
    const user = userEvent.setup();
    render(
      <MessageBubble
        message={botMsg({ sources: [{ id: 's-plain', title: 'Nguồn tĩnh', confidence: 0.7 }] })}
      />
    );

    await user.tab();
    expect(document.activeElement).toHaveAccessibleName('Sao chép');
    await user.tab();
    expect(document.activeElement).toHaveAccessibleName('Hay lắm! 👍');
  });
});

describe('ticket 23 criterion 1 — sidebar keeps every action on the keyboard path', () => {
  const entry: SessionHistoryEntry = {
    messageId: 'm-1',
    question: 'Học phí ngành Luật năm 2026?',
    timestamp: T0,
  };

  function renderSidebar() {
    return render(
      <ChatSidebar
        isOpen
        onClose={() => {}}
        onNewChat={() => {}}
        history={[entry]}
        activeMessageId="m-1"
        onSelectEntry={() => {}}
        onDeleteEntry={() => {}}
      />
    );
  }

  it('reaches the delete control one Tab after the entry control (the old nested button trapped it)', async () => {
    const user = userEvent.setup();
    renderSidebar();

    // The jump target is marked for AT without relying on the highlight colour.
    const select = screen.getByRole('button', { name: /Học phí ngành Luật/ });
    expect(select).toHaveAttribute('aria-current', 'true');

    select.focus();
    await user.tab();
    expect(document.activeElement).toHaveAccessibleName('Xóa cuộc trò chuyện');
  });

  it('never nests a button inside a button', () => {
    const { container } = renderSidebar();
    expect(container.querySelector('button button')).toBeNull();
  });
});

describe('ticket 23 criterion 4 — narrow screens keep citations', () => {
  it('wraps the citation list and renders every source as its own list item', () => {
    const sources: Source[] = [
      LINKED_SOURCE,
      { id: 's-2', title: 'Quy chế tuyển sinh đại học chính quy', confidence: 0.6 },
      { id: 's-3', title: 'Danh mục học bổng', data_year: '2026', confidence: 0.5 },
    ];
    render(<MessageBubble message={botMsg({ sources })} />);

    const list = screen.getByRole('list', { name: 'Nguồn tham khảo' });
    // flex-wrap is the mechanism that prevents 320px clipping/overlap — the
    // chips flow onto extra rows instead of being cut off.
    expect(list.className).toContain('flex-wrap');
    expect(within(list).getAllByRole('listitem')).toHaveLength(3);
  });
});

describe('ticket 23 criterion 2 — screen-reader announcements', () => {
  it('announces an abstain answer as an alert with the Vietnamese label', () => {
    render(
      <ChatMessages
        messages={[
          userMsg(),
          botMsg({ statusCode: 'INSUFFICIENT_DATA', statusReason: 'Kho dữ liệu chưa có học phí 2026' }),
        ]}
        isTyping={false}
      />
    );

    expect(screen.getByRole('alert').textContent).toContain('Thiếu dữ liệu để trả lời chắc chắn');
  });

  it('announces an out-of-scope answer as an alert', () => {
    render(
      <ChatMessages
        messages={[userMsg(), botMsg({ statusCode: 'NOT_IN_HUSC_SCOPE' })]}
        isTyping={false}
      />
    );

    expect(screen.getByRole('alert').textContent).toContain('Ngoài phạm vi tuyển sinh HUSC');
  });

  it('announces a generation failure with the retry hint', () => {
    render(
      <ChatMessages
        messages={[
          userMsg(),
          botMsg({ statusCode: 'GENERATION_FAILED', retryQuery: 'Học phí ngành CNTT?' }),
        ]}
        isTyping={false}
      />
    );

    const alert = screen.getByRole('alert').textContent ?? '';
    expect(alert).toContain('Không tạo được câu trả lời');
    expect(alert).toContain('bấm nút thử lại');
  });

  it('announces a transport error with the localized Vietnamese copy', () => {
    render(
      <ChatMessages
        messages={[userMsg(), botMsg({ content: '', errorKind: 'network' })]}
        isTyping={false}
      />
    );

    expect(screen.getByRole('alert').textContent).toContain('Không kết nối được máy chủ');
  });

  it('announces the processing stage politely', () => {
    render(
      <ChatMessages
        messages={[userMsg(), botMsg({ content: '', stage: 'Đang tra cứu dữ liệu tuyển sinh…' })]}
        isTyping
      />
    );

    expect(screen.getByRole('status').textContent).toContain('Đang tra cứu dữ liệu tuyển sinh…');
    expect(screen.getByRole('alert').textContent).toBe('');
  });

  it('announces a finished answer once, with its source count', () => {
    render(
      <ChatMessages
        messages={[userMsg(), botMsg({ statusCode: 'SUCCESS', sources: [LINKED_SOURCE, { id: 's-2', title: 'Quy chế', confidence: 0.5 }] })]}
        isTyping={false}
      />
    );

    const polite = screen.getByRole('status').textContent ?? '';
    expect(polite).toContain('Câu trả lời đã sẵn sàng');
    expect(polite).toContain('2 nguồn tham khảo');
  });

  it('does not announce streamed tokens: the transcript is a role=log with aria-live off', () => {
    render(
      <ChatMessages
        messages={[userMsg(), botMsg({ content: 'Học phí ngành CNTT là 15.000.0', stage: null })]}
        isTyping
      />
    );

    const log = screen.getByRole('log');
    expect(log).toHaveAttribute('aria-live', 'off');
    expect(log).toHaveAttribute('aria-label', 'Dòng hội thoại');
    // No live region fires while the answer is still streaming.
    expect(screen.getByRole('status').textContent).toBe('');
    expect(screen.getByRole('alert').textContent).toBe('');
  });

  it('marks the sidebar-jumped message with aria-current, not colour alone', () => {
    render(
      <ChatMessages messages={[botMsg()]} isTyping={false} activeMessageId="bot-1" />
    );

    const current = document.querySelector('[aria-current="true"]');
    expect(current).not.toBeNull();
    expect(current).toHaveAttribute('data-message-id', 'bot-1');
  });
});

describe('ticket 23 criterion 3 — status is never colour-only', () => {
  it('renders the warning banner with an icon plus the status code and reason text', () => {
    render(
      <MessageBubble
        message={botMsg({
          statusCode: 'INSUFFICIENT_DATA',
          statusReason: 'Kho dữ liệu chưa có học phí 2026',
        })}
      />
    );

    const banner = screen.getByTestId('status-banner');
    // icon (shape) + text — the amber fill is redundant, not load-bearing
    expect(banner.querySelector('svg')).not.toBeNull();
    expect(banner).toHaveTextContent('INSUFFICIENT_DATA');
    expect(banner).toHaveTextContent('Kho dữ liệu chưa có học phí 2026');
  });

  it('uses an icon next to the localized error copy instead of colour alone', () => {
    render(<MessageBubble message={botMsg({ content: '' })} errorKind="server" />);

    expect(screen.getByText(/Máy chủ gặp sự cố/).textContent).toContain('⚠️');
  });

  it('exposes the pressed state of like/dislike via aria-pressed and a solid icon', async () => {
    const user = userEvent.setup();
    render(<MessageBubble message={botMsg()} />);

    const like = screen.getByRole('button', { name: 'Hay lắm! 👍' });
    expect(like).toHaveAttribute('aria-pressed', 'false');

    await user.click(like);
    expect(like).toHaveAttribute('aria-pressed', 'true');
    // fill changes outline -> solid: a non-colour cue visible to sighted users
    expect(like.querySelector('svg')).toHaveAttribute('fill', 'currentColor');

    await user.click(like);
    expect(like).toHaveAttribute('aria-pressed', 'false');
    expect(like.querySelector('svg')).toHaveAttribute('fill', 'none');
  });
});
