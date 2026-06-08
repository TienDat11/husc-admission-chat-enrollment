/**
 * MARKDOWN / XSS RENDER TESTS for <MessageBubble>
 *
 * Uses adversarial content to surface edge cases in ReactMarkdown + remark-gfm rendering.
 * XSS guard: assert <script> tags are never injected into the live DOM.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MessageBubble } from '@/components/chat/MessageBubble';
import type { Message } from '@/lib/chat-types';

// ---------------------------------------------------------------------------
// Factory — builds a minimal assistant Message
// ---------------------------------------------------------------------------
function makeMessage(content: string, overrides: Partial<Message> = {}): Message {
  return {
    id: 'test-id',
    role: 'assistant',
    content,
    timestamp: new Date('2025-01-01T00:00:00Z'),
    sources: [],
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// (a) Valid GFM table
// ---------------------------------------------------------------------------
describe('MessageBubble — GFM table rendering', () => {
  it('(a) renders a valid GFM table without throwing', () => {
    const content = `| Ngành | Điểm chuẩn |
| ----- | ---------- |
| CNTT  | 22.5       |
| Toán  | 21.0       |`;
    expect(() => render(<MessageBubble message={makeMessage(content)} />)).not.toThrow();
  });

  it('(a) valid GFM table produces <table> element in DOM', () => {
    const content = `| Ngành | Điểm chuẩn |
| ----- | ---------- |
| CNTT  | 22.5       |`;
    render(<MessageBubble message={makeMessage(content)} />);
    expect(document.querySelector('table')).not.toBeNull();
  });
});

// ---------------------------------------------------------------------------
// (b) Mismatched-column table — remark-gfm should degrade gracefully
// ---------------------------------------------------------------------------
describe('MessageBubble — mismatched-column table', () => {
  it('(b) renders mismatched-column table without throwing', () => {
    const content = `| A | B | C |
| - | - |
| 1 | 2 | 3 | 4 |`;
    expect(() => render(<MessageBubble message={makeMessage(content)} />)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// (c) No-separator table (no --- row) — should NOT render as table
// ---------------------------------------------------------------------------
describe('MessageBubble — no-separator table', () => {
  it('(c) renders no-separator table without throwing', () => {
    const content = `| A | B |
| 1 | 2 |`;
    expect(() => render(<MessageBubble message={makeMessage(content)} />)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// (d) Bold + italic + strikethrough
// ---------------------------------------------------------------------------
describe('MessageBubble — bold / italic / strikethrough', () => {
  it('(d) renders **bold** without throwing', () => {
    const content = '**Điểm chuẩn** rất quan trọng';
    expect(() => render(<MessageBubble message={makeMessage(content)} />)).not.toThrow();
  });

  it('(d) renders _italic_ without throwing', () => {
    const content = '_Lưu ý:_ kiểm tra kỹ';
    expect(() => render(<MessageBubble message={makeMessage(content)} />)).not.toThrow();
  });

  it('(d) renders ~~strikethrough~~ without throwing', () => {
    const content = '~~Thông tin cũ~~ — đã lỗi thời';
    expect(() => render(<MessageBubble message={makeMessage(content)} />)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// (e) Unclosed **bold — should not crash, just render as literal text
// ---------------------------------------------------------------------------
describe('MessageBubble — unclosed bold', () => {
  it('(e) unclosed **bold does not throw', () => {
    const content = 'Giá trị **chưa đóng thẻ';
    expect(() => render(<MessageBubble message={makeMessage(content)} />)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// (f) Pipe inside table cell
// ---------------------------------------------------------------------------
describe('MessageBubble — pipe inside table cell', () => {
  it('(f) escaped pipe in cell renders without throwing', () => {
    const content = `| Môn | Ghi chú |
| --- | ------- |
| Toán \\| Lý | Xét tuyển |`;
    expect(() => render(<MessageBubble message={makeMessage(content)} />)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// (g) XSS: raw <script> tag — MUST NOT execute or appear in DOM as script
// ---------------------------------------------------------------------------
describe('MessageBubble — XSS guard', () => {
  it('(g) raw <script>alert(1)</script> does not create a live <script> element', () => {
    const content = '<script>alert(1)</script>';
    render(<MessageBubble message={makeMessage(content)} />);
    // ReactMarkdown sanitizes by not rendering raw HTML by default (no rehype-raw)
    const scripts = document.querySelectorAll('script');
    // Filter to only injected test scripts (not vitest internals that may exist)
    const alertScripts = Array.from(scripts).filter(s => s.textContent?.includes('alert(1)'));
    expect(alertScripts).toHaveLength(0);
  });

  it('(g) <img onerror> XSS vector does not execute', () => {
    const content = '<img src="x" onerror="window.__xss_fired=true">';
    render(<MessageBubble message={makeMessage(content)} />);
    expect((window as any).__xss_fired).toBeUndefined();
  });

  it('(g) javascript: href link does not appear as raw anchor with js: protocol', () => {
    const content = '[click me](javascript:alert(1))';
    render(<MessageBubble message={makeMessage(content)} />);
    const links = document.querySelectorAll('a[href^="javascript:"]');
    // ReactMarkdown strips javascript: URLs (react-markdown v10 default behavior)
    // INTENDED-RED if this fails — means XSS vector not neutralized
    expect(links).toHaveLength(0);
  });

  it('(g) markdown injection via data: URI does not leak', () => {
    const content = '[evil](data:text/html,<script>alert(1)</script>)';
    render(<MessageBubble message={makeMessage(content)} />);
    const dataLinks = document.querySelectorAll('a[href^="data:"]');
    // data: URIs in <a> are potentially dangerous; assert none survive
    // INTENDED-RED if this fails
    expect(dataLinks).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// (h) Empty string — should render without throwing
// ---------------------------------------------------------------------------
describe('MessageBubble — empty content', () => {
  it('(h) empty string renders without throwing', () => {
    expect(() => render(<MessageBubble message={makeMessage('')} />)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// User message role — content should be plain text, not markdown rendered
// ---------------------------------------------------------------------------
describe('MessageBubble — user role', () => {
  it('user message renders raw text content (no ReactMarkdown processing)', () => {
    const content = '**not bold** just text';
    render(<MessageBubble message={{ ...makeMessage(content), role: 'user' }} />);
    expect(screen.getByText('**not bold** just text')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Status banner — shown only when statusCode !== 'SUCCESS'
// ---------------------------------------------------------------------------
describe('MessageBubble — status banner', () => {
  it('shows status banner when statusCode is INSUFFICIENT_DATA', () => {
    const msg = makeMessage('Some answer', {
      statusCode: 'INSUFFICIENT_DATA',
      statusReason: 'Thiếu dữ liệu',
    });
    render(<MessageBubble message={msg} />);
    expect(screen.getByText('INSUFFICIENT_DATA')).toBeInTheDocument();
    expect(screen.getByText('Thiếu dữ liệu')).toBeInTheDocument();
  });

  it('does NOT show banner when statusCode is SUCCESS', () => {
    const msg = makeMessage('All good', { statusCode: 'SUCCESS' });
    render(<MessageBubble message={msg} />);
    expect(screen.queryByText('SUCCESS')).toBeNull();
  });

  it('does NOT show banner when statusCode is absent (undefined)', () => {
    const msg = makeMessage('Response from /v2/query with no statusCode');
    // statusCode is undefined — simulates /v2 response after naive migration
    render(<MessageBubble message={msg} />);
    // No banner element should appear
    const banners = document.querySelectorAll('.border-amber-300');
    expect(banners).toHaveLength(0);
  });
});
