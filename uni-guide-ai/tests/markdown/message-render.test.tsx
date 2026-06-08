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

// ---------------------------------------------------------------------------
// (i) Very long table — 50 rows — no crash, all <tr> rendered
// ---------------------------------------------------------------------------
describe('MessageBubble — very long table (50 rows)', () => {
  it('renders a 50-row table without throwing and emits 50 <tr> elements', () => {
    const header = `| Mã ngành | Tên ngành | Điểm chuẩn |`;
    const sep = `| --- | --- | --- |`;
    const rows = Array.from({ length: 50 }, (_, i) =>
      `| ${7500000 + i} | Ngành ${i} | ${15 + (i % 10)}.${i % 10} |`
    ).join('\n');
    const content = [header, sep, ...rows.split('\n')].join('\n');
    expect(() => render(<MessageBubble message={makeMessage(content)} />)).not.toThrow();
    const trs = document.querySelectorAll('tr');
    // 1 header + 50 data rows = 51
    expect(trs.length).toBeGreaterThanOrEqual(50);
  });
});

// ---------------------------------------------------------------------------
// (j) Markdown link inside a table cell — should render as <a>
// ---------------------------------------------------------------------------
describe('MessageBubble — markdown link inside a table cell', () => {
  it('renders a [link](url) inside a table cell as <a>', () => {
    const content = `| Tài liệu | Liên kết |
| --- | --- |
| Quy chế | [Xem tại đây](https://husc.edu.vn/quyche) |`;
    render(<MessageBubble message={makeMessage(content)} />);
    const anchor = document.querySelector('a[href="https://husc.edu.vn/quyche"]');
    expect(anchor).not.toBeNull();
    expect(anchor!.textContent).toBe('Xem tại đây');
  });
});

// ---------------------------------------------------------------------------
// (k) Vietnamese diacritics + emoji in content
// ---------------------------------------------------------------------------
describe('MessageBubble — Vietnamese diacritics + emoji', () => {
  it('renders diacritics and emoji without throwing', () => {
    const content = `Chào bạn! 👋 Trường **Đại học Khoa học** (HUSC) tuyển sinh năm nay với các ngành: Công nghệ thông tin, Trí tuệ nhân tạo, Khoa học dữ liệu 🎓.`;
    expect(() => render(<MessageBubble message={makeMessage(content)} />)).not.toThrow();
    expect(document.body.textContent).toContain('Đại học Khoa học');
    expect(document.body.textContent).toContain('🎓');
  });

  it('handles all-Vietnamese text with all diacritic forms', () => {
    const content = `Tiêu chí: Trắc nghiệm, tự luận, vấn đáp. Ngành: Y khoa, Dược học, Điều dưỡng.`;
    expect(() => render(<MessageBubble message={makeMessage(content)} />)).not.toThrow();
    expect(document.body.textContent).toContain('Trắc nghiệm');
    expect(document.body.textContent).toContain('Dược học');
  });
});

// ---------------------------------------------------------------------------
// (l) Real http link — anchor hardening (target/rel)
// ---------------------------------------------------------------------------
describe('MessageBubble — link hardening', () => {
  it('renders an http link with target=_blank and rel="noopener noreferrer"', () => {
    const content = `Xem chi tiết tại [trang tuyển sinh](https://husc.edu.vn/tuyensinh).`;
    render(<MessageBubble message={makeMessage(content)} />);
    const anchor = document.querySelector('a[href="https://husc.edu.vn/tuyensinh"]');
    expect(anchor).not.toBeNull();
    expect(anchor!.getAttribute('target')).toBe('_blank');
    // react-markdown v10 emits "noopener noreferrer" (no underscore) by default
    expect(anchor!.getAttribute('rel')).toBe('noopener noreferrer');
  });

  it('renders a relative link (with target=_blank as per MessageBubble custom renderer)', () => {
    const content = `[nội bộ](/faq)`;
    render(<MessageBubble message={makeMessage(content)} />);
    const anchor = document.querySelector('a[href="/faq"]');
    expect(anchor).not.toBeNull();
    // MessageBubble's custom `a` component (line 88-97) always sets
    // target=_blank + rel=noopener noreferrer — both for external and
    // internal links. This is intentional UX (every link opens new tab) but
    // can be surprising; pin the contract here.
    expect(anchor!.getAttribute('target')).toBe('_blank');
    expect(anchor!.getAttribute('rel')).toBe('noopener noreferrer');
  });
});

// ---------------------------------------------------------------------------
// (m) XSS — javascript: URL in markdown link — MUST NOT be a live href
// ---------------------------------------------------------------------------
describe('MessageBubble — XSS javascript: URL hardening', () => {
  it('javascript: URL in a markdown link is NOT rendered as a clickable javascript: href', () => {
    const content = `[click me](javascript:alert(1))`;
    render(<MessageBubble message={makeMessage(content)} />);
    // react-markdown v10 strips javascript: URLs by default — no live
    // <a href="javascript:..."> should be present.
    const live = document.querySelector('a[href^="javascript:"]');
    expect(live).toBeNull();
  });

  it('VBScript: URL is also stripped', () => {
    const content = `[click](vbscript:msgbox(1))`;
    render(<MessageBubble message={makeMessage(content)} />);
    const live = document.querySelector('a[href^="vbscript:"]');
    expect(live).toBeNull();
  });

  it('data:text/html URL in a markdown link is neutralized', () => {
    const content = `[evil](data:text/html,<script>alert(1)</script>)`;
    render(<MessageBubble message={makeMessage(content)} />);
    const live = document.querySelector('a[href^="data:"]');
    expect(live).toBeNull();
  });

  it('image-style XSS via markdown image with javascript: does not execute', () => {
    const content = `![alt](javascript:alert(1))`;
    expect(() => render(<MessageBubble message={makeMessage(content)} />)).not.toThrow();
    // No <img> with javascript: src
    const evilImg = document.querySelector('img[src^="javascript:"]');
    expect(evilImg).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// (n) Deeply nested markdown — bullet within bullet
// ---------------------------------------------------------------------------
describe('MessageBubble — nested markdown', () => {
  it('renders nested bullet list without throwing', () => {
    const content = `### Danh sách
- Cấp 1 A
  - Cấp 2 A1
  - Cấp 2 A2
- Cấp 1 B`;
    expect(() => render(<MessageBubble message={makeMessage(content)} />)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// (o) Code blocks — fenced (```)
// ---------------------------------------------------------------------------
describe('MessageBubble — code blocks', () => {
  it('renders a fenced code block without throwing', () => {
    const content = '```ts\nconst x: number = 1;\nfunction f() { return x; }\n```';
    expect(() => render(<MessageBubble message={makeMessage(content)} />)).not.toThrow();
    const code = document.querySelector('code');
    expect(code).not.toBeNull();
  });
});

// ---------------------------------------------------------------------------
// (p) Mixed heavy content — table + list + link + bold
// ---------------------------------------------------------------------------
describe('MessageBubble — mixed heavy content', () => {
  it('renders a kitchen-sink message without throwing', () => {
    const content = `### Ngành **Công nghệ thông tin** 🎓

- **Mã ngành:** 7480201
- **Chỉ tiêu:** 120
- **Tổ hợp:** A00, A01, D01

| Môn | Tổ hợp | Điểm |
| --- | --- | --- |
| Toán | A00 | 22.5 |
| Lý | A01 | 21.0 |

Xem thêm tại [trang tuyển sinh](https://husc.edu.vn/tuyensinh).`;
    expect(() => render(<MessageBubble message={makeMessage(content)} />)).not.toThrow();
    expect(document.querySelector('table')).not.toBeNull();
    expect(document.querySelector('a[href^="https://"]')).not.toBeNull();
  });
});
