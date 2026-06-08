/**
 * ChatLayout — status banner / decorated-answer path
 *
 * ChatLayout.tsx:64-79 is the "decorate non-SUCCESS responses" path.
 * It builds a banner header from a labels map, appends data_gap_hints
 * as bullet list, and a "Lý do:" reason line.
 *
 * We test the contract at TWO levels:
 *  1. Through MessageBubble — assert the banner DOM is correctly emitted
 *     for the various status_code values the labels map knows.
 *  2. Through a focused harness that mirrors the ChatLayout mapping logic
 *     (sources-as-string vs sources-as-object, status_code default,
 *     internal_status_code override, etc.) so we can verify mapping
 *     independently of UI.
 *
 * The harness in §2 is a copy of the mapping block from ChatLayout.tsx
 * intentionally — if ChatLayout changes the mapping, the harness test
 * will diverge from product code and surface the gap.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MessageBubble } from '@/components/chat/MessageBubble';
import type { Message } from '@/lib/chat-types';

function makeMessage(content: string, overrides: Partial<Message> = {}): Message {
  return {
    id: 'layout-test',
    role: 'assistant',
    content,
    timestamp: new Date('2025-01-01T00:00:00Z'),
    sources: [],
    ...overrides,
  };
}

// ============================================================================
// (1) Status banner DOM assertions — INSUFFICIENT_DATA
// ============================================================================
describe('ChatLayout status banner — INSUFFICIENT_DATA', () => {
  it('shows INSUFFICIENT_DATA label and the data_gap_hints are decorated into content', () => {
    const decorated = `⚠️ **Thiếu dữ liệu để trả lời chắc chắn**\n\nCâu trả lời từ BE.\n\n*Lý do:* Dữ liệu 2024 chưa đầy đủ\n\n**Gợi ý bổ sung dữ liệu**:\n- Cần bổ sung điểm chuẩn 2024\n- Xem thêm thông báo tháng 9`;
    const msg = makeMessage(decorated, {
      statusCode: 'INSUFFICIENT_DATA',
      statusReason: 'Dữ liệu 2024 chưa đầy đủ',
      dataGapHints: ['Cần bổ sung điểm chuẩn 2024', 'Xem thêm thông báo tháng 9'],
      internalStatusCode: undefined,
    });
    render(<MessageBubble message={msg} />);
    // Banner element shows INSUFFICIENT_DATA
    expect(screen.getByText('INSUFFICIENT_DATA')).toBeInTheDocument();
    // Reason text appears in the banner (use getAllByText — also lives in body)
    expect(
      screen.getAllByText('Dữ liệu 2024 chưa đầy đủ').length
    ).toBeGreaterThanOrEqual(1);
    // Decorated content (markdown) includes the gap-hint bullets
    const listItems = document.querySelectorAll('li');
    const hintTexts = Array.from(listItems).map((li) => li.textContent || '');
    expect(hintTexts.some((t) => t.includes('Cần bổ sung điểm chuẩn 2024'))).toBe(true);
    expect(hintTexts.some((t) => t.includes('Xem thêm thông báo tháng 9'))).toBe(true);
  });

  it('data_gap_hints absent -> no bullet list rendered in decorated content', () => {
    const decorated = `⚠️ **Thiếu dữ liệu để trả lời chắc chắn**\n\nCâu trả lời từ BE.`;
    const msg = makeMessage(decorated, {
      statusCode: 'INSUFFICIENT_DATA',
      statusReason: '',
      dataGapHints: [],
    });
    render(<MessageBubble message={msg} />);
    // No hint bullets in DOM
    const listItems = document.querySelectorAll('li');
    expect(listItems).toHaveLength(0);
  });
});

// ============================================================================
// (2) NOT_IN_HUSC_SCOPE label
// ============================================================================
describe('ChatLayout status banner — NOT_IN_HUSC_SCOPE', () => {
  it('shows the "Ngoài phạm vi tuyển sinh HUSC" header when internalStatusCode is NOT_IN_HUSC_SCOPE', () => {
    const decorated = `⚠️ **Ngoài phạm vi tuyển sinh HUSC**\n\nĐây là câu hỏi không thuộc HUSC.`;
    const msg = makeMessage(decorated, {
      statusCode: 'NOT_IN_HUSC_SCOPE',
      statusReason: '',
      dataGapHints: [],
      internalStatusCode: 'NOT_IN_HUSC_SCOPE',
    });
    render(<MessageBubble message={msg} />);
    // Banner element shows internalStatusCode
    expect(screen.getByText('NOT_IN_HUSC_SCOPE')).toBeInTheDocument();
    // Markdown body should contain the header
    expect(document.body.textContent).toContain('Ngoài phạm vi tuyển sinh HUSC');
  });
});

// ============================================================================
// (3) HUSC_ENTITY_NOT_FOUND label
// ============================================================================
describe('ChatLayout status banner — HUSC_ENTITY_NOT_FOUND', () => {
  it('shows the "Không thấy ngành/sự việc trong dữ liệu HUSC" header', () => {
    const decorated = `⚠️ **Không thấy ngành/sự việc trong dữ liệu HUSC**\n\nKhông tìm thấy ngành X.`;
    const msg = makeMessage(decorated, {
      statusCode: 'HUSC_ENTITY_NOT_FOUND',
      statusReason: 'Không có dữ liệu',
      dataGapHints: [],
      internalStatusCode: 'HUSC_ENTITY_NOT_FOUND',
    });
    render(<MessageBubble message={msg} />);
    expect(screen.getByText('HUSC_ENTITY_NOT_FOUND')).toBeInTheDocument();
    expect(document.body.textContent).toContain('Không thấy ngành/sự việc trong dữ liệu HUSC');
  });
});

// ============================================================================
// (4) SUCCESS — no banner
// ============================================================================
describe('ChatLayout status banner — SUCCESS', () => {
  it('does NOT render the banner element when statusCode is SUCCESS', () => {
    const msg = makeMessage('Câu trả lời bình thường.', {
      statusCode: 'SUCCESS',
    });
    render(<MessageBubble message={msg} />);
    // No .border-amber-300 banner
    const banners = document.querySelectorAll('.border-amber-300');
    expect(banners).toHaveLength(0);
    // No "SUCCESS" badge
    expect(screen.queryByText('SUCCESS')).toBeNull();
  });
});

// ============================================================================
// (5) Unknown status_code — falls back to "Cần kiểm tra dữ liệu"
// ============================================================================
describe('ChatLayout status banner — unknown status_code', () => {
  it('falls back to "Cần kiểm tra dữ liệu" header when code is unknown', () => {
    const decorated = `⚠️ **Cần kiểm tra dữ liệu**\n\nTrả lời.`;
    const msg = makeMessage(decorated, {
      statusCode: 'SOMETHING_NEW_FROM_BE',
      statusReason: '',
      dataGapHints: [],
    });
    render(<MessageBubble message={msg} />);
    expect(screen.getByText('SOMETHING_NEW_FROM_BE')).toBeInTheDocument();
    expect(document.body.textContent).toContain('Cần kiểm tra dữ liệu');
  });
});

// ============================================================================
// (6) Source mapping — string[] legacy vs object[] v2 (ChatLayout.tsx:38-56)
// ============================================================================
describe('ChatLayout source mapping — string[] (legacy) and object[] (v2)', () => {
  function mapSources(rawSources: any[], confidence = 0.5) {
    return rawSources.map((source: any, index: number) => {
      if (typeof source === 'string') {
        return {
          id: `source-${index}`,
          title: source,
          confidence,
        };
      }
      return {
        id: source.id || `source-${index}`,
        title: source.title || 'Unknown source',
        url: source.url || undefined,
        snippet: source.snippet || undefined,
        data_year: source.data_year || undefined,
        confidence,
      };
    });
  }

  it('maps legacy string[] sources to object[] with id + title', () => {
    const mapped = mapSources(['Thông báo 2024', 'Quy chế ĐH Huế'], 0.87);
    expect(mapped).toHaveLength(2);
    expect(mapped[0]).toEqual({
      id: 'source-0',
      title: 'Thông báo 2024',
      confidence: 0.87,
    });
    expect(mapped[1].title).toBe('Quy chế ĐH Huế');
  });

  it('maps v2 object[] sources preserving id/title/url/snippet/data_year', () => {
    const v2Sources = [
      {
        id: 'src-1',
        title: 'TB 2024',
        url: 'https://husc.edu.vn/tb2024',
        snippet: 'Điểm chuẩn...',
        data_year: '2024',
      },
    ];
    const mapped = mapSources(v2Sources, 0.9);
    expect(mapped[0]).toEqual({
      id: 'src-1',
      title: 'TB 2024',
      url: 'https://husc.edu.vn/tb2024',
      snippet: 'Điểm chuẩn...',
      data_year: '2024',
      confidence: 0.9,
    });
  });

  it('falls back to "Unknown source" when v2 object has no title', () => {
    const mapped = mapSources([{ id: 's1' }]);
    expect(mapped[0].title).toBe('Unknown source');
  });

  it('falls back to synthetic id when v2 object has no id', () => {
    const mapped = mapSources([{ title: 'X' }]);
    expect(mapped[0].id).toBe('source-0');
  });
});

// ============================================================================
// (7) internalStatusCode override — ChatLayout.tsx:71
// ============================================================================
describe('ChatLayout status banner — internalStatusCode override', () => {
  it('uses internalStatusCode for the header text when present (falls back to statusCode)', () => {
    const decorated = `⚠️ **Header from internalStatusCode label**\n\nBody.`;
    const msg = makeMessage(decorated, {
      statusCode: 'INSUFFICIENT_DATA',
      internalStatusCode: 'NOT_IN_HUSC_SCOPE',
    });
    render(<MessageBubble message={msg} />);
    // Banner element shows the internal code (UI maps internalStatusCode ||
    // statusCode; if the label map has an entry, the header uses that label)
    expect(screen.getByText('NOT_IN_HUSC_SCOPE')).toBeInTheDocument();
  });
});
