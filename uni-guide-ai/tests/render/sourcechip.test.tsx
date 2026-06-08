/**
 * SourceChip render tests — exercised through <MessageBubble>.
 *
 * SourceChip is a private (non-exported) subcomponent of MessageBubble.tsx
 * (line ~192). We verify the contract by rendering MessageBubble with
 * assistant messages that include crafted Source[] arrays, and then
 * inspecting the live DOM for the year badge, the link/button branching,
 * and the security attributes (target/rel).
 *
 * Matrix:
 *   - data_year present, non-'N/A'   -> year badge visible
 *   - data_year absent               -> no year badge
 *   - data_year = 'N/A'              -> no year badge
 *   - data_year = '' (empty)         -> no year badge
 *   - url present                    -> renders <a> with target=_blank, rel=noopener noreferrer
 *   - url absent                     -> renders <button>
 *   - title truncated                 -> title span has truncate class
 *   - confidence 0..1                 -> no badge
 */

import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { MessageBubble } from '@/components/chat/MessageBubble';
import type { Message, Source } from '@/lib/chat-types';

function makeMessage(sources: Source[]): Message {
  return {
    id: 'src-test',
    role: 'assistant',
    content: 'Some answer with sources.',
    timestamp: new Date('2025-01-01T00:00:00Z'),
    sources,
  };
}

describe('SourceChip — data_year badge visibility', () => {
  it('shows year badge when data_year is a normal string', () => {
    render(
      <MessageBubble
        message={makeMessage([
          { id: 's1', title: 'Doc 1', data_year: '2024', confidence: 0.9 },
        ])}
      />
    );
    expect(document.querySelector('span.bg-primary\\/10')).not.toBeNull();
    expect(document.body.textContent).toContain('2024');
  });

  it('does NOT show year badge when data_year is absent', () => {
    render(
      <MessageBubble
        message={makeMessage([
          { id: 's1', title: 'Doc 1', confidence: 0.9 },
        ])}
      />
    );
    // No bg-primary/10 element (which is the year-badge class)
    expect(document.querySelector('span.bg-primary\\/10')).toBeNull();
  });

  it('does NOT show year badge when data_year is "N/A"', () => {
    render(
      <MessageBubble
        message={makeMessage([
          { id: 's1', title: 'Doc 1', data_year: 'N/A', confidence: 0.9 },
        ])}
      />
    );
    expect(document.querySelector('span.bg-primary\\/10')).toBeNull();
  });

  it('does NOT show year badge when data_year is empty string', () => {
    render(
      <MessageBubble
        message={makeMessage([
          { id: 's1', title: 'Doc 1', data_year: '', confidence: 0.9 },
        ])}
      />
    );
    expect(document.querySelector('span.bg-primary\\/10')).toBeNull();
  });

  it('shows multiple year badges for multiple sources with different years', () => {
    render(
      <MessageBubble
        message={makeMessage([
          { id: 's1', title: 'A', data_year: '2023', confidence: 0.9 },
          { id: 's2', title: 'B', data_year: '2024', confidence: 0.8 },
        ])}
      />
    );
    const badges = document.querySelectorAll('span.bg-primary\\/10');
    expect(badges).toHaveLength(2);
    expect(badges[0].textContent).toBe('2023');
    expect(badges[1].textContent).toBe('2024');
  });
});

describe('SourceChip — url branching (anchor vs button)', () => {
  it('renders <a> with target=_blank and rel=noopener noreferrer when url is present', () => {
    render(
      <MessageBubble
        message={makeMessage([
          {
            id: 's1',
            title: 'TB 2024',
            url: 'https://husc.edu.vn/tb2024',
            confidence: 0.9,
          },
        ])}
      />
    );
    const anchor = document.querySelector('a[href="https://husc.edu.vn/tb2024"]');
    expect(anchor).not.toBeNull();
    expect(anchor!.getAttribute('target')).toBe('_blank');
    // Note: SourceChip sets rel="noopener noreferrer" (no underscore)
    expect(anchor!.getAttribute('rel')).toBe('noopener noreferrer');
  });

  it('renders <button> (not <a>) when url is absent', () => {
    render(
      <MessageBubble
        message={makeMessage([
          { id: 's1', title: 'Plain source', confidence: 0.9 },
        ])}
      />
    );
    // Source chip should render as a <button>, not a link
    // The container of the chip is a motion.button — find it by the title
    const buttons = Array.from(document.querySelectorAll('button')).filter(
      (b) => b.textContent?.includes('Plain source')
    );
    expect(buttons.length).toBeGreaterThan(0);
    // And no anchor with the title text
    const anchors = Array.from(document.querySelectorAll('a')).filter(
      (a) => a.textContent?.includes('Plain source')
    );
    expect(anchors).toHaveLength(0);
  });

  it('renders <button> when url is undefined explicitly', () => {
    render(
      <MessageBubble
        message={makeMessage([
          { id: 's1', title: 'Explicit undef', url: undefined, confidence: 0.9 },
        ])}
      />
    );
    const anchors = Array.from(document.querySelectorAll('a')).filter(
      (a) => a.textContent?.includes('Explicit undef')
    );
    expect(anchors).toHaveLength(0);
  });
});

describe('SourceChip — security hardening', () => {
  it('javascript: URL in source.url is rendered literally as a link href (XSS — known weakness)', () => {
    render(
      <MessageBubble
        message={makeMessage([
          {
            id: 's1',
            title: 'Evil',
            url: 'javascript:alert(1)',
            confidence: 0.9,
          },
        ])}
      />
    );
    // Document current behavior — SourceChip does NOT sanitize url; a
    // javascript: href from a malicious BE response would render as a live
    // <a href="javascript:...">. This test will FAIL if SourceChip ever
    // gains sanitization (which is what we want).
    const anchor = document.querySelector('a[href^="javascript:"]');
    // We mark this as INTENDED-RED-as-characterization: it currently passes
    // (the anchor exists) — exposing the gap. When the gap is closed, the
    // assertion flips to expose progress.
    expect(anchor).not.toBeNull();
  });

  it('data: URL in source.url is rendered literally as a link href (XSS — known weakness)', () => {
    render(
      <MessageBubble
        message={makeMessage([
          {
            id: 's1',
            title: 'Data',
            url: 'data:text/html,<script>alert(1)</script>',
            confidence: 0.9,
          },
        ])}
      />
    );
    const anchor = document.querySelector('a[href^="data:"]');
    // Same characterization as above — flag the gap.
    expect(anchor).not.toBeNull();
  });
});

describe('SourceChip — title rendering', () => {
  it('renders the title in a span with truncate class', () => {
    render(
      <MessageBubble
        message={makeMessage([
          {
            id: 's1',
            title: 'A long title that should be truncated visually',
            confidence: 0.9,
          },
        ])}
      />
    );
    const truncated = Array.from(document.querySelectorAll('span')).find(
      (s) =>
        s.className.includes('truncate') &&
        s.textContent === 'A long title that should be truncated visually'
    );
    expect(truncated).toBeDefined();
  });

  it('renders the document emoji prefix (📄)', () => {
    render(
      <MessageBubble
        message={makeMessage([
          { id: 's1', title: 'Any', confidence: 0.9 },
        ])}
      />
    );
    // 📄 is part of the source chip content
    expect(document.body.textContent).toContain('📄');
  });

  it('does not render any source chip when sources array is empty', () => {
    render(<MessageBubble message={makeMessage([])} />);
    // No truncate span (chip title)
    const truncated = document.querySelector('span.truncate');
    expect(truncated).toBeNull();
  });
});
