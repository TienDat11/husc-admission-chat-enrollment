/**
 * ChatMessages — double-bubble suppression.
 *
 * Reproduces the user-reported bug: after sending a question the user
 * saw TWO simultaneous loading bubbles:
 *   (1) the assistant placeholder showing "⏳ Đang phân tích câu hỏi…"
 *   (2) the standalone <TypingIndicator/> showing "Đang suy nghĩ…"
 *
 * The fix lives in ChatMessages: when the last message is the
 * streaming assistant placeholder (content starts with "⏳ "), the
 * <TypingIndicator/> MUST be hidden even when isTyping=true.
 *
 * We exercise the logic at two levels:
 *   (1) The pure helper `isAssistantPlaceholderLast` is exported from
 *       ChatMessages.tsx so its semantics are pinned by a direct unit
 *       test. This is the load-bearing decision; if it ever flips back
 *       to the old "isTyping-only" gate, this test will surface it.
 *   (2) A full render test of <ChatMessages> with the exact "post-send"
 *       state (user message + assistant placeholder, isTyping=true)
 *       asserts that the TypingIndicator is NOT in the DOM and the
 *       placeholder ⏳ text IS in the DOM — exactly one loading bubble.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ChatMessages, isAssistantPlaceholderLast } from '@/components/chat/ChatMessages';
import type { Message } from '@/lib/chat-types';

function makeMessage(overrides: Partial<Message>): Message {
  return {
    id: overrides.id ?? `m-${Math.random()}`,
    role: overrides.role ?? 'assistant',
    content: overrides.content ?? '',
    timestamp: overrides.timestamp ?? new Date('2025-01-01T00:00:00Z'),
    sources: overrides.sources ?? [],
  };
}

// ---------------------------------------------------------------------------
// (1) Pure helper — isAssistantPlaceholderLast
// ---------------------------------------------------------------------------
describe('isAssistantPlaceholderLast', () => {
  it('returns false on an empty message list', () => {
    expect(isAssistantPlaceholderLast([])).toBe(false);
  });

  it('returns false when the last message is a user message', () => {
    const msgs = [makeMessage({ role: 'user', content: 'hi' })];
    expect(isAssistantPlaceholderLast(msgs)).toBe(false);
  });

  it('returns true when the last message is the assistant ⏳ placeholder', () => {
    const msgs = [
      makeMessage({ role: 'user', content: 'Điểm chuẩn CNTT 2024?' }),
      makeMessage({ role: 'assistant', content: '⏳ Đang phân tích câu hỏi…' }),
    ];
    expect(isAssistantPlaceholderLast(msgs)).toBe(true);
  });

  it('returns false once the placeholder has been replaced with the real answer', () => {
    const msgs = [
      makeMessage({ role: 'user', content: 'Điểm chuẩn CNTT 2024?' }),
      makeMessage({ role: 'assistant', content: 'Điểm chuẩn CNTT năm 2024 là 22.5.' }),
    ];
    expect(isAssistantPlaceholderLast(msgs)).toBe(false);
  });

  it('returns false when the assistant message exists but does not start with ⏳', () => {
    const msgs = [makeMessage({ role: 'assistant', content: 'Xin chào!' })];
    expect(isAssistantPlaceholderLast(msgs)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// (2) Full render of <ChatMessages> — exactly one loading affordance
// ---------------------------------------------------------------------------
describe('ChatMessages — no double-bubble while assistant placeholder is loading', () => {
  it('with isTyping=true and the assistant placeholder as the last message, renders ONLY the placeholder (not TypingIndicator)', () => {
    const messages: Message[] = [
      makeMessage({ id: 'u1', role: 'user', content: 'Điểm chuẩn CNTT 2024?' }),
      makeMessage({
        id: 'a1',
        role: 'assistant',
        content: '⏳ Đang phân tích câu hỏi…',
      }),
    ];

    render(<ChatMessages messages={messages} isTyping={true} />);

    // (1) The placeholder ⏳ label IS in the DOM — this is the SINGLE
    //     visible loading bubble.
    expect(screen.getByText(/Đang phân tích câu hỏi…/)).toBeInTheDocument();

    // (2) The TypingIndicator copy "Đang suy nghĩ" is NOT in the DOM —
    //     that was the second bubble before the fix.
    expect(screen.queryByText(/Đang suy nghĩ/)).toBeNull();

    // (3) Tighter guard: count the animated dots the indicator uses.
    //     The placeholder bubble does not render those 3 bouncing dots.
    const typingDotParents = document.querySelectorAll('span.text-sm.text-muted-foreground.font-bold');
    expect(typingDotParents.length).toBe(0);
  });

  it('with isTyping=true and NO assistant placeholder, TypingIndicator IS rendered (regression guard for the normal case)', () => {
    // Defensive coverage: the suppression logic must not over-trigger.
    // If the last message is a user message (no placeholder), the
    // indicator should still render when isTyping=true.
    const messages: Message[] = [
      makeMessage({ id: 'u1', role: 'user', content: 'Câu hỏi đầu tiên' }),
    ];

    render(<ChatMessages messages={messages} isTyping={true} />);

    // The TypingIndicator copy IS in the DOM.
    expect(screen.getByText(/Đang suy nghĩ/)).toBeInTheDocument();
  });
});
