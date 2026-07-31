/**
 * ChatMessages — double-bubble suppression (stage-pill era).
 *
 * With Part C of the streaming plan, the assistant placeholder no longer
 * uses a "⏳ " content prefix. Instead, processing state is stored in a
 * separate `stage` field on the Message. The stage pill in MessageBubble
 * replaces the old "⏳ Đang phân tích…" bubble.
 *
 * The <TypingIndicator/> suppression now checks `hasAssistantStage()`:
 * when the last assistant message has a truthy `stage` field, the
 * standalone indicator is hidden to avoid double-affordance.
 *
 * We exercise the logic at two levels:
 *   (1) The pure helper `hasAssistantStage` is exported from
 *       ChatMessages.tsx so its semantics are pinned by a direct unit
 *       test.
 *   (2) A full render test of <ChatMessages> with the exact "post-send"
 *       state (user message + assistant placeholder with stage set)
 *       asserts that the TypingIndicator is NOT in the DOM and the
 *       stage pill text IS in the DOM — exactly one loading affordance.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ChatMessages, hasAssistantStage } from '@/components/chat/ChatMessages';
import type { Message } from '@/lib/chat-types';

function makeMessage(overrides: Partial<Message>): Message {
  return {
    id: overrides.id ?? `m-${Math.random()}`,
    role: overrides.role ?? 'assistant',
    content: overrides.content ?? '',
    stage: overrides.stage ?? undefined,
    timestamp: overrides.timestamp ?? new Date('2025-01-01T00:00:00Z'),
    sources: overrides.sources ?? [],
  };
}

// ---------------------------------------------------------------------------
// (1) Pure helper — hasAssistantStage
// ---------------------------------------------------------------------------
describe('hasAssistantStage', () => {
  it('returns false on an empty message list', () => {
    expect(hasAssistantStage([])).toBe(false);
  });

  it('returns false when the last message is a user message', () => {
    const msgs = [makeMessage({ role: 'user', content: 'hi' })];
    expect(hasAssistantStage(msgs)).toBe(false);
  });

  it('returns true when the last assistant message has a stage set', () => {
    const msgs = [
      makeMessage({ role: 'user', content: 'Diem chuan CNTT 2024?' }),
      makeMessage({ role: 'assistant', stage: 'Dang phan tich cau hoi…' }),
    ];
    expect(hasAssistantStage(msgs)).toBe(true);
  });

  it('returns false once stage is cleared (delta has arrived)', () => {
    const msgs = [
      makeMessage({ role: 'user', content: 'Diem chuan CNTT 2024?' }),
      makeMessage({ role: 'assistant', content: 'Diem chuan CNTT nam 2024 la 22.5.', stage: null }),
    ];
    expect(hasAssistantStage(msgs)).toBe(false);
  });

  it('returns false when assistant message has no stage field', () => {
    const msgs = [makeMessage({ role: 'assistant', content: 'Xin chao!' })];
    expect(hasAssistantStage(msgs)).toBe(false);
  });

  it('returns false when stage is an empty string', () => {
    const msgs = [makeMessage({ role: 'assistant', content: '', stage: '' })];
    expect(hasAssistantStage(msgs)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// (2) Full render of <ChatMessages> — exactly one loading affordance
// ---------------------------------------------------------------------------
describe('ChatMessages — no double-bubble while assistant stage is set', () => {
  it('with isTyping=true and the assistant has a stage, renders ONLY the stage pill (not TypingIndicator)', () => {
    const messages: Message[] = [
      makeMessage({ id: 'u1', role: 'user', content: 'Diem chuan CNTT 2024?' }),
      makeMessage({
        id: 'a1',
        role: 'assistant',
        content: '',
        stage: 'Dang phan tich cau hoi…',
      }),
    ];

    render(<ChatMessages messages={messages} isTyping={true} />);

    // (1) The stage pill label IS in the DOM — this is the SINGLE
    //     visible loading affordance.
    expect(screen.getByText(/Dang phan tich cau hoi…/)).toBeInTheDocument();

    // (2) The TypingIndicator copy "Đang suy nghĩ" is NOT in the DOM —
    //     that was the second bubble before the fix.
    expect(screen.queryByText(/Đang suy nghĩ/)).toBeNull();

    // (3) Tighter guard: count the animated dots the indicator uses.
    //     The stage pill does not render those 3 bouncing dots.
    const typingDotParents = document.querySelectorAll('span.text-sm.text-muted-foreground.font-bold');
    expect(typingDotParents.length).toBe(0);
  });

  it('with isTyping=true and NO assistant stage, TypingIndicator IS rendered (regression guard for the normal case)', () => {
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
