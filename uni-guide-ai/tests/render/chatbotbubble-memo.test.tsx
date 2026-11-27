import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import type { Message } from '@/lib/chat-types';

const markdownRenders = vi.hoisted(() => ({ count: 0 }));

vi.mock('react-markdown', async () => {
  const React = await import('react');
  return {
    default: ({ children }: { children: string }) => {
      markdownRenders.count += 1;
      return React.createElement('div', null, children);
    },
  };
});

import { ChatBotBubble } from '@/components/chat/ChatBotBubble';

const message = { id: 'a1', role: 'assistant', content: 'Xin chào', timestamp: new Date(0) } as Message;

describe('ChatBotBubble memo', () => {
  it('does not re-render when the same message is passed again', () => {
    const { rerender } = render(<ChatBotBubble message={message} />);
    const afterMount = markdownRenders.count;
    rerender(<ChatBotBubble message={message} />);
    expect(markdownRenders.count).toBe(afterMount);
  });

  it('re-renders when its message changes', () => {
    const { rerender } = render(<ChatBotBubble message={message} />);
    const afterMount = markdownRenders.count;
    rerender(<ChatBotBubble message={{ ...message, content: 'Xin chào!' }} />);
    expect(markdownRenders.count).toBeGreaterThan(afterMount);
  });
});
