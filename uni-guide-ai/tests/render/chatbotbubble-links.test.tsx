import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ChatBotBubble } from '@/components/chat/ChatBotBubble';
import type { Message } from '@/lib/chat-types';

const bot = (content: string): Message =>
  ({ id: 'a1', role: 'assistant', content, timestamp: new Date(0) }) as Message;

describe('ChatBotBubble markdown links', () => {
  it('renders protocol-relative and javascript: links as plain text', () => {
    render(<ChatBotBubble message={bot('Xem [evil](//evil.example) và [js](javascript:alert(1))')} />);
    expect(screen.queryAllByRole('link')).toHaveLength(0);
    expect(document.body.textContent).toContain('evil');
    expect(document.body.textContent).toContain('js');
  });

  it('keeps https links with a safe target and rel', () => {
    render(<ChatBotBubble message={bot('[ok](https://example.com/x)')} />);
    const link = screen.getByRole('link', { name: 'ok' });
    expect(link).toHaveAttribute('href', 'https://example.com/x');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', 'noopener noreferrer');
  });
});
