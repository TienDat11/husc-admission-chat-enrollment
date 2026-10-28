import type { Message } from './chat-types';

/**
 * Ticket 31: the sidebar shows the CURRENT session's real questions.
 * There is no persistence layer — a new chat empties the list.
 */
export interface SessionHistoryEntry {
  /** id of the user message that opened the exchange; also the scroll anchor. */
  messageId: string;
  question: string;
  timestamp: Date;
}

const QUESTION_MAX = 90;

function condense(text: string): string {
  const flat = text.replace(/\s+/g, ' ').trim();
  if (flat.length <= QUESTION_MAX) return flat;
  return `${flat.slice(0, QUESTION_MAX - 1).trimEnd()}…`;
}

/** One entry per user turn, in the order they were asked. */
export function buildSessionHistory(messages: Message[]): SessionHistoryEntry[] {
  const entries: SessionHistoryEntry[] = [];
  for (const message of messages) {
    if (message.role !== 'user') continue;
    if (!message.content.trim()) continue;
    entries.push({
      messageId: message.id,
      question: condense(message.content),
      timestamp: message.timestamp,
    });
  }
  return entries;
}

/**
 * Drop the exchange opened by `messageId`: the question itself plus the
 * assistant reply that immediately follows it. Later turns are untouched.
 */
export function removeExchange(messages: Message[], messageId: string): Message[] {
  const start = messages.findIndex((m) => m.id === messageId && m.role === 'user');
  if (start === -1) return messages;
  const end = messages[start + 1]?.role === 'assistant' ? start + 2 : start + 1;
  return [...messages.slice(0, start), ...messages.slice(end)];
}
