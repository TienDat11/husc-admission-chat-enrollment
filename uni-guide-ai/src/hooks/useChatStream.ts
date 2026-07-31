// @spec(plan-warmup-ux-20260705-111031 §3.2)
// useChatStream — integrates the classified QueryError from api.ts so
// MessageBubble can render localized Vietnamese copy on transport failure.
import { useState, useCallback, useRef } from 'react';
import type { Message, Source } from '@/lib/chat-types';
import {
  sendChatMessage,
  queryStream,
  classifyFetchError,
  type QueryError,
  type QueryErrorKind,
} from '@/lib/api';
import { mapSources, decorateAnswer } from '@/lib/chat-helpers';

export interface UseChatStreamResult {
  /** Append a user message and run the streaming + fallback pipeline. */
  send: (content: string) => Promise<void>;
  /** True while an assistant response is in-flight. */
  isTyping: boolean;
  /** Last transport/API error message, or null. */
  error: string | null;
  /** Clear the error state. */
  clearError: () => void;
}

export interface UseChatStreamOptions {
  /** Current message list — used to compute the bounded history window. */
  messages: Message[];
  /** Receive every message list update (optimistic user, streaming placeholder, final). */
  onMessagesChange: (messages: Message[]) => void;
  /** Optional id factory. */
  generateId?: () => string;
}

/**
 * Encapsulates the streaming /v2/query/stream flow plus the non-streaming
 * fallback. Preserves the exact runtime behavior that previously lived in
 * ChatLayout.tsx:
 *   - optimistic user message
 *   - assistant placeholder with ⏳ stage label
 *   - delta frames appended to the placeholder
 *   - final `done` frame replaces the content
 *   - on stream failure, reset placeholder and call sendChatMessage
 *   - on any error, replace placeholder with an error bot message
 */
export function useChatStream(options: UseChatStreamOptions): UseChatStreamResult {
  const { messages, onMessagesChange, generateId = defaultGenerateId } = options;
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const clearError = useCallback(() => setError(null), []);

  const appendMessages = useCallback(
    (...toAppend: Message[]) => {
      onMessagesChange([...messages, ...toAppend]);
    },
    [messages, onMessagesChange],
  );

  const send = useCallback(
    async (content: string) => {
      // Abort any previous in-flight request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();
      const signal = abortControllerRef.current.signal;

      const userMessage: Message = {
        id: generateId(),
        role: 'user',
        content,
        timestamp: new Date(),
      };

      // Multi-turn context slice: last 4 prior messages (2 user + 2 assistant
      // turns). Exclude the current user message — the BE treats `content` as
      // the live question.
      const historyWindow = messages.slice(-4).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const assistantId = generateId();
      const placeholder: Message = {
        id: assistantId,
        role: 'assistant',
        content: '',
        stage: 'Dang phan tich cau hoi…',
        timestamp: new Date(),
      };

      appendMessages(userMessage, placeholder);
      setIsTyping(true);
      setError(null);

      // Helper that always reads the latest `messages` from the closure
      // captured by the options object, not the stale state at send time.
      const replacePlaceholder = (updater: (current: Message[]) => Message[]) => {
        onMessagesChange(updater([...messages, userMessage, placeholder]));
      };

      let finalAnswer = '';
      let statusCode = 'SUCCESS';
      let statusReason = '';
      let dataGapHints: string[] = [];
      let internalStatusCode: string | undefined;
      let confidence = 0.5;
      let sources: Source[] = [];

      try {
        try {
          await queryStream(
            content,
            {
              onMeta: (meta) => {
                statusCode = meta.status_code || 'SUCCESS';
                dataGapHints = meta.data_gap_hints || [];
              },
              onStage: (stage) => {
                replacePlaceholder((list) =>
                  list.map((m) =>
                    m.id === assistantId ? { ...m, stage, content: m.content.startsWith('⏳ ') ? '' : m.content } : m,
                  ),
                );
              },
              onDelta: (text) => {
                finalAnswer += text;
                replacePlaceholder((list) =>
                  list.map((m) => {
                    if (m.id !== assistantId) return m;
                    return { ...m, stage: null, content: (m.content || '') + text };
                  }),
                );
              },
              onDone: (fullAnswer) => {
                finalAnswer = fullAnswer;
              },
            },
            historyWindow,
            signal,
          );

          const decorated = decorateAnswer(
            finalAnswer,
            statusCode,
            statusReason,
            dataGapHints,
            internalStatusCode,
          );
          replacePlaceholder((list) =>
            list.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    content: decorated,
                    statusCode,
                    statusReason,
                    dataGapHints,
                    internalStatusCode,
                  }
                : m,
            ),
          );
          setIsTyping(false);
          return;
        } catch (_streamErr) {
          replacePlaceholder((list) =>
            list.map((m) => (m.id === assistantId ? { ...m, content: '' } : m)),
          );
        }

        const response = await sendChatMessage(content, historyWindow);
        const rawSources = response.sources || [];
        sources = mapSources(rawSources as Array<SourceChip | string>, response.confidence ?? 0.5);
        statusCode = response.status_code || 'SUCCESS';
        statusReason = response.status_reason || '';
        dataGapHints = response.data_gap_hints || [];
        internalStatusCode = response.internal_status_code || undefined;
        confidence = response.confidence ?? 0.5;
        const decorated = decorateAnswer(
          response.answer || '',
          statusCode,
          statusReason,
          dataGapHints,
          internalStatusCode,
        );

        replacePlaceholder((list) =>
          list.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: decorated,
                  sources,
                  statusCode,
                  statusReason,
                  dataGapHints,
                  internalStatusCode,
                  confidence,
                }
              : m,
          ),
        );
        setIsTyping(false);
      } catch (err) {
        setIsTyping(false);
        // Classify the error so MessageBubble (below) can render the
        // correct Vietnamese copy. The `queryStream` retry loop already
        // ran its backoff; this is the final classified QueryError.
        const qe: QueryError = classifyFetchError(err);
        setError(qe.message);

        const errorMessageBot: Message = {
          id: assistantId,
          role: 'assistant',
          // content is kept minimal — MessageBubble uses `errorKind` to
          // render the full localized copy. The raw `err.message` (e.g.
          // "Failed to fetch") is NEVER displayed to the user.
          content: '',
          errorKind: qe.kind,
          timestamp: new Date(),
        };
        replacePlaceholder((list) =>
          list.map((m) => (m.id === assistantId ? errorMessageBot : m)),
        );
        console.error('Failed to send message:', err);
      }
    },
    [messages, appendMessages, onMessagesChange, generateId],
  );

  return { send, isTyping, error, clearError };
}

function defaultGenerateId(): string {
  return Math.random().toString(36).substring(2, 15);
}
