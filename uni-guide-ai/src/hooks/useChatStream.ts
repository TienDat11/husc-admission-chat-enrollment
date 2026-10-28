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
import type { SourceChip } from '@/lib/api-types';

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
        stage: 'Đang phân tích câu hỏi…',
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
      // Ticket 17: provenance of the answer text ("evidence-policy:<reason>"
      // on abstain/conflict, model name otherwise). Copied verbatim from the
      // meta frame / fallback response; never defaulted. Declared here because
      // both the streamed onMeta handler and the non-stream fallback assign it.
      let provider: string | null | undefined;
      // answer_confidence, verbatim: undefined means the BE omitted the field,
      // null means the generator explicitly declined to vouch for the text.
      // No default is ever assigned (ticket 43 criterion 3).
      let answerConfidence: number | null | undefined;
      // Ticket 17: provenance of the answer text ("evidence-policy:<reason>"
      // on abstain/conflict, model name otherwise). Copied verbatim from the
      // meta frame / fallback response; never defaulted.
      // Ticket T07: the BE stamps trace_id on the meta frame; the 👍/👎 buttons
      // in MessageBubble need it to POST /v2/feedback.
      let traceId: string | undefined;

      try {
        try {
          await queryStream(
            content,
            {
              onMeta: (meta) => {
                statusCode = meta.status_code || 'SUCCESS';
                // FIX-05 (ticket 44): these three were emitted by the BE but
                // dropped here, so a degraded generation rendered as a normal
                // answer on the streamed path only.
                statusReason = meta.status_reason || '';
                internalStatusCode = meta.internal_status_code || undefined;
                dataGapHints = meta.data_gap_hints || [];
                // Ticket 17: the real retrieval confidence. Read it BEFORE
                // mapSources — otherwise every chip carries the fabricated
                // default (0.5) while the non-stream path carries the real
                // number. Only fall back to the default when the BE omits it.
                if (typeof meta.confidence === 'number') confidence = meta.confidence;
                // Ticket 43: the self-trust the generator placed in its own
                // text — null on a failed generation. Copied verbatim.
                answerConfidence = meta.answer_confidence;
                // Ticket 17: provenance of the answer text — the FE keys the
                // abstain/conflict states off this string, never off a guess.
                provider = meta.provider;
                // The BE ships the citation set in the meta frame
                // (main.py:1888-1894). Without this the streamed path renders
                // an answer with no source chips, while the non-stream
                // fallback below renders them — an inconsistency the user
                // sees as "citations sometimes disappear".
                sources = mapSources(meta.sources, confidence);
                traceId = meta.trace_id;
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
                    sources,
                    confidence,
                    answerConfidence,
                    // The question that produced this answer, so a failed
                    // generation can be re-sent verbatim (ticket 43).
                    retryQuery: content,
                    provider,
                    statusCode,
                    statusReason,
                    dataGapHints,
                    internalStatusCode,
                    // Ticket T07: carry the trace id so the bubble's 👍/👎
                    // buttons can POST /v2/feedback for this answer.
                    traceId,
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
        // Ticket 43: read the generator's self-trust verbatim — no default.
        answerConfidence = response.answer_confidence;
        // Ticket 17: the fallback path carries the same provider string.
        provider = response.provider;
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
                  provider,
                  statusCode,
                  statusReason,
                  dataGapHints,
                  internalStatusCode,
                  confidence,
                  answerConfidence,
                  retryQuery: content,
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
          // Ticket 17 criterion 5: transport failures must offer a retry of
          // the SAME question, so the error bot carries the query verbatim.
          retryQuery: content,
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
