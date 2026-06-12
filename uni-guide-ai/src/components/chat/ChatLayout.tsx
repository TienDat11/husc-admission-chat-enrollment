import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Message, generateId, Source } from '@/lib/chat-types';
import { sendChatMessage, queryStream } from '@/lib/api';
import { ChatHeader } from './ChatHeader';
import { ChatSidebar } from './ChatSidebar';
import { ChatMessages } from './ChatMessages';
import { ChatInput } from './ChatInput';
import { SuggestedQuestions } from './SuggestedQuestions';

export function ChatLayout() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // SLICE-A.LATENCY — warmup moved to the landing page (Index.tsx) so the
  // embedding model preloads the instant the site loads, not when ChatLayout
  // mounts (which is only after the user clicks "Bắt đầu"). No warmup here.

  const handleSendMessage = useCallback(async (content: string) => {
    // Add user message immediately (optimistic UI)
    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setError(null);

    // Show typing indicator
    setIsTyping(true);

    // SLICE-B.LATENCY: stream-optimistic assistant message. We append an
    // empty assistant bubble that we mutate in-place as `delta` frames
    // arrive, so the user sees text appear progressively. On `done`, the
    // message is finalized (sources + status). On error, we fall back to
    // the non-stream `sendChatMessage()` and replace the bubble.
    //
    // BUGFIX (double-bubble): the placeholder bubble itself is now the
    // primary loading affordance — we seed it with the transient ⏳
    // label so there is never a blank flash, and the BE's `stage`
    // frame (or first `delta`) overwrites it. The standalone
    // <TypingIndicator/> is suppressed in ChatMessages when the last
    // message is this assistant placeholder, so the user sees EXACTLY
    // ONE loading bubble at a time (the placeholder with the ⏳ label,
    // not the "●●● Đang suy nghĩ…" indicator).
    const assistantId = generateId();
    const placeholder: Message = {
      id: assistantId,
      role: 'assistant',
      content: '⏳ Đang phân tích câu hỏi…',
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, placeholder]);

    // Convert /v2 sources shape (enriched SourceChip) to the FE Source type.
    const mapSources = (
      rawSources: any[] | undefined,
      confidence: number,
    ): Source[] => {
      const items = rawSources || [];
      return items.map((source: any, index: number) => {
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
    };

    const decorate = (
      rawAnswer: string,
      statusCode: string,
      statusReason: string,
      dataGapHints: string[],
      internalStatusCode: string | undefined,
    ): string => {
      if (statusCode === 'SUCCESS') return rawAnswer;
      const labels: Record<string, string> = {
        NOT_IN_HUSC_SCOPE: 'Ngoài phạm vi tuyển sinh HUSC',
        HUSC_ENTITY_NOT_FOUND: 'Không thấy ngành/sự việc trong dữ liệu HUSC',
        INSUFFICIENT_DATA: 'Thiếu dữ liệu để trả lời chắc chắn',
      };
      const displayCode = internalStatusCode || statusCode;
      const header = labels[displayCode] || labels[statusCode] || 'Cần kiểm tra dữ liệu';
      const hintLines = dataGapHints.length > 0
        ? `\n\n**Gợi ý bổ sung dữ liệu**:\n${dataGapHints.map((h) => `- ${h}`).join('\n')}`
        : '';
      const reasonLine = statusReason ? `\n\n*Lý do:* ${statusReason}` : '';
      return `⚠️ **${header}**\n\n${rawAnswer}${reasonLine}${hintLines}`;
    };

    try {
      // SLICE-B: try the streaming path first.
      let finalAnswer = '';
      let statusCode = 'SUCCESS';
      let statusReason = '';
      let dataGapHints: string[] = [];
      let internalStatusCode: string | undefined;
      let confidence = 0.5;
      let sources: Source[] = [];

      try {
        await queryStream(content, {
          onMeta: (meta) => {
            statusCode = meta.status_code || 'SUCCESS';
            dataGapHints = meta.data_gap_hints || [];
            // sources / statusReason are finalized in done; meta carries
            // the pre-decorated surface only.
          },
          // PART 1 LATENCY: BE emits a `stage` frame BEFORE the first
          // delta so we can replace the bubble's empty content with a
          // transient progress label. The first delta below clears it.
          onStage: (stage) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: `⏳ ${stage}` } : m,
              ),
            );
          },
          onDelta: (text) => {
            finalAnswer += text;
            setMessages((prev) =>
              prev.map((m) => {
                if (m.id !== assistantId) return m;
                // Strip the transient stage prefix on the first delta.
                const wasStage = m.content.startsWith('⏳ ');
                const base = wasStage ? '' : m.content;
                return { ...m, content: base + text };
              }),
            );
          },
          onDone: (fullAnswer) => {
            finalAnswer = fullAnswer;
          },
        });
        // After stream ends, we don't have a final sources / statusReason /
        // confidence payload from the SSE wire (the BE only sends route +
        // sources in the meta event). Derive sources from the meta we
        // captured. If the meta frame carried sources, surface them;
        // otherwise leave sources empty until a fallback re-query.
        // The post-guard answer text is the stream's `finalAnswer`; we
        // decorate it on done using the captured status fields.
        const decorated = decorate(
          finalAnswer,
          statusCode,
          statusReason,
          dataGapHints,
          internalStatusCode,
        );
        setMessages((prev) =>
          prev.map((m) =>
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
        // Streaming path failed — fall back to the non-streaming call.
        // Reset the placeholder content so the fallback message replaces it.
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: '' } : m,
          ),
        );
      }

      // Fallback: non-streaming /v2/query (original behavior).
      const response = await sendChatMessage(content);
      const rawSources = response.sources || [];
      sources = mapSources(rawSources, response.confidence ?? 0.5);
      statusCode = response.status_code || 'SUCCESS';
      statusReason = response.status_reason || '';
      dataGapHints = response.data_gap_hints || [];
      internalStatusCode = response.internal_status_code || undefined;
      confidence = response.confidence ?? 0.5;
      const decorated = decorate(
        response.answer || '',
        statusCode,
        statusReason,
        dataGapHints,
        internalStatusCode,
      );

      setMessages((prev) =>
        prev.map((m) =>
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
      const errorMessage = err instanceof Error ? err.message : 'Không thể kết nối đến máy chủ';
      setError(errorMessage);

      // Replace the placeholder assistant bubble with an error message.
      const errorMessageBot: Message = {
        id: assistantId,
        role: 'assistant',
        content: `⚠️ Có lỗi xảy ra: ${errorMessage}\n\nVui lòng kiểm tra kết nối hoặc thử lại sau.`,
        timestamp: new Date(),
      };
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? errorMessageBot : m)),
      );

      console.error('Failed to send message:', err);
    }
  }, []);

  const handleSelectQuestion = useCallback((question: string) => {
    handleSendMessage(question);
  }, [handleSendMessage]);

  const handleNewChat = useCallback(() => {
    setMessages([]);
    setActiveConversationId(null);
    setIsSidebarOpen(false);
    setError(null);
  }, []);

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background mesh-gradient-bg">
      {/* Sidebar */}
      <ChatSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        onNewChat={handleNewChat}
        activeConversationId={activeConversationId}
        onSelectConversation={setActiveConversationId}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        <ChatHeader onMenuClick={() => setIsSidebarOpen(true)} />

        <ChatMessages messages={messages} isTyping={isTyping} />

        {/* Suggested Questions */}
        <AnimatePresence>
          {messages.length === 0 && !isTyping && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
            >
              <SuggestedQuestions onSelect={handleSelectQuestion} />
            </motion.div>
          )}
        </AnimatePresence>

        <ChatInput onSend={handleSendMessage} isLoading={isTyping} />
      </div>
    </div>
  );
}