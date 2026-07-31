// @spec(plan-warmup-ux-20260705-111031 §3.2 / §5 Phase 3)
// ChatLayout — top-level container that wires backend readiness UX.
import { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useChatStream } from '@/hooks/useChatStream';
import { useBackendReadiness, type ReadinessStage } from '@/hooks/useBackendReadiness';
import { ChatHeader } from './ChatHeader';
import { ChatSidebar } from './ChatSidebar';
import { ChatMessages } from './ChatMessages';
import { ChatInput } from './ChatInput';
import { SuggestedQuestions } from './SuggestedQuestions';

interface BannerState {
  visible: boolean;
  seconds: number;
}

function WarmingBanner({ stage }: { stage: ReadinessStage }) {
  const mountedAtRef = useRef<number>(Date.now());
  const [seconds, setSeconds] = useState<number>(() =>
    Math.floor((Date.now() - mountedAtRef.current) / 1000),
  );

  useEffect(() => {
    mountedAtRef.current = Date.now();
    setSeconds(0);
    const id = setInterval(() => {
      setSeconds(Math.floor((Date.now() - mountedAtRef.current) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [stage]);

  if (stage === 'ready') return null;
  const isFailed = stage === 'failed';
  const copy = isFailed
    ? '🔴 Máy chủ không phản hồi. Sẽ thử lại...'
    : `⏳ Đang kết nối tới máy chủ... (${seconds}s)`;
  const tail = isFailed ? '' : '';
  const cls = isFailed
    ? 'border-red-400 bg-red-50 text-red-900 dark:border-red-700 dark:bg-red-950/40 dark:text-red-100'
    : 'border-blue-400 bg-blue-50 text-blue-900 dark:border-blue-700 dark:bg-blue-950/40 dark:text-blue-100';

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      data-testid="readiness-banner"
      data-stage={stage}
      role="status"
      aria-live="polite"
      className={`max-w-4xl mx-auto mt-3 mx-3 md:mx-auto px-4 py-2 rounded-2xl border-2 text-sm font-semibold flex items-center gap-2 ${cls}`}
    >
      <span>{copy}</span>
      {tail && <span className="text-xs opacity-80">{tail}</span>}
    </motion.div>
  );
}

export function ChatLayout() {
  const [messages, setMessages] = useState([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState(null);

  const readiness = useBackendReadiness({ pollMs: 2000 });

  const { send, isTyping } = useChatStream({
    messages,
    onMessagesChange: setMessages,
  });

  const handleSendMessage = useCallback(
    (content) => {
      send(content);
    },
    [send],
  );

  const handleSelectQuestion = useCallback(
    (question) => {
      handleSendMessage(question);
    },
    [handleSendMessage],
  );

  const handleNewChat = useCallback(() => {
    setMessages([]);
    setActiveConversationId(null);
    setIsSidebarOpen(false);
  }, []);

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background mesh-gradient-bg">
      <ChatSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        onNewChat={handleNewChat}
        activeConversationId={activeConversationId}
        onSelectConversation={setActiveConversationId}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <ChatHeader onMenuClick={() => setIsSidebarOpen(true)} />

        {/* Readiness banner: shown above ChatMessages while the BE
            is warming or has gone away. Plan §3.2 + N-goal: graceful
            degradation — typing is NOT blocked, only Send is disabled. */}
        <AnimatePresence>
          {!readiness.ready && (
            <WarmingBanner stage={readiness.stage} />
          )}
        </AnimatePresence>

        <ChatMessages
          messages={messages}
          isTyping={isTyping}
          backendReady={readiness.ready}
        />

        <AnimatePresence>
          {messages.length === 0 && !isTyping && readiness.ready && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
            >
              <SuggestedQuestions onSelect={handleSelectQuestion} />
            </motion.div>
          )}
        </AnimatePresence>

        <ChatInput
          onSend={handleSendMessage}
          isLoading={isTyping}
          backendReady={readiness.ready}
          readinessStage={readiness.stage}
        />
      </div>
    </div>
  );
}
