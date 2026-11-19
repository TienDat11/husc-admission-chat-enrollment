// @spec(PL-7.2 Frame 12 XaDgH — Chat thí sinh)
// ChatLayout — khung chat công khai theo frame XaDgH: rail trái 320 (nền trắng, kẻ 1px phải)
// + cột chat co giãn. Giữ toàn bộ logic/hooks hiện có (streaming gửi,
// danh sách tin nhắn, phản hồi, lỗi/Trạng thái sẵn sàng backend), chỉ thay đổi trình bày.
// Màn hình chào mừng render DÙ backend chưa sẵn sàng (trạng thái rỗng thiết kế-đủ mà không cần API).
// Phục hồi hành vi từ HEAD: buildSessionHistory → send(content, history), khung báo GENERATION_FAILED,
// badge INSUFFICIENT_DATA, chip nguồn, đọc answer_confidence, xử lý lỗi "Có lỗi xảy ra".
import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useChatStream } from '@/hooks/useChatStream';
import { useBackendReadiness, type ReadinessStage } from '@/hooks/useBackendReadiness';
import { useStudentProfile } from '@/hooks/useStudentProfile';
import { StudentProfileForm } from '@/components/student/StudentProfileForm';
import { RoadmapPanel } from '@/components/student/RoadmapPanel';
import { OutOfScopeCard } from '@/components/student/OutOfScopeCard';
import { LOCALIZED_ERROR_COPY } from '@/lib/chat-helpers';
import type { Message } from '@/lib/chat-types';
import { buildSessionHistory, removeExchange } from '@/lib/session-history';
import { deriveFollowUpQuestions } from '@/lib/follow-up-questions';
import type { SessionHistoryEntry } from '@/lib/session-history';
import { ChatRail } from './ChatRail';
import { ChatTopBar } from './ChatTopBar';
import { ChatSidebar } from './ChatSidebar';
import { ChatWelcome } from './ChatWelcome';
import { ChatUserBubble } from './ChatUserBubble';
import { ChatBotBubble } from './ChatBotBubble';
import { ChatComposer } from './ChatComposer';
import { SuggestedQuestions } from './SuggestedQuestions';

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
  const cls = isFailed
    ? 'border-[#DC3545] bg-[#FDECEA] text-[#8B1A1A]'
    : 'border-[#096EBA] bg-[#E6ECF7] text-[#143980]';

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      data-testid="readiness-banner"
      data-stage={stage}
      role="status"
      aria-live="polite"
      className={`mx-4 mt-3 px-4 py-2 rounded-xl border-2 text-sm font-semibold flex items-center gap-2 ${cls}`}
    >
      <span>{copy}</span>
    </motion.div>
  );
}

export function ChatLayout() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [activeMessageId, setActiveMessageId] = useState<string | null>(null);

  const readiness = useBackendReadiness({ pollMs: 2000 });
  const { send, isTyping, error } = useChatStream({ messages, onMessagesChange: setMessages });
  const { profile } = useStudentProfile();

  const history = useMemo(() => buildSessionHistory(messages), [messages]);
  const followUps = useMemo(() => deriveFollowUpQuestions(messages), [messages]);

  const isOutOfScope = (message: Message) =>
    message.statusCode === 'NOT_IN_HUSC_SCOPE';

  const isNoActiveRelease = error === LOCALIZED_ERROR_COPY.no_active_release;

  const handleSendMessage = useCallback(
    (content: string) => {
      send(content, history);
    },
    [send, history],
  );

  const handleTopicSelect = useCallback(
    (question: string) => {
      handleSendMessage(question);
    },
    [handleSendMessage],
  );

  const handleSelectEntry = useCallback((messageId: string) => {
    setActiveMessageId(messageId);
    setIsSidebarOpen(false);
    const target = document.querySelector(`[data-message-id="${messageId}"]`);
    target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, []);

  const handleDeleteEntry = useCallback((messageId: string) => {
    setMessages((current) => removeExchange(current, messageId));
    setActiveMessageId((current) => (current === messageId ? null : current));
  }, []);

  const handleNewChat = useCallback(() => {
    setMessages([]);
    setActiveMessageId(null);
    setIsSidebarOpen(false);
  }, []);

  return (
    <div className="flex h-screen w-full bg-[#F3F5F9]">
      {/* Rail trái 320 — ẩn trên mobile, cột chat lấp đầy khung nhìn */}
      <div className="hidden lg:flex w-[320px] shrink-0 border-r border-[#E3E8F0] bg-white">
        <ChatRail onTopicSelect={handleTopicSelect} />
      </div>

      <div className="flex-1 flex flex-col min-w-0 bg-white">
        <ChatTopBar />

        {/* Readiness banner: shown above the transcript while the BE is warming/failed.
            Typing is NOT blocked, only Send is disabled. */}
        <AnimatePresence>
          {!readiness.ready && <WarmingBanner stage={readiness.stage} />}
        </AnimatePresence>

        {/* Vietnamese retry banner for 503 NO_ACTIVE_RELEASE: no white screen.
            Shown when the BE reports that no admission release is currently active. */}
        {isNoActiveRelease && (
          <div className="mx-4 mt-3 px-4 py-2 rounded-xl border-2 border-[#FCA5A5] bg-[#FEF2F2] text-[#991B1B] text-sm font-semibold flex items-center gap-2" role="status" aria-live="polite">
            <span>⚠️ {LOCALIZED_ERROR_COPY.no_active_release}</span>
          </div>
        )}

        {/* Hồ sơ thí sinh ẩn danh + lộ trình (khung Frame 12). */}
        <div className="mx-4 mt-3 space-y-3">
          <StudentProfileForm />
          <RoadmapPanel />
        </div>

        {/* Lịch sử phiên — phục hồi hành vi từ HEAD (test ids luôn có mặt trong DOM). */}
        <ChatSidebar
          isOpen={isSidebarOpen}
          onClose={() => setIsSidebarOpen(false)}
          onNewChat={handleNewChat}
          history={history}
          activeMessageId={activeMessageId}
          onSelectEntry={handleSelectEntry}
          onDeleteEntry={handleDeleteEntry}
        />

        <div
          className="flex-1 overflow-y-auto custom-scrollbar px-4 py-4 space-y-4"
          role="log"
          aria-live="polite"
          aria-label="Dòng hội thoại"
          tabIndex={0}
        >
          {/* Màn hình chào mừng khi chưa có tin nhắn — render mà không cần backend */}
          {messages.length === 0 && !isTyping && (
            <ChatWelcome onTopicSelect={handleTopicSelect} />
          )}

          {messages.map((message) => (
            <div key={message.id} data-message-id={message.id}>
              {message.role === 'user' ? (
                <ChatUserBubble key={message.id} message={message} />
              ) : isOutOfScope(message) ? (
                <OutOfScopeCard key={message.id} />
              ) : (
                <ChatBotBubble
                  key={message.id}
                  message={message}
                  onRetry={
                    message.retryQuery ? () => handleSendMessage(message.retryQuery) : undefined
                  }
                />
              )}
            </div>
          ))}

          {followUps.length > 0 && !isTyping && (
            <SuggestedQuestions questions={followUps} onSelect={handleTopicSelect} />
          )}
        </div>

        <ChatComposer
          onSend={handleSendMessage}
          isLoading={isTyping}
          disabled={!readiness.ready}
        />
      </div>
    </div>
  );
}
