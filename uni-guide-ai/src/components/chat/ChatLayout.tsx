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
import { EligibilityCard } from '@/components/student/EligibilityCard';
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
import { WARMING_BANNER } from '@/lib/copy';

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
    ? `🔴 ${WARMING_BANNER.failed}`
    : `⏳ ${WARMING_BANNER.connecting(seconds)}`;
  const cls = isFailed
    ? 'border-brand-danger bg-[#FDECEA] text-[#8B1A1A]'
    : 'border-brand bg-[#E6ECF7] text-brand-navy';

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

// Breakpoint lg (1024px): hồ sơ nằm ở rail trên desktop và trong khối thu gọn phía trên
// khung chat trên màn hình nhỏ. Không có matchMedia (jsdom) thì coi như không phải desktop.
function useIsDesktop(): boolean {
  const query = '(min-width: 1024px)';
  const [isDesktop, setIsDesktop] = useState(
    () => typeof window.matchMedia === 'function' && window.matchMedia(query).matches,
  );
  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return;
    const media = window.matchMedia(query);
    const update = () => setIsDesktop(media.matches);
    update();
    media.addEventListener?.('change', update);
    return () => media.removeEventListener?.('change', update);
  }, []);
  return isDesktop;
}

export function ChatLayout() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [activeMessageId, setActiveMessageId] = useState<string | null>(null);
  const [profileOpen, setProfileOpen] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const isDesktop = useIsDesktop();

  const readiness = useBackendReadiness({ pollMs: 2000 });
  const { send, isTyping, error } = useChatStream({ messages, onMessagesChange: setMessages });
  const { profile } = useStudentProfile();

  const history = useMemo(() => buildSessionHistory(messages), [messages]);
  const followUps = useMemo(() => deriveFollowUpQuestions(messages), [messages]);
  const lastAnswerId = useMemo(
    () =>
      messages.filter((m) => m.role !== 'user' && m.statusCode !== 'NOT_IN_HUSC_SCOPE').at(-1)?.id ??
      null,
    [messages],
  );

  const isOutOfScope = (message: Message) =>
    message.statusCode === 'NOT_IN_HUSC_SCOPE';

  const isNoActiveRelease = error === LOCALIZED_ERROR_COPY.no_active_release;

  // Stable send: latest send/history are read through a ref so memoised bubbles
  // do not receive a new callback on every streamed token.
  const latestSendRef = useRef({ send, history });
  latestSendRef.current = { send, history };
  const handleSendMessage = useCallback((content: string) => {
    const latest = latestSendRef.current;
    latest.send(content, latest.history);
  }, []);

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

  // Keep the newest exchange in view inside the message list (never the document):
  // a bot answer is aligned to the top of the list so its text and source chips
  // both fit; a user message scrolls to the bottom. Reduced motion skips smoothing.
  const messageCount = messages.length;
  const lastId = messages.at(-1)?.id;
  const lastRole = messages.at(-1)?.role;

  // Announce only when a streamed answer completes (tokens are not announced).
  // seq bumps on every announcement so the live-region text node is re-created
  // even when the message is identical (identical text is otherwise not re-read).
  const [announcement, setAnnouncement] = useState({ seq: 0, text: '' });
  const wasTypingRef = useRef(false);
  useEffect(() => {
    if (wasTypingRef.current && !isTyping) setAnnouncement((prev) => ({ seq: prev.seq + 1, text: 'Đã nhận được câu trả lời.' }));
    wasTypingRef.current = isTyping;
  }, [isTyping]);

  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const reduceMotion =
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const behavior: ScrollBehavior = reduceMotion ? 'auto' : 'smooth';
    const anchor =
      lastRole === 'assistant' && lastId
        ? list.querySelector<HTMLElement>(`[data-message-id="${lastId}"]`)
        : null;
    if (anchor) {
      const offset =
        anchor.getBoundingClientRect().top - list.getBoundingClientRect().top + list.scrollTop;
      list.scrollTo?.({ top: offset - 16, behavior });
    } else {
      list.scrollTo?.({ top: list.scrollHeight, behavior });
    }
  }, [messageCount, isTyping, lastId, lastRole]);

  return (
    <div className="flex h-dvh w-full overflow-hidden bg-[#F3F5F9]">
      {/* Rail trái 320 (lg trở lên): câu hỏi mới, lịch sử phiên, chủ đề và hồ sơ. */}
      <div className="hidden lg:flex w-[320px] shrink-0">
        <ChatRail
          onTopicSelect={handleTopicSelect}
          onNewChat={handleNewChat}
          history={history}
          activeMessageId={activeMessageId}
          onSelectEntry={handleSelectEntry}
          profile={isDesktop ? <EligibilityCard /> : null}
        />
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

      <div className="flex-1 flex flex-col min-w-0 min-h-0 bg-white">
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

        {/* Hồ sơ và lộ trình trên màn hình nhỏ: thu gọn. Trên lg hồ sơ ở rail, lộ trình trong luồng chat. */}
        {!isDesktop && (
          <div className="mx-4 mt-3 shrink-0">
            <button
              type="button"
              aria-expanded={profileOpen}
              aria-controls="chat-profile-panel"
              onClick={() => setProfileOpen((open) => !open)}
              className="flex w-full items-center justify-between rounded-8 border border-[#D6DEEA] bg-white px-3 py-3 text-left text-sm font-semibold text-brand-navy"
            >
              <span>Hồ sơ thí sinh và lộ trình</span>
              <span aria-hidden="true">{profileOpen ? '▲' : '▼'}</span>
            </button>
            <div
              id="chat-profile-panel"
              className={`${profileOpen ? 'block' : 'hidden'} mt-3 max-h-[35dvh] space-y-3 overflow-y-auto`}
            >
              <StudentProfileForm />
              <RoadmapPanel />
            </div>
          </div>
        )}


        <div
          ref={listRef}
          className="flex-1 min-h-0 overflow-y-auto custom-scrollbar px-4 py-4 space-y-4"
          role="log"
          aria-live="off"
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
                  onRetry={handleSendMessage}
                  followUps={!isTyping && message.id === lastAnswerId ? followUps : undefined}
                  onFollowUp={handleTopicSelect}
                />
              )}
            </div>
          ))}
          {isDesktop && (
            <div className="max-w-[960px]">
              <RoadmapPanel />
            </div>
          )}

        </div>

        <div className="sr-only" role="status" aria-live="polite"><span key={announcement.seq}>{announcement.text}</span></div>
        <ChatComposer
          onSend={handleSendMessage}
          isLoading={isTyping}
          disabled={!readiness.ready}
        />
      </div>
    </div>
  );
}
