// @spec(plan-warmup-ux-20260705-111031 §3.2 / §5 Phase 4)
// ChatMessages — main message stream. When the BE is not ready and the
// user has not yet sent anything, render 3 skeleton bubbles so the surface
// acknowledges activity instead of either crashing or showing "no content".
import { useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowDown } from 'lucide-react';
import { Message } from '@/lib/chat-types';
import type { QueryErrorKind } from '@/lib/api';
import { MessageBubble } from './MessageBubble';
import { TypingIndicator } from './TypingIndicator';

interface ChatMessagesProps {
  messages: Message[];
  isTyping: boolean;
  /** When false, render a skeleton welcome instead of the prompt hero. */
  backendReady?: boolean;
}

/**
 * Returns true when the last message is an assistant message with a `stage`
 * field set (i.e. the streaming placeholder is in its processing state).
 * Used to suppress the standalone <TypingIndicator/> so the stage pill in
 * MessageBubble is the single visible processing affordance.
 */
export function hasAssistantStage(messages: Message[]): boolean {
  if (messages.length === 0) return false;
  const last = messages[messages.length - 1];
  if (last.role !== 'assistant') return false;
  return !!last.stage;
}

function SkeletonBubbles() {
  // Plan §3.2 last section: three skeleton bubbles with shimmer animation.
  // Pure CSS gradient sweep — no per-bubble state, simple to test.
  return (
    <div
      data-testid="chat-skeleton"
      className="space-y-6"
      role="status"
      aria-label="Đang chờ máy chủ sẵn sàng"
    >
      {[0, 1, 2].map((i) => (
        <div key={i} className="flex gap-3 justify-start">
          <div className="flex-shrink-0 w-12 h-12 rounded-2xl bg-muted border-2 border-border" />
          <div
            className="h-16 rounded-2xl bg-muted/70 border-2 border-border max-w-[60%]"
            style={{
              width: `${68 - i * 12}%`,
              backgroundImage:
                'linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.45) 50%, rgba(255,255,255,0) 100%)',
              backgroundSize: '200% 100%',
              animation: `skeleton-shimmer 1.6s ease-in-out infinite`,
              animationDelay: `${i * 0.18}s`,
            }}
          />
        </div>
      ))}
      <style>{`
        @keyframes skeleton-shimmer {
          0%   { background-position: 100% 0; }
          100% { background-position: -100% 0; }
        }
      `}</style>
    </div>
  );
}

export function ChatMessages({
  messages,
  isTyping,
  backendReady = true,
}: ChatMessagesProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isTyping]);

  const showSkeleton = messages.length === 0 && !isTyping && !backendReady;

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto custom-scrollbar mesh-gradient-bg"
    >
      <div className="max-w-4xl mx-auto p-4 md:p-8 space-y-6">
        {/* Skeleton welcome — visible only while the BE is warming/failed
            AND no user messages have started the conversation. */}
        {showSkeleton && <SkeletonBubbles />}

        {/* Welcome Message */}
        {messages.length === 0 && !isTyping && backendReady && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            data-testid="chat-welcome"
            className="text-center py-12"
          >
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.1, type: 'spring', stiffness: 200 }}
              className="w-24 h-24 mx-auto mb-6 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center shadow-glow"
            >
              <span className="text-5xl">🎓</span>
            </motion.div>
            <motion.h2
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="font-display text-2xl md:text-3xl font-bold text-foreground mb-4"
            >
              Xin chào! Tôi là Trợ lý Tuyển sinh
            </motion.h2>
            <motion.p
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="text-muted-foreground max-w-md mx-auto text-base leading-relaxed"
            >
              Tôi sẵn sàng giải đáp mọi thắc mắc về tuyển sinh, ngành học, học phí và đời sống sinh viên tại Đại học Khoa học Huế.
            </motion.p>
          </motion.div>
        )}

        {/* Messages */}
        <AnimatePresence mode="popLayout">
          {messages.map((message, index) => (
            <MessageBubble
              key={message.id}
              message={message}
              errorKind={message.errorKind as QueryErrorKind | undefined}
            />
          ))}
        </AnimatePresence>

        {/* Typing Indicator — suppressed when the last assistant message
            has a `stage` field (the stage pill in MessageBubble handles
            the processing state natively, avoiding double-bubble). */}
        <AnimatePresence>
          {isTyping && !hasAssistantStage(messages) && (
            <TypingIndicator />
          )}
        </AnimatePresence>

        {/* Scroll anchor */}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}