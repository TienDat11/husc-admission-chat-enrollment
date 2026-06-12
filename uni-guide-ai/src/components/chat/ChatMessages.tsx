import { useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowDown } from 'lucide-react';
import { Message } from '@/lib/chat-types';
import { MessageBubble } from './MessageBubble';
import { TypingIndicator } from './TypingIndicator';

interface ChatMessagesProps {
  messages: Message[];
  isTyping: boolean;
}

/**
 * The streaming assistant placeholder bubble is the one ChatLayout pushes
 * onto `messages` immediately after the user sends a question (content
 * starts as "⏳ Đang phân tích câu hỏi…" and gets replaced as deltas
 * arrive). While that bubble is the last message, ChatLayout also keeps
 * `isTyping=true`, so the standalone <TypingIndicator/> must be hidden
 * to avoid the user-reported double-bubble ("Đang phân tích…" + "Đang
 * suy nghĩ…"). Once the BE's `done` (or the fallback path) writes a
 * non-loading content into the bubble, the placeholder is gone and the
 * indicator can render again (e.g. for the next pending request).
 */
export function isAssistantPlaceholderLast(messages: Message[]): boolean {
  if (messages.length === 0) return false;
  const last = messages[messages.length - 1];
  if (last.role !== 'assistant') return false;
  return last.content.startsWith('⏳ ');
}

export function ChatMessages({ messages, isTyping }: ChatMessagesProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isTyping]);

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto custom-scrollbar mesh-gradient-bg"
    >
      <div className="max-w-4xl mx-auto p-4 md:p-8 space-y-6">
        {/* Welcome Message */}
        {messages.length === 0 && !isTyping && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
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
              isLast={index === messages.length - 1}
            />
          ))}
        </AnimatePresence>

        {/* Typing Indicator — suppressed when the last message is the
            streaming assistant placeholder (its own bubble already shows
            the transient ⏳ label). Showing both produces a visible
            double-bubble ("Đang phân tích…" + "Đang suy nghĩ…"). */}
        <AnimatePresence>
          {isTyping && !isAssistantPlaceholderLast(messages) && (
            <TypingIndicator />
          )}
        </AnimatePresence>

        {/* Scroll anchor */}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}