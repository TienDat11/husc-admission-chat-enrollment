// @spec(plan-warmup-ux-20260705-111031 §3.2)
// ChatInput — sticky composer; gates Send while the backend is warming
// or has failed (plan §3.2 G2). Typing is intentionally NOT blocked so the
// user can draft ahead of the BE coming online.
import { useState, useRef, useEffect, KeyboardEvent } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import type { ReadinessStage } from '@/hooks/useBackendReadiness';

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading?: boolean;
  disabled?: boolean;
  backendReady?: boolean;
  readinessStage?: ReadinessStage;
}

const WARMING_PLACEHOLDER = 'Đang kết nối tới máy chủ...';
const FAILED_PLACEHOLDER = 'Máy chủ không phản hồi — đang thử lại...';
const READY_PLACEHOLDER = 'Hỏi bất cứ điều gì về tuyển sinh... 💬';

export function ChatInput({
  onSend,
  isLoading,
  disabled,
  backendReady,
  readinessStage,
}: ChatInputProps) {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Resolve placeholder via a strict precedence: warming > failed > ready/
  // undefined. (Unreadiness also short-circuits the Send-disabled gate.)
  const placeholder =
    readinessStage === 'warming'
      ? WARMING_PLACEHOLDER
      : readinessStage === 'failed'
        ? FAILED_PLACEHOLDER
        : READY_PLACEHOLDER;

  const sendDisabled = !message.trim() || isLoading || disabled || backendReady === false;

  const handleSend = () => {
    if (!sendDisabled) {
      onSend(message.trim());
      setMessage('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
    }
  }, [message]);

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      data-testid="chat-input"
      data-readiness-stage={readinessStage ?? 'ready'}
      className="sticky bottom-0 bg-background border-t-4 border-border p-4"
    >
      <div className="max-w-4xl mx-auto">
        {/* Input Container - Locket style */}
        <div className="relative flex items-end gap-3">
          {/* Textarea Container */}
          <div className="relative flex-1">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              // Typing stays enabled even when !ready (N-goal: graceful
              // degradation). Only the SUBMIT action is gated.
              disabled={disabled || isLoading}
              rows={1}
              className="w-full px-5 py-4 pr-16 rounded-2xl bg-muted border-3 border-border focus:border-primary focus:bg-background resize-none text-[15px] leading-relaxed font-medium placeholder:text-muted-foreground/70 transition-all duration-200 outline-none shadow-[0_3px_0_hsl(var(--border))] focus:shadow-[0_4px_0_hsl(var(--primary-dark))] disabled:opacity-50"
            />

            {/* Send Button - Locket style */}
            <motion.button
              whileHover={{ scale: 1.1, y: -2 }}
              whileTap={{ scale: 0.95, y: 2 }}
              onClick={handleSend}
              disabled={sendDisabled}
              aria-disabled={sendDisabled || undefined}
              data-testid="send-button"
              className={`absolute right-3 bottom-3 p-2.5 rounded-xl transition-all border-3 ${
                !sendDisabled
                  ? 'bg-gradient-to-r from-primary to-secondary text-white border-primary-dark shadow-[0_3px_0_hsl(var(--primary-dark))]'
                  : 'bg-muted text-muted-foreground border-border cursor-not-allowed'
              }`}
              aria-label="Gửi tin nhắn"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </motion.button>
          </div>
        </div>

        {/* Helper Text */}
        <p className="text-xs text-muted-foreground text-center mt-3 font-medium">
          ⌨️ Enter để gửi • Shift + Enter để xuống dòng
        </p>
      </div>
    </motion.div>
  );
}
