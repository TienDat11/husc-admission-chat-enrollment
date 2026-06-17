import { useState, useRef, useEffect, KeyboardEvent } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading?: boolean;
  disabled?: boolean;
}

export function ChatInput({ onSend, isLoading, disabled }: ChatInputProps) {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    if (message.trim() && !isLoading && !disabled) {
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
              placeholder="Hỏi bất cứ điều gì về tuyển sinh... 💬"
              disabled={disabled || isLoading}
              rows={1}
              className="w-full px-5 py-4 pr-16 rounded-2xl bg-muted border-3 border-border focus:border-primary focus:bg-background resize-none text-[15px] leading-relaxed font-medium placeholder:text-muted-foreground/70 transition-all duration-200 outline-none shadow-[0_3px_0_hsl(var(--border))] focus:shadow-[0_4px_0_hsl(var(--primary-dark))] disabled:opacity-50"
            />

            {/* Send Button - Locket style */}
            <motion.button
              whileHover={{ scale: 1.1, y: -2 }}
              whileTap={{ scale: 0.95, y: 2 }}
              onClick={handleSend}
              disabled={!message.trim() || isLoading || disabled}
              className={`absolute right-3 bottom-3 p-2.5 rounded-xl transition-all border-3 ${
                message.trim() && !isLoading
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
