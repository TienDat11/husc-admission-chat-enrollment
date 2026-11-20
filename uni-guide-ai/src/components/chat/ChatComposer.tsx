// @spec(PL-7.2 Frame 12 XaDgH — ô nhập dính đáy: khối cao 56 bo 12 viền 1px #D6DEEA lề trong 14, placeholder, nút gửi tròn 36 nền #143980 mũi tên trắng (primary duy nhất), đáy 12 #6B7686
// ChatComposer — ô nhập tin nhắn dính đáy (dumb/presentational). Giữ logic gửi (Enter, auto-resize, disable khi stream).
import { useCallback, useEffect, useRef, useState, KeyboardEvent } from 'react';
import { Loader2 } from 'lucide-react';

interface ChatComposerProps {
  onSend: (message: string) => void;
  isLoading?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatComposer({
  onSend,
  isLoading = false,
  disabled = false,
  placeholder = 'Nhập câu hỏi về tuyển sinh năm 2026…',
}: ChatComposerProps) {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const sendDisabled = disabled || isLoading || !message.trim();

  const handleSend = useCallback(() => {
    if (sendDisabled) return;
    onSend(message.trim());
    setMessage('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [message, onSend, sendDisabled]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
    }
  }, [message]);

  return (
    <div className="flex flex-col gap-2 pb-[env(safe-area-inset-bottom)]">
      <div className="flex items-end gap-2 rounded-12 border border-[#D6DEEA] bg-white p-[14px]">
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={1}
          // Typing is intentionally never blocked: only the Submit action is
          // gated via `disabled`/`isLoading` on the send button.
          disabled={disabled || isLoading}
          className="w-full resize-none outline-none border-transparent text-[14px] leading-[1.6] text-[#0F1B2D] placeholder:text-[#6B7686] placeholder-font-medium min-h-[20px] max-h-[150px] focus:border-transparent focus:ring-0 disabled:pointer-events-none disabled:opacity-50"
          aria-label="Nhập câu hỏi tuyển sinh"
        />
        <button
          type="button"
          onClick={handleSend}
          disabled={sendDisabled}
          className={`h-9 shrink-0 rounded-full px-4 text-[14px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#143980] focus-visible:ring-offset-2 ${
            sendDisabled
              ? 'bg-[#F7F9FC] text-[#6B7686] border border-[#D6DEEA]'
              : 'bg-[#143980] text-white border border-transparent'
          }`}
          aria-label="Gửi tin nhắn"
        >
          {isLoading ? <Loader2 className="w-4 h-4 animate-spin" aria-label="Đang gửi" /> : 'Gửi →'}
        </button>
      </div>
    </div>
  );
}
