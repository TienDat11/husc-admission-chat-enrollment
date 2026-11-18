// @spec(PL-7.2 Frame 12 XaDgH — bong bóng người dùng: canh phải, nền #143980, chữ trắng 14/1.6, bo 12 một góc 4, tối đa 560, lề trong 12/14)
// ChatUserBubble — bong bubbles của người học (dumb/presentational).
import type { Message } from '@/lib/chat-types';

interface ChatUserBubbleProps {
  message: Message;
}

export function ChatUserBubble({ message }: ChatUserBubbleProps) {
  return (
    <div className="flex justify-end mb-4">
      <div
        className="max-w-[560px] rounded-xl rounded-br-sm bg-[#143980] px-[14px] py-3 text-white text-[14px] leading-[1.6] shadow-sm break-words"
      >
        {message.content}
      </div>
      <span className="ml-2 mt-0.5 text-[11px] leading-[14px] text-[#6B7686] font-medium">
        {message.timestamp.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })}
      </span>
    </div>
  );
}
