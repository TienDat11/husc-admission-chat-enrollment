// @spec(PL-7.2 Frame 12 XaDgH — câu trả lời Bot: KHÔNG bong bóng nền, avatar tròn 28 nền #E6ECF7 khiên 14 #143980 trái, nội dung 14/1.65 #0F1B2D tối đa 640
// ChatBotBubble — câu trả lời của Bot. Giữ khung Frame 12 và phục hồi các hành vi từ HEAD:
// khung báo trạng thái (INSUFFICIENT_DATA / HUSC_ENTITY_NOT_FOUND / GENERATION_FAILED / không xác định),
// chip nguồn, đọc answer_confidence, nút thử lại cho GENERATION_FAILED, bản báo lỗi "Có lỗi xảy ra".
import { useState } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Shield, AlertTriangle, RefreshCw, ThumbsDown, ThumbsUp } from 'lucide-react';
import type { Message } from '@/lib/chat-types';
import { sendFeedback } from '@/lib/api';
import { isGenerationFailure, LOCALIZED_ERROR_COPY } from '@/lib/chat-helpers';
import { SourceChip } from '@/components/student/SourceChip';

interface ChatBotBubbleProps {
  message: Message;
  /** Ticket 43: gửi lại ngay câu hỏi trước đó qua handleSendMessage của ChatLayout. */
  onRetry?: () => void;
}

const ANSWER_COMPONENTS: Components = {
  table: ({ children }) => (
    <div className="my-3 w-full max-w-full overflow-x-auto rounded-8 border border-[#EDF1F6]">
      <table className="min-w-full border-collapse text-[13px] leading-[18px]">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="bg-[#F7F9FC] px-3 py-2 text-left text-[13px] font-medium text-[#4A5568]">{children}</th>
  ),
  td: ({ children }) => (
    <td className="border-t border-[#EDF1F6] px-3 py-2 align-top text-[14px] text-[#0F1B2D]">{children}</td>
  ),
};

export function ChatBotBubble({ message, onRetry }: ChatBotBubbleProps) {
  const generationFailure = isGenerationFailure(message);
  const isError = !!message.errorKind;
  const [rated, setRated] = useState<'helpful' | 'unclear' | null>(null);
  const canRate = !!message.traceId && !isError && !generationFailure && !!message.content;
  const rate = (choice: 'helpful' | 'unclear') => {
    const next = rated === choice ? null : choice;
    setRated(next);
    if (next === null || !message.traceId) return;
    sendFeedback({
      trace_id: message.traceId,
      helpful: next === 'helpful',
      question: message.retryQuery ?? null,
    }).catch(() => setRated(null));
  };
  // Dùng bản sao đã được trang trí (decorateAnswer đã chạy trong useChatStream);
  // khi transport lỗi, hiển thị bản dịch theo mã lỗi thay vì nội dung trống/raw err.message.
  const body = isError
    ? `⚠️ ${LOCALIZED_ERROR_COPY[message.errorKind as keyof typeof LOCALIZED_ERROR_COPY] ?? LOCALIZED_ERROR_COPY.unknown}`
    : message.content;

  return (
    <div className="flex gap-3 mb-4">
      <div className="flex-shrink-0 w-7 h-7 rounded-full bg-[#E6ECF7] flex items-center justify-center shrink-0">
        <Shield className="w-3.5 h-3.5 text-[#143980]" aria-hidden="true" />
      </div>
      <div className="flex-1 max-w-[640px] text-[#0F1B2D] text-[14px] leading-[1.65]">
        {/* Khung báo trạng thái — riêng biệt so với câu trả lời bình thường, KHÔNG phải 'Cần kiểm tra dữ liệu' cho GENERATION_FAILED. */}
        {message.statusCode && message.statusCode !== 'SUCCESS' && (
          <div className="mb-3 flex items-start gap-2 rounded-xl border-2 border-[#D97706] bg-[#FFFBEB] px-3 py-2 text-xs text-[#78350F]">
            <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" aria-hidden="true" />
            <div className="min-w-0">
              <div className="font-bold text-[12px]">
                Lưu ý về câu trả lời
              </div>
              {message.statusReason && /\s/.test(message.statusReason) && (
                <div className="mt-1 opacity-90 text-[12px]">{message.statusReason}</div>
              )}
            </div>
          </div>
        )}

        {isError ? (
          <p className="mb-3">{body}</p>
        ) : (
          message.content || generationFailure ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={ANSWER_COMPONENTS}>{body}</ReactMarkdown>
          ) : null
        )}

        {/* Chip nguồn (Nguồn: ...) */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-4 pt-3 border-t border-[#D6DEEA] space-y-2">
            {message.sources.map((source) => (
              <SourceChip key={source.id} source={source} />
            ))}
          </div>
        )}

        {/* Đọc answer_confidence chỉ khi BE gửi một số thực; null hoặc bỏ trống -> không hiển thị. */}
        {typeof message.answerConfidence === 'number' && (
          <div className="mt-3 text-[12px] leading-[16px] font-medium text-[#6B7686]" data-testid="answer-confidence">
            Độ tin cậy câu trả lời: {Math.round(message.answerConfidence * 100)}%
          </div>
        )}
        {canRate && (
          <div className="mt-4 flex flex-wrap items-center gap-2 rounded-8 bg-[#E6ECF7]/60 px-3 py-2 text-[12px] leading-[16px] text-[#4A5568]">
            <span className="font-medium">Câu trả lời này có giúp ích cho bạn không?</span>
            <button
              type="button"
              aria-pressed={rated === 'helpful'}
              onClick={() => rate('helpful')}
              className={`inline-flex items-center gap-1 rounded-8 border px-2.5 py-1 font-semibold ${
                rated === 'helpful' ? 'border-[#143980] bg-[#143980] text-white' : 'border-[#D6DEEA] bg-white text-[#143980]'
              }`}
            >
              <ThumbsUp className="h-3.5 w-3.5" aria-hidden="true" />
              Hữu ích
            </button>
            <button
              type="button"
              aria-pressed={rated === 'unclear'}
              onClick={() => rate('unclear')}
              className={`inline-flex items-center gap-1 rounded-8 border px-2.5 py-1 font-semibold ${
                rated === 'unclear' ? 'border-[#143980] bg-[#143980] text-white' : 'border-[#D6DEEA] bg-white text-[#143980]'
              }`}
            >
              <ThumbsDown className="h-3.5 w-3.5" aria-hidden="true" />
              Báo thông tin chưa rõ
            </button>
          </div>
        )}

        {/* Nút thử lại chỉ xuất hiện khi sinh thất bại; gửi lại ngay câu hỏi trước đó, không bao giờ rỗng. */}
        {generationFailure && onRetry && (
          <button
            type="button"
            onClick={onRetry}
            aria-label={LOCALIZED_ERROR_COPY.server}
            className="mt-3 inline-flex items-center gap-1 text-[12.5px] leading-[16px] text-[#143980] font-medium hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#143980]"
          >
            <RefreshCw className="w-3 h-3" aria-hidden="true" />
            Thử lại
          </button>
        )}
      </div>
      <span className="ml-2 mt-0.5 text-[11px] leading-[14px] text-[#6B7686] font-medium">
        {message.timestamp.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })}
      </span>
    </div>
  );
}
