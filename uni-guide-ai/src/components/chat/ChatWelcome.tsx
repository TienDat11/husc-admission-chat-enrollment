// @spec(PL-7.2 Frame 12 XaDgH — trạng thái rộng/chưa có tin nhắn: avatar halo lớn, tiêu đề "Xin chào!", mô tả 3 dòng, 3 hàng chủ đề toàn chiều rộng
// ChatWelcome — màn hình chào mừng trước khi có tin nhắn nào (dumb/presentational). Không phụ backend.
import { Shield } from 'lucide-react';
import { useCallback } from 'react';

interface ChatWelcomeProps {
  onTopicSelect: (question: string) => void;
}

const DESCRIPTION_LINES = [
  'Tôi là Trợ lý Tuyển sinh của Đại học Khoa học – Đại học Huế.',
  'Tôi giúp bạn tra cứu điểm chuẩn, ngành học, học phí và học bổng năm 2026.',
  'Chỉ cần nhập câu hỏi vào ô bên dưới, tôi sẽ trả lời ngay.',
];

const TOPIC_ROWS = [
  { label: 'Các ngành đào tạo năm 2025', question: 'Các ngành đào tạo năm 2025 là gì?' },
  { label: 'Phương thức xét tuyển', question: 'Phương thức xét tuyển năm 2026 là gì?' },
  { label: 'Học bổng & ưu đãi', question: 'Học bổng và ưu đãi sinh viên năm 2026' },
];

export function ChatWelcome({ onTopicSelect }: ChatWelcomeProps) {
  const handleTopic = useCallback(
    (question: string) => {
      onTopicSelect(question);
    },
    [onTopicSelect],
  );

  return (
    <div className="flex flex-col items-center justify-center h-full text-center py-20">
      <div className="w-24 h-24 rounded-full bg-[#E6ECF7] flex items-center justify-center mb-6 shadow-[0_0_40px_hsl(219,73%,29%/0.25)]">
        <Shield className="w-11 h-11 text-[#143980]" aria-hidden="true" />
      </div>
      <h2 className="font-semibold text-[24px] leading-[30px] text-[#0F1B2D] mb-6">
        Xin chào!
      </h2>
      <p className="text-[14px] leading-[1.65] text-[#4A5568] max-w-[200px] mx-auto mb-10">
        {DESCRIPTION_LINES.map((line, index) => (
          <>
            {line}
            {index < DESCRIPTION_LINES.length - 1 ? <br /> : null}
          </>
        ))}
      </p>
      <div className="w-full max-w-[280px] space-y-2">
        {TOPIC_ROWS.map((row) => (
          <button
            type="button"
            key={row.label}
            onClick={() => handleTopic(row.question)}
            className="w-full flex items-center justify-between gap-3 rounded-xl border border-[#E3E8F0] bg-white px-4 py-3 text-left hover:bg-[#F7F9FC] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#143980] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
            aria-label={row.label}
          >
            <span className="text-[14px] leading-[1.6] text-[#0F1B2D] font-medium">
              {row.label}
            </span>
            <span className="shrink-0 text-[12px] text-[#143980] font-semibold">›</span>
          </button>
        ))}
      </div>
    </div>
  );
}
