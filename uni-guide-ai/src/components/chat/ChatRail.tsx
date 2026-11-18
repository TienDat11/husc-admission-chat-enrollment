// @spec(PL-7.2 Frame 12 XaDgH — rail trái 320, nền trắng, kẻ 1px phải)
// ChatRail — panneau bên trái của màn hình chat thí sinh (không sidebar quản trị).
// Thành phần dumb/presentational: chỉ hiển thị và gọi lại onTopicSelect khi nhấn chip/hàng hôm nay.
import { MessageCircle, Shield } from 'lucide-react';
import { useCallback } from 'react';

interface ChatRailProps {
  /** Được gọi khi học sinh chọn một câu hỏi gợi ý / chip chủ đề. */
  onTopicSelect: (question: string) => void;
}

const TODAY_QUESTIONS = [
  'Học phí ngành CNTT năm 2026',
  'Điều kiện xét tuyển thẳng IELTS',
];

const TOPIC_CHIPS = ['Hồ sơ cần gì?', 'Lịch xét tuyển đợt 1', 'Học bổng'];

export function ChatRail({ onTopicSelect }: ChatRailProps) {
  const handleTopic = useCallback(
    (question: string) => {
      onTopicSelect(question);
    },
    [onTopicSelect],
  );

  return (
    <aside className="flex flex-col h-full w-[320px] border-r border-[#E3E8F0] bg-white shrink-0">
      {/* Khối thương hiệu */}
      <div className="flex items-center gap-3 px-4 py-4 border-b border-[#E3E8F0]">
        <Shield className="w-6 h-[28px] text-[#143980]" aria-hidden="true" />
        <span className="text-[15px] font-semibold leading-[20px] text-[#0F1B2D]">
          Trợ lý tuyển sinh
        </span>
      </div>

      {/* Hôm nay */}
      <div className="px-4 py-4">
        <p className="text-[13px] font-semibold leading-[18px] text-[#6B7686] mb-3">
          Hôm nay
        </p>
        <div className="space-y-2">
          {TODAY_QUESTIONS.map((question) => (
            <button
              type="button"
              key={question}
              onClick={() => handleTopic(question)}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-8 text-left hover:bg-[#F3F5F9] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#143980] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
              aria-label={`Hôm nay: ${question}`}
            >
              <MessageCircle className="w-4 h-4 shrink-0 text-[#143980]" aria-hidden="true" />
              <span className="text-[13px] leading-[18px] text-[#4A5568] font-medium">
                {question}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Chủ đề nhanh (chips bo 999, cao 32) */}
      <div className="px-4 py-4 border-b border-[#E3E8F0]">
        <p className="text-[13px] font-semibold leading-[18px] text-[#6B7686] mb-3">
          Chủ đề nhanh
        </p>
        <div className="flex flex-wrap gap-2">
          {TOPIC_CHIPS.map((chip) => (
            <button
              type="button"
              key={chip}
              onClick={() => handleTopic(chip)}
              className="flex-shrink-0 rounded-full border border-[#E3E8F0] bg-[#F7F9FC] px-3 py-1.5 text-[12.5px] leading-[16px] text-[#4A5568] font-medium hover:bg-[#143980] hover:text-white transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#143980] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
              aria-label={`Chủ đề: ${chip}`}
            >
              {chip}
            </button>
          ))}
        </div>
      </div>

      {/* Hồ sơ của bạn */}
      <div className="flex-1 px-4 py-4">
        <div className="rounded-8 bg-[#F7F9FC] p-3">
          <p className="text-[12px] leading-[16px] text-[#6B7686] mb-2">
            Chỉ lưu trên trình duyệt này, không cần tài khoản
          </p>
          <p className="text-[12.5px] leading-[18px] text-[#0F1B2D] mb-1">
            Điểm dự kiến 24.75 · A00
          </p>
          <p className="text-[12.5px] leading-[18px] text-[#0F1B2D] mb-1">
            Ưu tiên KV1 · Đối tượng 2
          </p>
          <p className="text-[12.5px] leading-[18px] text-[#0F1B2D] mb-3">
            Quan tâm: CNTT, Quản trị kinh doanh
          </p>
          <button
            type="button"
            onClick={() => {}}
            className="text-[12.5px] leading-[16px] text-[#143980] font-medium hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#143980] disabled:pointer-events-none disabled:opacity-50"
            aria-label="Chỉnh sửa hồ sơ"
          >
            Chỉnh sửa hồ sơ
          </button>
        </div>
      </div>

      {/* Đáy rail */}
      <div className="px-4 py-4">
        <p className="text-[12px] leading-[18px] text-[#4A5568]">
          Phòng Đào tạo &amp; Công tác sinh viên · 0234.3823290 ·
          tuyensinh@husc.edu.vn · 07:30–17:00 (T2–T6)
        </p>
      </div>
    </aside>
  );
}
