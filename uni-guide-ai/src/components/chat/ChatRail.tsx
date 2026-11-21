// @spec(PL-7.2 Frame 12 XaDgH — rail trái 320, nền trắng, kẻ 1px phải)
// ChatRail — rail trái của màn hình chat thí sinh (lg trở lên): câu hỏi mới, lịch sử
// của phiên hiện tại, chủ đề nhanh, hồ sơ ở đáy. Không có cột menu riêng.
import { MessageCircle, Shield } from 'lucide-react';
import type { ReactNode } from 'react';
import type { SessionHistoryEntry } from '@/lib/session-history';

interface ChatRailProps {
  /** Được gọi khi học sinh chọn một chủ đề nhanh. */
  onTopicSelect: (question: string) => void;
  onNewChat: () => void;
  /** Câu hỏi của phiên hiện tại, mới nhất sau cùng. */
  history: SessionHistoryEntry[];
  activeMessageId: string | null;
  onSelectEntry: (messageId: string) => void;
  /** Hồ sơ thí sinh ở đáy rail; null khi rail không hiển thị hồ sơ. */
  profile: ReactNode;
}

const TOPIC_CHIPS = ['Hồ sơ cần gì?', 'Lịch xét tuyển đợt 1', 'Học bổng'];

export function ChatRail({
  onTopicSelect,
  onNewChat,
  history,
  activeMessageId,
  onSelectEntry,
  profile,
}: ChatRailProps) {
  return (
    <aside className="flex flex-col h-full w-[320px] overflow-y-auto border-r border-[#E3E8F0] bg-white shrink-0">
      {/* Khối thương hiệu */}
      <div className="flex items-center gap-3 px-4 py-4 border-b border-[#E3E8F0]">
        <Shield className="w-6 h-[28px] text-[#143980]" aria-hidden="true" />
        <span className="text-[15px] font-semibold leading-[20px] text-[#0F1B2D]">
          Trợ lý tuyển sinh
        </span>
      </div>

      <div className="px-4 pt-4 shrink-0">
        <button
          type="button"
          onClick={onNewChat}
          className="w-full h-10 rounded-8 border border-[#D6DEEA] bg-white text-[14px] font-semibold text-[#143980] hover:bg-[#F3F5F9] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#143980] focus-visible:ring-offset-2"
        >
          Câu hỏi mới
        </button>
      </div>

      {/* Hôm nay: câu hỏi của phiên hiện tại */}
      <div className="shrink-0 px-4 py-4">
        <p className="text-[13px] font-semibold leading-[18px] text-[#6B7686] mb-3">Hôm nay</p>
        {history.length === 0 ? (
          <p className="text-[13px] leading-[18px] text-[#6B7686]">
            Chưa có câu hỏi nào trong phiên này.
          </p>
        ) : (
          <div className="space-y-2">
            {history.map((entry) => {
              const isActive = activeMessageId === entry.messageId;
              return (
                <button
                  type="button"
                  key={entry.messageId}
                  data-testid="rail-history-item"
                  onClick={() => onSelectEntry(entry.messageId)}
                  aria-current={isActive ? 'true' : undefined}
                  className={`w-full flex items-start gap-3 px-3 py-2.5 rounded-8 text-left transition-colors hover:bg-[#F3F5F9] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#143980] ${
                    isActive ? 'bg-[#EFF4FB]' : ''
                  }`}
                >
                  <MessageCircle className="mt-0.5 w-4 h-4 shrink-0 text-[#143980]" aria-hidden="true" />
                  <span className="min-w-0 break-words text-[13px] leading-[18px] text-[#4A5568] font-medium">
                    {entry.question}
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Chủ đề nhanh: hàng viền 1px */}
      <div className="shrink-0 px-4 py-4 border-t border-[#E3E8F0]">
        <p className="text-[13px] font-semibold leading-[18px] text-[#6B7686] mb-3">Chủ đề nhanh</p>
        <div className="space-y-2">
          {TOPIC_CHIPS.map((chip) => (
            <button
              type="button"
              key={chip}
              onClick={() => onTopicSelect(chip)}
              className="w-full h-9 rounded-8 border border-[#D6DEEA] bg-white px-3 text-left text-[13px] font-medium text-[#143980] hover:bg-[#F3F5F9] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#143980] disabled:pointer-events-none disabled:opacity-50"
              aria-label={`Chủ đề: ${chip}`}
            >
              {chip}
            </button>
          ))}
        </div>
      </div>

      {/* Hồ sơ xét tuyển của thí sinh */}
      {profile !== null ? (
        <div className="shrink-0 max-h-[45%] overflow-y-auto border-t border-[#E3E8F0] p-4">
          {profile}
        </div>
      ) : null}

      {/* Thông tin liên hệ: thẻ navy ở đáy rail */}
      <div className="shrink-0 m-4 rounded-12 bg-[#143980] p-4 text-white">
        <p className="text-[13px] font-semibold leading-[18px]">Phòng Đào tạo &amp; Công tác sinh viên</p>
        <p className="mt-0.5 text-[12px] leading-[16px] text-white/80">Trường ĐH Khoa học, ĐH Huế</p>
        <dl className="mt-3 space-y-1 text-[12px] leading-[16px]">
          <div className="flex gap-2"><dt className="w-16 shrink-0 text-white/70">Hotline</dt><dd>0234.3823290</dd></div>
          <div className="flex gap-2"><dt className="w-16 shrink-0 text-white/70">Email</dt><dd className="break-all">tuyensinh@husc.edu.vn</dd></div>
          <div className="flex gap-2"><dt className="w-16 shrink-0 text-white/70">Giờ làm</dt><dd>07:30–17:00 (T2–T6)</dd></div>
        </dl>
      </div>
    </aside>
  );
}
