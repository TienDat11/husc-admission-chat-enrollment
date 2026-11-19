// @spec(PL-7.2 Frame 12 — thẻ "Ngoài phạm vi")
// OutOfScopeCard — hiển thị khi câu hỏi ngoài phạm vi tư vấn tuyển sinh.
// Bao gồm liên kết gọi nóng tel: Hotline.
import { AlertTriangle, Phone } from 'lucide-react';

const HOTLINE = '1900-1234';

export function OutOfScopeCard({
  reason = 'Câu hỏi này nằm ngoài phạm vi tư vấn tuyển sinh của trường.',
}: {
  reason?: string;
}) {
  return (
    <div className="mt-4 border border-[#D6DEEA] rounded-xl bg-[#FFF7F2] p-4">
      <div className="flex items-start gap-2">
        <AlertTriangle className="w-4 h-4 mt-0.5 text-[#9A3412] shrink-0" aria-hidden="true" />
        <div className="flex-1 min-w-0">
          <p className="text-[13px] leading-[18px] text-[#0F1B2D]">
            {reason}
          </p>
          <p className="mt-2 text-[12px] leading-[16px] text-[#6B7686]">
            Bạn có thể liên hệ phòng tuyển sinh để được hỗ trợ trực tiếp.
          </p>
          <a
            href={`tel:${HOTLINE.replace(/\D/g, '')}`}
            className="mt-2 inline-flex items-center gap-1.5 text-[12px] font-medium text-[#143980] underline hover:text-[#0f3d6e]"
          >
            <Phone className="w-3 h-3" aria-hidden="true" />
            Gọi Hotline: {HOTLINE}
          </a>
        </div>
      </div>
    </div>
  );
}
