// @spec(PL-7.2 Frame 12 XaDgH — khối "Lộ trình 5 bước của bạn": panel trắng viền 1px bo 12, đầu cao 44, năm hàng cao 40, chấm 20 trái
// ChatRoadmap — lộ trình 5 bước (dumb/presentational).
import { Check } from 'lucide-react';

interface Step {
  title: string;
  detail: string;
  status: 'done' | 'current' | 'pending';
}

interface ChatRoadmapProps {
  steps: Step[];
  readyCount: number;
}

export function ChatRoadmap({ steps, readyCount }: ChatRoadmapProps) {
  return (
    <section className="mt-6 border border-[#D6DEEA] rounded-xl bg-white">
      <div className="flex items-center justify-between h-[44px] border-b border-[#D6DEEA] px-3">
        <h4 className="text-[14px] font-semibold leading-[20px] text-[#0F1B2D]">
          Lộ trình xét tuyển &amp; nhập học
        </h4>
        <p className="text-[12px] leading-[16px] text-[#6B7686]">
          {readyCount} / 5 bước đã sẵn sàng
        </p>
      </div>
      <div className="divide-y divide-[#EDF1F6]">
        {steps.map((step, index) => {
          const done = step.status === 'done';
          const current = step.status === 'current';
          return (
            <div
              key={index}
              className={`h-[40px] flex items-center gap-2 px-3 ${
                current ? 'bg-[#FFF7E6] text-[#8A5A00]' : ''
              }`}
            >
              {done ? (
                <div className="w-5 h-5 rounded-full bg-[#2E7D5B] flex items-center justify-center">
                  <Check className="w-2.5 h-2.5 text-white" aria-hidden="true" />
                </div>
              ) : (
                <div
                  className={`w-5 h-5 rounded-full flex items-center justify-center border ${
                    current ? 'border-2 border-[#143980]' : 'border-[#D6DEEA]'
                  }`}
                />
              )}
              <div className="flex-1 min-w-0">
                <div className="text-[14px] font-semibold leading-[20px]">
                  {step.title}
                </div>
                <div className="text-[12.5px] leading-[18px] text-[#4A5568]">
                  {step.detail}
                </div>
              </div>
              {current && (
                <button
                  type="button"
                  className="text-[12.5px] leading-[16px] text-[#143980] font-medium hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#143980] disabled:pointer-events-none disabled:opacity-50"
                  aria-label="Cách đặt nguyện vọng"
                >
                  Cách đặt nguyện vọng
                </button>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
