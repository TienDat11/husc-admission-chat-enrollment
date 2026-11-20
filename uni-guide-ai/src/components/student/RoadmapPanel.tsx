// @spec(PL-7.2 Frame 12 — lộ trình 5 bước TT06)
// RoadmapPanel — hiển thị 5 bước lộ trình và trạng thái done / needs_input / blocked.
// Bước đầu tiên thiếu đầu vào -> needs_input; các bước sau -> blocked.
import { useEffect, useState } from 'react';
import { Check, AlertTriangle, Clock } from 'lucide-react';
import { useStudentProfile } from '@/hooks/useStudentProfile';
import { getApiBaseUrl } from '@/lib/api';

interface RoadmapStep {
  n: number;
  key: string;
  status: 'done' | 'needs_input' | 'blocked';
  needs_input: { field: string; hint: string };
  source?: { document_id: string; page: number } | null;
}

const LABELS: Record<string, { title: string; hint?: string }> = {
  hoso_diem: { title: '1. Hồ sơ điểm', hint: 'Nhập điểm thi tốt nghiệp (0–30).' },
  chon_phuong_thuc: {
    title: '2. Chọn phương thức',
    hint: 'Chọn phương thức xét tuyển (thi THPT / trực tiếp / ...).',
  },
  chon_to_hop: {
    title: '3. Chọn tổ hợp',
    hint: 'Chọn tổ hợp hợp lệ cho ngành mục tiêu.',
  },
  so_diem_chuan: {
    title: '4. Số điểm chuẩn',
    hint: 'Tra số điểm chuẩn năm gần nhất cho ngành mục tiêu.',
  },
  moc_thoi_gian: {
    title: '5. Mốc thời gian',
    hint: 'Xem khung thời gian đăng ký từ THOI_GIAN của đợt.',
  },
};

export function RoadmapPanel() {
  const { sessionId, profile, setProfile } = useStudentProfile();
  const [steps, setSteps] = useState<RoadmapStep[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    setLoading(true);
    fetch(`${getApiBaseUrl()}/v2/admission-roadmap`, {
      headers: { 'X-Student-Session': sessionId },
    })
      .then(async (res) => {
        if (res.status === 400) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err?.message || 'Phiên học viên không hợp lệ.');
        }
        if (!res.ok) throw new Error(`Lấy lộ trình thất bại (${res.status}).`);
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        setSteps(data?.steps ?? []);
        setError(null);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  if (loading) return <div className="mt-4 text-[12px] text-[#6B7686]">Đang tải lộ trình...</div>;
  if (error) return <div className="mt-4 text-[12px] text-[#991B1B]">{error}</div>;
  if (!steps) return null;

  const firstMissing = steps.findIndex((s) => s.status !== 'done');

  return (
    <section className="mt-4 border border-[#D6DEEA] rounded-xl bg-white">
      <div className="border-b border-[#D6DEEA] px-3 py-2">
        <h4 className="text-[13px] font-semibold text-[#0F1B2D]">
          Lộ trình xét tuyển &amp; nhập học (5 bước)
        </h4>
        <p className="text-[11px] text-[#6B7686]">
          Bước đầu tiên thiếu đầu vào sẽ hiển thị nhập liệu; các bước sau bị khóa cho đến khi xong bước trước.
        </p>
      </div>
      <div className="divide-y divide-[#EDF1F6]">
        {steps.map((step) => {
          const done = step.status === 'done';
          const needsInput = step.status === 'needs_input';
          const blocked = step.status === 'blocked';
          const label = LABELS[step.key] || { title: step.key };
          const isCurrent = needsInput && firstMissing === steps.indexOf(step);
          return (
            <div
              key={step.n}
              className={`h-[44px] flex items-center gap-2 px-3 ${
                isCurrent ? 'bg-[#FFF7E6] text-[#8A5A00]' : ''
              }`}
            >
              {done ? (
                <div className="w-5 h-5 rounded-full bg-[#2E7D5B] flex items-center justify-center shrink-0">
                  <Check className="w-2.5 h-2.5 text-white" aria-hidden="true" />
                </div>
              ) : blocked ? (
                <div className="w-5 h-5 rounded-full border border-[#D6DEEA] flex items-center justify-center shrink-0">
                  <Clock className="w-2.5 h-2.5 text-[#9CA3AF]" aria-hidden="true" />
                </div>
              ) : (
                <div className="w-5 h-5 rounded-full border-2 border-[#143980] flex items-center justify-center shrink-0" />
              )}
              <div className="flex-1 min-w-0">
                <div className="text-[13px] font-semibold leading-[18px]">
                  {label.title}
                </div>
                {needsInput && (
                  <div className="text-[11px] leading-[14px] text-[#991B1B] font-medium">
                    {label.hint || step.needs_input.hint}
                  </div>
                )}
                {blocked && (
                  <div className="text-[11px] leading-[14px] text-[#6B7686]">
                    Chờ hoàn thành bước trước.
                  </div>
                )}
              </div>
              {step.source && (
                <span
                  className="ml-2 text-[10px] px-1.5 py-0.5 rounded-full border border-[#D6DEEA] text-[#4A5568]"
                >
                  {step.source.page ? `Dữ liệu tuyển sinh ${step.source.page}` : 'Dữ liệu tuyển sinh'}
                </span>
              )}
            </div>
          );
        })}
      </div>
      {profile && (
        <div className="px-3 py-2 border-t border-[#EDF1F6] bg-[#F7F9FC]">
          <p className="text-[11px] text-[#6B7686]">
            Điểm: {profile.raw_score ?? '—'} &nbsp;|&nbsp; Phương thức:{' '}
            {profile.method ?? '—'} &nbsp;|&nbsp; Tổ hợp:{' '}
            {profile.to_hop ?? '—'} &nbsp;|&nbsp; Ngành: {(
              profile.target_majors ?? []
            ).join(', ') || '—'}
          </p>
        </div>
      )}
    </section>
  );
}
