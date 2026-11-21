// Thẻ "Hồ sơ xét tuyển của bạn" ở rail trái: tóm tắt chỉ đọc từ hồ sơ hiện có;
// "Chỉnh sửa" mở StudentProfileForm ngay trong thẻ và thu gọn lại sau khi lưu.
import { useState, type ReactNode } from 'react';
import { useStudentProfile } from '@/hooks/useStudentProfile';
import { DT_OPTIONS, METHOD_OPTIONS, StudentProfileForm } from './StudentProfileForm';

const EMPTY = 'Chưa nhập';

function labelOf(options: { value: string; label: string }[], value: string | null | undefined): string {
  if (!value) return EMPTY;
  return options.find((option) => option.value === value)?.label ?? value;
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex gap-2 text-[12px] leading-[18px]">
      <dt className="w-24 shrink-0 text-[#6B7686]">{label}</dt>
      <dd className="min-w-0 break-words font-medium text-[#0F1B2D]">{children}</dd>
    </div>
  );
}

export function EligibilityCard() {
  const { profile } = useStudentProfile();
  const [editing, setEditing] = useState(false);
  const majors = profile?.target_majors ?? [];

  return (
    <section aria-label="Hồ sơ xét tuyển của bạn" className="rounded-12 border border-[#D6DEEA] bg-[#F7F9FC] p-4">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-[14px] font-semibold leading-[20px] text-[#0F1B2D]">Hồ sơ xét tuyển của bạn</h2>
        <button
          type="button"
          aria-expanded={editing}
          onClick={() => setEditing((open) => !open)}
          className="shrink-0 rounded-8 border border-[#D6DEEA] bg-white px-2.5 py-1 text-[12px] font-semibold text-[#143980] hover:bg-[#F3F5F9]"
        >
          {editing ? 'Đóng' : 'Chỉnh sửa'}
        </button>
      </div>

      {editing ? (
        <div className="mt-3">
          <StudentProfileForm onSaved={() => setEditing(false)} />
        </div>
      ) : (
        <dl className="mt-3 space-y-2">
          <Row label="Điểm thi">{profile?.raw_score != null ? String(profile.raw_score) : EMPTY}</Row>
          <Row label="Phương thức">{labelOf(METHOD_OPTIONS, profile?.method)}</Row>
          <Row label="Tổ hợp">{profile?.to_hop || EMPTY}</Row>
          <Row label="Ngành quan tâm">{majors.length > 0 ? majors.join(', ') : EMPTY}</Row>
          <Row label="Khu vực">{profile?.uu_tien_kv || EMPTY}</Row>
          <Row label="Đối tượng">{labelOf(DT_OPTIONS, profile?.uu_tien_dt)}</Row>
        </dl>
      )}
    </section>
  );
}
