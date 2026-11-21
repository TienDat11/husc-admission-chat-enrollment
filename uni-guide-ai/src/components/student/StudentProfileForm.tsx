// @spec(PL-7.2 Frame 12 — anonymous student profile form)
// StudentProfileForm — nhập / cập nhật / xóa hồ sơ thí sinh ẩn danh.
// Không lưu tên/email/phone/cccd/ip (xem §PL-4 student_profiles cấm các cột này).
import { useCallback, useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import {
  Button,
} from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { useStudentProfile } from '@/hooks/useStudentProfile';
import { toast } from 'sonner';

const schema = z.object({
  raw_score: z.coerce.number().min(0, 'Điểm phải ≥ 0').max(30, 'Điểm phải ≤ 30'),
  method: z.enum(['thi_thpt', 'xe_tuyen_thang', 'cn_sinh_vien', 'khac']),
  to_hop: z.string().min(1, 'Chọn tổ hợp'),
  uu_tien_kv: z.enum(['KV1', 'KV2NT', 'KV2', 'KV3', 'none']),
  uu_tien_dt: z.enum(['UT1', 'UT2', 'none']),
  target_majors: z
    .string()
    .trim()
    .refine(
      (v) => {
        if (!v) return true;
        const items = v.split(',').map((s) => s.trim()).filter(Boolean);
        return items.length <= 10;
      },
      'Tối đa 10 ngành',
    ),
  admission_year: z.coerce.number().int().min(2020).max(2030).optional(),
});

type FormValues = z.infer<typeof schema>;

const KV_OPTIONS = [
  { value: 'none', label: 'Không (KV3)' },
  { value: 'KV1', label: 'KV1 — ưu tiên 0,75' },
  { value: 'KV2NT', label: 'KV2 NT — ưu tiên 0,50' },
  { value: 'KV2', label: 'KV2 — ưu tiên 0,25' },
];

export const DT_OPTIONS = [
  { value: 'none', label: 'Không' },
  { value: 'UT1', label: 'UT1 — ưu tiên 2,0' },
  { value: 'UT2', label: 'UT2 — ưu tiên 1,0' },
];

export const METHOD_OPTIONS = [
  { value: 'thi_thpt', label: 'Xét điểm thi tốt nghiệp' },
  { value: 'xe_tuyen_thang', label: 'Xét tuyển thẳng' },
  { value: 'cn_sinh_vien', label: 'Cử nhân cùng ngành' },
  { value: 'khac', label: 'Khác' },
];

export function StudentProfileForm({ onSaved }: { onSaved?: () => void } = {}) {
  const { profile, setProfile, clearProfile, hasProfile } = useStudentProfile();
  const [saving, setSaving] = useState(false);
  const [rateLimitedMs, setRateLimitedMs] = useState(0); // 0 = not limited
  const [message, setMessage] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);

  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      raw_score: '',
      method: 'thi_thpt',
      to_hop: '',
      uu_tien_kv: 'none',
      uu_tien_dt: 'none',
      target_majors: '',
      admission_year: undefined,
    },
  });

  const canSave = !saving && rateLimitedMs <= 0;

  const countdown = useCallback(() => {
    if (rateLimitedMs <= 0) return '';
    const remaining = Math.max(0, rateLimitedMs - Date.now());
    if (remaining <= 0) {
      setRateLimitedMs(0);
      return '';
    }
    return `(${Math.ceil(remaining / 1000)}s)`;
  }, [rateLimitedMs]);

  const handleSave = async (values: FormValues) => {
    setMessage(null);
    setSaving(true);
    try {
      const targetMajors = values.target_majors
        ? values.target_majors.split(',').map((s) => s.trim()).filter(Boolean)
        : [];
      const res = await fetch('/v2/student-profile', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          raw_score: values.raw_score,
          method: values.method,
          to_hop: values.to_hop,
          uu_tien_kv: values.uu_tien_kv === 'none' ? null : values.uu_tien_kv,
          uu_tien_dt: values.uu_tien_dt === 'none' ? null : values.uu_tien_dt,
          target_majors,
          admission_year: values.admission_year,
        }),
      });
      if (res.status === 429) {
        const retryAfter = res.headers.get('Retry-After');
        const ms = retryAfter ? parseInt(retryAfter, 10) * 1000 : 60_000;
        setRateLimitedMs(Date.now() + ms);
        setMessage({ kind: 'err', text: `Quá nhiều yêu cầu. Vui lòng đợi ${countdown()}.` });
        return;
      }
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setMessage({ kind: 'err', text: err?.message || 'Lưu hồ sơ thất bại.' });
        return;
      }
      const data = await res.json();
      setProfile(data as never);
      setMessage({ kind: 'ok', text: 'Đã lưu hồ sơ thí sinh.' });
      toast.success('Đã lưu hồ sơ thí sinh');
      onSaved?.();
    } catch (err) {
      setMessage({ kind: 'err', text: 'Không thể kết nối máy chủ. Vui lòng thử lại.' });
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!window.confirm('Xóa hồ sơ thí sinh hiện tại?')) return;
    setSaving(true);
    try {
      const res = await fetch('/v2/student-profile', { method: 'DELETE' });
      if (res.status === 429) {
        const retryAfter = res.headers.get('Retry-After');
        const ms = retryAfter ? parseInt(retryAfter, 10) * 1000 : 60_000;
        setRateLimitedMs(Date.now() + ms);
        setMessage({ kind: 'err', text: `Quá nhiều yêu cầu. Vui lòng đợi ${countdown()}.` });
        return;
      }
      if (!res.ok) {
        setMessage({ kind: 'err', text: 'Xóa hồ sơ thất bại.' });
        return;
      }
      clearProfile();
      setMessage({ kind: 'ok', text: 'Đã xóa hồ sơ.' });
      toast.success('Đã xóa hồ sơ thí sinh');
    } catch (err) {
      setMessage({ kind: 'err', text: 'Không thể kết nối máy chủ.' });
    } finally {
      setSaving(false);
    }
  };

  const disabledSave = !canSave || !form.formState.isValid;
  const saveLabel = canSave
    ? hasProfile ? 'Cập nhật hồ sơ' : 'Lưu hồ sơ'
    : `Đang chờ ${countdown()}`;

  return (
    <Card className="mt-4">
      <CardHeader className="flex flex-row items-center justify-between py-2">
        <CardTitle className="text-[13px] leading-none">Hồ sơ thí sinh (ẩn danh)</CardTitle>
        <CardDescription className="text-[11px]">
          {hasProfile ? 'Đã có hồ sơ' : 'Chưa có hồ sơ'}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {message && (
          <div
            className={`rounded border px-2 py-1.5 text-[11.5px] ${
              message.kind === 'ok'
                ? 'border-[#B9E3C9] bg-[#F0FDF4] text-[#14532D]'
                : 'border-[#FCA5A5] bg-[#FEF2F2] text-[#991B1B]'
            }`}
          >
            {message.text}
          </div>
        )}
        <form onSubmit={form.handleSubmit(handleSave)} className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label className="text-[11.5px] font-medium">Điểm thi (0–30)</Label>
              <Input
                {...form.register('raw_score')}
                type="number"
                min={0}
                max={30}
                step="0.1"
                placeholder="24"
                disabled={!canSave}
                className="h-7 text-[12px]"
              />
              {form.formState.errors.raw_score && (
                <span className="text-[10px] text-[#991B1B]">
                  {form.formState.errors.raw_score.message}
                </span>
              )}
            </div>
            <div className="space-y-1">
              <Label className="text-[11.5px] font-medium">Phương thức</Label>
              <Select
                {...form.register('method')}
                disabled={!canSave}
                onValueChange={(v) => form.setValue('method', v)}
              >
                <SelectTrigger className="h-7 text-[12px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {METHOD_OPTIONS.map((o) => (
                    <SelectItem key={o.value} value={o.value}>
                      {o.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label className="text-[11.5px] font-medium">Tổ hợp</Label>
              <Input
                {...form.register('to_hop')}
                placeholder="A00 / A01 / ..."
                disabled={!canSave}
                className="h-7 text-[12px]"
              />
            </div>
            <div className="space-y-1">
              <Label className="text-[11.5px] font-medium">Năm tuyển sinh</Label>
              <Input
                {...form.register('admission_year')}
                type="number"
                placeholder="2025"
                disabled={!canSave}
                className="h-7 text-[12px]"
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label className="text-[11.5px] font-medium">Ưu tiên KV</Label>
              <Select
                {...form.register('uu_tien_kv')}
                disabled={!canSave}
                onValueChange={(v) => form.setValue('uu_tien_kv', v)}
              >
                <SelectTrigger className="h-7 text-[12px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {KV_OPTIONS.map((o) => (
                    <SelectItem key={o.value} value={o.value}>
                      {o.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <Label className="text-[11.5px] font-medium">Ưu tiên ĐT</Label>
              <Select
                {...form.register('uu_tien_dt')}
                disabled={!canSave}
                onValueChange={(v) => form.setValue('uu_tien_dt', v)}
              >
                <SelectTrigger className="h-7 text-[12px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {DT_OPTIONS.map((o) => (
                    <SelectItem key={o.value} value={o.value}>
                      {o.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="space-y-1">
            <Label className="text-[11.5px] font-medium">Ngành mục tiêu (tối đa 10)</Label>
            <Input
              {...form.register('target_majors')}
              placeholder="CNTT, KTPM, HTTT"
              disabled={!canSave}
              className="h-7 text-[12px]"
            />
            <span className="text-[10px] text-[#6B7686]">
              Nhập tên ngành cách nhau bởi dấu phẩy (tối đa 10).
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="submit"
              disabled={disabledSave}
              className="px-3 py-1.5 text-[12px]"
            >
              {saveLabel}
            </Button>
            {hasProfile && (
              <Button
                variant="destructive"
                onClick={handleDelete}
                disabled={!canSave}
                className="px-3 py-1.5 text-[12px]"
              >
                Xóa hồ sơ
              </Button>
            )}
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
