// @spec(PL-7.2 Frame 12 — recommender + anonymous student profile)
// MajorRecommender — gợi ý ngành phù hợp.
// Dùng useStudentProfile để lấy điểm ưu tiên KV/ĐT từ hồ sơ thí sinh ẩn danh
// thay vì nhập thủ công ô uuTien.
import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Sparkles,
  Loader2,
  AlertTriangle,
  ShieldCheck,
  AlertCircle,
  Skull,
  Info,
  FlaskConical,
} from 'lucide-react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useStudentProfile } from '@/hooks/useStudentProfile';
import {
  getRecommendations,
  getWhatIf,
  type RecommendItem,
  type RecommendLabel,
  type WhatIfResponse,
} from '@/lib/api';

const TO_HOP_OPTIONS = [
  { value: '__none__', label: 'Không chọn (tất cả tổ hợp)' },
  { value: 'A00', label: 'A00 — Toán, Lý, Hóa' },
  { value: 'A01', label: 'A01 — Toán, Lý, Anh' },
  { value: 'B00', label: 'B00 — Toán, Hóa, Sinh' },
  { value: 'C00', label: 'C00 — Văn, Sử, Địa' },
  { value: 'D01', label: 'D01 — Toán, Văn, Anh' },
];

const KV_OPTIONS = [
  { value: '__none__', label: 'Không (KV3)' },
  { value: 'KV1', label: 'KV1' },
  { value: 'KV2NT', label: 'KV2 NT' },
  { value: 'KV2', label: 'KV2' },
];

const DT_OPTIONS = [
  { value: '__none__', label: 'Không' },
  { value: 'UT1', label: 'UT1' },
  { value: 'UT2', label: 'UT2' },
];

// Map BE label → display (VI) + Tailwind class for the colored badge.
const LABEL_STYLES: Record<
  RecommendLabel,
  { label: string; icon: React.ReactNode; className: string }
> = {
  an_toan: {
    label: 'An toàn',
    icon: <ShieldCheck className="w-3.5 h-3.5" />,
    className:
      'bg-emerald-100 text-emerald-800 border-emerald-300 dark:bg-emerald-900/40 dark:text-emerald-200 dark:border-emerald-700',
  },
  can_nhac: {
    label: 'Cân nhắc',
    icon: <AlertCircle className="w-3.5 h-3.5" />,
    className:
      'bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-900/40 dark:text-amber-200 dark:border-amber-700',
  },
  mao_hiem: {
    label: 'Mạo hiểm',
    icon: <Skull className="w-3.5 h-3.5" />,
    className:
      'bg-rose-100 text-rose-800 border-rose-300 dark:bg-rose-900/40 dark:text-rose-200 dark:border-rose-700',
  },
};

interface MajorRecommenderProps {
  /**
   * When true, the component renders an additional compact "What-if" widget
   * below the recommend results. Demo can toggle this via prop.
   */
  showWhatIf?: boolean;
}

export function MajorRecommender({ showWhatIf = true }: MajorRecommenderProps) {
  const { profile, setProfile } = useStudentProfile();

  // ── Recommend form state ───────────────────────────────────────────────
  const [score, setScore] = useState<string>(profile?.raw_score ? String(profile.raw_score) : '24');
  const [toHop, setToHop] = useState<string>(profile?.to_hop ?? '__none__');
  const [kv, setKv] = useState<string>(profile?.uu_tien_kv || '__none__');
  const [dt, setDt] = useState<string>(profile?.uu_tien_dt || '__none__');

  // ── Result state ───────────────────────────────────────────────────────
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<RecommendItem[] | null>(null);
  const [basis, setBasis] = useState<string>('');
  const [disclaimer, setDisclaimer] = useState<string>('');

  const parsedScore = useMemo(() => Number.parseFloat(score), [score]);
  const kvValue = useMemo(() => (kv && kv !== '__none__' ? kv : null), [kv]);
  const dtValue = useMemo(() => (dt && dt !== '__none__' ? dt : null), [dt]);

  const scoreValid = useMemo(
    () => Number.isFinite(parsedScore) && parsedScore >= 0 && parsedScore <= 30,
    [parsedScore],
  );

  const canSubmit = scoreValid && !loading;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await getRecommendations({
        score: parsedScore,
        kv: kvValue,
        dt: dtValue,
        ...(toHop && toHop !== '__none__' ? { to_hop: toHop } : {}),
      });
      setResults(resp.recommendations);
      setBasis(resp.basis);
      setDisclaimer(resp.disclaimer);

      // Sync the profile with the current inputs so KV/ĐT persist.
      setProfile({
        ...profile,
        raw_score: parsedScore,
        to_hop: toHop !== '__none__' ? toHop : null,
        uu_tien_kv: kvValue,
        uu_tien_dt: dtValue,
      } as never);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Không thể kết nối máy chủ. Vui lòng thử lại.',
      );
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveProfile = () => {
    setProfile({
      ...profile,
      raw_score: parsedScore,
      to_hop: toHop !== '__none__' ? toHop : null,
      uu_tien_kv: kvValue,
      uu_tien_dt: dtValue,
    } as never);
  };

  return (
    <Card className="w-full max-w-3xl mx-auto border-2 border-border shadow-[0_6px_0_hsl(var(--border))] bg-card">
      <CardHeader>
        <div className="flex items-center gap-2">
          <span className="inline-flex w-10 h-10 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-secondary text-white shadow-md">
            <Sparkles className="w-5 h-5" />
          </span>
          <div>
            <CardTitle className="text-xl font-bold">
              Gợi ý ngành phù hợp
            </CardTitle>
            <CardDescription>
              Điền điểm của bạn — hệ thống sẽ xếp hạng các ngành theo mức độ
              phù hợp (tính toán tức thì, không cần LLM).
            </CardDescription>
          </div>
        </div>
        {profile && (
          <div className="text-[11px] leading-[14px] text-[#6B7686]">
            Đang dùng hồ sơ: điểm {profile.raw_score}, phương thức{' '}
            {profile.method ?? '—'}
          </div>
        )}
      </CardHeader>

      <CardContent className="space-y-5">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Score */}
            <div className="space-y-1.5">
              <Label htmlFor="mr-score" className="font-semibold">
                Điểm của bạn
              </Label>
              <Input
                id="mr-score"
                type="number"
                min={0}
                max={30}
                step={0.25}
                value={score}
                onChange={(e) => setScore(e.target.value)}
                placeholder="0 – 30"
                className="font-semibold"
              />
              <p className="text-xs text-muted-foreground">Thang 30, bước 0.25</p>
            </div>

            {/* Tổ hợp */}
            <div className="space-y-1.5">
              <Label htmlFor="mr-tohop" className="font-semibold">
                Tổ hợp môn
              </Label>
              <Select value={toHop} onValueChange={setToHop}>
                <SelectTrigger id="mr-tohop" className="font-medium">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {TO_HOP_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">Không bắt buộc</p>
            </div>

            {/* Ưu tiên KV */}
            <div className="space-y-1.5">
              <Label htmlFor="mr-kv" className="font-semibold">
                Khu vực ưu tiên (KV)
              </Label>
              <Select value={kv} id="mr-kv" onValueChange={setKv}>
                <SelectTrigger className="font-medium">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {KV_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Ưu tiên ĐT */}
            <div className="space-y-1.5">
              <Label htmlFor="mr-dt" className="font-semibold">
                Đối tượng ưu tiên (ĐT)
              </Label>
              <Select value={dt} id="mr-dt" onValueChange={setDt}>
                <SelectTrigger className="font-medium">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {DT_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {!scoreValid && (
            <p className="text-xs text-rose-600 dark:text-rose-400 font-medium">
              ⚠️ Điểm phải nằm trong khoảng 0 – 30.
            </p>
          )}

          <div className="flex items-center justify-between">
            <Button
              type="submit"
              disabled={!canSubmit}
              className="w-full sm:w-auto font-bold"
              size="lg"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Đang tính toán…
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 mr-2" />
                  Gợi ý ngành
                </>
              )}
            </Button>
            {profile && (
              <Button
                type="button"
                onClick={handleSaveProfile}
                variant="outline"
                size="sm"
                className="text-[12px] px-3"
              >
                Lưu vào hồ sơ
              </Button>
            )}
          </div>
        </form>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex items-start gap-2 rounded-xl border-2 border-rose-300 bg-rose-50 px-3 py-2 text-sm text-rose-800 dark:border-rose-700 dark:bg-rose-950/40 dark:text-rose-200"
            >
              <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <span className="font-medium">{error}</span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results */}
        <AnimatePresence mode="wait">
          {results && results.length > 0 && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="space-y-4"
            >
              <div className="flex items-center justify-between flex-wrap gap-2">
                <h3 className="text-base font-bold">
                  Kết quả ({results.length} ngành)
                </h3>
                <p className="text-xs text-muted-foreground">
                  Sắp xếp theo mức độ phù hợp
                </p>
              </div>

              <ul className="space-y-3">
                {results.map((item, idx) => {
                  const style = LABEL_STYLES[item.label];
                  return (
                    <motion.li
                      key={`${item.major_code}-${idx}`}
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: idx * 0.04 }}
                    >
                      <div className="rounded-2xl border-2 border-border bg-background p-4 hover:border-primary/40 transition-colors">
                        <div className="flex items-start justify-between gap-3 flex-wrap">
                          <div className="min-w-0 flex-1">
                            <h4 className="font-bold text-base leading-snug">
                              {item.major_name}
                            </h4>
                            <p className="text-xs text-muted-foreground mt-0.5 font-mono">
                              {item.major_code}
                            </p>
                          </div>
                          <Badge
                            variant="outline"
                            className={`${style.className} flex items-center gap-1 px-2.5 py-1 text-xs font-bold border-2`}
                          >
                            {style.icon}
                            {style.label}
                          </Badge>
                        </div>

                        <div className="mt-3 grid grid-cols-2 sm:grid-cols-3 gap-2 text-sm">
                          <Stat
                            label="Điểm chuẩn"
                            value={item.latest_diem_chuan?.toFixed(2) ?? '—'}
                          />
                          <Stat label="Năm" value={String(item.latest_year)} />
                          <Stat
                            label="Chênh lệch"
                            value={
                              typeof item.delta === 'number'
                                ? `${item.delta > 0 ? '+' : ''}${item.delta.toFixed(2)}`
                                : '—'
                            }
                            tone={
                              item.delta > 0
                                ? 'good'
                                : item.delta < 0
                                  ? 'bad'
                                  : 'neutral'
                            }
                          />
                        </div>

                        {item.to_hop && item.to_hop.length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-1.5">
                            {item.to_hop.map((t) => (
                              <span
                                key={t}
                                className="inline-flex items-center px-2 py-0.5 rounded-full bg-muted text-muted-foreground text-[10px] font-mono font-semibold"
                              >
                                {t}
                              </span>
                            ))}
                          </div>
                        )}

                        <p className="mt-3 text-sm text-foreground/80 leading-relaxed">
                          {item.explanation}
                        </p>
                      </div>
                    </motion.li>
                  );
                })}
              </ul>

              {/* Basis + disclaimer — prominent for trust */}
              <div className="rounded-2xl border-2 border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950/30 dark:text-amber-100">
                <div className="flex items-start gap-2">
                  <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  <div className="space-y-1">
                    <p className="font-bold">Cơ sở tính toán</p>
                    <p className="text-xs leading-relaxed">{basis}</p>
                  </div>
                </div>
                <div className="flex items-start gap-2 mt-2 pt-2 border-t border-amber-300/60 dark:border-amber-700/60">
                  <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-bold">Lưu ý</p>
                    <p className="text-xs leading-relaxed">{disclaimer}</p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {results && results.length === 0 && (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="rounded-2xl border-2 border-border bg-muted px-4 py-6 text-center text-sm text-muted-foreground"
            >
              Không tìm thấy ngành phù hợp với điểm và tổ hợp đã chọn. Hãy thử
              thay đổi organizer điểm hoặc bỏ chọn tổ hợp.
            </motion.div>
          )}
        </AnimatePresence>

        {/* What-if widget — secondary, optional */}
        {showWhatIf && results && results.length > 0 && (
          <WhatIfWidget
            score={parsedScore}
            kv={kvValue}
            dt={dtValue}
            toHop={toHop}
            majors={results.map((r) => ({
              major_code: r.major_code,
              major_name: r.major_name,
            }))}
          />
        )}
      </CardContent>
    </Card>
  );
}

// ── What-if sub-component ────────────────────────────────────────────────
function WhatIfWidget({
  score,
  kv,
  dt,
  toHop,
  majors,
}: {
  score: number;
  kv: string | null;
  dt: string | null;
  toHop: string;
  majors: { major_code: string; major_name: string }[];
}) {
  const [majorCode, setMajorCode] = useState<string>('');
  const [result, setResult] = useState<WhatIfResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    if (!majorCode) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await getWhatIf({
        score,
        major_code: majorCode,
        kv,
        dt,
        ...(toHop && toHop !== '__none__' ? { to_hop: toHop } : {}),
      });
      setResult(resp);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Không thể truy vấn what-if.',
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-2xl border-2 border-dashed border-primary/30 bg-primary/5 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <FlaskConical className="w-4 h-4 text-primary" />
        <h4 className="font-bold text-sm">What-if? Thử một ngành cụ thể</h4>
      </div>
      <div className="flex flex-col sm:flex-row gap-2">
        <Select value={majorCode} onValueChange={setMajorCode}>
          <SelectTrigger className="flex-1 font-medium">
            <SelectValue placeholder="Chọn một ngành từ kết quả ở trên…" />
          </SelectTrigger>
          <SelectContent>
            {majors.map((m) => (
              <SelectItem key={m.major_code} value={m.major_code}>
                {m.major_code} — {m.major_name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button
          type="button"
          onClick={run}
          disabled={!majorCode || loading}
          variant="secondary"
          className="font-bold"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Xem'}
        </Button>
      </div>
      {result && (
        <div className="rounded-lg border border-[#EDF1F6] bg-white p-3 space-y-1">
          <p className="text-xs font-semibold">Xác suất trúng tuyển ước tính</p>
          <div className="text-xl font-mono font-bold">
            {(result.probability ?? 0).toFixed(2)}%
          </div>
          <p className="text-xs text-[#6B7686]">
            Cơ sở: {result.basis ?? '—'} &nbsp;|&nbsp; Mức rủi ro:{' '}
            {result.risk_level ?? '—'}
          </p>
        </div>
      )}
      {error && (
        <p className="text-xs text-rose-700 dark:text-rose-300 font-medium">
          {error}
        </p>
      )}
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: 'good' | 'bad' | 'neutral' }) {
  const toneClass =
    tone === 'good'
      ? 'text-[#14532D]'
      : tone === 'bad'
        ? 'text-[#991B1B]'
        : 'text-[#6B7686]';
  return (
    <div className="text-center">
      <div className="text-[10px] text-[#6B7686] font-medium">{label}</div>
      <div className={`text-[12px] font-bold ${toneClass}`}>{value}</div>
    </div>
  );
}
