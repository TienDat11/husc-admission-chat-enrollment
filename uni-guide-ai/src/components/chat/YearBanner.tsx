// @spec(S13.7) Frontend banner showing current admission year + freshness
import { useEffect, useState } from "react";

export interface MetaResponse {
  current_admission_year: number;
  latest_crawl_date: string | null;
  total_notifications: number | null;
  freshness_lag_days: number | null;
  freshness_alert: boolean;
}

interface YearBannerProps {
  /** Override the fetch URL (mainly for tests/storybook). */
  apiUrl?: string;
  /** Override locale; defaults to "vi". Pass "en" for the English variant. */
  locale?: "vi" | "en";
}

const STRINGS = {
  vi: {
    label: (year: number) => `Tư vấn tuyển sinh năm ${year}`,
    updated: (date: string) => `Cập nhật ${date}`,
    stale: (days: number) => `Dữ liệu cũ ${days} ngày`,
    error: "Không tải được trạng thái dữ liệu",
  },
  en: {
    label: (year: number) => `${year} admissions advisor`,
    updated: (date: string) => `Updated ${date}`,
    stale: (days: number) => `${days}-day stale`,
    error: "Could not load data freshness",
  },
} as const;


function formatDate(iso: string | null, locale: "vi" | "en"): string | null {
  if (!iso) return null;
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return null;
    return d.toLocaleDateString(locale === "vi" ? "vi-VN" : "en-US", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    });
  } catch {
    return null;
  }
}


export function YearBanner({ apiUrl = "/api/meta", locale = "vi" }: YearBannerProps) {
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const t = STRINGS[locale];

  useEffect(() => {
    let cancelled = false;
    fetch(apiUrl)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<MetaResponse>;
      })
      .then((data) => {
        if (!cancelled) setMeta(data);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [apiUrl]);

  if (error) {
    return (
      <div
        role="status"
        aria-label={t.error}
        className="text-xs text-muted-foreground/70 px-3 py-1.5"
        data-testid="year-banner-error"
      >
        {t.error}
      </div>
    );
  }

  if (!meta) {
    return (
      <div
        role="status"
        aria-busy="true"
        aria-label={locale === "vi" ? "Đang tải trạng thái dữ liệu" : "Loading data freshness status"}
        className="text-xs text-muted-foreground/50 px-3 py-1.5 animate-pulse"
        data-testid="year-banner-loading"
      >
        ...
      </div>
    );
  }

  const formattedDate = formatDate(meta.latest_crawl_date, locale);

  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-md bg-primary/5 text-primary"
      data-testid="year-banner"
    >
      <span className="font-medium">{t.label(meta.current_admission_year)}</span>
      {formattedDate && (
        <>
          <span aria-hidden="true">•</span>
          <span className="text-muted-foreground" data-testid="year-banner-updated">
            {t.updated(formattedDate)}
          </span>
        </>
      )}
      {meta.freshness_alert && meta.freshness_lag_days !== null && (
        <span
          className="ml-auto inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-200"
          role="alert"
          data-testid="year-banner-stale"
        >
          {t.stale(meta.freshness_lag_days)}
        </span>
      )}
    </div>
  );
}

export default YearBanner;
