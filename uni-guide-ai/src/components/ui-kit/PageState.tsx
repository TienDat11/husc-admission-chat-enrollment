import type { ReactNode } from "react";

/** Loading placeholder for a list. Announces itself as a status region. */
export const ListSkeleton = ({
  rows = 3,
  label = "Đang tải dữ liệu…",
}: {
  rows?: number;
  label?: string;
}) => (
  <div role="status" aria-live="polite" className="space-y-2">
    <span className="sr-only">{label}</span>
    {Array.from({ length: rows }, (_, index) => (
      <div key={index} aria-hidden="true" className="h-12 animate-pulse rounded-[8px] bg-zinc-100" />
    ))}
  </div>
);

/** Shown when a query succeeded but there is nothing to display yet. */
export const EmptyState = ({
  title,
  hint,
  action,
}: {
  title: string;
  hint?: string;
  action?: ReactNode;
}) => (
  <div className="rounded-[12px] border border-dashed border-[#D6DEEA] bg-white p-6 text-center">
    <p className="font-display text-[15px] font-semibold text-[#0F1B2D]">{title}</p>
    {hint ? <p className="mt-1 text-[14px] text-[#474747]">{hint}</p> : null}
    {action ? <div className="mt-3">{action}</div> : null}
  </div>
);

/** Shown when a query failed. Pass onRetry (usually refetch) to offer a retry. */
export const ErrorState = ({
  message,
  onRetry,
  retryLabel = "Thử lại",
}: {
  message: string;
  onRetry?: () => void;
  retryLabel?: string;
}) => (
  <div role="alert" className="note bad">
    <p>{message}</p>
    {onRetry ? (
      <button type="button" className="btn mt-2" onClick={onRetry}>
        {retryLabel}
      </button>
    ) : null}
  </div>
);
