// @spec(PL-7.2 Frame 12 — chip nguồn)
// SourceChip — chip nguồn mở tài liệu PDF tại trang N: `...pdf#page=N`.
import { ExternalLink } from 'lucide-react';

interface SourceChipProps {
  source: {
    id: string;
    title: string;
    url?: string;
    page?: number;
    data_year?: string;
  };
  className?: string;
}

export function SourceChip({
  source,
  className,
}: SourceChipProps) {
  const base = source.url || '';
  const href = base && typeof source.page === 'number'
    ? `${base}#page=${source.page}`
    : base;
  const chipClass =
    'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-xl border border-[#D6DEEA] bg-[#F7F9FC] text-[12px] leading-[16px] text-[#4A5568] font-medium hover:bg-[#E6ECF7] transition-colors ' +
    (className ?? '');

  if (!href) {
    return (
      <span className={chipClass} title={source.title}>
        <span aria-hidden="true">📄</span>
        <span className="truncate max-w-[140px] block">{source.title}</span>
      </span>
    );
  }

  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={chipClass}
      title={`${source.title}${source.page ? ` — trang ${source.page}` : ''}`}
    >
      <span aria-hidden="true">📄</span>
      <span className="truncate max-w-[140px] block">{source.title}</span>
      {source.data_year && source.data_year !== 'N/A' && (
        <span className="ml-1.5 text-[10px] px-1 py-0.5 rounded-full bg-[#E6ECF7] text-[#143980] font-semibold">
          {source.data_year}
        </span>
      )}
      <ExternalLink className="w-3 h-3 text-[#143980]" aria-hidden="true" />
    </a>
  );
}
