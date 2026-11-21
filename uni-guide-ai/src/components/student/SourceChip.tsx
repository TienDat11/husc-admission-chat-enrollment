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
    'inline-flex max-w-full items-center gap-1.5 rounded-8 bg-[#E6ECF7] px-2.5 py-1 text-[12px] leading-[16px] font-semibold text-[#143980] transition-colors ' +
    (href ? 'hover:bg-[#D6DEEA] ' : '') +
    (className ?? '');
  const content = (
    <>
      <span>Nguồn:</span>
      <span className="min-w-0 truncate">{source.title}</span>
      {typeof source.page === 'number' && <span className="shrink-0">· Trang {source.page}</span>}
    </>
  );

  if (!href) {
    return (
      <span className={chipClass} title={source.title}>
        {content}
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
      {content}
      <ExternalLink className="h-3 w-3 shrink-0" aria-hidden="true" />
    </a>
  );
}
