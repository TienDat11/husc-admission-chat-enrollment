import { Check } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export type PillTone = "ok" | "pending" | "overdue" | "blocked" | "muted";

/** The overdue marker. Always shown as text next to the overdue colour. */
export const OVERDUE_LABEL = "Quá hạn";

const TONE: Record<PillTone, { glyph: ReactNode; className: string }> = {
  ok: { glyph: <Check className="h-3.5 w-3.5" aria-hidden="true" />, className: "border-emerald-700 bg-emerald-50 text-emerald-950" },
  pending: { glyph: "…", className: "border-sky-700 bg-sky-50 text-sky-950" },
  overdue: { glyph: "!", className: "border-red-700 bg-red-50 text-red-950" },
  blocked: { glyph: "✕", className: "border-red-700 bg-white text-red-900" },
  muted: { glyph: "•", className: "border-zinc-500 bg-zinc-100 text-zinc-900" },
};

/**
 * A status label. The glyph and the label text carry the meaning; the colour
 * only reinforces it.
 */
export const StatusPill = ({ tone, label }: { tone: PillTone; label: string }) => (
  <span
    data-tone={tone}
    className={cn(
      "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-sm font-medium",
      TONE[tone].className,
    )}
  >
    <span aria-hidden="true">{TONE[tone].glyph}</span>
    {label}
  </span>
);
