import type { ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

/**
 * The one primary action of a work area (`#3730A3`). Invariant: every
 * `<section data-workarea>` renders at most one `[data-variant="primary"]`.
 */
export const PrimaryButton = ({ className, type = "button", ...props }: ButtonHTMLAttributes<HTMLButtonElement>) => (
  <button
    type={type}
    data-variant="primary"
    className={cn(
      "inline-flex min-h-11 items-center justify-center gap-2 rounded-8 bg-brand-navy px-4 text-base font-semibold text-white hover:bg-[#102B61] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-navy disabled:cursor-not-allowed disabled:opacity-50",
      className,
    )}
    {...props}
  />
);
