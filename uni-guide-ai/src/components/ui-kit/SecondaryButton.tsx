import type { ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export const SecondaryButton = ({ className, type = "button", ...props }: ButtonHTMLAttributes<HTMLButtonElement>) => (
  <button
    type={type}
    data-variant="secondary"
    className={cn(
      "inline-flex min-h-11 items-center justify-center gap-2 rounded-8 border border-zinc-400 bg-white px-4 text-base font-semibold text-zinc-900 hover:bg-zinc-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-navy disabled:cursor-not-allowed disabled:opacity-50",
      className,
    )}
    {...props}
  />
);
