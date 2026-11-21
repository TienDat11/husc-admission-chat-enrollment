import type { ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export const GhostButton = ({ className, type = "button", ...props }: ButtonHTMLAttributes<HTMLButtonElement>) => (
  <button
    type={type}
    data-variant="ghost"
    className={cn(
      "inline-flex min-h-11 items-center justify-center gap-2 rounded-8 px-3 text-base font-medium text-[#143980] underline-offset-4 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#143980] disabled:cursor-not-allowed disabled:opacity-50",
      className,
    )}
    {...props}
  />
);
