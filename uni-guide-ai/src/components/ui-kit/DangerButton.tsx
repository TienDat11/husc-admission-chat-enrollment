import type { ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export const DangerButton = ({ className, type = "button", ...props }: ButtonHTMLAttributes<HTMLButtonElement>) => (
  <button
    type={type}
    data-variant="danger"
    className={cn(
      "inline-flex min-h-11 items-center justify-center gap-2 rounded-8 border border-red-700 bg-white px-4 text-base font-semibold text-red-800 hover:bg-red-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-700 disabled:cursor-not-allowed disabled:opacity-50",
      className,
    )}
    {...props}
  />
);
