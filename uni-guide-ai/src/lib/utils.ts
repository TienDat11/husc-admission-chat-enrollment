import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { DEFAULT_ERROR_MESSAGE } from "@/lib/copy";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const VI_LOCALE = "vi-VN";
const VI_DATE_OPTIONS: Intl.DateTimeFormatOptions = { day: "2-digit", month: "2-digit", year: "numeric" };
const VI_TIME_OPTIONS: Intl.DateTimeFormatOptions = { hour: "2-digit", minute: "2-digit", hour12: false };

type DateInput = string | number | Date | null | undefined;

function toValidDate(value: DateInput): Date | null {
  if (value === null || value === undefined || value === "") return null;
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

/** dd/mm/yyyy in vi-VN; `fallback` for empty or invalid input. */
export function formatViDate(value: DateInput, fallback = "—"): string {
  const date = toValidDate(value);
  return date ? date.toLocaleDateString(VI_LOCALE, VI_DATE_OPTIONS) : fallback;
}

/** dd/mm/yyyy hh:mm in vi-VN; `fallback` for empty or invalid input. */
export function formatViDateTime(value: DateInput, fallback = "—"): string {
  const date = toValidDate(value);
  if (!date) return fallback;
  return `${date.toLocaleDateString(VI_LOCALE, VI_DATE_OPTIONS)} ${date.toLocaleTimeString(VI_LOCALE, VI_TIME_OPTIONS)}`;
}

/** Message of an Error (or a non-empty string); otherwise `fallback`. */
export function getErrorMessage(err: unknown, fallback: string = DEFAULT_ERROR_MESSAGE): string {
  if (err instanceof Error && err.message.trim()) return err.message;
  if (typeof err === "string" && err.trim()) return err;
  return fallback;
}
