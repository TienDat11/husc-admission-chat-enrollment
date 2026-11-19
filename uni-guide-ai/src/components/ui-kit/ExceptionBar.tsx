import type { ReactNode } from "react";
import { SecondaryButton } from "./SecondaryButton";

export type ExceptionSeverity = "error" | "warning" | "info";

const PREFIX: Record<ExceptionSeverity, string> = {
  error: "Lỗi",
  warning: "Cảnh báo",
  info: "Thông tin",
};

const GLYPH: Record<ExceptionSeverity, string> = {
  error: "✕",
  warning: "!",
  info: "i",
};

const SURFACE: Record<ExceptionSeverity, string> = {
  error: "border-red-700 bg-red-50 text-red-950",
  warning: "border-amber-700 bg-amber-50 text-amber-950",
  info: "border-sky-700 bg-sky-50 text-sky-950",
};

/**
 * A system-state banner (PL-7 errors, lockout, frozen writes). The severity is
 * carried by a glyph and the Vietnamese prefix as well as the colour, so a
 * reader who cannot see the colour still gets the state. The optional action is
 * a secondary button: a work area keeps exactly one primary button.
 */
export const ExceptionBar = ({
  severity = "error",
  title,
  detail,
  action,
  children,
}: {
  severity?: ExceptionSeverity;
  title: string;
  detail?: string;
  action?: { label: string; onClick: () => void; disabled?: boolean };
  children?: ReactNode;
}) => (
  <div
    role={severity === "error" ? "alert" : "status"}
    data-severity={severity}
    className={`flex flex-col gap-2 rounded-md border-l-4 p-3 text-base ${SURFACE[severity]}`}
  >
    <p className="font-semibold">
      <span aria-hidden="true" className="mr-2 inline-block w-5 text-center font-bold">
        {GLYPH[severity]}
      </span>
      <span>
        {PREFIX[severity]}: {title}
      </span>
    </p>
    {detail ? <p>{detail}</p> : null}
    {children}
    {action ? (
      <div>
        <SecondaryButton onClick={action.onClick} disabled={action.disabled}>
          {action.label}
        </SecondaryButton>
      </div>
    ) : null}
  </div>
);
