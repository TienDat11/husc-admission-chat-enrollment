import type { KeyboardEvent } from "react";

/**
 * Keyboard handling for a hand-built tablist (role="tablist" with role="tab"
 * buttons). Left and Right step through the tabs and wrap at the ends; Home and
 * End jump to the first and last tab. The tab that receives focus is also
 * activated, matching the Radix tabs used elsewhere. Other keys are ignored.
 */
export function handleTablistKey(event: KeyboardEvent<HTMLElement>, activate: (index: number) => void): void {
  const tabs = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('[role="tab"]'));
  if (tabs.length === 0) return;
  const selected = tabs.findIndex((tab) => tab.getAttribute("aria-selected") === "true");
  const focused = tabs.indexOf(event.target as HTMLElement);
  const from = focused !== -1 ? focused : Math.max(0, selected);

  let next: number;
  switch (event.key) {
    case "ArrowRight":
      next = (from + 1) % tabs.length;
      break;
    case "ArrowLeft":
      next = (from - 1 + tabs.length) % tabs.length;
      break;
    case "Home":
      next = 0;
      break;
    case "End":
      next = tabs.length - 1;
      break;
    default:
      return;
  }
  event.preventDefault();
  tabs[next].focus();
  activate(next);
}
