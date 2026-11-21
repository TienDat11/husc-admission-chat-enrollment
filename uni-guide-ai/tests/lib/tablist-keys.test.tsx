import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { handleTablistKey } from "@/lib/tablistKeys";

const LABELS = ["Một", "Hai", "Ba"];

function Demo() {
  const [selected, setSelected] = useState(0);
  return (
    <div role="tablist" onKeyDown={(event) => handleTablistKey(event, setSelected)}>
      {LABELS.map((label, index) => (
        <button
          key={label}
          type="button"
          role="tab"
          aria-selected={selected === index}
          tabIndex={selected === index ? 0 : -1}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

const tab = (label: string) => screen.getByRole("tab", { name: label });

describe("handleTablistKey — tablist tự làm việc bằng bàn phím", () => {
  it("ArrowRight and ArrowLeft move focus and selection, wrapping at the ends", () => {
    render(<Demo />);
    tab("Một").focus();

    fireEvent.keyDown(tab("Một"), { key: "ArrowRight" });
    expect(tab("Hai")).toHaveAttribute("aria-selected", "true");
    expect(document.activeElement).toBe(tab("Hai"));

    fireEvent.keyDown(tab("Hai"), { key: "ArrowLeft" });
    fireEvent.keyDown(tab("Một"), { key: "ArrowLeft" });
    expect(tab("Ba")).toHaveAttribute("aria-selected", "true");
    expect(document.activeElement).toBe(tab("Ba"));

    fireEvent.keyDown(tab("Ba"), { key: "ArrowRight" });
    expect(tab("Một")).toHaveAttribute("aria-selected", "true");
  });

  it("Home and End jump to the first and last tab", () => {
    render(<Demo />);
    tab("Một").focus();

    fireEvent.keyDown(tab("Một"), { key: "End" });
    expect(tab("Ba")).toHaveAttribute("aria-selected", "true");
    expect(document.activeElement).toBe(tab("Ba"));

    fireEvent.keyDown(tab("Ba"), { key: "Home" });
    expect(tab("Một")).toHaveAttribute("aria-selected", "true");
    expect(document.activeElement).toBe(tab("Một"));
  });

  it("roving tabindex: only the selected tab is in the tab order", () => {
    render(<Demo />);
    expect(tab("Một")).toHaveAttribute("tabindex", "0");
    expect(tab("Hai")).toHaveAttribute("tabindex", "-1");

    tab("Một").focus();
    fireEvent.keyDown(tab("Một"), { key: "ArrowRight" });
    expect(tab("Hai")).toHaveAttribute("tabindex", "0");
    expect(tab("Một")).toHaveAttribute("tabindex", "-1");
  });

  it("ignores keys that are not tab navigation keys", () => {
    render(<Demo />);
    tab("Một").focus();
    fireEvent.keyDown(tab("Một"), { key: "a" });
    expect(tab("Một")).toHaveAttribute("aria-selected", "true");
    expect(tab("Hai")).toHaveAttribute("aria-selected", "false");
  });
});
