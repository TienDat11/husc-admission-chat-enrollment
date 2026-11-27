import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { CanonicalRecord } from "@/lib/adminApi";
import type { ReviewActions } from "@/hooks/useReviewActions";
import { ReviewDecisionBar } from "@/components/teacher/review/ReviewDecisionBar";

const record = { record_id: "HOC_PHI:2026", record_type: "HOC_PHI", admission_year: 2026, version: 3, authored_by: "x" } as unknown as CanonicalRecord;

const makeReview = (overrides: Partial<ReviewActions> = {}) =>
  ({
    commit: vi.fn(),
    setConfirming: vi.fn(),
    setRejectReason: vi.fn(),
    rejectReason: "",
    confirming: false,
    actionError: null,
    pendingAction: null,
    ...overrides,
  }) as unknown as ReviewActions;

const renderBar = (review: ReviewActions) =>
  render(<ReviewDecisionBar review={review} record={record} canReview isOwnWork={false} diffPending={false} />);

describe("ReviewDecisionBar reject-reason collapse", () => {
  it("keeps the reason textarea mounted but collapsed on mobile; reject stays disabled with a hint", () => {
    renderBar(makeReview());
    const reject = screen.getByRole("button", { name: "Trả lại kèm lý do" });
    expect(reject).toBeDisabled();
    expect(reject).toHaveAttribute("aria-describedby", "duyet-ly-do-hint");
    expect(screen.getByText("Nhập lý do trả lại để bật nút")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Duyệt & Phát hành/ })).toBeInTheDocument();
    const wrapper = screen.getByLabelText(/Lý do trả lại/).closest("div");
    expect(wrapper).toHaveClass("hidden", "md:block");
    expect(screen.getByRole("button", { name: "Ghi chú trả lại" })).toHaveAttribute("aria-expanded", "false");
  });

  it("the toggle expands and collapses the textarea wrapper and flips aria-expanded", async () => {
    renderBar(makeReview());
    const toggle = screen.getByRole("button", { name: "Ghi chú trả lại" });
    const wrapper = screen.getByLabelText(/Lý do trả lại/).closest("div") as HTMLElement;
    expect(toggle).toHaveAttribute("aria-controls", wrapper.id);
    await userEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(wrapper).toHaveClass("block");
    await userEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(wrapper).toHaveClass("hidden", "md:block");
  });

  it("shows a truncated preview of the typed reason when collapsed, and no hint", () => {
    renderBar(makeReview({ rejectReason: "  thiếu số liệu  " }));
    expect(document.querySelector("p.truncate")).toHaveTextContent("thiếu số liệu");
    expect(screen.queryByText("Nhập lý do trả lại để bật nút")).toBeNull();
    expect(screen.getByRole("button", { name: "Trả lại kèm lý do" })).toBeEnabled();
  });

  it("reject with a reason commits the trimmed reason", async () => {
    const review = makeReview({ rejectReason: "  thiếu số liệu  " });
    renderBar(review);
    await userEvent.click(screen.getByRole("button", { name: "Trả lại kèm lý do" }));
    expect(review.commit).toHaveBeenCalledWith(record, "reject", "thiếu số liệu");
  });
});
