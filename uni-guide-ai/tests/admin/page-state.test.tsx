import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { EmptyState, ErrorState, ListSkeleton } from "@/components/ui-kit/PageState";

describe("PageState", () => {
  it("ListSkeleton announces loading as a status region", () => {
    render(<ListSkeleton label="Đang tải danh sách" />);
    expect(screen.getByRole("status")).toHaveTextContent("Đang tải danh sách");
  });

  it("EmptyState shows its title and hint", () => {
    render(<EmptyState title="Chưa có mục nào" hint="Hãy quay lại sau." />);
    expect(screen.getByText("Chưa có mục nào")).toBeInTheDocument();
    expect(screen.getByText("Hãy quay lại sau.")).toBeInTheDocument();
  });

  it("ErrorState is an alert and its retry button calls onRetry", async () => {
    const onRetry = vi.fn();
    render(<ErrorState message="Không tải được dữ liệu." onRetry={onRetry} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Không tải được dữ liệu.");
    await userEvent.click(screen.getByRole("button", { name: "Thử lại" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("ErrorState without onRetry renders no retry button", () => {
    render(<ErrorState message="Lỗi." />);
    expect(screen.queryByRole("button", { name: "Thử lại" })).toBeNull();
  });
});
