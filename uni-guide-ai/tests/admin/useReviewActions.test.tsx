import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { CanonicalRecord } from "@/lib/adminApi";
import { useReviewActions } from "@/hooks/useReviewActions";

const teacher = vi.hoisted(() => ({
  approveAndPublish: vi.fn(),
  approveAndPublishMany: vi.fn(),
  claimDraft: vi.fn(),
  rejectWithReason: vi.fn(),
}));
const keySeq = vi.hoisted(() => ({ n: 0 }));
const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }));
vi.mock("@/lib/teacherApi", () => teacher);
vi.mock("sonner", () => ({ toast }));
vi.mock("@/lib/adminApi", () => ({ newIdempotencyKey: () => "key-" + ++keySeq.n }));

const record = {
  record_id: "HOC_PHI:2026",
  record_type: "HOC_PHI",
  admission_year: 2026,
  version: 3,
  authored_by: "someone-else",
  text: "Học phí.",
} as unknown as CanonicalRecord;
const identity = { actor_id: "me", actions: ["records.approve"] };

const setup = () => {
  const queryClient = new QueryClient();
  return renderHook(() => useReviewActions({ queryClient, identity }));
};

describe("useReviewActions", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    keySeq.n = 0;
  });

  it("double-click guard: a second commit while one is in flight sends nothing", async () => {
    let finish!: () => void;
    teacher.approveAndPublish.mockReturnValue(new Promise<void>((resolve) => (finish = resolve)));
    const { result } = setup();
    act(() => {
      result.current.commit(record, "approve");
      result.current.commit(record, "approve");
    });
    expect(teacher.approveAndPublish).toHaveBeenCalledTimes(1);
    await act(async () => finish());
    await waitFor(() => expect(result.current.pendingAction).toBeNull());
  });

  it("batch error resets the in-flight key and confirm state", async () => {
    teacher.approveAndPublishMany.mockRejectedValue(new Error("boom"));
    const { result } = setup();
    act(() => {
      result.current.toggleSelect(record.record_id);
      result.current.setBatchConfirming(true);
    });
    await act(async () => {
      await result.current.runBatchApprove([record]);
    });
    expect(result.current.batchKey).toBeNull();
    expect(result.current.batchConfirming).toBe(false);
    expect(toast.error).toHaveBeenCalledWith("boom");
  });

  it("idempotency key is stable across a retry of the same reject", async () => {
    teacher.rejectWithReason
      .mockRejectedValueOnce(new Error("network down"))
      .mockResolvedValueOnce(undefined);
    const { result } = setup();
    act(() => result.current.commit(record, "reject", "thiếu số liệu"));
    await waitFor(() => expect(result.current.actionError).toBe("network down"));
    act(() => result.current.commit(record, "reject", "thiếu số liệu"));
    await waitFor(() => expect(teacher.rejectWithReason).toHaveBeenCalledTimes(2));
    const firstKey = teacher.rejectWithReason.mock.calls[0][4];
    const secondKey = teacher.rejectWithReason.mock.calls[1][4];
    expect(firstKey).toBe(secondKey);
    expect(teacher.rejectWithReason.mock.calls[1][3]).toBe("thiếu số liệu");
  });
});
