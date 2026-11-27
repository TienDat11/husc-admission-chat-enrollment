import { useRef, useState } from "react";
import type { QueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { newIdempotencyKey, type CanonicalRecord } from "@/lib/adminApi";
import { invalidateRecordTransition } from "@/lib/recordTransitions";
import { approveAndPublish, approveAndPublishMany, claimDraft, rejectWithReason } from "@/lib/teacherApi";
import { getErrorMessage } from "@/lib/utils";

export type ReviewAction = "approve" | "reject";
export type BatchResult = {
  succeeded: string[];
  failures: Array<{ recordId: string; error: string }>;
};

const FALLBACK = "Hành động thất bại.";

type Options = {
  queryClient: QueryClient;
  identity: { actor_id: string; actions: readonly string[] } | null | undefined;
};

type Intent = { idempotencyKey: string; recordId: string; action: ReviewAction; reason: string | null };

/**
 * Mutation flows and their dialog state for the PendingReview page: single
 * approve/reject (one idempotency key per intention, kept across retries of
 * the same click), batch publish, and draft claim. Request payloads are
 * unchanged from the page this was extracted from.
 */
export function useReviewActions({ queryClient, identity }: Options) {
  const [confirming, setConfirming] = useState(false);
  const [rejectReason, setRejectReason] = useState("");
  const [actionError, setActionError] = useState<string | null>(null);
  const [pendingAction, setPendingAction] = useState<ReviewAction | null>(null);
  // One key per intention, minted when the teacher commits to an action,
  // kept across retries of that same click.
  const [intent, setIntent] = useState<Intent | null>(null);
  // Batch: one key per run, so a retry of the same batch replays.
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [batchConfirming, setBatchConfirming] = useState(false);
  const [batchKey, setBatchKey] = useState<string | null>(null);
  const [batchResult, setBatchResult] = useState<BatchResult | null>(null);
  // Claim: per-record confirm state so the 409 reason renders per card.
  const [claimConfirmingId, setClaimConfirmingId] = useState<string | null>(null);
  const [claimKey, setClaimKey] = useState<string | null>(null);
  const [claimError, setClaimError] = useState<string | null>(null);

  // Synchronous double-click guards: state updates land after the second click.
  const commitBusy = useRef(false);
  const batchBusy = useRef(false);
  const claimBusy = useRef(false);

  const commit = (record: CanonicalRecord, action: ReviewAction, reason?: string) => {
    if (commitBusy.current) return;
    // Four-eyes is re-checked here as well as on the button: a confirm dialog can
    // outlive the selection it was opened on.
    if (
      action === "approve" &&
      (record.authored_by === identity?.actor_id || !identity?.actions.includes("records.approve"))
    ) {
      setConfirming(false);
      setActionError("Bạn không thể duyệt mục này.");
      return;
    }
    const reasonValue = reason ?? null;
    const reused =
      intent &&
      intent.recordId === record.record_id &&
      intent.action === action &&
      intent.reason === reasonValue
        ? intent.idempotencyKey
        : null;
    const key = reused ?? newIdempotencyKey();
    setIntent({ idempotencyKey: key, recordId: record.record_id, action, reason: reasonValue });
    setActionError(null);
    const run =
      action === "approve"
        ? approveAndPublish(null, record.record_id, record.version, key)
        : rejectWithReason(null, record.record_id, record.version, reason ?? "", key);
    commitBusy.current = true;
    setPendingAction(action);
    run
      .then(() => {
        setConfirming(false);
        setRejectReason("");
        setIntent(null);
        invalidateRecordTransition(queryClient);
        toast.success(action === "approve" ? "Đã xuất bản mục này." : "Đã trả lại mục này.");
      })
      .catch((failure: unknown) => {
        const message = getErrorMessage(failure, FALLBACK);
        setActionError(message);
        toast.error(message);
      })
      .finally(() => {
        commitBusy.current = false;
        setPendingAction(null);
      });
  };

  const runBatchApprove = async (rows: CanonicalRecord[]) => {
    if (batchBusy.current) return;
    batchBusy.current = true;
    const key = batchKey ?? newIdempotencyKey();
    setBatchKey(key);
    setBatchConfirming(false);
    try {
      const result = await approveAndPublishMany(
        null,
        rows
          .filter((r) => selectedIds.includes(r.record_id))
          .map((r) => ({ recordId: r.record_id, version: r.version })),
        (recordId) => key + ":" + recordId,
      );
      setBatchResult(result);
      setSelectedIds([]);
      if (result.failures.length > 0) {
        toast.error(result.failures.length + " mục xuất bản thất bại.");
      } else {
        toast.success("Đã xuất bản " + result.succeeded.length + " mục.");
      }
    } catch (failure: unknown) {
      toast.error(getErrorMessage(failure, FALLBACK));
    } finally {
      batchBusy.current = false;
      setBatchKey(null);
      invalidateRecordTransition(queryClient);
    }
  };

  const runClaim = async (record: CanonicalRecord) => {
    if (claimBusy.current) return;
    claimBusy.current = true;
    const key = claimKey ?? newIdempotencyKey();
    setClaimKey(key);
    setClaimConfirmingId(null);
    setClaimError(null);
    try {
      await claimDraft(null, record.record_id, record.version, key);
      toast.success("Đã nhận nháp.");
      invalidateRecordTransition(queryClient);
    } catch (failure: unknown) {
      const message = getErrorMessage(failure, FALLBACK);
      setClaimError(message);
      toast.error(message);
    } finally {
      claimBusy.current = false;
      setClaimKey(null);
    }
  };

  const toggleSelect = (recordId: string) => {
    setSelectedIds((ids) =>
      ids.includes(recordId) ? ids.filter((id) => id !== recordId) : [...ids, recordId],
    );
    setBatchResult(null);
  };

  // A pending confirm or a typed reject reason belongs to the record it was opened on.
  const resetForSelection = () => {
    setConfirming(false);
    setRejectReason("");
    setActionError(null);
  };

  return {
    confirming, setConfirming, rejectReason, setRejectReason, actionError, pendingAction,
    commit, resetForSelection, selectedIds, toggleSelect, batchConfirming, setBatchConfirming,
    batchKey, batchResult, runBatchApprove, claimConfirmingId, setClaimConfirmingId,
    claimKey, claimError, runClaim,
  };
}

export type ReviewActions = ReturnType<typeof useReviewActions>;
