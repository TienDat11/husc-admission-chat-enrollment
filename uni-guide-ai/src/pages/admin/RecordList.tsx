// Admissions record list — filter by ngành / năm / trạng thái, act on a record
// through the lifecycle — ticket 19.
//
// Two rules shape this screen:
//
// 1. An action a role does not hold is *absent*, not disabled-and-403ing. The
//    grants come from the backend (`useAdminSession().can`), which reads the
//    same matrix the routes enforce.
// 2. Nothing here hard-deletes. The lifecycle's retreat moves are
//    `supersede` (published -> archived, only when a published replacement
//    exists), `withdraw` (published -> archived with no replacement) and
//    `revoke_supersession` (archived -> published, un-doing a wrong
//    replacement without touching its bytes).
import { Fragment, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useAdminSession } from "@/hooks/useAdminSession";
import {
  AdminApiError,
  listRecords,
  newIdempotencyKey,
  structuredValues,
  transitionRecord,
  type CanonicalRecord,
  type LifecycleAction,
  type RecordFilters,
  type RecordStatus,
} from "@/lib/adminApi";
import {
  RECORD_STATUSES,
  RECORD_TYPE_LABELS,
  STATUS_LABELS,
  canPublish,
} from "@/lib/adminValidation";

interface TransitionSpec {
  action: LifecycleAction;
  label: string;
  /** The grant the backend requires for this action. */
  permission: string;
  /** Statuses this action is legal from (src/domain/canonical/lifecycle.py). */
  from: RecordStatus[];
  /** Whether a reason must be supplied before the action is sent. */
  reasonRequired: boolean;
  /** Whether the action is consequential enough to confirm first. */
  needsConfirm: boolean;
  hint?: string;
}

const TRANSITIONS: TransitionSpec[] = [
  {
    action: "submit",
    label: "Gửi duyệt",
    permission: "records.submit",
    from: ["draft"],
    reasonRequired: false,
    needsConfirm: false,
  },
  {
    action: "approve",
    label: "Duyệt",
    permission: "records.approve",
    from: ["review"],
    reasonRequired: false,
    needsConfirm: false,
    hint: "Không thể tự duyệt bản do chính bạn viết.",
  },
  {
    action: "reject",
    label: "Từ chối",
    permission: "records.reject",
    from: ["review"],
    reasonRequired: true,
    needsConfirm: true,
  },
  {
    action: "publish",
    label: "Công bố",
    permission: "records.publish",
    from: ["approved"],
    reasonRequired: false,
    needsConfirm: true,
  },
  {
    // Supersede replaces: it only works when a published replacement exists,
    // and the archive it causes is the replacement's effect, not the verb.
    action: "supersede",
    label: "Thay thế",
    permission: "records.supersede",
    from: ["published"],
    reasonRequired: false,
    needsConfirm: true,
    hint: "Chỉ thay thế được khi đã có một bản published khác ghi đè.",
  },
  {
    // Withdraw is the real archive: it retires a published record with no
    // replacement standing behind it.
    action: "withdraw",
    label: "Lưu trữ",
    permission: "records.withdraw",
    from: ["published"],
    reasonRequired: true,
    needsConfirm: true,
    hint: "Rút bản published mà không có bản thay thế. Bắt buộc ghi lý do.",
  },
  {
    action: "revoke_supersession",
    label: "Gỡ thay thế",
    permission: "records.revoke_supersession",
    from: ["archived"],
    reasonRequired: true,
    needsConfirm: true,
    hint: "Đưa bản bị thay thế sai trở lại published; bytes của bản thay thế không đổi.",
  },
];

function nganhLabel(record: CanonicalRecord): string {
  const values = structuredValues(record);
  const name = values.ten_nganh;
  const code = values.ma_nganh;
  if (typeof name === "string" && name) return typeof code === "string" && code ? `${name} (${code})` : name;
  if (typeof code === "string" && code) return code;
  return "—";
}

const RecordList = () => {
  const { can } = useAdminSession();
  const queryClient = useQueryClient();

  const [draft, setDraft] = useState({ nganh: "", year: "", status: "" });
  const [filters, setFilters] = useState<RecordFilters>({});
  const [pending, setPending] = useState<{ record: CanonicalRecord; spec: TransitionSpec } | null>(null);
  const [reason, setReason] = useState("");
  const [actionError, setActionError] = useState<string | null>(null);
  // One key per confirmed intention: minted when the user commits to an
  // action, kept across retries of *that same* action, cleared once it lands.
  // Retrying after a dropped response must carry the same key or the backend
  // runs the transition twice; a different action is a different intention and
  // must carry a different key, or the backend refuses it as a mismatched reuse.
  const [intent, setIntent] = useState<{
    key: string;
    recordId: string;
    action: LifecycleAction;
  } | null>(null);

  const list = useQuery({
    queryKey: ["admin", "records", filters],
    queryFn: () => listRecords(null, filters),
    retry: false,
  });

  const transition = useMutation({
    mutationFn: (input: {
      record: CanonicalRecord;
      action: LifecycleAction;
      reason?: string;
      idempotencyKey: string;
    }) =>
      transitionRecord(
        null,
        input.record.record_id,
        {
          action: input.action,
          expected_version: input.record.version,
          ...(input.reason ? { reason: input.reason } : {}),
        },
        { idempotencyKey: input.idempotencyKey },
      ),
    onSuccess: () => {
      setPending(null);
      setReason("");
      setActionError(null);
      setIntent(null);
      void queryClient.invalidateQueries({ queryKey: ["admin", "records"] });
    },
    onError: (failure: unknown) => {
      // The key survives the failure on purpose: the retry must reuse it.
      setActionError(
        failure instanceof AdminApiError ? failure.message : "Hành động thất bại.",
      );
    },
  });

  const applyFilters = () => {
    setFilters({
      ...(draft.nganh.trim() ? { nganh: draft.nganh.trim() } : {}),
      ...(draft.year.trim() ? { year: Number(draft.year) } : {}),
      ...(draft.status ? { status: draft.status as RecordStatus } : {}),
    });
  };

  const clearFilters = () => {
    setDraft({ nganh: "", year: "", status: "" });
    setFilters({});
  };

  const rows = list.data?.records ?? [];

  /**
   * Commit to an action, reusing the key of an in-flight intention.
   *
   * The key is minted on the first commit and survives a failure, so a retry
   * after a dropped response replays instead of re-executing. A different
   * record or action is a new intention and gets a fresh key.
   */
  const commit = (record: CanonicalRecord, action: LifecycleAction, reasonText?: string) => {
    const reused =
      intent && intent.recordId === record.record_id && intent.action === action
        ? intent.key
        : null;
    const key = reused ?? newIdempotencyKey();
    setIntent({ key, recordId: record.record_id, action });
    transition.mutate({
      record,
      action,
      ...(reasonText ? { reason: reasonText } : {}),
      idempotencyKey: key,
    });
  };

  return (
    <div className="space-y-4">
      <form
        className="flex flex-wrap items-end gap-3"
        onSubmit={(event) => {
          event.preventDefault();
          applyFilters();
        }}
      >
        <div className="space-y-1">
          <Label htmlFor="filter-nganh" className="text-xs text-foreground dark:text-zinc-300">
            Ngành
          </Label>
          <Input
            id="filter-nganh"
            className="h-8 w-56 !border-zinc-500 text-xs"
            placeholder="Tên ngành hoặc mã ngành"
            value={draft.nganh}
            onChange={(event) => setDraft({ ...draft, nganh: event.target.value })}
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="filter-year" className="text-xs text-foreground dark:text-zinc-300">
            Năm
          </Label>
          <Input
            id="filter-year"
            type="number"
            className="h-8 w-24 !border-zinc-500 text-xs"
            value={draft.year}
            onChange={(event) => setDraft({ ...draft, year: event.target.value })}
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="filter-status" className="text-xs text-foreground dark:text-zinc-300">
            Trạng thái
          </Label>
          <select
            id="filter-status"
            className="h-8 rounded-md border !border-zinc-500 bg-background px-2 text-xs text-foreground dark:text-zinc-100"
            value={draft.status}
            onChange={(event) => setDraft({ ...draft, status: event.target.value })}
          >
            <option value="">Tất cả</option>
            {RECORD_STATUSES.map((status) => (
              <option key={status} value={status}>
                {STATUS_LABELS[status]}
              </option>
            ))}
          </select>
        </div>
        <Button type="submit" size="sm" variant="outline" className="!border-zinc-500">
          Lọc
        </Button>
        <Button type="button" size="sm" variant="ghost" onClick={clearFilters}>
          Xóa lọc
        </Button>
        {can("records.create") ? (
          <Button asChild size="sm" className="bg-blue-600 text-white">
            <Link to="/admin/records/new">Tạo record mới</Link>
          </Button>
        ) : null}
      </form>

      {list.isPending ? (
        <p role="status" className="text-xs text-muted-foreground dark:text-zinc-400">
          Đang tải danh sách…
        </p>
      ) : null}

      {list.isError ? (
        <p role="alert" className="text-xs text-red-600 dark:text-red-400">
          {list.error instanceof AdminApiError
            ? `Không tải được danh sách: ${list.error.message}`
            : "Không tải được danh sách record."}
        </p>
      ) : null}

      {list.isSuccess && rows.length === 0 ? (
        <p role="status" className="text-xs text-muted-foreground dark:text-zinc-400">
          Không có record nào khớp bộ lọc.
        </p>
      ) : null}

      {rows.length > 0 ? (
        <div className="rounded-sm border border-border dark:border-zinc-800">
          <Table>
            <TableHeader>
              <TableRow className="border-border dark:border-zinc-800 hover:bg-muted/50 dark:hover:bg-zinc-900/50">
                <TableHead className="h-8 px-3 text-xs text-muted-foreground dark:text-zinc-400">Mã record</TableHead>
                <TableHead className="h-8 px-3 text-xs text-muted-foreground dark:text-zinc-400">Loại</TableHead>
                <TableHead className="h-8 px-3 text-xs text-muted-foreground dark:text-zinc-400">Năm</TableHead>
                <TableHead className="h-8 px-3 text-xs text-muted-foreground dark:text-zinc-400">Ngành</TableHead>
                <TableHead className="h-8 px-3 text-xs text-muted-foreground dark:text-zinc-400">Trạng thái</TableHead>
                <TableHead className="h-8 px-3 text-xs text-muted-foreground dark:text-zinc-400">Version</TableHead>
                <TableHead className="h-8 px-3 text-xs text-muted-foreground dark:text-zinc-400">Hành động</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((record) => {
                const actions = TRANSITIONS.filter(
                  (spec) => spec.from.includes(record.status) && can(spec.permission),
                );
                return (
                  <Fragment key={record.record_id}>
                    <TableRow className="border-border dark:border-zinc-800">
                      <TableCell className="px-3 py-1.5 font-mono text-xs text-foreground dark:text-zinc-100">
                        {record.record_id}
                      </TableCell>
                      <TableCell className="px-3 py-1.5 text-xs text-foreground dark:text-zinc-300">
                        {RECORD_TYPE_LABELS[record.record_type] ?? record.record_type}
                      </TableCell>
                      <TableCell className="px-3 py-1.5 text-xs text-foreground dark:text-zinc-300">
                        {record.admission_year}
                      </TableCell>
                      <TableCell className="px-3 py-1.5 text-xs text-foreground dark:text-zinc-300">
                        {nganhLabel(record)}
                      </TableCell>
                      <TableCell className="px-3 py-1.5 text-xs">
                        <span data-status={record.status} className="text-foreground dark:text-zinc-300">
                          {STATUS_LABELS[record.status] ?? record.status}
                        </span>
                      </TableCell>
                      <TableCell className="px-3 py-1.5 text-xs text-muted-foreground dark:text-zinc-400">
                        v{record.version}
                      </TableCell>
                      <TableCell className="px-3 py-1.5">
                        <div className="flex flex-wrap gap-1">
                          {can("records.edit") ? (
                            <Button asChild size="sm" variant="outline" className="h-7 !border-zinc-500 text-xs">
                              <Link to={`/admin/records/${encodeURIComponent(record.record_id)}/edit`}>
                                Sửa
                              </Link>
                            </Button>
                          ) : null}
                          {/* Ticket 25: the diff + preview screen is the
                              pre-submission stop for anything not yet
                              published or archived. */}
                          {record.status === "draft" ||
                          record.status === "review" ||
                          record.status === "approved" ? (
                            <Button asChild size="sm" variant="outline" className="h-7 !border-zinc-500 text-xs">
                              <Link
                                to={`/admin/records/${encodeURIComponent(record.record_id)}/review`}
                              >
                                Xem trước
                              </Link>
                            </Button>
                          ) : null}
                          {actions.map((spec) => {
                            const blockedByProvenance =
                              spec.action === "publish" && !canPublish(record);
                            return (
                              <Button
                                key={spec.action}
                                size="sm"
                                variant={
                                  spec.action === "supersede" ||
                                  spec.action === "withdraw" ||
                                  spec.action === "revoke_supersession"
                                    ? "ghost"
                                    : "outline"
                                }
                                className="h-7 !border-zinc-500 text-xs"
                                disabled={blockedByProvenance}
                                title={
                                  blockedByProvenance
                                    ? "Thiếu source_document_id: bản công bố bắt buộc phải có nguồn."
                                    : spec.hint
                                }
                                onClick={() => {
                                  setActionError(null);
                                  setReason("");
                                  if (spec.needsConfirm) setPending({ record, spec });
                                  else commit(record, spec.action);
                                }}
                              >
                                {spec.label}
                              </Button>
                            );
                          })}
                        </div>
                      </TableCell>
                    </TableRow>
                    {pending?.record.record_id === record.record_id ? (
                      <TableRow key={`${record.record_id}-confirm`} className="border-border dark:border-zinc-800">
                        <TableCell colSpan={7} className="bg-muted/40 dark:bg-zinc-900/40 px-3 py-3">
                          <div className="space-y-2">
                            <p className="text-xs text-foreground dark:text-zinc-300">
                              {pending.spec.action === "supersede"
                                ? `Thay thế ${record.record_id}@v${record.version}: bản published chuyển sang archived vì đã có bản khác ghi đè. Bản đã công bố không bị sửa hay xóa, chỉ đổi trạng thái.`
                                : pending.spec.action === "withdraw"
                                  ? `Lưu trữ ${record.record_id}@v${record.version}: rút bản published mà không có bản thay thế. Bản đã công bố không bị sửa hay xóa, chỉ đổi trạng thái.`
                                  : pending.spec.action === "revoke_supersession"
                                    ? `Gỡ thay thế ${record.record_id}@v${record.version}: đưa bản archived trở lại published. Bytes của bản đã thay thế nó không đổi.`
                                    : `${pending.spec.label} ${record.record_id}@v${record.version}?`}
                            </p>
                            <div className="space-y-1">
                              <Label
                                htmlFor="transition-reason"
                                className="text-xs text-foreground dark:text-zinc-300"
                              >
                                {pending.spec.reasonRequired
                                  ? "Lý do (bắt buộc)"
                                  : "Lý do (không bắt buộc)"}
                              </Label>
                              <Textarea
                                id="transition-reason"
                                className="min-h-[60px] !border-zinc-500 text-xs"
                                value={reason}
                                onChange={(event) => setReason(event.target.value)}
                              />
                            </div>
                            {actionError ? (
                              <p role="alert" className="text-xs text-red-600 dark:text-red-400">
                                {actionError}
                              </p>
                            ) : null}
                            <div className="flex gap-2">
                              <Button
                                size="sm"
                                className="bg-blue-600 text-white"
                                disabled={
                                  transition.isPending ||
                                  (pending.spec.reasonRequired && !reason.trim())
                                }
                                onClick={() =>
                                  commit(
                                    record,
                                    pending.spec.action,
                                    reason.trim() ? reason.trim() : undefined,
                                  )
                                }
                              >
                                Xác nhận
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => {
                                  setPending(null);
                                  setReason("");
                                  setActionError(null);
                                }}
                              >
                                Hủy
                              </Button>
                            </div>
                          </div>
                        </TableCell>
                      </TableRow>
                    ) : null}
                  </Fragment>
                );
              })}
            </TableBody>
          </Table>
        </div>
      ) : null}

      {actionError && pending === null ? (
      <p role="alert" className="text-xs text-red-600 dark:text-red-400">
          {actionError}
        </p>
      ) : null}
    </div>
  );
};

export default RecordList;
