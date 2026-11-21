// Lịch sử phát hành & Nhật ký kiểm toán — the release history (what the Bot serves,
// the release behind it and the one-step rollback) beside the audit ledger.
// The ledger's hash chain is checked here: rows arrive newest first, and each
// row's prev_hash must equal the entry_hash of the row after it.
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchReleaseStatus, newIdempotencyKey, rollbackRelease, type ReleaseView } from "@/lib/adminApi";
import { fetchAuditEvents, type AuditEventRecord } from "@/lib/teacherApi";

const ACTION_TITLES: Record<string, string> = {
  publish: "Phát hành lên Bot",
  rollback: "Khôi phục bản trước",
  submit: "Gửi duyệt",
  approve: "Duyệt mục",
  reject: "Trả lại mục",
  create: "Thêm tài liệu",
  edit: "Sửa thông tin",
  supersede: "Thay thế bản cũ",
  withdraw: "Thu hồi",
  revoke_supersession: "Hoàn tác thay thế",
};

/** One ledger row in plain words. Empty parts are left out, never printed as "". */
export function auditLine(event: Record<string, unknown>): { title: string; detail: string | null } {
  const action = String(event.action ?? "").trim();
  const title = ACTION_TITLES[action] ?? (action ? `Hoạt động: ${action}` : "Ghi nhận hoạt động");
  const target = String(event.entity_id ?? "").trim();
  const reason = String(event.reason ?? "").trim();
  const parts = [target ? `Đối tượng: ${target}` : "", reason ? `Lý do: ${reason}` : ""].filter(Boolean);
  return { title, detail: parts.length > 0 ? parts.join(" · ") : null };
}

/** `HH:mm dd/mm/yyyy` in local time, or "" when the value is not a date. */
export function stamp(value: unknown): string {
  const date = new Date(String(value ?? ""));
  if (Number.isNaN(date.getTime())) return "";
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(date.getHours())}:${pad(date.getMinutes())} ${pad(date.getDate())}/${pad(date.getMonth() + 1)}/${date.getFullYear()}`;
}

/** Rows are newest first: each prev_hash must match the entry_hash of the older row after it. */
export function chainIntact(rows: AuditEventRecord[]): boolean {
  return rows.every((row, index) => index === rows.length - 1 || row.prev_hash === rows[index + 1].entry_hash);
}

function ReleaseCard({ release, tone, title }: { release: ReleaseView; tone: "active" | "previous"; title: string }) {
  const active = tone === "active";
  return (
    <div
      className={`space-y-1 rounded-xl border p-4 ${
        active ? "border-[#1E6B34] bg-[#E7F4E8] text-[#1E6B34]" : "border-[#d6deea] bg-[#f7f9fc] text-[#0f1b2d]"
      }`}
    >
      <p className="font-semibold">
        Phiên bản {release.release_id} · {title}
      </p>
      {stamp(release.created_at) ? <p className="text-sm">Phát hành lúc {stamp(release.created_at)}</p> : null}
      {active ? (
        <p className="text-sm">
          {release.document_count} tài liệu đang phục vụ · {release.chunk_count} đoạn nội dung
        </p>
      ) : null}
    </div>
  );
}

const History = () => {
  const queryClient = useQueryClient();
  const [rollbackOpen, setRollbackOpen] = useState(false);
  const [rollbackReason, setRollbackReason] = useState("");
  const [rollbackDone, setRollbackDone] = useState(false);
  const [rollbackError, setRollbackError] = useState<string | null>(null);
  const [rollbackKey, setRollbackKey] = useState<{ key: string; reason: string } | null>(null);

  const events = useQuery({
    queryKey: ["admin", "audit-events"],
    queryFn: () => fetchAuditEvents(null, 50),
    retry: false,
  });

  const release = useQuery({
    queryKey: ["admin", "releases"],
    queryFn: () => fetchReleaseStatus(null),
    retry: false,
  });

  const rollback = useMutation({
    // The key is minted per click and passed in as a variable. Reading state here
    // would be a stale closure that reuses the previous intention's key.
    mutationFn: ({ reason, key }: { reason: string; key: string }) =>
      rollbackRelease(null, { reason }, { idempotencyKey: key }),
    onSuccess: () => {
      setRollbackKey(null);
      setRollbackDone(true);
      setRollbackOpen(false);
      setRollbackReason("");
      void queryClient.invalidateQueries({ queryKey: ["admin", "releases"] });
      void queryClient.invalidateQueries({ queryKey: ["admin", "audit-events"] });
    },
    onError: (failure: unknown) =>
      setRollbackError(failure instanceof Error ? failure.message : "Khôi phục thất bại."),
  });

  const rows = events.data?.events ?? [];
  const previous = release.data?.previous ?? null;
  const active = release.data?.active ?? null;

  if (events.isPending) {
    return (
      <p role="status" aria-live="polite" className="text-zinc-600">
        Đang tải lịch sử…
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold text-[#0f1b2d]">Lịch sử phát hành &amp; Nhật ký kiểm toán</h1>
        <p className="text-base text-[#4a5568]">
          Theo dõi mọi phiên bản dữ liệu được đưa lên Bot, khôi phục về phiên bản trước khi có sự cố, và tra cứu vết
          kiểm toán bất biến.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="space-y-4">
          <section aria-labelledby="releases-title" className="space-y-3 rounded-xl border border-[#d6deea] bg-white p-5">
            <h2 id="releases-title" className="text-lg font-semibold text-[#0f1b2d]">
              Lịch sử phiên bản phát hành lên Bot
            </h2>
            {active ? <ReleaseCard release={active} tone="active" title="Bot đang sử dụng" /> : null}
            {previous ? (
              <div className="space-y-3 rounded-xl border border-[#d6deea] bg-[#f7f9fc] p-4">
                <ReleaseCard release={previous} tone="previous" title="Bản lưu trước đó" />
                <button
                  type="button"
                  className="min-h-11 rounded-lg border border-[#143980] bg-white px-4 text-base font-semibold text-[#143980] hover:bg-[#eef3fb] disabled:opacity-50"
                  onClick={() => {
                    setRollbackDone(false);
                    setRollbackError(null);
                    setRollbackOpen(true);
                  }}
                  disabled={rollback.isPending}
                >
                  Khôi phục về phiên bản này (Rollback)
                </button>
              </div>
            ) : (
              <p className="text-base text-[#4a5568]">Chưa có phiên bản trước để khôi phục.</p>
            )}
            {!active && !previous && release.data ? (
              <p className="text-base text-[#4a5568]">Chưa có phiên bản nào được phát hành.</p>
            ) : null}
            {release.isError ? (
              <p role="alert" className="text-base text-[#9b1c1c]">
                Không tải được phiên bản phát hành.
              </p>
            ) : null}
          </section>

          {rollbackDone ? (
            <p role="status" className="rounded-lg border border-[#BFDCC4] bg-[#E7F4E8] p-4 text-base text-[#1E6B34]">
              Đã khôi phục phiên bản trước.
            </p>
          ) : null}
          {rollbackError ? (
            <p role="alert" className="rounded-lg border border-[#F3C4C4] bg-[#FDECEA] p-4 text-base text-[#9b1c1c]">
              {rollbackError}
            </p>
          ) : null}

          {rollbackOpen && previous ? (
            <div
              role="alertdialog"
              aria-labelledby="khoi-phuc-h"
              aria-describedby="khoi-phuc-d"
              className="space-y-4 rounded-xl border border-[#8a5a00] bg-white p-5"
            >
              <h3 id="khoi-phuc-h" className="text-lg font-semibold text-[#6b4300]">
                Khôi phục Bot về Phiên bản {previous.release_id}
              </h3>
              <p id="khoi-phuc-d" className="text-base text-[#0f1b2d]">
                Bot sẽ trả lời lại theo bản cũ. Bản hiện tại vẫn được giữ trong lịch sử và nhật ký.
              </p>
              <div className="space-y-1.5">
                <label htmlFor="khoi-phuc-ly-do" className="block text-sm font-semibold">
                  Lý do khôi phục (bắt buộc)
                </label>
                <input
                  id="khoi-phuc-ly-do"
                  type="text"
                  value={rollbackReason}
                  placeholder="Ví dụ: Phát hiện bảng phụ lục trang 9 của thông báo mới cần đính chính"
                  onChange={(event) => setRollbackReason(event.target.value)}
                  className="min-h-11 w-full rounded-lg border border-[#143980] bg-white px-3 text-base"
                />
              </div>
              <div className="flex flex-wrap gap-3">
                <button
                  type="button"
                  className="min-h-11 rounded-lg border border-[#d6deea] bg-white px-4 text-base font-semibold text-[#0f1b2d] hover:bg-[#f7f9fc]"
                  onClick={() => setRollbackOpen(false)}
                >
                  Không khôi phục
                </button>
                <button
                  type="button"
                  className="min-h-11 rounded-lg bg-[#6b4300] px-4 text-base font-semibold text-white disabled:opacity-50"
                  disabled={!rollbackReason.trim() || rollback.isPending}
                  onClick={() => {
                    const reason = rollbackReason.trim();
                    // Same reason as the last attempt is a retry of that intention and keeps its key.
                    const key = rollbackKey?.reason === reason ? rollbackKey.key : newIdempotencyKey();
                    setRollbackKey({ key, reason });
                    rollback.mutate({ reason, key });
                  }}
                >
                  {rollback.isPending ? "Đang khôi phục…" : "Xác nhận khôi phục phiên bản trước"}
                </button>
              </div>
            </div>
          ) : null}
        </div>

        <section aria-labelledby="audit-title" className="space-y-3 rounded-xl border border-[#d6deea] bg-white p-5">
          <h2 id="audit-title" className="text-lg font-semibold text-[#0f1b2d]">
            Nhật ký kiểm toán bất biến
          </h2>
          {rows.length > 0 ? (
            chainIntact(rows) ? (
              <p className="inline-block rounded-full bg-[#E7F4E8] px-3 py-1 text-sm font-semibold text-[#1E6B34]">
                Chuỗi chữ ký kiểm toán toàn vẹn (Đã xác thực)
              </p>
            ) : (
              <p className="inline-block rounded-full bg-[#FDECEA] px-3 py-1 text-sm font-semibold text-[#9b1c1c]">
                Chuỗi chữ ký có đoạn không khớp trong các dòng đang xem
              </p>
            )
          ) : null}
          {events.isError ? (
            <p role="alert" className="text-base text-[#9b1c1c]">
              Không tải được nhật ký kiểm toán.
            </p>
          ) : null}
          {!events.isError && rows.length === 0 ? (
            <p className="text-base text-[#4a5568]">Chưa có dòng nhật ký nào.</p>
          ) : null}
          <ol aria-label="Dòng thời gian" className="m-0 max-h-[70vh] list-none space-y-2 overflow-y-auto p-0 pr-1">
            {rows.map((row) => {
              const line = auditLine(row.event);
              const who = String(row.event.actor_id ?? "").trim();
              const when = stamp(row.event.timestamp ?? row.event.occurred_at);
              return (
                <li key={row.entry_hash} className="space-y-1 rounded-lg bg-[#f7f9fc] p-4">
                  <p className="font-mono text-xs text-[#4a5568]">
                    {[when, who].filter(Boolean).join(" · ")}
                  </p>
                  <p className="text-base font-semibold text-[#0f1b2d]">{line.title}</p>
                  {line.detail ? <p className="text-sm text-[#4a5568]">{line.detail}</p> : null}
                </li>
              );
            })}
          </ol>
        </section>
      </div>
    </div>
  );
};

export default History;
