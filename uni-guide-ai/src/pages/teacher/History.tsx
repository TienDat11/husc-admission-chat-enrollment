import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchReleaseStatus,
  newIdempotencyKey,
  rollbackRelease,
} from "@/lib/adminApi";
import { fetchAuditEvents, type AuditEventRecord } from "@/lib/teacherApi";

const ACTION_NAMES: Record<string, string> = {
  publish: "xuất bản",
  rollback: "khôi phục bản trước",
  submit: "gửi duyệt",
  approve: "duyệt",
  reject: "trả lại",
  create: "thêm tài liệu",
  edit: "sửa thông tin",
  supersede: "thay thế bản cũ",
  withdraw: "thu hồi",
  revoke_supersession: "hoàn tác thay thế",
};

const eventText = (event: AuditEventRecord): string => {
  const payload = event.event;
  const action = ACTION_NAMES[String(payload.action ?? "")] ?? String(payload.action ?? "");
  const target = String(payload.entity_id ?? "");
  const reason = payload.reason ? ` (lý do: ${String(payload.reason)})` : "";
  const who = String(payload.actor_id ?? "");
  return `${who} ${action} "${target}"${reason}`;
};

/**
 * Ticket T09 — "Lịch sử & khôi phục" (mockup 08, 05 §4.7): the timeline in
 * plain words, and the "Khôi phục bản trước" button behind a confirmation
 * box that names which version it reverts to and asks for a reason.
 */
const History = () => {
  const queryClient = useQueryClient();
  const [rollbackTarget, setRollbackTarget] = useState(false);
  const [rollbackReason, setRollbackReason] = useState("");
  const [rollbackDone, setRollbackDone] = useState(false);
  const [rollbackError, setRollbackError] = useState<string | null>(null);
  const [rollbackKey, setRollbackKey] = useState<string | null>(null);

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
    mutationFn: async (reason: string) => {
      const key = rollbackKey ?? newIdempotencyKey();
      setRollbackKey(key);
      return rollbackRelease(null, { reason }, { idempotencyKey: key });
    },
    onSuccess: () => {
      setRollbackDone(true);
      setRollbackTarget(false);
      setRollbackReason("");
      void queryClient.invalidateQueries({ queryKey: ["admin", "releases"] });
      void queryClient.invalidateQueries({ queryKey: ["admin", "audit-events"] });
    },
    onError: (failure: unknown) =>
      setRollbackError(
        failure instanceof Error ? failure.message : "Khôi phục thất bại.",
      ),
  });

  const rows = events.data?.events ?? [];
  const previous = release.data?.previous;

  const startRollback = () => {
    setRollbackDone(false);
    setRollbackError(null);
    setRollbackTarget(true);
  };

  if (events.isPending) {
    return (
      <p role="status" aria-live="polite" className="text-zinc-600">
        <span aria-hidden="true">⏳</span> Đang tải lịch sử…
      </p>
    );
  }

  return (
    <div className="space-y-5">
      <h1 className="text-2xl font-bold">Lịch sử &amp; khôi phục</h1>

      <ol className="timeline" aria-label="Dòng thời gian">
        {rows.map((event) => (
          <li key={event.entry_hash}>
            <strong>
              {new Date(String(event.event.timestamp ?? event.event.occurred_at ?? "")).toLocaleString(
                "vi-VN",
                { day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" },
              )}
            </strong>{" "}
            — {eventText(event)}
          </li>
        ))}
      </ol>

      {rollbackDone ? (
        <p role="status" className="note ok">
          <span aria-hidden="true">✓</span> Đã khôi phục. Bot trả lời theo bản
          trước trong vòng 5–10 phút.
        </p>
      ) : null}

      <div className="actions">
        <button
          type="button"
          className="btn"
          onClick={startRollback}
          disabled={!previous || rollback.isPending}
        >
          Khôi phục bản trước
        </button>
      </div>

      {rollbackError ? (
        <p role="alert" className="note bad">
          <span aria-hidden="true">⚠</span> {rollbackError}
        </p>
      ) : null}

      {rollbackTarget ? (
        <div
          role="alertdialog"
          aria-labelledby="khoi-phuc-h"
          aria-describedby="khoi-phuc-d"
          className="dialog"
        >
          <h3 id="khoi-phuc-h">
            Khôi phục về bản trước ({previous?.release_id})?
          </h3>
          <p id="khoi-phuc-d">
            Bot sẽ trả lời lại theo bản cũ. Bản hiện tại vẫn được giữ trong
            lịch sử.
          </p>
          <div className="field">
            <label htmlFor="khoi-phuc-ly-do">Lý do khôi phục (bắt buộc)</label>
            <input
              id="khoi-phuc-ly-do"
              type="text"
              value={rollbackReason}
              onChange={(event) => setRollbackReason(event.target.value)}
            />
          </div>
          <div className="actions">
            <button
              type="button"
              className="btn"
              onClick={() => setRollbackTarget(false)}
            >
              Không khôi phục
            </button>
            <button
              type="button"
              className="btn btn-primary"
              disabled={!rollbackReason.trim() || rollback.isPending}
              onClick={() => {
                setRollbackKey(null);
                rollback.mutate(rollbackReason.trim());
              }}
            >
              {rollback.isPending ? "Đang khôi phục…" : "Khôi phục"}
            </button>
          </div>
        </div>
      ) : null}

      <p className="text-sm text-zinc-600">
        Về <Link to="/quan-ly">Tổng quan</Link>
      </p>
    </div>
  );
};

export default History;
