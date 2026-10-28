import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { listUnanswered, type UnansweredItem } from "@/lib/teacherApi";

const REASON_LABELS: Record<string, string> = {
  OUT_OF_DOMAIN: "Ngoài phạm vi",
  LOW_CONFIDENCE: "Thí sinh đánh giá chưa tốt",
  NO_INFO: "Không có thông tin",
  GENERATION_FAILED: "Không sinh được câu trả lời",
};

/**
 * Ticket T09 — "Câu hỏi bot chưa trả lời được" (mockup 05, 05 §4.5):
 * the open questions the bot could not answer, with the reason in plain
 * words, and one action per row — "Bổ sung thông tin" (pre-filled into the
 * QA form) or "Không cần trả lời" (short required reason).
 */
const Unanswered = () => {
  const [noNeedOpen, setNoNeedOpen] = useState<UnansweredItem | null>(null);
  const [noNeedReason, setNoNeedReason] = useState("");
  const [noNeedConfirmed, setNoNeedConfirmed] = useState(false);

  const list = useQuery({
    queryKey: ["admin", "unanswered"],
    queryFn: () => listUnanswered(null),
    retry: false,
  });

  const rows = list.data?.items.filter((item) => !item.resolved) ?? [];

  const reasonFor = (item: UnansweredItem) =>
    REASON_LABELS[item.status_code] ?? "Không có thông tin";

  if (list.isPending) {
    return (
      <p role="status" aria-live="polite" className="text-zinc-600">
        <span aria-hidden="true">⏳</span> Đang lấy danh sách câu hỏi…
      </p>
    );
  }

  if (list.isError) {
    return (
      <div role="alert" className="note bad">
        <span aria-hidden="true">⚠</span> Chưa lấy được danh sách câu hỏi.
        Kiểm tra kết nối mạng rồi thử lại.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">
        Câu hỏi bot chưa trả lời được ({rows.length})
      </h1>
      <p className="text-zinc-600">
        Thông tin cá nhân trong câu hỏi đã được che. Bổ sung thông tin để bot
        trả lời được lần sau.
      </p>

      {rows.length === 0 ? (
        <p role="status" className="note ok">
          <span aria-hidden="true">✓</span> Bot đã trả lời được mọi câu hỏi
          trong 7 ngày qua.
        </p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Câu hỏi</th>
                <th scope="col">Lý do</th>
                <th scope="col">Lần cuối</th>
                <th scope="col">Việc cần làm</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((item) => (
                <tr key={item.trace_id}>
                  <td>{item.question}</td>
                  <td>{reasonFor(item)}</td>
                  <td>
                    {new Date(item.asked_at).toLocaleString("vi-VN", {
                      day: "2-digit",
                      month: "2-digit",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </td>
                  <td>
                    <div className="flex gap-2">
                      <Link
                        to={`/quan-ly/them/qa?question=${encodeURIComponent(item.question)}&trace_id=${encodeURIComponent(item.trace_id)}`}
                        className="btn btn-primary"
                      >
                        Bổ sung thông tin
                      </Link>
                      <button
                        type="button"
                        className="btn"
                        onClick={() => {
                          setNoNeedOpen(item);
                          setNoNeedReason("");
                          setNoNeedConfirmed(false);
                        }}
                      >
                        Không cần trả lời
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {noNeedOpen ? (
        <div className="space-y-3" role="dialog" aria-label="Không cần trả lời">
          <p className="note info">
            <strong>{noNeedOpen.question}</strong>
          </p>
          <div className="field">
            <label htmlFor="co-can-trali-ly-do">Lý do ngắn (bắt buộc)</label>
            <input
              id="co-can-trali-ly-do"
              type="text"
              value={noNeedReason}
              onChange={(event) => setNoNeedReason(event.target.value)}
            />
          </div>
          <div className="actions">
            <button
              type="button"
              className="btn btn-primary"
              disabled={!noNeedReason.trim()}
              onClick={() => setNoNeedConfirmed(true)}
            >
              Xác nhận
            </button>
            <button type="button" className="btn" onClick={() => setNoNeedOpen(null)}>
              Hủy
            </button>
          </div>
          {noNeedConfirmed ? (
            <p role="status" className="note ok">
              <span aria-hidden="true">✓</span> Đã đánh dấu câu hỏi này là
              không cần trả lời.
            </p>
          ) : null}
        </div>
      ) : null}

      <p className="text-sm text-zinc-600">
        Về <Link to="/quan-ly">Tổng quan</Link>
      </p>
    </div>
  );
};

export default Unanswered;
