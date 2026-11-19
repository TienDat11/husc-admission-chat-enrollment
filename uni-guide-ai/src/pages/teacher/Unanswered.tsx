import { Link } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useAdminSession } from "@/hooks/useAdminSession";
import {
  listUnanswered,
  markOutOfScope,
  promoteToGolden,
  type UnansweredItem,
} from "@/lib/teacherApi";

const REASON_LABELS: Record<string, string> = {
  OUT_OF_DOMAIN: "Ngoài phạm vi",
  LOW_CONFIDENCE: "Thí sinh đánh giá chưa tốt",
  NO_INFO: "Không có thông tin",
  GENERATION_FAILED: "Không sinh được câu trả lời",
};

const overdueStyle = {
  background: "#8b1a1a",
  color: "#fff",
  padding: "2px 8px",
  borderRadius: 4,
  fontSize: "0.85rem",
  fontWeight: 600,
  display: "inline-block",
} as React.CSSProperties;

const okStyle = {
  ...overdueStyle,
  background: "#1b5e20",
} as React.CSSProperties;

const chipStyle = {
  background: "#6b7280",
  color: "#fff",
  padding: "1px 7px",
  borderRadius: 4,
  fontSize: "0.8rem",
  display: "inline-block",
} as React.CSSProperties;

const scopeChipStyle = {
  ...chipStyle,
  background: "#9aa0a8",
} as React.CSSProperties;

/**
 * Frame 7 — "Câu hỏi bot chưa trả lời được": open unanswered questions with
 * SLA and occurrence markers, plus the "Ngoài phạm vi" and
 * "Trả lời & Đưa vào bộ câu hỏi chuẩn" actions.
 */
const Unanswered = () => {
  const { can } = useAdminSession();
  const canEdit = can("records.edit");
  const canCreate = can("records.create");

  const list = useQuery({
    queryKey: ["admin", "unanswered"],
    queryFn: () => listUnanswered(null),
    retry: false,
  });

  const rows = (list.data?.items ?? []).filter((item) => !item.resolved);

  const outOfScope = useMutation({
    mutationFn: (traceId: string) => markOutOfScope(null, traceId),
    onSuccess: () => list.refetch(),
  });

  const promote = useMutation({
    mutationFn: (traceId: string) =>
      promoteToGolden(null, traceId, {
        admission_year: 2026,
        answer: "",
        coordinate_key: `gq-${traceId}`,
        scope: "tuyen_sinh",
        subject: "cau_hoi_chua_tra_loi",
      }),
    onSuccess: () => list.refetch(),
  });

  const reasonLabel = (item: UnansweredItem) =>
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
                <th scope="col">SLA</th>
                <th scope="col">Lần hỏi</th>
                <th scope="col">Việc cần làm</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((item) => {
                const overdue = item.is_overdue ?? new Date(item.sla_due_at).getTime() < Date.now();
                return (
                  <tr key={item.trace_id}>
                    <td>{item.question}</td>
                    <td>{reasonLabel(item)}</td>
                    <td>
                      <span
                        style={overdue ? overdueStyle : okStyle}
                        aria-label={overdue ? "Quá hạn xử lý" : "Còn hạn xử lý"}
                      >
                        {overdue ? "Quá han" : "Còn hạn"}
                      </span>
                    </td>
                    <td>
                      <span
                        style={chipStyle}
                        aria-label={`${item.occurrence_count} lần hỏi`}
                      >
                        {item.occurrence_count} lần
                      </span>
                    </td>
                    <td>
                      <div className="flex flex-col gap-2 sm:flex-row sm:gap-2">
                        {item.out_of_scope ? (
                          <span
                            style={scopeChipStyle}
                            aria-label="Đã đánh dấu ngoài phạm vi"
                          >
                            Ngoài phạm vi
                          </span>
                        ) : (
                          <>
                            {canEdit ? (
                              <button
                                type="button"
                                className="btn btn-outline"
                                disabled={outOfScope.isPending}
                                onClick={() => void outOfScope.mutate(item.trace_id)}
                              >
                                Ngoài phạm vi
                              </button>
                            ) : (
                              <button type="button" className="btn" disabled>
                                Ngoài phạm vi
                              </button>
                            )}
                            {canCreate ? (
                              <Link
                                to={`/quan-ly/them/qa?question=${encodeURIComponent(item.question)}&trace_id=${encodeURIComponent(item.trace_id)}&promote=1`}
                                className="btn btn-primary"
                                disabled={promote.isPending}
                              >
                                Trả lời & Đưa vào bộ câu hỏi chuẩn
                              </Link>
                            ) : (
                              <button
                                type="button"
                                className="btn"
                                disabled
                              >
                                Trả lời & Đưa vào bộ câu hỏi chuẩn
                              </button>
                            )}
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <p className="text-sm text-zinc-600">
        Về <Link to="/quan-ly">Tổng quan</Link>
      </p>
    </div>
  );
};

export default Unanswered;
