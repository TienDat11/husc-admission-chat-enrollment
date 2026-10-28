import { useQuery } from "@tanstack/react-query";
import { fetchReleaseStatus, listRecords, type CanonicalRecord } from "@/lib/adminApi";
import { Link } from "react-router-dom";

/**
 * Ticket T09 — "Tổng quan": the three work cards from 05 §4.1 plus the
 * release line that tells the teacher which information the bot is serving.
 */
const Overview = () => {

  const review = useQuery({
    queryKey: ["admin", "records", { status: "review" }],
    queryFn: () => listRecords(null, { status: "review" }),
    retry: false,
  });

  const published = useQuery({
    queryKey: ["admin", "records", { status: "published" }],
    queryFn: () => listRecords(null, { status: "published" }),
    retry: false,
  });

  const release = useQuery({
    queryKey: ["admin", "releases"],
    queryFn: () => fetchReleaseStatus(null),
    retry: false,
  });

  const isExpiring = (record: CanonicalRecord) => {
    if (!record.effective_to) return false;
    const end = new Date(record.effective_to);
    const limit = new Date();
    limit.setDate(limit.getDate() + 30);
    return end.getTime() <= limit.getTime();
  };

  const pendingCount = review.data?.records.length ?? 0;
  const expiringCount =
    published.data?.records.filter(isExpiring).length ?? 0;
  const empty = pendingCount === 0 && expiringCount === 0;

  if (review.isPending || published.isPending || release.isPending) {
    return (
      <p role="status" aria-live="polite" className="note">
        <span aria-hidden="true">⏳</span> Đang lấy danh sách việc cần làm…
      </p>
    );
  }

  if (review.isError || published.isError || release.isError) {
    return (
      <div role="alert" className="note bad">
        <span aria-hidden="true">⚠</span> Chưa lấy được danh sách việc cần
        làm. Kiểm tra kết nối mạng rồi thử lại.
      </div>
    );
  }

  const botLine =
    release.data?.active?.created_at ?? release.data?.active_release_id;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-zinc-900">Việc cần làm hôm nay</h1>

      {botLine ? (
        <p role="status" className="note ok">
          <span aria-hidden="true">✓</span> Bot đang trả lời theo thông tin
          cập nhật lúc {new Date(botLine).toLocaleString("vi-VN")}.
        </p>
      ) : null}

      {empty ? (
        <p role="status" className="note ok">
          <span aria-hidden="true">✓</span> Không có việc cần làm. Bot đang
          hoạt động bình thường.
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="card">
            <div className="num">{pendingCount}</div>
            <p>thông tin chờ thầy duyệt</p>
            <Link to="/quan-ly/cho-duyet" className="btn btn-primary">
              Duyệt ngay
            </Link>
          </div>
          <div className="card">
            <div className="num">{release.data?.problems.length ?? 0}</div>
            <p>vấn đề cần kỹ thuật xem</p>
            <Link to="/quan-ly/lich-su" className="btn">
              Xem
            </Link>
          </div>
          <div className="card">
            <div className="num">{expiringCount}</div>
            <p>thông tin hết hạn trong 30 ngày tới</p>
            <Link to="/quan-ly/kho" className="btn">
              Xem
            </Link>
          </div>
        </div>
      )}
    </div>
  );
};

export default Overview;
