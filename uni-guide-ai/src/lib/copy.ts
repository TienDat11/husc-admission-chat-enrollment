/** Shared Vietnamese UI copy. Keep labels short and use the same verb for the same action. */

/** Fallback shown when an error has no usable message. */
export const DEFAULT_ERROR_MESSAGE = "Thao tác thất bại. Vui lòng thử lại.";

/** One verb per action across admin and chat screens. */
export const ACTION_LABELS = {
  save: "Lưu",
  cancel: "Hủy",
  confirm: "Xác nhận",
  close: "Đóng",
  retry: "Thử lại",
  refresh: "Làm mới",
  approve: "Duyệt",
  reject: "Từ chối",
  publish: "Xuất bản",
  delete: "Xóa",
  search: "Tìm kiếm",
  back: "Quay lại",
} as const;

/** Status words used in pills and table cells. */
export const STATUS_LABELS = {
  loading: "Đang tải...",
  pending: "Chờ xử lý",
  done: "Hoàn tất",
  failed: "Thất bại",
  empty: "Chưa có dữ liệu",
} as const;

/** Text of the chat WarmingBanner (icon is added by the component). */
export const WARMING_BANNER = {
  connecting: (seconds: number) => `Đang kết nối tới máy chủ... (${seconds}s)`,
  failed: "Máy chủ không phản hồi. Sẽ thử lại...",
} as const;
