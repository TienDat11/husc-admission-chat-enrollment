// @spec(PL-7.2 Frame 12 XaDgH — thanh trên cột chat, cao 56px, kẻ 1px dưới)
// ChatTopBar — thanh tiêu đề cột chat (dumb/presentational).
export function ChatTopBar() {
  return (
    <header className="h-[56px] border-b border-[#E3E8F0] bg-white flex items-center justify-between px-4 shrink-0">
      <div className="flex flex-col gap-1">
        <h3 className="text-[15px] font-semibold leading-[20px] text-[#0F1B2D]">
          Thông tin tuyển sinh năm 2026
        </h3>
        <p className="text-[12px] leading-[16px] text-[#6B7686]">
          Trả lời từ Đề án và Thông báo chính thức 2026
        </p>
      </div>
      <div className="shrink-0 whitespace-nowrap rounded-full bg-[#E7F4E8] px-3 py-1">
        <span className="text-[12.5px] leading-[16px] text-[#1B5E20] font-medium">
          Dữ liệu 2026
        </span>
      </div>
    </header>
  );
}
