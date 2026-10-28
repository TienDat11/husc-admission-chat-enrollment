// Ticket T09 — the 7 teacher-facing screens share one shell: the left nav
// from 05 §3, the "Cần giúp?" bar from 05 §2, and the HUSC token set
// (#143980 accent, 18 px body, Montserrat) that the mockups check 30/30.
//
// The shell is intentionally separate from AdminLayout: that one is the
// dark zinc technical surface (Config/Release/Debt), this one is the
// light office surface the UX spec writes for 50–65 year old staff.
import { NavLink, Outlet } from "react-router-dom";
import { useAdminSession } from "@/hooks/useAdminSession";

const ROLE_NAMES: Record<string, string> = {
  editor: "Người soạn",
  reviewer_publisher: "Người duyệt và xuất bản",
  admin: "Quản trị",
  ops: "Kỹ thuật",
  viewer: "Chỉ xem",
};

const navItems = [
  { to: "/quan-ly", label: "Tổng quan", icon: "🏠", end: true },
  { to: "/quan-ly/them", label: "Thêm thông tin", icon: "➕" },
  { to: "/quan-ly/cho-duyet", label: "Chờ duyệt", icon: "📝" },
  { to: "/quan-ly/kho", label: "Kho thông tin", icon: "📚" },
  { to: "/quan-ly/cau-hoi", label: "Câu hỏi bot chưa trả lời được", icon: "❓" },
  { to: "/quan-ly/hoi-thu", label: "Hỏi thử bot", icon: "💬" },
  { to: "/quan-ly/lich-su", label: "Lịch sử & khôi phục", icon: "🕘" },
];

const TeacherLayout = () => {
  const { status, identity } = useAdminSession();

  return (
    <div className="husc-admin flex min-h-screen flex-col bg-white text-zinc-900">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-2 focus:z-50 focus:rounded focus:bg-white focus:px-3 focus:py-2 focus:text-sm"
      >
        Bỏ qua, đến nội dung chính
      </a>
      <div className="flex flex-1 flex-col lg:flex-row">
        <aside className="side w-full shrink-0 border-b border-[#5f6670]/40 lg:w-[270px] lg:border-b-0">
          <p className="brand mb-4 px-2 text-lg font-bold text-[#143980]">Tuyển sinh HUSC</p>
          <nav aria-label="Điều hướng chính">
            <ul className="m-0 list-none space-y-1 p-0">
              {navItems.map((item) => (
                <li key={item.to}>
                  <NavLink to={item.to} end={item.end}>
                    <span aria-hidden="true" className="ico">
                      {item.icon}
                    </span>
                    {item.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
          <hr />
          <nav aria-label="Quản trị kỹ thuật">
            <ul className="m-0 list-none p-0">
              <li className="tech">
                <NavLink to="/admin">
                  <span aria-hidden="true" className="ico">
                    ⚙
                  </span>
                  Quản trị kỹ thuật
                </NavLink>
              </li>
            </ul>
          </nav>
        </aside>
        <main id="main-content" className="min-w-0 flex-1 p-7">
          <p className="mb-5 text-base text-zinc-600">
            {status === "ready" && identity ? (
              <>
                Đang đăng nhập: {identity.actor_id} —{" "}
                {ROLE_NAMES[identity.role] ?? identity.role}
              </>
            ) : (
              "Chưa đăng nhập"
            )}
          </p>
          <div className="max-w-4xl">
            <Outlet />
          </div>
        </main>
      </div>
      <footer
        role="contentinfo"
        className="bg-[#143980] px-7 py-3 text-sm text-white"
      >
        Cần giúp? Gọi anh Minh (kỹ thuật viên) — máy lẻ 115
      </footer>
    </div>
  );
};


export default TeacherLayout;
