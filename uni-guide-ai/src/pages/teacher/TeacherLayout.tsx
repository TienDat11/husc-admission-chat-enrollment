// Ticket T09 — the 7 teacher-facing screens share one shell: the left nav
// from 05 §3, the "Cần giúp?" bar from 05 §2, and the HUSC token set
// (#143980 accent, 18 px body, Montserrat) that the mockups check 30/30.
//
// The shell is intentionally separate from AdminLayout: that one is the
// dark zinc technical surface (Config/Release/Debt), this one is the
// light office surface the UX spec writes for 50–65 year old staff.
import { NavLink, Outlet } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useAdminSession } from "@/hooks/useAdminSession";
import { fetchAdminOverview } from "@/lib/adminOpsApi";
import { SidebarBadge } from "@/components/ui-kit/SidebarBadge";

const ROLE_NAMES: Record<string, string> = {
  editor: "Người soạn",
  reviewer_publisher: "Người duyệt và xuất bản",
  admin: "Quản trị",
  ops: "Kỹ thuật",
  viewer: "Chỉ xem",
};

type NavItem = {
  to: string;
  label: string;
  icon: string;
  end?: boolean;
  /** Count shown in the item's badge, read from GET /admin/overview. */
  badge?: "review";
  /** Permission the item needs; the item is absent without it. */
  action?: string;
};

const navItems: NavItem[] = [
  { to: "/quan-ly", label: "Tổng quan", icon: "🏠", end: true },
  { to: "/quan-ly/them", label: "Thêm thông tin", icon: "➕" },
  { to: "/quan-ly/cho-duyet", label: "Chờ duyệt", icon: "📝", badge: "review" },
  { to: "/quan-ly/kho", label: "Kho thông tin", icon: "📚" },
  { to: "/quan-ly/cau-hoi", label: "Câu hỏi bot chưa trả lời được", icon: "❓" },
  { to: "/quan-ly/hoi-thu", label: "Hỏi thử bot", icon: "💬" },
  { to: "/quan-ly/lich-su", label: "Lịch sử & khôi phục", icon: "🕘" },
  { to: "/quan-ly/van-hanh", label: "Vận hành", icon: "🛠", action: "ops.warmup" },
  { to: "/quan-ly/tai-khoan", label: "Tài khoản", icon: "👥", action: "roles.manage" },
];

const TeacherLayout = () => {
  const { status, identity, can } = useAdminSession();
  const overview = useQuery({
    queryKey: ["admin", "overview"],
    queryFn: () => fetchAdminOverview(null),
    enabled: status === "ready",
    retry: false,
    refetchInterval: 30_000,
  });
  const reviewCount = overview.data?.review_count ?? 0;
  const visibleItems = navItems.filter((item) => !item.action || can(item.action));

  return (
    <div className="husc-admin flex min-h-screen flex-col overflow-x-hidden bg-white text-zinc-900">
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
              {visibleItems.map((item) => {
                const count = item.badge === "review" ? reviewCount : 0;
                return (
                  <li key={item.to}>
                    <NavLink to={item.to} end={item.end}>
                      <span aria-hidden="true" className="ico">
                        {item.icon}
                      </span>
                      {count > 0 ? <SidebarBadge label={item.label} count={count} /> : item.label}
                    </NavLink>
                  </li>
                );
              })}
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
