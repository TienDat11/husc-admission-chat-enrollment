import { ADMIN_OVERVIEW_POLL_MS } from "@/lib/constants";
// Ticket T09 — the 7 teacher-facing screens share one shell: the left nav
// from 05 §3, the "Cần giúp?" line in the nav footer from 05 §2, and the HUSC
// token set (#143980 accent, 18 px body, Montserrat) that the mockups check 30/30.
//
// The shell is intentionally separate from AdminLayout: that one is the
// dark zinc technical surface (Config/Release/Debt), this one is the
// light office surface the UX spec writes for 50–65 year old staff.
import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  Bot,
  ClipboardCheck,
  History,
  HelpCircle,
  Home,
  Library,
  MessageSquare,
  Plus,
  Settings,
  Users,
  Wrench,
  type LucideIcon,
} from "lucide-react";
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
  Icon: LucideIcon;
  end?: boolean;
  /** Count shown in the item's badge, read from GET /admin/overview. */
  badge?: "review";
  /** Permission the item needs; the item is absent without it. */
  action?: string;
};

const navItems: NavItem[] = [
  { to: "/quan-ly", label: "Tổng quan", Icon: Home, end: true },
  { to: "/quan-ly/them", label: "Thêm thông tin", Icon: Plus },
  { to: "/quan-ly/cho-duyet", label: "Chờ duyệt", Icon: ClipboardCheck, badge: "review" },
  { to: "/quan-ly/kho", label: "Kho thông tin", Icon: Library },
  { to: "/quan-ly/cau-hoi", label: "Câu hỏi chưa trả lời", Icon: HelpCircle },
  { to: "/quan-ly/phan-hoi", label: "Phản hồi & Golden", Icon: MessageSquare },
  { to: "/quan-ly/tai-khoan", label: "Tài khoản & RBAC", Icon: Users, action: "roles.manage" },
  { to: "/quan-ly/lich-su", label: "Lịch sử", Icon: History },
  // Live screens the design frames do not draw; kept reachable below the frame set.
  { to: "/quan-ly/hoi-thu", label: "Hỏi thử bot", Icon: Bot },
  { to: "/quan-ly/van-hanh", label: "Vận hành", Icon: Wrench, action: "ops.warmup" },
];

const TeacherLayout = () => {
  const { status, identity, can } = useAdminSession();
  const [navOpen, setNavOpen] = useState(false);
  const overview = useQuery({
    queryKey: ["admin", "overview"],
    queryFn: () => fetchAdminOverview(null),
    enabled: status === "ready" && can("records.read"),
    retry: false,
    refetchInterval: ADMIN_OVERVIEW_POLL_MS,
    refetchIntervalInBackground: false,
  });
  const reviewCount = overview.data?.review_count ?? 0;
  const visibleItems = navItems.filter((item) => !item.action || can(item.action));

  return (
    <div className="husc-admin flex min-h-screen flex-col overflow-x-hidden bg-[#f3f5f9] text-[#0f1b2d]">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-2 focus:z-50 focus:rounded focus:bg-white focus:px-3 focus:py-2 focus:text-sm"
      >
        Bỏ qua, đến nội dung chính
      </a>
      <div className="flex flex-1 flex-col lg:flex-row">
        <aside className="side w-full shrink-0 lg:min-h-screen lg:w-[248px]">
          <div className="flex items-center justify-between gap-3">
            <p className="brand px-2 pb-4 text-lg font-bold text-white">HUSC Tuyển sinh</p>
            <button
              type="button"
              aria-expanded={navOpen}
              aria-controls="admin-nav"
              onClick={() => setNavOpen((open) => !open)}
              className="mb-4 min-h-11 rounded-lg border border-white/30 px-3 text-sm font-semibold text-white lg:hidden"
            >
              {navOpen ? "Đóng menu" : "Mở menu"}
            </button>
          </div>
          <div id="admin-nav" className={navOpen ? "flex flex-col" : "hidden lg:flex lg:flex-1 lg:flex-col"}>
            <nav aria-label="Điều hướng chính">
              <ul className="m-0 list-none p-0">
                {visibleItems.map((item) => {
                  const count = item.badge === "review" ? reviewCount : 0;
                  return (
                    <li key={item.to}>
                      <NavLink to={item.to} end={item.end}>
                        <span aria-hidden="true" className="ico">
                          <item.Icon />
                        </span>
                        <span className="min-w-0">{item.label}</span>
                        {count > 0 ? <SidebarBadge count={count} /> : null}
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
                      <Settings />
                    </span>
                    Quản trị kỹ thuật
                  </NavLink>
                </li>
              </ul>
            </nav>
            {status === "ready" && identity ? (
              <div className="who mt-auto">
                <span aria-hidden="true" className="avatar">
                  {identity.actor_id.charAt(0).toUpperCase()}
                </span>
                <span>
                  <span className="block font-semibold">{identity.actor_id}</span>
                  <span className="block opacity-80">{ROLE_NAMES[identity.role] ?? identity.role}</span>
                </span>
              </div>
            ) : null}
            <footer role="contentinfo" className="help mt-4">
              Cần giúp? Gọi anh Minh (kỹ thuật viên), máy lẻ 115
            </footer>
          </div>
        </aside>
        <main id="main-content" className="min-w-0 flex-1 p-8">
          <div className="max-w-5xl">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};

export default TeacherLayout;
