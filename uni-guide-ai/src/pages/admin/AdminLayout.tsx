import { NavLink, Outlet } from "react-router-dom";
import { useTheme } from "@/hooks/useTheme";

const navItems = [
  { to: "/admin/config", label: "Config", end: false },
  { to: "/admin/ingest", label: "Ingest", end: false },
  { to: "/admin/retrieval", label: "Retrieval", end: false },
  { to: "/admin/debt", label: "Debt", end: false },
];

const AdminLayout = () => {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  return (
    <div className="flex h-[calc(100vh-4rem)]">
      <aside
        className={`w-48 shrink-0 border-r ${
          isDark ? "border-zinc-800 bg-zinc-950" : "border-zinc-200 bg-white"
        }`}
      >
        <nav className="flex flex-col gap-0.5 p-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `block rounded-sm px-3 py-1.5 text-xs transition-colors ${
                  isActive
                    ? "bg-amber-500/10 text-amber-500 font-medium"
                    : isDark
                    ? "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100"
                    : "text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main
        className={`flex-1 overflow-auto ${
          isDark ? "bg-zinc-950 text-zinc-100" : "bg-white text-zinc-900"
        }`}
      >
        <div className="p-4">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default AdminLayout;
