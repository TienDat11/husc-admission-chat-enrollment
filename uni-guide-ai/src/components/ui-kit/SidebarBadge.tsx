/**
 * A count next to a sidebar item, rendered as `Nhãn (n)`. A zero count renders
 * nothing, so an empty queue does not look like a pending one.
 */
export const SidebarBadge = ({ label, count }: { label: string; count: number }) => {
  if (count <= 0) return null;
  return (
    <span
      data-testid="sidebar-badge"
      className="ml-auto whitespace-nowrap rounded-full bg-[#3730A3] px-2 py-0.5 text-sm font-semibold text-white"
    >
      {label} ({count})
    </span>
  );
};
