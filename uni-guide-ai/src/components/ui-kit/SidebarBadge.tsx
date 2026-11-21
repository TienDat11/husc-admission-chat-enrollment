/**
 * The count pill on a sidebar item, right-aligned after the item's label. A zero
 * count renders nothing, so an empty queue does not look like a pending one.
 */
export const SidebarBadge = ({ count }: { count: number }) => {
  if (count <= 0) return null;
  return (
    <span
      data-testid="sidebar-badge"
      className="ml-auto shrink-0 rounded-full bg-white px-2 py-0.5 text-sm font-semibold text-[#103157]"
    >
      <span>{count}</span>
      <span className="sr-only"> mục cần xử lý</span>
    </span>
  );
};
