import { Link, useLocation } from "react-router-dom";
import { navGroups } from "../../navigation";

export default function SideNavigation({
  collapsed,
  mobileOpen,
  onToggleCollapse,
  onCloseMobile,
}: {
  collapsed: boolean;
  mobileOpen: boolean;
  onToggleCollapse: () => void;
  onCloseMobile: () => void;
}) {
  const location = useLocation();

  return (
    <nav
      className={`side-nav${collapsed ? " collapsed" : ""}${mobileOpen ? " open" : ""}`}
      aria-label="Primary"
    >
      <button type="button" className="nav-toggle" onClick={onToggleCollapse} aria-expanded={!collapsed}>
        {collapsed ? "»" : "«"}
      </button>
      {navGroups.map((group) => (
        <div key={group.id}>
          {!collapsed ? <div className="nav-group">{group.label}</div> : null}
          {group.items.map((item) => {
            const active =
              location.pathname === item.path || location.pathname.startsWith(`${item.path}/`);
            return (
              <Link
                key={item.id}
                to={item.path}
                className={active ? "active" : undefined}
                title={item.label}
                onClick={onCloseMobile}
              >
                {collapsed ? item.label.slice(0, 1) : item.label}
              </Link>
            );
          })}
        </div>
      ))}
    </nav>
  );
}
