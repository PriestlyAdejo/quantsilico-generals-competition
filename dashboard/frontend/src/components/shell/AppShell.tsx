import { useMemo, useState } from "react";
import { Outlet } from "react-router-dom";
import SideNavigation from "./SideNavigation";
import TopStatusBar from "./TopStatusBar";

export default function AppShell() {
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem("qs-nav-collapsed") === "true";
    } catch {
      return false;
    }
  });
  const [mobileOpen, setMobileOpen] = useState(false);

  const shellClass = useMemo(
    () => `shell${collapsed ? " nav-collapsed" : ""}`,
    [collapsed],
  );

  const toggleCollapse = () => {
    setCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem("qs-nav-collapsed", String(next));
      } catch {
        /* ignore */
      }
      return next;
    });
  };

  return (
    <div className={shellClass}>
      <TopStatusBar />
      <button
        type="button"
        className="menu-btn mobile-only"
        aria-label="Open navigation"
        onClick={() => setMobileOpen(true)}
        style={{ position: "fixed", top: 4, right: 8, zIndex: 45 }}
      >
        Menu
      </button>
      <div
        className={`nav-backdrop${mobileOpen ? " open" : ""}`}
        onClick={() => setMobileOpen(false)}
        aria-hidden={!mobileOpen}
      />
      <SideNavigation
        collapsed={collapsed}
        mobileOpen={mobileOpen}
        onToggleCollapse={toggleCollapse}
        onCloseMobile={() => setMobileOpen(false)}
      />
      <main className="shell-main">
        <div className="shell-main-inner">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
