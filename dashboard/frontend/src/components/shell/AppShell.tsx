import React, { useState } from "react";
import { Outlet, useLocation } from "react-router";
import TopStatusBar from "./TopStatusBar";
import SideNavigation from "./SideNavigation";
import CommandPalette from "../CommandPalette";

const LS_KEY = "qs-nav-collapsed";

export default function AppShell() {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    try { return localStorage.getItem(LS_KEY) === "true"; } catch { return false; }
  });

  const navWidth = collapsed ? 48 : 220;

  const handleToggle = () => {
    setCollapsed((prev) => {
      const next = !prev;
      try { localStorage.setItem(LS_KEY, String(next)); } catch {}
      return next;
    });
  };

  return (
    <div className="min-h-screen bg-[#090D11]">
      <TopStatusBar />
      <SideNavigation
        collapsed={collapsed}
        onToggle={handleToggle}
        activePath={location.pathname}
      />
      <CommandPalette />
      <main
        className="transition-all duration-200 overflow-y-auto"
        style={{ paddingTop: 32, paddingLeft: navWidth, minHeight: "100vh" }}
      >
        <div className="p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
