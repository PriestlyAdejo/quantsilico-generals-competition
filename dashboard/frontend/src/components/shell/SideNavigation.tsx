import React from "react";
import { NavLink } from "react-router";
import {
  LayoutDashboard, Swords, FlaskConical, PlayCircle, ClipboardCheck, Cpu,
  TestTube, Brain, Network, Microscope, Trophy, Upload, Medal, GitBranch,
  BookOpen, ChevronLeft, ChevronRight,
} from "lucide-react";
import { navGroups } from "../../app/navigation";

const ICONS: Record<string, React.ComponentType<{ size?: number; className?: string }>> = {
  LayoutDashboard, Swords, FlaskConical, PlayCircle, ClipboardCheck, Cpu,
  TestTube, Brain, Network, Microscope, Trophy, Upload, Medal, GitBranch,
  BookOpen,
};

interface Props {
  collapsed: boolean;
  onToggle: () => void;
  activePath: string;
}

export default function SideNavigation({ collapsed, onToggle, activePath }: Props) {
  const width = collapsed ? 48 : 220;

  return (
    <div
      className="fixed left-0 bottom-0 bg-[#11161C] border-r border-[#1E2630] flex flex-col overflow-hidden z-40 transition-all duration-200"
      style={{ top: 32, width }}
    >
      <div className="flex-1 overflow-y-auto overflow-x-hidden py-2">
        {navGroups.map((group) => (
          <div key={group.id} className="mb-1">
            {!collapsed && (
              <div
                className="px-3 pt-3 pb-1 text-[#6F7C89] uppercase tracking-widest"
                style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}
              >
                {group.label}
              </div>
            )}
            {collapsed && <div className="h-2" />}
            {group.items.map((item) => {
              const Icon = ICONS[item.icon] ?? LayoutDashboard;
              const isActive = activePath === item.path || activePath.startsWith(item.path + "/");
              return (
                <NavLink
                  key={item.id}
                  to={item.path}
                  title={collapsed ? item.label : undefined}
                  className={() =>
                    [
                      "flex items-center gap-2 px-3 py-2 text-xs transition-colors duration-100 relative",
                      isActive
                        ? "bg-[#161C24] text-[#FFB000] border-l-2 border-[#FFB000]"
                        : "text-[#8593A1] hover:text-[#CDD6DF] hover:bg-[#0C1116] border-l-2 border-transparent",
                    ].join(" ")
                  }
                >
                  <Icon size={14} className="flex-shrink-0" />
                  {!collapsed && (
                    <span style={{ fontFamily: "var(--font-body)", fontSize: 12 }}>{item.label}</span>
                  )}
                  {item.badge && !collapsed && (
                    <span className="ml-auto text-[9px] bg-[#FFB000] text-[#090D11] px-1 rounded-sm font-bold">
                      {item.badge}
                    </span>
                  )}
                </NavLink>
              );
            })}
          </div>
        ))}
      </div>
      <button
        onClick={onToggle}
        className="flex items-center justify-center h-9 border-t border-[#1E2630] text-[#6F7C89] hover:text-[#CDD6DF] hover:bg-[#0C1116] transition-colors"
        title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
      </button>
    </div>
  );
}
