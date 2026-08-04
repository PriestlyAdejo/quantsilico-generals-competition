import React, { ReactNode } from "react";
import DataSourceBadge from "../status/DataSourceBadge";
import { DataSourceKind } from "../../types/common";

interface Props {
  title: string;
  eyebrow?: string;
  children: ReactNode;
  badge?: ReactNode;
  className?: string;
  dataKind?: DataSourceKind;
}

export default function ChartCard({ title, eyebrow, children, badge, className = "", dataKind }: Props) {
  return (
    <div className={`bg-[#11161C] border border-[#1E2630] rounded-sm ${className}`}>
      <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-[#1E2630]">
        <div className="flex items-center gap-2">
          {eyebrow && (
            <span
              className="text-[#FFB000] uppercase tracking-widest"
              style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}
            >
              $ {eyebrow}
            </span>
          )}
          <span className="text-[#EAF0F6] font-bold" style={{ fontFamily: "var(--font-display)", fontSize: 13 }}>
            {title}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {badge}
          {dataKind && <DataSourceBadge kind={dataKind} pill />}
        </div>
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}
