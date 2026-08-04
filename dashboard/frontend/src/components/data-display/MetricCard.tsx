import React, { ReactNode } from "react";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface Props {
  label: string;
  value: string | number;
  unit?: string;
  delta?: number;
  deltaDirection?: "up" | "down" | "neutral";
  sublabel?: string;
  badge?: ReactNode;
}

export default function MetricCard({ label, value, unit, delta, deltaDirection, sublabel, badge }: Props) {
  const DeltaIcon =
    deltaDirection === "up" ? TrendingUp : deltaDirection === "down" ? TrendingDown : Minus;
  const deltaColor =
    deltaDirection === "up" ? "text-[#3FB950]" : deltaDirection === "down" ? "text-[#F85149]" : "text-[#6F7C89]";

  return (
    <div className="bg-[#11161C] border border-[#1E2630] rounded-sm p-4 flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span
          className="text-[#8593A1] uppercase tracking-widest"
          style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}
        >
          {label}
        </span>
        {badge}
      </div>
      <div className="flex items-end gap-1">
        <span
          className="text-2xl font-bold text-[#EAF0F6] leading-none"
          style={{ fontFamily: "var(--font-mono)" }}
        >
          {value}
        </span>
        {unit && (
          <span className="text-xs text-[#6F7C89] mb-0.5" style={{ fontFamily: "var(--font-mono)" }}>
            {unit}
          </span>
        )}
      </div>
      {(delta !== undefined || sublabel) && (
        <div className="flex items-center gap-1 mt-1">
          {delta !== undefined && deltaDirection && (
            <span className={`flex items-center gap-0.5 ${deltaColor}`} style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>
              <DeltaIcon size={10} />
              {Math.abs(delta)}
            </span>
          )}
          {sublabel && (
            <span className="text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>
              {sublabel}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
