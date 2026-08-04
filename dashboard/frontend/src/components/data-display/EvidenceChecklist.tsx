import React from "react";
import { ChecklistRow, ChecklistStatus } from "../../types/champion";
import { Check, X, AlertTriangle, Clock, Minus } from "lucide-react";

interface Props {
  rows: ChecklistRow[];
  className?: string;
}

export default function EvidenceChecklist({ rows, className = "" }: Props) {
  return (
    <div className={`border border-[#1E2630] rounded-sm overflow-hidden ${className}`}>
      <table className="w-full">
        <thead>
          <tr className="border-b border-[#1E2630] bg-[#0C1116]">
            <th className="px-3 py-2 text-left" style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "#6F7C89", textTransform: "uppercase", letterSpacing: "0.08em" }}>Gate</th>
            <th className="px-3 py-2 text-left" style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "#6F7C89", textTransform: "uppercase", letterSpacing: "0.08em" }}>Status</th>
            <th className="px-3 py-2 text-left" style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "#6F7C89", textTransform: "uppercase", letterSpacing: "0.08em" }}>Detail</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-[#1E2630] last:border-b-0 hover:bg-[#0C1116] transition-colors">
              <td className="px-3 py-2.5">
                <span className="text-[#CDD6DF]" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>{row.gate}</span>
              </td>
              <td className="px-3 py-2.5">
                <StatusChip status={row.status} />
              </td>
              <td className="px-3 py-2.5">
                <span className="text-[#8593A1]" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>
                  {row.detail}
                  {row.blockerReason && <span className="text-[#F85149] ml-1">— {row.blockerReason}</span>}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatusChip({ status }: { status: ChecklistStatus }) {
  const configs: Record<ChecklistStatus, { icon: React.ReactNode; label: string; color: string }> = {
    PASS: { icon: <Check size={9} />, label: "PASS", color: "#3FB950" },
    FAIL: { icon: <X size={9} />, label: "FAIL", color: "#F85149" },
    BLOCKED: { icon: <AlertTriangle size={9} />, label: "BLOCKED", color: "#F85149" },
    PENDING: { icon: <Clock size={9} />, label: "PENDING", color: "#6F7C89" },
    NOT_EVALUATED: { icon: <Minus size={9} />, label: "NOT EVAL", color: "#4A5568" },
  };
  const { icon, label, color } = configs[status];
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-sm border" style={{ color, borderColor: `${color}66`, fontFamily: "var(--font-mono)", fontSize: 9, letterSpacing: "0.08em" }}>
      {icon}{label}
    </span>
  );
}
