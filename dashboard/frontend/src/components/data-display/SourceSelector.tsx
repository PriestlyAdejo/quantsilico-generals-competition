import React from "react";
import { DataSourceKind } from "../../types/common";

interface SourceOption {
  kind: DataSourceKind;
  label: string;
  count?: number;
}

interface Props {
  options: SourceOption[];
  value: DataSourceKind;
  onChange: (kind: DataSourceKind) => void;
}

export default function SourceSelector({ options, value, onChange }: Props) {
  return (
    <div className="flex gap-0 border border-[#1E2630] rounded-sm overflow-hidden">
      {options.map(opt => (
        <button
          key={opt.kind}
          onClick={() => onChange(opt.kind)}
          className={`flex items-center gap-1.5 px-3 py-1.5 transition-colors ${
            value === opt.kind ? "bg-[#1E2630] text-[#EAF0F6]" : "text-[#6F7C89] hover:text-[#8593A1] hover:bg-[#0C1116]"
          }`}
          style={{ fontFamily: "var(--font-mono)", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.08em" }}
        >
          {opt.label}
          {opt.count !== undefined && (
            <span className="bg-[#2D3748] rounded-sm px-1 text-[#8593A1]" style={{ fontSize: 9 }}>{opt.count}</span>
          )}
        </button>
      ))}
    </div>
  );
}
