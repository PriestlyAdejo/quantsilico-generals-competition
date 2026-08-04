import React, { useEffect, useState } from "react";
import { useDataSource } from "../../app/DataSourceProvider";
import { ApplicationStatusRecord } from "../../types/overview";

export default function TopStatusBar({ onMenuClick }: { onMenuClick?: () => void }) {
  const ds = useDataSource();
  const [status, setStatus] = useState<ApplicationStatusRecord | null>(null);

  useEffect(() => {
    ds.getApplicationStatus().then(setStatus).catch(() => {});
  }, [ds]);

  const branch = status?.branch ?? "———";
  const sha = status ? status.engineSha.slice(0, 8) : "———";
  const candidate = status?.currentCandidate ?? "———";
  const champion = status?.currentChampion ?? "NOT RECORDED";

  return (
    <div className="fixed top-0 left-0 right-0 z-50 h-8 bg-[#0C1116] border-b border-[#1E2630] flex items-center justify-between px-3">
      <div className="flex items-center gap-2">
        {onMenuClick && (
          <button
            onClick={onMenuClick}
            className="lg:hidden text-[#8593A1] hover:text-[#FFB000] mr-1 flex items-center"
            aria-label="Open navigation"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <rect y="2" width="16" height="1.5" rx="0.75" />
              <rect y="7" width="16" height="1.5" rx="0.75" />
              <rect y="12" width="16" height="1.5" rx="0.75" />
            </svg>
          </button>
        )}
        <span className="font-mono text-xs font-bold text-[#FFB000]">QUANTSILICO</span>
        <span className="text-[#1E2630] font-mono text-xs hidden sm:block">/</span>
        <span className="font-mono text-xs text-[#8593A1] hidden sm:block">GENERALS RESEARCH CONSOLE</span>
      </div>
      <div className="hidden lg:flex items-center gap-0">
        <Chip label="BRANCH" value={branch} />
        <Divider />
        <Chip label="SHA" value={sha} />
        <Divider />
        <Chip label="CANDIDATE" value={candidate.length > 20 ? candidate.slice(0, 20) + "…" : candidate} valueClass="text-[#FFB000]" />
        <Divider />
        <Chip label="CHAMPION" value={champion} valueClass={champion === "NOT RECORDED" ? "text-[#6F7C89]" : "text-[#22D3EE]"} />
      </div>
      <div className="lg:hidden">
        <Chip label="BRANCH" value={branch} />
      </div>
    </div>
  );
}

function Divider() {
  return <div className="w-px h-4 bg-[#1E2630] mx-2" />;
}

function Chip({ label, value, valueClass = "text-[#CDD6DF]" }: { label: string; value: string; valueClass?: string }) {
  return (
    <div className="flex items-center gap-1" style={{ fontFamily: "var(--font-mono)", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.08em" }}>
      <span className="text-[#6F7C89]">{label}:</span>
      <span className={valueClass}>{value}</span>
    </div>
  );
}
