import React, { useEffect, useState } from "react";
import { useDataSource } from "../app/DataSourceProvider";
import { ChampionWorkspace } from "../types/champion";
import EvidenceChecklist from "../components/data-display/EvidenceChecklist";
import DataSourceBadge from "../components/status/DataSourceBadge";
import { AlertTriangle } from "lucide-react";
import { fmtWDL, fmtPct } from "../utils/formatting";

export default function ChampionPage() {
  const ds = useDataSource();
  const [workspace, setWorkspace] = useState<ChampionWorkspace | null>(null);

  useEffect(() => { ds.getChampionWorkspace().then(setWorkspace); }, [ds]);

  if (!workspace) return (
    <div className="p-6">
      <p className="text-[#FFB000] font-mono text-xs uppercase tracking-widest mb-1">$ champion/</p>
      <h1 className="text-2xl font-bold text-[#EAF0F6]" style={{ fontFamily: "var(--font-heading)" }}>Champion</h1>
      <p className="text-[#8593A1] text-sm mt-4">Loading…</p>
    </div>
  );

  return (
    <div className="p-6 space-y-6">
      <header>
        <p className="text-[#FFB000] font-mono text-xs uppercase tracking-widest mb-1">$ champion/</p>
        <h1 className="text-2xl font-bold text-[#EAF0F6]" style={{ fontFamily: "var(--font-heading)" }}>Champion Workspace</h1>
      </header>

      <DataSourceBadge kind={workspace.kind} />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="border border-[#1E2630] rounded-sm p-4 bg-[#0C1116]">
          <p className="text-[#6F7C89] font-mono text-xs uppercase tracking-wider mb-1">Current Champion</p>
          <p className="text-xl font-bold font-mono" style={{ color: workspace.currentChampion ? "#3FB950" : "#4A5568" }}>
            {workspace.currentChampion ?? "NOT RECORDED"}
          </p>
          <p className="text-[#6F7C89] font-mono text-xs mt-1">
            {workspace.currentChampion ? "Promoted to champion." : "No champion has been recorded in imported evidence."}
          </p>
        </div>

        <div className="border border-[#FFB000] border-opacity-40 rounded-sm p-4 bg-[#0C1116]">
          <p className="text-[#6F7C89] font-mono text-xs uppercase tracking-wider mb-1">Current Candidate</p>
          <p className="text-[#FFB000] font-bold font-mono text-sm">{workspace.currentCandidate}</p>
          <p className="text-[#8593A1] font-mono text-xs mt-1">CHALLENGER — EVALUATED — BLOCKED</p>
          <p className="text-[#F85149] font-mono text-xs mt-1">Discovery gate FAILED: 0.438 &lt; threshold</p>
        </div>

        <div className="border border-[#1E2630] rounded-sm p-4 bg-[#0C1116]">
          <p className="text-[#6F7C89] font-mono text-xs uppercase tracking-wider mb-1">Submitted Baseline</p>
          <p className="font-bold font-mono text-sm" style={{ color: workspace.currentSubmittedBaseline ? "#CDD6DF" : "#4A5568" }}>
            {workspace.currentSubmittedBaseline ?? "NOT RECORDED"}
          </p>
          <p className="text-[#6F7C89] font-mono text-xs mt-1">
            {workspace.currentSubmittedBaseline ? "Verified submission record." : "No submission baseline recorded in imported evidence."}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="border border-[#1E2630] rounded-sm p-4">
          <p className="text-[#8593A1] font-mono text-xs uppercase tracking-wider mb-1">Development Result</p>
          <p className="text-[#FFB000] font-bold font-mono text-2xl">21W / 27D / 0L</p>
          <DataSourceBadge kind="IMPORTED_PROJECT_EVIDENCE" pill />
        </div>
        <div className="border border-[#1E2630] rounded-sm p-4">
          <p className="text-[#8593A1] font-mono text-xs uppercase tracking-wider mb-1">Discovery Rate</p>
          <p className="text-[#F85149] font-bold font-mono text-2xl">0.438</p>
          <p className="text-[#F85149] font-mono text-xs">Below threshold — gate FAILED</p>
        </div>
      </div>

      <section>
        <p className="text-[#8593A1] font-mono text-xs uppercase tracking-widest mb-3">Promotion Gate Checklist</p>
        <EvidenceChecklist rows={workspace.checklist.rows} />
      </section>

      <section className="border border-[#F85149] border-opacity-30 rounded-sm p-4 bg-[#0C1116]">
        <div className="flex items-start gap-3">
          <AlertTriangle size={16} className="text-[#F85149] flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-[#F85149] font-mono text-xs font-bold uppercase mb-1">Promotion Blocked</p>
            <p className="text-[#8593A1] font-mono text-xs">The discovery development gate has FAILED. All downstream gates (holdout, package, submission) are blocked. PPO training has not started. No promotion is possible until the discovery rate threshold is met.</p>
          </div>
        </div>
      </section>

      <button
        disabled
        className="w-full py-2.5 rounded-sm border border-[#1E2630] text-[#4A5568] cursor-not-allowed"
        style={{ fontFamily: "var(--font-mono)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.08em" }}
      >
        Promote to Champion — Disabled (Discovery Gate FAILED)
      </button>
    </div>
  );
}
