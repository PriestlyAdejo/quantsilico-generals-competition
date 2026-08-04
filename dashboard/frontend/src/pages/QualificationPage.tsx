import React, { useEffect, useState } from "react";
import { useParams } from "react-router";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { useDataSource } from "../app/DataSourceProvider";
import DataSourceBadge from "../components/status/DataSourceBadge";
import StatusBadge from "../components/status/StatusBadge";
import PageHeader from "../components/typography/PageHeader";
import Panel from "../components/data-display/Panel";
import LoadingState from "../components/feedback/LoadingState";
import { QualCandidate } from "../types/qualification";
import { Phase9QStep, StepStatus } from "../types/qualification";

const CHART_STYLE = { fontFamily: "var(--font-mono)", fontSize: 10, fill: "#6F7C89" };

const STEP_LABELS: Record<Phase9QStep, string> = {
  screening: "Screening",
  development: "Development",
  holdout: "Holdout",
  package: "Package",
  linux_parity: "Linux Parity",
  upload_ready: "Upload Ready",
  portal: "Portal",
};

const STEP_ORDER: Phase9QStep[] = [
  "screening", "development", "holdout", "package", "linux_parity", "upload_ready", "portal",
];

function stepStatusColor(status: StepStatus): string {
  switch (status) {
    case "complete": return "#3FB950";
    case "active":   return "#FFB000";
    case "failed":   return "#F85149";
    default:         return "#6F7C89";
  }
}

function Phase9QStepper({ candidate }: { candidate: QualCandidate }) {
  const steps = candidate.phase9q.steps;
  const stepMap = Object.fromEntries(steps.map((s) => [s.step, s]));

  return (
    <Panel title="Phase 9Q Qualification Stepper" eyebrow="phase-9q/" className="mb-6">
      <div className="flex items-start gap-0">
        {STEP_ORDER.map((stepId, idx) => {
          const stepData = stepMap[stepId];
          const status: StepStatus = stepData?.status ?? "pending";
          const color = stepStatusColor(status);
          const isLast = idx === STEP_ORDER.length - 1;

          return (
            <React.Fragment key={stepId}>
              <div className="flex flex-col items-center flex-1 min-w-0">
                <div
                  className="w-7 h-7 rounded-full border-2 flex items-center justify-center mb-2 flex-shrink-0 text-xs font-bold transition-all"
                  style={{
                    borderColor: color,
                    color: status === "active" ? "#090D11" : color,
                    backgroundColor: status === "active" ? "#FFB000" : "transparent",
                    boxShadow: status === "active" ? "0 0 8px rgba(255,176,0,0.5)" : "none",
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  {status === "complete" ? "✓" : idx + 1}
                </div>
                <div
                  className="text-center"
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 9,
                    color,
                    maxWidth: 72,
                    wordBreak: "break-word",
                  }}
                >
                  {STEP_LABELS[stepId]}
                </div>
                {stepData?.completedAt && (
                  <div className="text-[#6F7C89] text-center mt-0.5" style={{ fontFamily: "var(--font-mono)", fontSize: 8 }}>
                    {stepData.completedAt}
                  </div>
                )}
              </div>
              {!isLast && (
                <div
                  className="flex-shrink-0 mt-3.5 h-px w-4"
                  style={{ backgroundColor: status === "complete" ? "#3FB950" : "#1E2630" }}
                />
              )}
            </React.Fragment>
          );
        })}
      </div>
    </Panel>
  );
}

function CandidateCharts({ candidate }: { candidate: QualCandidate }) {
  const wdlData = [
    { phase: "Screening", Wins: candidate.screeningWDL.wins, Draws: candidate.screeningWDL.draws, Losses: candidate.screeningWDL.losses },
    { phase: "Development", Wins: candidate.developmentWDL.wins, Draws: candidate.developmentWDL.draws, Losses: candidate.developmentWDL.losses },
  ];

  const discoveryData = [
    { label: "Discovered", value: Math.round((candidate.discoveryRate ?? 0) * 100) },
    { label: "Converted", value: Math.round((candidate.conversionRate ?? 0) * 100) },
  ];

  const terminalData = [
    { label: "P50", turns: candidate.terminalTurnP50 },
    { label: "P95", turns: candidate.terminalTurnP95 },
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-4">
      <Panel title="W/D/L by Phase" eyebrow="charts/">
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={wdlData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1E2630" />
            <XAxis dataKey="phase" tick={CHART_STYLE} axisLine={false} tickLine={false} />
            <YAxis tick={CHART_STYLE} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2630", fontFamily: "var(--font-mono)", fontSize: 10 }} />
            <Bar dataKey="Wins" stackId="a" fill="#3FB950" />
            <Bar dataKey="Draws" stackId="a" fill="#8593A1" />
            <Bar dataKey="Losses" stackId="a" fill="#F85149" />
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Discovery Funnel" eyebrow="charts/">
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={discoveryData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1E2630" />
            <XAxis dataKey="label" tick={CHART_STYLE} axisLine={false} tickLine={false} />
            <YAxis tick={CHART_STYLE} axisLine={false} tickLine={false} domain={[0, 100]} unit="%" />
            <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2630", fontFamily: "var(--font-mono)", fontSize: 10 }} formatter={(v) => `${v}%`} />
            <Bar dataKey="value" fill="#22D3EE" />
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Terminal Turn" eyebrow="charts/">
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={terminalData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1E2630" />
            <XAxis dataKey="label" tick={CHART_STYLE} axisLine={false} tickLine={false} />
            <YAxis tick={CHART_STYLE} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2630", fontFamily: "var(--font-mono)", fontSize: 10 }} />
            <Bar dataKey="turns" fill="#FFB000" />
          </BarChart>
        </ResponsiveContainer>
      </Panel>
    </div>
  );
}

export default function QualificationPage() {
  const { candidateId } = useParams<{ candidateId?: string }>();
  const ds = useDataSource();
  const [candidates, setCandidates] = useState<QualCandidate[] | null>(null);
  const [selected, setSelected] = useState<QualCandidate | null>(null);

  useEffect(() => {
    ds.listCandidates().then((list) => {
      setCandidates(list);
      // Default: select the Expander (IMPORTED_PROJECT_EVIDENCE) or the requested one
      const target = candidateId
        ? list.find((c) => c.id === candidateId)
        : list.find((c) => c.kind === "IMPORTED_PROJECT_EVIDENCE") ?? list[0];
      setSelected(target ?? null);
    });
  }, [ds, candidateId]);

  if (!candidates) return <LoadingState />;

  const expander = candidates.find((c) => c.kind === "IMPORTED_PROJECT_EVIDENCE");

  return (
    <div>
      <PageHeader eyebrow="qualification/" title="Qualification" subtitle="Phase 9Q candidate tracking." />

      {/* Stepper for selected candidate */}
      {selected && <Phase9QStepper candidate={selected} />}

      {/* Candidate table */}
      <Panel title="Candidates" eyebrow="candidates/" className="mb-6">
        <div className="overflow-x-auto">
          <table className="w-full" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>
            <thead>
              <tr className="border-b border-[#1E2630]">
                {["Name", "Kind", "Phase", "Screening WDL", "Dev WDL", "Discovery%", "Failure Class"].map((h) => (
                  <th key={h} className="text-left py-2 px-3 text-[#6F7C89] uppercase tracking-wider" style={{ fontSize: 9 }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {candidates.map((c) => {
                const isExpander = c.kind === "IMPORTED_PROJECT_EVIDENCE";
                const isSelected = selected?.id === c.id;
                return (
                  <React.Fragment key={c.id}>
                    <tr
                      className={`border-b border-[#1E2630] cursor-pointer transition-colors ${
                        isSelected ? "bg-[#161C24]" : "hover:bg-[#0C1116]"
                      }`}
                      onClick={() => setSelected(c)}
                    >
                      <td className="py-2 px-3 text-[#EAF0F6]">{c.name}</td>
                      <td className="py-2 px-3">
                        <DataSourceBadge kind={c.kind} pill />
                      </td>
                      <td className="py-2 px-3 text-[#8593A1]">{c.phase9q.currentStep.replace(/_/g, " ")}</td>
                      <td className="py-2 px-3 text-[#8593A1]">
                        {c.screeningWDL.wins}W/{c.screeningWDL.draws}D/{c.screeningWDL.losses}L
                      </td>
                      <td className="py-2 px-3 text-[#8593A1]">
                        {c.developmentWDL.wins}W/{c.developmentWDL.draws}D/{c.developmentWDL.losses}L
                      </td>
                      <td className="py-2 px-3 text-[#8593A1]">
                        {Math.round((c.discoveryRate ?? 0) * 100)}%
                      </td>
                      <td className="py-2 px-3">
                        {c.failureClass ? (
                          <StatusBadge variant="warning" label={c.failureClass} />
                        ) : (
                          <span className="text-[#6F7C89]">—</span>
                        )}
                      </td>
                    </tr>
                    {/* Expanded row for Expander */}
                    {isExpander && isSelected && (
                      <tr className="bg-[#0C1116]">
                        <td colSpan={7} className="px-4 py-3">
                          <div className="flex flex-wrap gap-3 text-xs">
                            <span className="text-[#6F7C89]">Terminal P50: <span className="text-[#EAF0F6]">{c.terminalTurnP50}</span></span>
                            <span className="text-[#6F7C89]">Terminal P95: <span className="text-[#EAF0F6]">{c.terminalTurnP95}</span></span>
                            <span className="text-[#6F7C89]">Conversion: <span className="text-[#EAF0F6]">{Math.round((c.conversionRate ?? 0) * 100)}%</span></span>
                            <span className="text-[#6F7C89]">Checkpoint: <span className="text-[#22D3EE]">{c.checkpoint}</span></span>
                            {c.notes && <span className="text-[#6F7C89]">Notes: <span className="text-[#8593A1]">{c.notes}</span></span>}
                          </div>
                          {c.failureClass === "DEATHTOUCH_NOT_EXPLOITED" && (
                            <div className="mt-2">
                              <StatusBadge variant="warning" label="DEATHTOUCH_NOT_EXPLOITED" dot />
                            </div>
                          )}
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* Charts for selected candidate */}
      {selected && (
        <div>
          <div className="flex items-center gap-3 mb-2">
            <span className="text-[#EAF0F6] font-bold" style={{ fontFamily: "var(--font-display)", fontSize: 14 }}>
              {selected.name} — Detail
            </span>
            <DataSourceBadge kind={selected.kind} pill />
          </div>
          <CandidateCharts candidate={selected} />
        </div>
      )}
    </div>
  );
}
