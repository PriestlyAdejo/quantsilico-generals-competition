import React, { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { useDataSource } from "../app/DataSourceProvider";
import DataSourceBadge from "../components/status/DataSourceBadge";
import StatusBadge from "../components/status/StatusBadge";
import PageHeader from "../components/typography/PageHeader";
import Panel from "../components/data-display/Panel";
import LoadingState from "../components/feedback/LoadingState";
import { QualCandidate, Phase9QStep, StepStatus, QualStageInfo } from "../types/qualification";
import { fmtAvailablePct, fmtWDL } from "../utils/formatting";
import { SUBMITTED_CANDIDATE_ID } from "../types/common";

const CHART_STYLE = { fontFamily: "var(--font-mono)", fontSize: 10, fill: "#6F7C89" };

const DEFAULT_LABELS: Record<Phase9QStep, string> = {
  screening: "Screening Evaluation",
  development: "Development Evaluation",
  holdout: "Holdout Evaluation",
  package: "Package Build",
  linux_parity: "Linux Validation",
  upload_ready: "Upload Ready",
  portal: "Portal Accepted",
};

function stepStatusColor(status: StepStatus): string {
  switch (status) {
    case "complete": return "#3FB950";
    case "active": return "#FFB000";
    case "failed": return "#F85149";
    default: return "#6F7C89";
  }
}

function QualificationStepper({
  candidate,
  selectedStageId,
  onSelect,
}: {
  candidate: QualCandidate;
  selectedStageId: string;
  onSelect: (id: string) => void;
}) {
  const steps = candidate.phase9q.steps;
  return (
    <Panel title="Delivery pipeline" eyebrow="stages/" className="mb-6">
      <p className="text-[#6F7C89] font-mono text-[10px] mb-3">
        Select a stage to inspect evidence. Advanced identifiers live in the evidence drawer.
      </p>
      <div className="flex items-start gap-0">
        {steps.map((stepData, idx) => {
          const status = stepData.status;
          const color = stepStatusColor(status);
          const label = stepData.label ?? DEFAULT_LABELS[stepData.step] ?? stepData.step;
          const isLast = idx === steps.length - 1;
          const selected = selectedStageId === stepData.step;
          return (
            <React.Fragment key={stepData.step}>
              <button
                type="button"
                className="flex flex-col items-center flex-1 min-w-0 bg-transparent border-0 cursor-pointer p-0"
                onClick={() => onSelect(stepData.step)}
              >
                <div
                  className="w-7 h-7 rounded-full border-2 flex items-center justify-center mb-2 flex-shrink-0 text-xs font-bold"
                  style={{
                    borderColor: selected ? "#FFB000" : color,
                    color: status === "active" ? "#090D11" : color,
                    backgroundColor: status === "active" ? "#FFB000" : "transparent",
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  {status === "complete" ? "✓" : idx + 1}
                </div>
                <div className="text-center" style={{ fontFamily: "var(--font-mono)", fontSize: 9, color, maxWidth: 80 }}>
                  {label}
                </div>
                <div className="text-[#4A5568] text-center mt-0.5" style={{ fontFamily: "var(--font-mono)", fontSize: 8 }}>
                  {stepData.step}
                </div>
              </button>
              {!isLast && (
                <div className="flex-shrink-0 mt-3.5 h-px w-4" style={{ backgroundColor: status === "complete" ? "#3FB950" : "#1E2630" }} />
              )}
            </React.Fragment>
          );
        })}
      </div>
    </Panel>
  );
}

export default function QualificationPage() {
  const { candidateId } = useParams<{ candidateId?: string }>();
  const ds = useDataSource();
  const [candidates, setCandidates] = useState<QualCandidate[] | null>(null);
  const [selected, setSelected] = useState<QualCandidate | null>(null);
  const [suite, setSuite] = useState("development");
  const [stageId, setStageId] = useState("development");

  useEffect(() => {
    ds.listCandidates().then((list) => {
      setCandidates(list);
      const target = candidateId
        ? list.find((c) => c.id === candidateId)
        : list.find((c) => c.id === SUBMITTED_CANDIDATE_ID || c.name === SUBMITTED_CANDIDATE_ID)
          ?? list.find((c) => c.kind === "IMPORTED_PROJECT_EVIDENCE")
          ?? list[0];
      setSelected(target ?? null);
      if (target?.phase9q.currentStep) setStageId(target.phase9q.currentStep);
    });
  }, [ds, candidateId]);

  const stageDetail: QualStageInfo | null = useMemo(() => {
    if (!selected?.stages?.length) return null;
    return selected.stages.find((s) => s.id === stageId) ?? selected.stages[0] ?? null;
  }, [selected, stageId]);

  if (!candidates) return <LoadingState />;

  const wdlData = selected
    ? [
        {
          phase: "Screening",
          Wins: selected.screeningAvailability === "RECORDED" ? selected.screeningWDL?.wins ?? null : null,
          Draws: selected.screeningAvailability === "RECORDED" ? selected.screeningWDL?.draws ?? null : null,
          Losses: selected.screeningAvailability === "RECORDED" ? selected.screeningWDL?.losses ?? null : null,
        },
        {
          phase: "Development",
          Wins: selected.developmentAvailability === "RECORDED" ? selected.developmentWDL?.wins ?? null : null,
          Draws: selected.developmentAvailability === "RECORDED" ? selected.developmentWDL?.draws ?? null : null,
          Losses: selected.developmentAvailability === "RECORDED" ? selected.developmentWDL?.losses ?? null : null,
        },
      ].filter((row) => row.Wins != null)
    : [];

  const discoveryData = selected
    ? [
        ...(selected.discovery.availability === "RECORDED" && selected.discovery.value != null
          ? [{ label: "Discovery", value: Math.round(selected.discovery.value * 1000) / 10 }]
          : []),
        ...(selected.conversion.availability === "RECORDED" && selected.conversion.value != null
          ? [{ label: "Conversion", value: Math.round(selected.conversion.value * 1000) / 10 }]
          : []),
      ]
    : [];

  return (
    <div>
      <PageHeader
        eyebrow="qualification/"
        title="Candidate Qualification"
        subtitle="Delivery and evaluation evidence for the selected candidate."
      />
      <p className="px-1 mb-4">
        <a href="/documentation/qualification" className="text-[#22D3EE] font-mono text-xs hover:underline">About this page</a>
        {" · "}
        <a href="/documentation/glossary" className="text-[#22D3EE] font-mono text-xs hover:underline">Glossary</a>
      </p>

      <Panel title="Evidence selectors" eyebrow="filters/" className="mb-4">
        <div className="flex flex-wrap gap-3 items-end">
          <label className="text-[#6F7C89] font-mono text-[10px]">
            Candidate
            <select
              className="block mt-1 bg-[#0C1116] border border-[#1E2630] text-[#EAF0F6] font-mono text-xs px-2 py-1.5 rounded-sm min-w-[16rem]"
              value={selected?.id ?? ""}
              onChange={(e) => {
                const c = candidates.find((x) => x.id === e.target.value) ?? null;
                setSelected(c);
              }}
            >
              {candidates.map((c) => (
                <option key={c.id} value={c.id}>{c.name}</option>
              ))}
            </select>
          </label>
          <label className="text-[#6F7C89] font-mono text-[10px]">
            Evaluation suite
            <select
              className="block mt-1 bg-[#0C1116] border border-[#1E2630] text-[#EAF0F6] font-mono text-xs px-2 py-1.5 rounded-sm"
              value={suite}
              onChange={(e) => setSuite(e.target.value)}
            >
              <option value="screening">Screening Evaluation</option>
              <option value="development">Development Evaluation</option>
              <option value="persistent-state-diagnostic">Persistent-state diagnostic</option>
              <option value="pre-ppo">Pre-PPO submission comparison</option>
              <option value="portal">Portal submission</option>
              <option value="learning-readiness">Learning readiness</option>
              <option value="learned-promotion">Learned promotion</option>
            </select>
          </label>
          <div className="text-[#8593A1] font-mono text-[10px] max-w-md">
            Selected suite controls which metrics are emphasised below. Suites are never merged unlabeled.
            Current suite: <span className="text-[#FFB000]">{suite}</span>
          </div>
        </div>
      </Panel>

      {selected && (
        <QualificationStepper candidate={selected} selectedStageId={stageId} onSelect={setStageId} />
      )}

      {stageDetail && (
        <Panel title={stageDetail.label} eyebrow="stage-detail/" className="mb-6">
          <p className="text-[#4A5568] font-mono text-[10px] mb-2">Internal ID: {stageDetail.internalId}</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs" style={{ fontFamily: "var(--font-mono)" }}>
            <div><span className="text-[#6F7C89]">What it tests:</span> <span className="text-[#CDD6DF]">{stageDetail.explains}</span></div>
            <div><span className="text-[#6F7C89]">Evidence:</span> <span className="text-[#CDD6DF]">{stageDetail.evidence}</span></div>
            <div><span className="text-[#6F7C89]">PASS means:</span> <span className="text-[#CDD6DF]">{stageDetail.passMeans}</span></div>
            <div><span className="text-[#6F7C89]">FAIL means:</span> <span className="text-[#CDD6DF]">{stageDetail.failMeans}</span></div>
            <div><span className="text-[#6F7C89]">Blocks next:</span> <span className="text-[#CDD6DF]">{stageDetail.blocksNext ? "yes" : "no"}</span></div>
            <div><span className="text-[#6F7C89]">Perspective:</span> <span className="text-[#CDD6DF]">{stageDetail.perspective}</span></div>
          </div>
          {stageDetail.reasons && stageDetail.reasons.length > 0 && (
            <div className="mt-3">
              <StatusBadge variant="warning" label={stageDetail.reasons[0]} />
            </div>
          )}
        </Panel>
      )}

      <Panel title="Candidates" eyebrow="candidates/" className="mb-6">
        <div className="overflow-x-auto">
          <table className="w-full" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>
            <thead>
              <tr className="border-b border-[#1E2630]">
                {["Name", "Kind", "Stage", "Screening W/D/L", "Development W/D/L", "Discovery", "Conversion"].map((h) => (
                  <th key={h} className="text-left py-2 px-3 text-[#6F7C89] uppercase tracking-wider" style={{ fontSize: 9 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {candidates.map((c) => {
                const isSelected = selected?.id === c.id;
                return (
                  <tr
                    key={c.id}
                    className={`border-b border-[#1E2630] cursor-pointer ${isSelected ? "bg-[#161C24]" : "hover:bg-[#0C1116]"}`}
                    onClick={() => setSelected(c)}
                  >
                    <td className="py-2 px-3 text-[#EAF0F6]">{c.name}</td>
                    <td className="py-2 px-3"><DataSourceBadge kind={c.kind} pill /></td>
                    <td className="py-2 px-3 text-[#8593A1]">{DEFAULT_LABELS[c.phase9q.currentStep]}</td>
                    <td className="py-2 px-3 text-[#8593A1]">{fmtWDL(c.screeningWDL, c.screeningAvailability)}</td>
                    <td className="py-2 px-3 text-[#8593A1]">{fmtWDL(c.developmentWDL, c.developmentAvailability)}</td>
                    <td className="py-2 px-3 text-[#8593A1]">{fmtAvailablePct(c.discovery)}</td>
                    <td className="py-2 px-3 text-[#8593A1]">{fmtAvailablePct(c.conversion)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>

      {selected && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
          <Panel title="W/D/L by phase" eyebrow="charts/">
            {wdlData.length === 0 ? (
              <p className="text-[#6F7C89] font-mono text-xs p-4">NOT RECORDED for the selected suites.</p>
            ) : (
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
            )}
          </Panel>
          <Panel title="Discovery and conversion" eyebrow="charts/">
            {discoveryData.length === 0 ? (
              <p className="text-[#6F7C89] font-mono text-xs p-4">NOT RECORDED</p>
            ) : (
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={discoveryData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E2630" />
                  <XAxis dataKey="label" tick={CHART_STYLE} axisLine={false} tickLine={false} />
                  <YAxis tick={CHART_STYLE} axisLine={false} tickLine={false} domain={[0, 100]} unit="%" />
                  <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2630", fontFamily: "var(--font-mono)", fontSize: 10 }} formatter={(v) => `${v}%`} />
                  <Bar dataKey="value" fill="#22D3EE" />
                </BarChart>
              </ResponsiveContainer>
            )}
            <p className="text-[#6F7C89] font-mono text-[10px] mt-2">
              Suite provenance: {selected.discovery.source ?? selected.conversion.source ?? "NOT RECORDED"}
            </p>
          </Panel>
        </div>
      )}
    </div>
  );
}
