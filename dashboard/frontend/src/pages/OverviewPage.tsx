import React, { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from "recharts";
import { useDataSource } from "../app/DataSourceProvider";
import { OverviewRecord } from "../types/overview";
import PageHeader from "../components/typography/PageHeader";
import MetricCard from "../components/data-display/MetricCard";
import ChartCard from "../components/data-display/ChartCard";
import DataSourceBadge from "../components/status/DataSourceBadge";
import LoadingState from "../components/feedback/LoadingState";
import { chartTheme, rechartsDefaults } from "../utils/chartTheme";
import { fmtWDL, fmtPct } from "../utils/formatting";
import { AlertTriangle } from "lucide-react";

export default function OverviewPage() {
  const ds = useDataSource();
  const [record, setRecord] = useState<OverviewRecord | null>(null);

  useEffect(() => { ds.getOverview().then(setRecord); }, [ds]);

  if (!record) return <LoadingState />;

  const wdlChartData = record.wdlHistory.map(h => ({
    name: h.dateLabel ?? h.week,
    wins: h.wdl.wins,
    draws: h.wdl.draws,
    losses: h.wdl.losses,
  }));

  return (
    <div className="p-6 space-y-6">
      <PageHeader eyebrow="overview/" title="Overview" />
      <DataSourceBadge kind={record.kind} />

      {record.blocker && (
        <div className="border border-[#F85149] border-opacity-50 rounded-sm p-3 bg-[#0C1116] flex items-start gap-2">
          <AlertTriangle size={14} className="text-[#F85149] flex-shrink-0 mt-0.5" />
          <div>
            <span className="text-[#F85149] font-mono text-xs font-bold">BLOCKER: {record.blocker}</span>
            <p className="text-[#8593A1] font-mono text-xs mt-0.5">Discovery rate {fmtPct(Number.isFinite(record.discoveryRate) ? record.discoveryRate : null)} — PPO {record.ppoStatus.replace("_", " ")}</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="Current Result" value={fmtWDL(record.currentResult, record.currentResult ? "RECORDED" : "MISSING")} sublabel="Development" />
        <MetricCard label="Discovery Rate" value={fmtPct(Number.isFinite(record.discoveryRate) ? record.discoveryRate : null)} sublabel="From qualification suite" />
        <MetricCard label="Conversion" value={fmtPct(Number.isFinite(record.conversionRate) ? record.conversionRate : null)} sublabel="Post-discovery" />
        <MetricCard label="PPO Status" value={record.ppoStatus.replace("_", " ")} sublabel="Training" />
      </div>

      <div className="border border-[#1E2630] rounded-sm p-4 bg-[#0C1116]">
        <p className="text-[#6F7C89] font-mono text-xs uppercase mb-1">Current Candidate</p>
        <p className="text-[#FFB000] font-mono font-bold">{record.currentCandidate}</p>
        <p className="text-[#6F7C89] font-mono text-[10px] mt-2">
          <a className="text-[#22D3EE] hover:underline" href="/documentation/overview">About Overview</a>
          {" · "}
          <a className="text-[#22D3EE] hover:underline" href="/documentation/glossary">Glossary</a>
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartCard title="W/D/L History">
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={wdlChartData} {...rechartsDefaults}>
                <CartesianGrid {...rechartsDefaults.cartesianGrid} />
                <XAxis dataKey="name" {...rechartsDefaults.axisStyle} tick={{ ...rechartsDefaults.axisStyle.tick, fontSize: 9 }} />
                <YAxis {...rechartsDefaults.axisStyle} />
                <Tooltip contentStyle={{ background: chartTheme.tooltip.bg, border: `1px solid ${chartTheme.tooltip.border}`, color: chartTheme.tooltip.text, fontFamily: "var(--font-mono)", fontSize: 10 }} />
                <Bar dataKey="wins" fill={chartTheme.positive} name="Wins" />
                <Bar dataKey="draws" fill={chartTheme.neutral} name="Draws" />
                <Bar dataKey="losses" fill={chartTheme.negative} name="Losses" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[#6F7C89] font-mono text-xs mt-1">
            {record.currentResult
              ? `Current recorded DEVELOPMENT: ${fmtWDL(record.currentResult, "RECORDED")}. Chart history may be incomplete.`
              : "Current DEVELOPMENT W/D/L NOT RECORDED for this overview source."}
          </p>
        </ChartCard>

        <ChartCard title="Qualification Funnel">
          <div className="space-y-1.5 mt-2">
            {record.qualificationFunnel.map(({ stage, count }) => (
              <div key={stage} className="flex items-center gap-2">
                <span className="w-28 text-[#8593A1]" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>{stage}</span>
                <div className="flex-1 h-4 bg-[#1E2630] rounded-sm overflow-hidden">
                  <div className="h-full bg-[#22D3EE] rounded-sm opacity-70" style={{ width: `${(count / 14) * 100}%` }} />
                </div>
                <span className="w-4 text-[#CDD6DF] text-right" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>{count}</span>
              </div>
            ))}
          </div>
        </ChartCard>
      </div>

      <ChartCard title="Experiment Timeline">
        <div className="space-y-2 mt-2">
          {record.experimentTimeline.map(entry => (
            <div key={entry.id} className="flex items-center gap-3 py-1.5 border-b border-[#1E2630] last:border-b-0">
              <div className={`w-2 h-2 rounded-full flex-shrink-0 ${entry.status === "complete" ? "bg-[#3FB950]" : entry.status === "running" ? "bg-[#FFB000] animate-pulse" : "bg-[#4A5568]"}`} />
              <span className="flex-1 text-[#CDD6DF]" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>{entry.label}</span>
              <span className="text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>{entry.dateLabel ?? entry.startedAt}</span>
            </div>
          ))}
        </div>
      </ChartCard>

      <ChartCard title="Active Jobs">
        <div className="space-y-2 mt-2">
          {record.activeJobs.map(job => (
            <div key={job.id} className="flex items-center gap-3">
              <span className="flex-1 text-[#8593A1]" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>{job.label}</span>
              <div className="w-32 h-3 bg-[#1E2630] rounded-sm overflow-hidden">
                <div className="h-full rounded-sm" style={{ width: `${Number.isFinite(job.progress) ? job.progress * 100 : 0}%`, backgroundColor: job.status === "complete" ? "#3FB950" : "#FFB000" }} />
              </div>
              <span className="text-[#6F7C89] w-10 text-right" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>{Number.isFinite(job.progress) ? `${(job.progress * 100).toFixed(0)}%` : "—"}</span>
            </div>
          ))}
        </div>
      </ChartCard>
    </div>
  );
}
