import React, { useEffect, useState, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { useDataSource } from "../app/DataSourceProvider";
import DataSourceBadge from "../components/status/DataSourceBadge";
import StatusBadge from "../components/status/StatusBadge";
import PageHeader from "../components/typography/PageHeader";
import Panel from "../components/data-display/Panel";
import MetricCard from "../components/data-display/MetricCard";
import LoadingState from "../components/feedback/LoadingState";
import { TrainingBlockedState, TrainingRun, TrainingMetric } from "../types/training";
import { toast } from "sonner";
import { CapabilityDisabledError } from "../services/apiErrors";

const CHART_STYLE = { fontFamily: "var(--font-mono)", fontSize: 10, fill: "#6F7C89" };

function makeMetric(step: number): TrainingMetric {
  const progress = Math.min(step / 10000, 1);
  return {
    step,
    policyLoss: 2.5 - progress * 1.8 + Math.random() * 0.1,
    valueLoss: 1.2 - progress * 0.8 + Math.random() * 0.05,
    entropy: 1.8 - progress * 0.6 + Math.random() * 0.05,
    klDiv: 0.02 + Math.random() * 0.01,
    gradNorm: 0.5 + Math.random() * 0.2,
    reward: progress * 0.6 + Math.random() * 0.05,
    winRate: progress * 0.55 + Math.random() * 0.05,
    drawRate: 0.3 + Math.random() * 0.05,
    lossRate: (1 - progress) * 0.3 + Math.random() * 0.03,
    stepsPerSec: 420 + Math.random() * 30,
    gpuUtil: 0.87 + Math.random() * 0.08,
  };
}

export default function TrainingPage() {
  const { runId } = useParams<{ runId?: string }>();
  const ds = useDataSource();
  const navigate = useNavigate();

  const [blocked, setBlocked] = useState<TrainingBlockedState | null | undefined>(undefined);
  const [completedRuns, setCompletedRuns] = useState<TrainingRun[]>([]);
  const [activeRun, setActiveRun] = useState<TrainingRun | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const stepRef = useRef(0);

  const stopInterval = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
  }, []);

  useEffect(() => {
    ds.getTrainingBlockedState().then(setBlocked);
    ds.listTrainingRuns().then((runs) => {
      setCompletedRuns(runs.filter((r) => r.status === "complete"));
    });
  }, [ds]);

  useEffect(() => {
    if (!runId) return;
    ds.getTrainingRunById(runId).then((run) => {
      if (run) {
        setActiveRun(run);
        setMetrics(run.metrics);
      }
    });
  }, [ds, runId]);

  // Live metric updates when run is active
  useEffect(() => {
    if (!activeRun || activeRun.status !== "running") return;
    stopInterval();
    intervalRef.current = setInterval(async () => {
      stepRef.current += 250;
      const metric = makeMetric(stepRef.current);
      await ds.appendTrainingMetric(activeRun.id, metric);
      setMetrics((prev) => [...prev, metric]);
      if (stepRef.current >= activeRun.totalSteps) stopInterval();
    }, 800);
    return stopInterval;
  }, [activeRun?.id, activeRun?.status]);

  useEffect(() => () => stopInterval(), [stopInterval]);

  const handleSmoke = async () => {
    try {
      const id = await ds.startDemoTrainingRun("smoke");
      const run = await ds.getTrainingRunById(id);
      if (run) { run.status = "running"; setActiveRun(run); setMetrics([]); stepRef.current = 0; }
      navigate(`/training/${id}`);
    } catch (e) {
      const message =
        e instanceof CapabilityDisabledError
          ? e.message
          : e instanceof Error
            ? e.message
            : "Training smoke launch failed";
      toast.error(message);
    }
  };

  const handleStop = () => {
    stopInterval();
    if (activeRun) setActiveRun({ ...activeRun, status: "paused" });
  };

  if (blocked === undefined) return <LoadingState />;

  const latestMetric = metrics[metrics.length - 1];
  const policyData = metrics.map((m) => ({ step: m.step, loss: +m.policyLoss.toFixed(4) }));
  const winRateData = metrics.map((m) => ({ step: m.step, rate: +m.winRate.toFixed(4) }));

  return (
    <div>
      <PageHeader eyebrow="training/" title="Training" subtitle="Manage and monitor training runs." />

      {/* Blocked Banner */}
      {blocked && !runId && (
        <Panel className="mb-6 border-[#FFB000]">
          <DataSourceBadge kind="DEMO" />
          <div className="mb-3">
            <div
              className="text-[#FFB000] font-bold text-xl mb-2"
              style={{ fontFamily: "var(--font-display)" }}
            >
              TRAINING BLOCKED
            </div>
            <div className="text-[#CDD6DF] mb-2" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
              {blocked.reason}
            </div>
            <div className="text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>
              Gate failed: {blocked.gateFailedAt}
            </div>
            <div className="text-[#8593A1] mt-2" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>
              Required: {blocked.requiredAction}
            </div>
          </div>
          <button
            disabled
            title={blocked.requiredAction}
            className="py-2 px-4 text-xs font-bold uppercase tracking-wider rounded-sm border border-[#1E2630] text-[#6F7C89] bg-[#0C1116] cursor-not-allowed opacity-60"
            style={{ fontFamily: "var(--font-mono)" }}
          >
            START TRAINING (BLOCKED)
          </button>
        </Panel>
      )}

      {/* Smoke test button */}
      {!activeRun && (
        <Panel title="Demo Training" eyebrow="demo/" className="mb-6">
          <DataSourceBadge kind="DEMO" />
          <div className="text-[#8593A1] mb-3" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>
            Run a smoke test (10k steps) to verify the training pipeline.
          </div>
          <button
            onClick={handleSmoke}
            className="py-2 px-4 text-xs font-bold uppercase tracking-wider rounded-sm border border-[#FFB000] text-[#FFB000] bg-[#161C24] hover:bg-[#FFB000] hover:text-[#090D11] transition-colors"
            style={{ fontFamily: "var(--font-mono)" }}
          >
            SMOKE TEST
          </button>
        </Panel>
      )}

      {/* Active run metrics */}
      {activeRun && (
        <div className="mb-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-[#EAF0F6] font-bold" style={{ fontFamily: "var(--font-display)", fontSize: 14 }}>
                {activeRun.label}
              </span>
              <DataSourceBadge kind={activeRun.kind} pill />
              <StatusBadge
                variant={activeRun.status === "running" ? "warning" : activeRun.status === "complete" ? "success" : "neutral"}
                label={activeRun.status}
                dot
              />
            </div>
            {activeRun.status === "running" && (
              <button
                onClick={handleStop}
                className="py-1.5 px-3 text-xs font-bold uppercase tracking-wider rounded-sm border border-[#F85149] text-[#F85149] hover:bg-[#F85149] hover:text-white transition-colors"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                STOP
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <MetricCard label="Policy Loss" value={latestMetric?.policyLoss.toFixed(3) ?? "—"} />
            <MetricCard label="Value Loss" value={latestMetric?.valueLoss.toFixed(3) ?? "—"} />
            <MetricCard label="Entropy" value={latestMetric?.entropy.toFixed(3) ?? "—"} />
            <MetricCard label="Steps/sec" value={latestMetric ? Math.round(latestMetric.stepsPerSec) : "—"} />
            <MetricCard label="Win Rate" value={latestMetric ? `${(latestMetric.winRate * 100).toFixed(1)}%` : "—"} />
            <MetricCard label="GPU Util" value={latestMetric ? `${(latestMetric.gpuUtil * 100).toFixed(0)}%` : "—"} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Panel title="Policy Loss" eyebrow="chart/">
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={policyData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E2630" />
                  <XAxis dataKey="step" tick={CHART_STYLE} axisLine={false} tickLine={false} />
                  <YAxis tick={CHART_STYLE} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2630", fontFamily: "var(--font-mono)", fontSize: 10 }} />
                  <Line type="monotone" dataKey="loss" stroke="#FFB000" dot={false} strokeWidth={1.5} />
                </LineChart>
              </ResponsiveContainer>
            </Panel>
            <Panel title="Win Rate" eyebrow="chart/">
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={winRateData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E2630" />
                  <XAxis dataKey="step" tick={CHART_STYLE} axisLine={false} tickLine={false} />
                  <YAxis tick={CHART_STYLE} axisLine={false} tickLine={false} domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2630", fontFamily: "var(--font-mono)", fontSize: 10 }} />
                  <Line type="monotone" dataKey="rate" stroke="#3FB950" dot={false} strokeWidth={1.5} />
                </LineChart>
              </ResponsiveContainer>
            </Panel>
          </div>
        </div>
      )}

      {/* Completed runs table */}
      {completedRuns.length > 0 && (
        <Panel title="Completed Runs" eyebrow="history/">
          <div className="overflow-x-auto">
            <table className="w-full" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>
              <thead>
                <tr className="border-b border-[#1E2630]">
                  {["Name", "Kind", "Status", "Steps", "Checkpoint"].map((h) => (
                    <th key={h} className="text-left py-2 px-3 text-[#6F7C89] uppercase tracking-wider" style={{ fontSize: 9 }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {completedRuns.map((run) => (
                  <tr key={run.id} className="border-b border-[#1E2630] hover:bg-[#0C1116]">
                    <td className="py-2 px-3 text-[#EAF0F6]">{run.label}</td>
                    <td className="py-2 px-3"><DataSourceBadge kind={run.kind} pill /></td>
                    <td className="py-2 px-3">
                      <StatusBadge variant="success" label={run.status} dot />
                    </td>
                    <td className="py-2 px-3 text-[#8593A1]">{run.currentStep.toLocaleString()}</td>
                    <td className="py-2 px-3 text-[#22D3EE]">{run.checkpoint ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}
