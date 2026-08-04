import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { useDataSource } from "../data/DataSourceContext";
import { ApiError } from "../data/types";
import { ChartCard, PageHeader, Panel } from "../components/data-display/Panel";
import { MetricChart } from "../components/data-display/MetricChart";
import { RawRecordDrawer } from "../components/feedback/RawRecordDrawer";
import { BackendUnavailable, LoadingState } from "../components/feedback/States";

const TABS = [
  "Overview",
  "Optimisation",
  "Performance",
  "Auxiliary Heads",
  "Hardware",
  "Evaluation",
  "Logs",
] as const;

type ChartDto = {
  id: string;
  title: string;
  series_keys: string[];
  points: Record<string, number | string>[];
  missing?: string[];
  note?: string | null;
  producer?: string;
};

export default function TrainingPage() {
  const ds = useDataSource();
  const { runId } = useParams();
  const [tab, setTab] = useState<(typeof TABS)[number]>("Overview");
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [down, setDown] = useState(false);
  const [rawOpen, setRawOpen] = useState(false);

  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/training", ac.signal)
      .then(setData)
      .catch((err: unknown) => {
        if (err instanceof ApiError && err.kind === "backend_unavailable") setDown(true);
      });
    return () => ac.abort();
  }, [ds]);

  if (down) return <BackendUnavailable />;
  if (!data) return <LoadingState />;

  const smoke = (data.smoke || {}) as Record<string, unknown>;
  const labels = (data.labels || {}) as Record<string, string>;
  const bc = smoke.bc_tiny as Record<string, unknown> | undefined;
  const reports = (bc?.reports || {}) as Record<string, { final_train_action_acc?: number }>;
  const charts = (data.charts as ChartDto[]) || [];
  const latency = smoke.competition_size_latency_gate as Record<string, unknown> | undefined;
  const campaigns = (data.campaigns as Record<string, unknown>[]) || [];

  return (
    <div>
      <PageHeader
        eyebrow="$ TRAINING /"
        title="Training Cockpit"
        subtitle="Producer charts only. No long campaign launch from this console."
      />
      {runId ? <p className="muted">Selected run: {runId}</p> : null}
      <div className="tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t}
            type="button"
            role="tab"
            className={tab === t ? "active" : undefined}
            aria-selected={tab === t}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>
      <Panel
        title={tab}
        actions={
          <button type="button" className="btn ghost" onClick={() => setRawOpen(true)}>
            View raw record
          </button>
        }
      >
        {tab === "Overview" ? (
          <>
            <p className="muted">{labels.bc_accuracies}</p>
            <p className="muted">{labels.ppo_smoke}</p>
            <p className="muted">{labels.charts}</p>
            <div className="metric-grid">
              {Object.entries(reports).map(([arch, rep]) => (
                <div key={arch} className="metric-card">
                  <div className="label">{arch}</div>
                  <div className="value">{String(rep.final_train_action_acc ?? "NOT RECORDED")}</div>
                  <div className="sublabel">BC smoke train_acc</div>
                </div>
              ))}
            </div>
            {campaigns.length ? (
              <ul className="stack">
                {campaigns.map((c) => (
                  <li key={String(c.id)}>
                    <strong>{String(c.kind)}</strong> · graph allowed={String(c.graph_training_allowed)} ·{" "}
                    {String(c.path)}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted">No DEVELOPMENT campaign summary yet.</p>
            )}
            <div className="warning" style={{ marginTop: "0.75rem" }}>
              {String(data.graph_latency_warning || "")}
            </div>
          </>
        ) : null}
        {tab === "Optimisation" || tab === "Performance" ? (
          <>
            {charts.length ? (
              charts.map((c) => (
                <ChartCard
                  key={c.id}
                  title={c.title}
                  provenance={{
                    manifestId: c.id,
                    kind: "PPO_TELEMETRY",
                    missingFields: c.missing,
                  }}
                >
                  <MetricChart
                    title={c.title}
                    points={c.points}
                    seriesKeys={c.series_keys}
                    missing={c.missing}
                    note={c.note}
                  />
                </ChartCard>
              ))
            ) : (
              <p className="muted">NOT RECORDED — no PPO producer charts</p>
            )}
          </>
        ) : null}
        {tab === "Hardware" ? (
          <ChartCard
            title="Competition-size latency gate"
            provenance={{
              manifestId: "competition_size_latency_gate.json",
              kind: "COMPETITION_SIZE_LATENCY_GATE",
            }}
          >
            <dl className="kv">
              <dt>CNN</dt>
              <dd>{String((latency?.classification as Record<string, string> | undefined)?.recurrent_cnn_v2 || "NOT RECORDED")}</dd>
              <dt>Graph</dt>
              <dd>
                {String(
                  (latency?.classification as Record<string, string> | undefined)?.recurrent_graph_belief_v2 ||
                    "NOT RECORDED",
                )}
              </dd>
            </dl>
          </ChartCard>
        ) : null}
        {tab === "Evaluation" ? (
          <p className="muted">Equal-budget comparison is available via raw drawer when recorded — not a win-rate claim.</p>
        ) : null}
        {tab === "Logs" || tab === "Auxiliary Heads" ? (
          <p className="muted">Structured log streams are NOT RECORDED. Use raw drawer for smoke manifests.</p>
        ) : null}
      </Panel>
      <RawRecordDrawer
        open={rawOpen}
        onClose={() => setRawOpen(false)}
        title="Training raw"
        recordId="training"
        schema="TRAINING_SMOKE_DASHBOARD"
        provenance="LIVE_REPOSITORY"
        raw={data}
      />
    </div>
  );
}
