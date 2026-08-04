import { useEffect, useState } from "react";
import { useDataSource } from "../data/DataSourceContext";
import type { OverviewResponse } from "../data/types";
import { ApiError } from "../data/types";
import { PageHeader, Panel, MetricStrip, MetricCard } from "../components/data-display/Panel";
import { RawRecordDrawer } from "../components/feedback/RawRecordDrawer";
import { BackendUnavailable, LoadingState, SchemaMismatch } from "../components/feedback/States";
import { StatusBadge, ProvenanceBadge } from "../components/status/StatusBadge";

export default function OverviewPage() {
  const ds = useDataSource();
  const [data, setData] = useState<OverviewResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [rawOpen, setRawOpen] = useState(false);

  const load = () => {
    const ac = new AbortController();
    setError(null);
    ds.getOverview(ac.signal)
      .then(setData)
      .catch((err: unknown) => {
        if (err instanceof ApiError) setError(err);
        else setError(new ApiError("http", String(err)));
      });
    return () => ac.abort();
  };

  useEffect(load, [ds]);

  if (error?.kind === "backend_unavailable") return <BackendUnavailable onRetry={load} />;
  if (error?.kind === "schema_mismatch") return <SchemaMismatch detail={error.message} />;
  if (!data) return <LoadingState />;

  const pkg = data.active_submitted_package;
  const current = data.gate_status?.current;
  const board = data.gate_board || {};
  const m = data.metrics || {};
  const hist = data.gate_status?.historical_observations || [];

  return (
    <div className="page-overview">
      <PageHeader
        eyebrow="$ OVERVIEW /"
        title="Overview"
        subtitle="Current project state and authentic research gates — not promotional fiction."
      />
      <div className="row gap">
        <ProvenanceBadge provenance="LIVE_REPOSITORY" />
        <button type="button" className="btn ghost" onClick={() => setRawOpen(true)}>
          View raw record
        </button>
      </div>

      {current?.heuristic_development === "FAIL" ? (
        <div className="banner failure" role="status">
          <strong>DEVELOPMENT GATE FAIL</strong>
          <span className="muted">
            {" "}
            Heuristic development discovery suite remains FAIL. Learning readiness is a separate gate (
            {current.learning_readiness}). Learned promotion stays {current.learned_promotion}.
          </span>
        </div>
      ) : null}

      <div className="metric-grid">
        <MetricCard label="Submitted candidate" value={String(m.submitted_candidate || pkg.candidate)} tone="accent" />
        <MetricCard
          label="Learning readiness"
          value={String(m.learning_readiness || current?.learning_readiness || "…")}
          sublabel="Current manifest"
          tone={String(m.learning_readiness) === "PASS" ? "success" : "muted"}
        />
        <MetricCard label="CNN latency" value={String(m.cnn_latency || "NOT RECORDED")} sublabel="Competition-size" />
        <MetricCard label="Graph latency" value={String(m.graph_latency || "NOT RECORDED")} sublabel="Competition-size" />
        <MetricCard
          label="Learned promotion"
          value={String(m.learned_promotion || current?.learned_promotion || "NONE")}
          sublabel="No learned champion"
        />
        <MetricCard
          label="DEVELOPMENT PPO"
          value={m.development_campaign ? "RECORDED" : "NOT RECORDED"}
          sublabel={m.development_elapsed_s != null ? `${Number(m.development_elapsed_s).toFixed(0)}s wall` : undefined}
        />
      </div>

      <div className="grid-2">
        <Panel title="Current gate board">
          <div className="stack">
            {Object.entries(board).map(([k, v]) => (
              <div key={k} className="row between">
                <code>{k}</code>
                <StatusBadge value={String(v)} />
              </div>
            ))}
          </div>
          <p className="muted">
            Top bar and this board use <strong>current</strong> manifests only. Upload-time PENDING_AT_RECORD_TIME is
            historical.
          </p>
        </Panel>
        <Panel title="Active jobs">
          {data.active_jobs?.length ? (
            <ul className="stack">
              {data.active_jobs.map((j) => (
                <li key={j.job_id} className="mono">
                  {j.job_id} · {j.state} · {j.candidate} vs {j.opponent}
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">No active jobs.</p>
          )}
        </Panel>
      </div>

      <Panel title="Submitted package">
        <MetricStrip
          items={[
            { label: "Lifecycle", value: pkg.lifecycle || "SUBMITTED" },
            { label: "Policy source", value: pkg.authoritative_policy_source_commit },
            { label: "Embedded bot_commit", value: `${pkg.embedded_bot_commit} (${pkg.embedded_metadata_status})` },
            { label: "Config hash", value: pkg.config_hash || "—" },
          ]}
        />
        <p className="muted">{pkg.metadata_note}</p>
      </Panel>

      <Panel title="Historical upload observation">
        {hist.length ? (
          <ul className="stack">
            {hist.map((h, i) => (
              <li key={i}>
                <StatusBadge value="HISTORICAL" />{" "}
                <span className="muted">
                  {String(h.source)} @ {String(h.observed_at || "unknown")} — learning_readiness at record time:{" "}
                  {String(h.learning_readiness)}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">No historical gate observations recorded.</p>
        )}
      </Panel>

      <RawRecordDrawer
        open={rawOpen}
        onClose={() => setRawOpen(false)}
        title="Overview raw record"
        recordId="overview"
        schema={`schema_version=${data.schema_version}`}
        provenance="LIVE_REPOSITORY"
        formatted={
          <dl className="kv">
            <dt>Branch</dt>
            <dd className="mono">{data.branch}</dd>
            <dt>Commit</dt>
            <dd className="mono">{data.commit}</dd>
            <dt>Champion</dt>
            <dd>{data.learned_champion_note}</dd>
          </dl>
        }
        raw={data}
      />
    </div>
  );
}
