import { useEffect, useState } from "react";
import { useDataSource } from "../data/DataSourceContext";
import type { OverviewResponse } from "../data/types";
import { ApiError } from "../data/types";
import { PageHeader, Panel, MetricStrip } from "../components/data-display/Panel";
import { BackendUnavailable, LoadingState, SchemaMismatch } from "../components/feedback/States";
import { StatusBadge } from "../components/status/StatusBadge";

export default function OverviewPage() {
  const ds = useDataSource();
  const [data, setData] = useState<OverviewResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

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

  if (error?.kind === "backend_unavailable") {
    return <BackendUnavailable onRetry={load} />;
  }
  if (error?.kind === "schema_mismatch") {
    return <SchemaMismatch detail={error.message} />;
  }
  if (!data) return <LoadingState />;

  const pkg = data.active_submitted_package;

  return (
    <div>
      <PageHeader
        title="Overview"
        subtitle="Live repository status and authentic research gates — not promotional fiction."
      />
      <MetricStrip
        items={[
          { label: "Branch", value: data.branch },
          { label: "Commit", value: data.commit.slice(0, 7) },
          { label: "Submitted", value: pkg.candidate },
          { label: "Learned champion", value: data.learned_champion_note },
        ]}
      />
      <Panel title="Gate board">
        <div className="row">
          {Object.entries(data.gate_status).map(([k, v]) => (
            <span key={k} className="chip">
              {k}: <StatusBadge value={String(v)} />
            </span>
          ))}
        </div>
        <p className="muted" style={{ marginTop: "0.75rem" }}>
          Research gates retain FAIL / NONE when that is the evidence. Dashboard integration gates are separate.
        </p>
      </Panel>
      <Panel title="Active submitted package">
        <div className="pre">
          {`candidate: ${pkg.candidate}
path: ${pkg.package_path}
sha256: ${pkg.package_sha256}
authoritative_policy_source_commit: ${pkg.authoritative_policy_source_commit}
embedded_bot_commit: ${pkg.embedded_bot_commit} (${pkg.embedded_metadata_status})
repository_completion_commit: ${pkg.repository_completion_commit}
`}
        </div>
        <p className="muted">{pkg.metadata_note}</p>
      </Panel>
      <Panel title="Learning smoke">
        <p className="muted">Phase 5/6 smoke passed. Not competitive performance.</p>
        <div className="pre">{JSON.stringify(data.learning_smoke || {}, null, 2)}</div>
      </Panel>
    </div>
  );
}
