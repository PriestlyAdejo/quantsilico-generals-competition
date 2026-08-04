import { useEffect, useState } from "react";
import { useDataSource } from "../data/DataSourceContext";
import { ApiError } from "../data/types";
import { MetricCard, PageHeader, Panel } from "../components/data-display/Panel";
import { RawRecordDrawer } from "../components/feedback/RawRecordDrawer";
import { BackendUnavailable, EmptyState, LoadingState } from "../components/feedback/States";
import { ProvenanceBadge, StatusBadge } from "../components/status/StatusBadge";

export default function CompetitionPage() {
  const ds = useDataSource();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [down, setDown] = useState(false);
  const [rawOpen, setRawOpen] = useState(false);

  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/competition", ac.signal)
      .then(setData)
      .catch((err: unknown) => {
        if (err instanceof ApiError && err.kind === "backend_unavailable") setDown(true);
      });
    return () => ac.abort();
  }, [ds]);

  if (down) return <BackendUnavailable />;
  if (!data) return <LoadingState />;

  const snap = (data.profile_snapshot || null) as Record<string, unknown> | null;
  const active = (data.active_portal_submission || data.active_submission || {}) as Record<string, unknown>;

  return (
    <div>
      <PageHeader
        eyebrow="$ COMPETITION /"
        title="Competition"
        subtitle="Official portal observations and manually recorded submissions."
      />
      <div className="banner warning">
        Portal profile snapshots are non-live. Do not imply exact per-match package attribution.
      </div>
      <div className="metric-grid">
        <MetricCard label="Active candidate" value={String(active.candidate_id || active.candidate || "—")} tone="accent" />
        <MetricCard label="Portal verdict" value={String(data.portal_verdict || "—")} />
        <MetricCard label="Attribution" value={String(snap?.attribution_method || "MANUAL_OPERATOR_ASSIGNMENT")} />
      </div>
      <div className="grid-2">
        <Panel title="Portal profile snapshot" actions={<button type="button" className="btn ghost" onClick={() => setRawOpen(true)}>Raw</button>}>
          {snap ? (
            <>
              <ProvenanceBadge provenance={String(snap.provenance || "MANUALLY_RECORDED")} observedAt={String(snap.observed_at || "")} />
              <dl className="kv">
                <dt>Live</dt>
                <dd>{String(snap.live ?? false)}</dd>
                <dt>Rank</dt>
                <dd>{String(snap.rank ?? "NOT RECORDED")}</dd>
                <dt>Source</dt>
                <dd className="mono">{String(snap.source_reference || "—")}</dd>
              </dl>
            </>
          ) : (
            <EmptyState title="NO PORTAL OBSERVATION IMPORTED" detail="Import a manual portal observation record to populate this card." />
          )}
        </Panel>
        <Panel title="Active submission summary">
          <dl className="kv">
            <dt>Package hash</dt>
            <dd className="mono">{String(active.package_sha256 || "—")}</dd>
            <dt>Config hash</dt>
            <dd className="mono">{String(active.config_hash || "—")}</dd>
            <dt>Gate at observation</dt>
            <dd>
              <StatusBadge value="HISTORICAL" /> upload-time only
            </dd>
          </dl>
        </Panel>
      </div>
      <RawRecordDrawer open={rawOpen} onClose={() => setRawOpen(false)} title="Competition raw" recordId="competition" schema="COMPETITION" provenance="PORTAL_OBSERVATION" raw={data} />
    </div>
  );
}
