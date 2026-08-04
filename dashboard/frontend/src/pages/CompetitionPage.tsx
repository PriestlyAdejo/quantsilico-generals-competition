import { useEffect, useState } from "react";
import { useDataSource } from "../data/DataSourceContext";
import { ApiError } from "../data/types";
import { PageHeader, Panel } from "../components/data-display/Panel";
import { BackendUnavailable, LoadingState } from "../components/feedback/States";
import { ProvenanceBadge } from "../components/status/StatusBadge";
import { StatusBadge } from "../components/status/StatusBadge";

export default function CompetitionPage() {
  const ds = useDataSource();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [down, setDown] = useState(false);

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

  const active = (data.active_submission || {}) as Record<string, unknown>;
  const snap = (data.profile_snapshot || null) as Record<string, unknown> | null;
  const probe = (data.attribution_probe || {}) as Record<string, unknown>;

  return (
    <div>
      <PageHeader
        title="Competition"
        subtitle="Manual portal observations only. Snapshots are not live Elo."
      />
      <Panel title="Active submission">
        <div className="pre">{JSON.stringify(active, null, 2)}</div>
      </Panel>
      {snap ? (
        <Panel title="PORTAL PROFILE SNAPSHOT">
          <div className="row" style={{ marginBottom: "0.5rem" }}>
            <ProvenanceBadge
              provenance={String(snap.provenance || "MANUALLY_RECORDED")}
              observedAt={String(snap.observed_at || "")}
            />
            <StatusBadge value={String(snap.attribution_method || "UNATTRIBUTED")} />
          </div>
          <p className="muted">
            Observed {String(snap.observed_at)} · source {String(snap.source_reference)} · not live
          </p>
          <div className="pre">
            {`rank: ${snap.rank} / ${snap.of}
elo: ${snap.elo}
games: ${snap.games}
record: ${JSON.stringify(snap.record)}
`}
          </div>
        </Panel>
      ) : null}
      <Panel title="Attribution probe">
        <p>{String(probe.finding || "No probe")}</p>
        <p className="muted">{String(probe.warning || "")}</p>
      </Panel>
      <Panel title="Match archive">
        <p className="muted">
          Many matches remain UNATTRIBUTED or MANUAL_OPERATOR_ASSIGNMENT because the public portal does
          not expose package hash beside each replay.
        </p>
      </Panel>
    </div>
  );
}
