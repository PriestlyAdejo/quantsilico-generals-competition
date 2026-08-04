import { useEffect, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useDataSource } from "../data/DataSourceContext";
import { ApiError } from "../data/types";
import { PageHeader, Panel } from "../components/data-display/Panel";
import { BackendUnavailable, EmptyState, LoadingState } from "../components/feedback/States";
import { StatusBadge } from "../components/status/StatusBadge";

export default function ModelsPage() {
  const ds = useDataSource();
  const { modelId } = useParams();
  const [params] = useSearchParams();
  const compare = params.get("compare");
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/models", ac.signal)
      .then(setData)
      .catch((err: unknown) => {
        if (err instanceof ApiError && err.kind === "backend_unavailable") setDown(true);
      });
    return () => ac.abort();
  }, [ds]);

  if (down) return <BackendUnavailable />;
  if (!data) return <LoadingState />;

  const models = (data.models as Record<string, unknown>[]) || [];
  const warning = String(data.graph_latency_warning || "");
  const latency = data.competition_size_latency_gate as Record<string, unknown> | undefined;
  const classification = (latency?.classification || {}) as Record<string, string>;

  return (
    <div>
      <PageHeader
        title="Models"
        subtitle="Smoke-tested architectures are not champions. Lifecycle and role are separate dimensions."
      />
      <div className="warning" data-testid="graph-latency-warning">
        {warning}
      </div>
      <Panel title="Learned champion">
        <strong>{String(data.learned_champion_note || "NO LEARNED CHAMPION")}</strong>
      </Panel>
      <Panel title="Competition-size latency">
        {latency ? (
          <div className="pre">
            {`recurrent_cnn_v2: ${classification.recurrent_cnn_v2 || "NOT RECORDED"}
recurrent_graph_belief_v2: ${classification.recurrent_graph_belief_v2 || "NOT RECORDED"}`}
          </div>
        ) : (
          <EmptyState title="NOT RECORDED" detail="Run competition-size latency gate before claiming CPU readiness." />
        )}
      </Panel>
      <Panel title="Registry">
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>ID</th>
                <th>Architecture</th>
                <th>Lifecycle</th>
                <th>Role</th>
                <th>Delivery</th>
                <th>Compatibility</th>
                <th>Latency</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={String(m.id)} data-selected={modelId === m.id || undefined}>
                  <td className="mono">{String(m.id)}</td>
                  <td>{String(m.architecture)}</td>
                  <td>
                    <StatusBadge value={String(m.lifecycle)} />
                  </td>
                  <td>
                    <StatusBadge value={String(m.competitive_role)} />
                  </td>
                  <td>
                    <StatusBadge value={String(m.delivery_status)} />
                  </td>
                  <td>
                    <StatusBadge value={String(m.compatibility)} />
                  </td>
                  <td>
                    <StatusBadge value={String(m.competition_size_latency || "n/a")} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {compare ? <p className="muted">Compare query: {compare}</p> : null}
        {!models.length ? <EmptyState title="No model records" /> : null}
      </Panel>
    </div>
  );
}
