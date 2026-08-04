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
