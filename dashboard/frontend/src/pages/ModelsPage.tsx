import { useEffect, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useDataSource } from "../data/DataSourceContext";
import { ApiError } from "../data/types";
import { ChartCard, PageHeader, Panel } from "../components/data-display/Panel";
import { MetricChart } from "../components/data-display/MetricChart";
import { RawRecordDrawer } from "../components/feedback/RawRecordDrawer";
import { BackendUnavailable, EmptyState, LoadingState } from "../components/feedback/States";
import { StatusBadge } from "../components/status/StatusBadge";

export default function ModelsPage() {
  const ds = useDataSource();
  const { modelId } = useParams();
  const [params] = useSearchParams();
  const compare = params.get("compare");
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [down, setDown] = useState(false);
  const [rawOpen, setRawOpen] = useState(false);

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
  const latency = data.competition_size_latency_gate as {
    classification?: Record<string, string>;
    results?: Record<string, { board: string; dense: boolean; stats_ms: { p99: number } }[]>;
  } | undefined;
  const classification = latency?.classification || {};
  const cnnPoints = (latency?.results?.recurrent_cnn_v2 || []).map((r, i) => ({
    update: i,
    board: r.board,
    p99: r.stats_ms.p99,
  }));

  return (
    <div>
      <PageHeader
        eyebrow="$ MODELS /"
        title="Models"
        subtitle="Smoke-tested architectures are not champions. Lifecycle and role are separate dimensions."
      />
      <div className="warning" data-testid="graph-latency-warning">
        {warning}
      </div>
      <Panel title="Learned champion">
        <strong>{String(data.learned_champion_note || "NO LEARNED CHAMPION")}</strong>
      </Panel>
      <ChartCard
        title="Competition-size latency (p99 ms)"
        provenance={{
          manifestId: "competition_size_latency_gate.json",
          kind: "COMPETITION_SIZE_LATENCY_GATE",
          architecture: "recurrent_cnn_v2",
        }}
      >
        {cnnPoints.length ? (
          <MetricChart title="CNN p99 by board condition" points={cnnPoints} seriesKeys={["p99"]} xKey="update" />
        ) : (
          <EmptyState title="NOT RECORDED" detail="Run competition-size latency gate first." />
        )}
        <div className="pre" style={{ marginTop: "0.5rem" }}>
          {`recurrent_cnn_v2: ${classification.recurrent_cnn_v2 || "NOT RECORDED"}
recurrent_graph_belief_v2: ${classification.recurrent_graph_belief_v2 || "NOT RECORDED"}`}
        </div>
      </ChartCard>
      <Panel title="Registry" actions={<button type="button" className="btn ghost" onClick={() => setRawOpen(true)}>Raw</button>}>
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
      <RawRecordDrawer open={rawOpen} onClose={() => setRawOpen(false)} title="Models raw" recordId="models" schema="MODELS" provenance="LIVE_REPOSITORY" raw={data} />
    </div>
  );
}
