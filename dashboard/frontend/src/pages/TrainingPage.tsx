import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { useDataSource } from "../data/DataSourceContext";
import { ApiError } from "../data/types";
import { PageHeader, Panel } from "../components/data-display/Panel";
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

export default function TrainingPage() {
  const ds = useDataSource();
  const { runId } = useParams();
  const [tab, setTab] = useState<(typeof TABS)[number]>("Overview");
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [down, setDown] = useState(false);

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

  return (
    <div>
      <PageHeader
        title="Training Cockpit"
        subtitle="Smoke evidence only. No long campaign launch from this console."
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
      <Panel title={tab}>
        {tab === "Overview" ? (
          <>
            <p className="muted">{labels.bc_accuracies}</p>
            <p className="muted">{labels.ppo_smoke}</p>
            <div className="pre">
              {Object.entries(reports)
                .map(([arch, rep]) => `${arch} train_acc(smoke)=${rep.final_train_action_acc}`)
                .join("\n") || "No BC tiny report"}
            </div>
            <div className="warning" style={{ marginTop: "0.75rem" }}>
              {String(data.graph_latency_warning || "")}
            </div>
          </>
        ) : null}
        {tab === "Hardware" ? (
          <div className="pre">{JSON.stringify(smoke.official_venv_cpu_load || {}, null, 2)}</div>
        ) : null}
        {tab === "Evaluation" ? (
          <div className="pre">{JSON.stringify(smoke.equal_budget_dev_comparison || {}, null, 2)}</div>
        ) : null}
        {tab === "Logs" || tab === "Optimisation" || tab === "Performance" || tab === "Auxiliary Heads" ? (
          <div className="pre">{JSON.stringify(smoke.ppo || [], null, 2)}</div>
        ) : null}
      </Panel>
    </div>
  );
}
