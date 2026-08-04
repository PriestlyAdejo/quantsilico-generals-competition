import { useEffect, useState } from "react";
import { useDataSource } from "../data/DataSourceContext";
import { ApiError } from "../data/types";
import { PageHeader, Panel } from "../components/data-display/Panel";
import { BackendUnavailable, EmptyState, LoadingState } from "../components/feedback/States";
import { StatusBadge } from "../components/status/StatusBadge";

export default function QualificationPage() {
  const ds = useDataSource();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/qualification", ac.signal)
      .then(setData)
      .catch((err: unknown) => {
        if (err instanceof ApiError && err.kind === "backend_unavailable") setDown(true);
      });
    return () => ac.abort();
  }, [ds]);

  if (down) return <BackendUnavailable />;
  if (!data) return <LoadingState />;

  const gates = (data.gates || {}) as Record<string, string>;

  return (
    <div>
      <PageHeader title="Qualification" subtitle="Named gates only — never unqualified QUALIFIED." />
      <Panel title="Gate board">
        <div className="stack">
          {Object.entries(gates).map(([name, value]) => (
            <div key={name} className="row">
              <code>{name}</code>
              <StatusBadge value={value} />
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

export function PopulationPage() {
  const ds = useDataSource();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/population", ac.signal).then(setData).catch(() => undefined);
    return () => ac.abort();
  }, [ds]);
  if (!data) return <LoadingState />;
  return (
    <div>
      <PageHeader title="Population" />
      <EmptyState title={String(data.state)} detail={String(data.note || "")} />
    </div>
  );
}

export function ExplainabilityPage() {
  const ds = useDataSource();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/explainability", ac.signal).then(setData).catch(() => undefined);
    return () => ac.abort();
  }, [ds]);
  if (!data) return <LoadingState />;
  return (
    <div>
      <PageHeader title="Explainability" />
      <EmptyState title={String(data.state)} detail={String(data.note || "")} />
    </div>
  );
}

export function ChampionPage() {
  const ds = useDataSource();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/champion", ac.signal).then(setData).catch(() => undefined);
    return () => ac.abort();
  }, [ds]);
  if (!data) return <LoadingState />;
  return (
    <div>
      <PageHeader title="Champion" subtitle="Submitted heuristic remains active. No learned champion." />
      <Panel title="Roles">
        <div className="pre">
          {`heuristic_baseline: ${data.heuristic_baseline}
learned_champion: ${data.learned_champion_note}
`}
        </div>
      </Panel>
      <Panel title="Promotion checklist (incomplete)">
        <div className="pre">{JSON.stringify(data.promotion_checklist, null, 2)}</div>
      </Panel>
    </div>
  );
}

export function EnvironmentLabPage() {
  const ds = useDataSource();
  const [caps, setCaps] = useState<Record<string, unknown> | null>(null);
  const [env, setEnv] = useState<Record<string, unknown> | null>(null);
  useEffect(() => {
    const ac = new AbortController();
    Promise.all([ds.getCapabilities(ac.signal), ds.getJson("/api/environment", ac.signal)]).then(
      ([c, e]) => {
        setCaps(c as unknown as Record<string, unknown>);
        setEnv(e);
      },
    );
    return () => ac.abort();
  }, [ds]);
  if (!caps || !env) return <LoadingState />;
  const capabilities = (caps.capabilities || {}) as Record<string, { enabled: boolean; reason: string }>;
  return (
    <div>
      <PageHeader title="Environment Lab" subtitle="Read-only official/replay inspection until a session service exists." />
      <Panel title="Capabilities">
        <p>Reset: {capabilities.environment_reset?.enabled ? "enabled" : "disabled"}</p>
        <p className="muted">{capabilities.environment_reset?.reason}</p>
        <p>Step: {capabilities.environment_step?.enabled ? "enabled" : "disabled"}</p>
        <p className="muted">{capabilities.environment_step?.reason}</p>
        <button className="btn" type="button" disabled>
          Reset
        </button>{" "}
        <button className="btn" type="button" disabled>
          Step
        </button>
      </Panel>
      <EmptyState title={String(env.state)} detail={String(env.reason || "")} />
    </div>
  );
}

export function RepositoryPage() {
  const ds = useDataSource();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/repository", ac.signal).then(setData).catch(() => undefined);
    return () => ac.abort();
  }, [ds]);
  if (!data) return <LoadingState />;
  return (
    <div>
      <PageHeader title="Repository" subtitle="Read-only. No git mutation controls." />
      <Panel>
        <div className="pre">{JSON.stringify(data, null, 2)}</div>
      </Panel>
    </div>
  );
}

export function DocumentationPage() {
  return (
    <div>
      <PageHeader title="Documentation" subtitle="Structured console guidance (no raw HTML injection)." />
      <Panel title="Startup">
        <p>
          Use <code>scripts/dashboard/start.ps1</code> with <code>.venv-training</code>. API binds to
          127.0.0.1:8765.
        </p>
      </Panel>
      <Panel title="Gates">
        <p>
          Distinguish HEURISTIC_DEVELOPMENT_GATE, PRE_PPO_SUBMISSION_GATE, PORTAL_SUBMISSION_GATE,
          LEARNING_READINESS_GATE, and LEARNED_PROMOTION_GATE. Never show unqualified QUALIFIED.
        </p>
      </Panel>
      <Panel title="Attribution">
        <p>
          Use EXACT_PORTAL_VERSION_ID, EXACT_PACKAGE_HASH, INFERRED_ACTIVE_UPLOAD_WINDOW,
          MANUAL_OPERATOR_ASSIGNMENT, or UNATTRIBUTED. Do not call inferred attribution exact.
        </p>
      </Panel>
      <Panel title="Manual upload">
        <p>Uploads are operator-manual. Credentials never enter this application.</p>
      </Panel>
    </div>
  );
}

export function ExperimentsPage() {
  const ds = useDataSource();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/experiments", ac.signal).then(setData).catch(() => undefined);
    return () => ac.abort();
  }, [ds]);
  if (!data) return <LoadingState />;
  const experiments = (data.experiments as Record<string, unknown>[]) || [];
  return (
    <div>
      <PageHeader title="Experiments" />
      <Panel>
        <p className="muted">{experiments.length} manifests (missing values stay unavailable, not zero).</p>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>ID</th>
                <th>Kind</th>
                <th>Path</th>
              </tr>
            </thead>
            <tbody>
              {experiments.slice(0, 50).map((e) => (
                <tr key={String(e.id)}>
                  <td className="mono">{String(e.id)}</td>
                  <td>{String(e.kind)}</td>
                  <td className="mono">{String(e.path)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

export function ReplayLabPage() {
  const ds = useDataSource();
  const [list, setList] = useState<Record<string, unknown> | null>(null);
  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/replays", ac.signal).then(setList).catch(() => undefined);
    return () => ac.abort();
  }, [ds]);
  if (!list) return <LoadingState />;
  const replays = (list.replays as Record<string, unknown>[]) || [];
  return (
    <div>
      <PageHeader title="Replay Lab" subtitle="Real replay IDs only. Missing fields render NOT RECORDED." />
      {!replays.length ? (
        <EmptyState title="No private replays on disk" detail="Arena may complete with REPLAY NOT RECORDED." />
      ) : (
        <Panel>
          <ul>
            {replays.map((r) => (
              <li key={String(r.id)} className="mono">
                {String(r.id)}
              </li>
            ))}
          </ul>
        </Panel>
      )}
    </div>
  );
}

export function NotFoundPage() {
  return <EmptyState title="Not found" detail="No route matched this path." />;
}
