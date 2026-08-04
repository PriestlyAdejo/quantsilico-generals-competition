import { useEffect, useState } from "react";
import { useDataSource } from "../../data/DataSourceContext";
import type { OverviewResponse } from "../../data/types";
import { ApiError } from "../../data/types";
import { StatusBadge } from "../status/StatusBadge";

function short(sha?: string) {
  return sha ? sha.slice(0, 7) : "—";
}

export default function TopStatusBar() {
  const ds = useDataSource();
  const [overview, setOverview] = useState<OverviewResponse | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    ds.getOverview(ac.signal)
      .then(setOverview)
      .catch((err: unknown) => {
        if (err instanceof ApiError && err.kind === "backend_unavailable") {
          setOverview(null);
        }
      });
    return () => ac.abort();
  }, [ds]);

  const gates = overview?.gate_status || {};
  const pkg = overview?.active_submitted_package;

  return (
    <header className="top-status" aria-label="Repository status">
      <div className="brand">QuantSilico</div>
      <span className="chip">{overview?.branch || "…"}</span>
      <span className="chip mono">{short(overview?.commit)}</span>
      <span className="chip mono">engine {short(overview?.engine_commit)}</span>
      <span className="chip">{pkg?.candidate || "package…"}</span>
      <span className="chip">{overview?.research_phase || "phase…"}</span>
      <span className="chip">jobs {overview?.active_jobs?.length ?? 0}</span>
      {Object.entries(gates).map(([name, value]) => (
        <span key={name} className="chip" title={name}>
          {name.replace(/_GATE$/, "")} <StatusBadge value={String(value)} />
        </span>
      ))}
    </header>
  );
}
