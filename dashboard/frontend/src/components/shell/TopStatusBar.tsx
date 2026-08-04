import { useEffect, useState } from "react";
import { useDataSource } from "../../data/DataSourceContext";
import type { OverviewResponse } from "../../data/types";
import { ApiError } from "../../data/types";
import { StatusBadge } from "../status/StatusBadge";

function short(sha?: string | null) {
  return sha ? String(sha).slice(0, 7) : "—";
}

export default function TopStatusBar() {
  const ds = useDataSource();
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [buildMismatch, setBuildMismatch] = useState<string | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    ds.getOverview(ac.signal)
      .then(setOverview)
      .catch((err: unknown) => {
        if (err instanceof ApiError && err.kind === "backend_unavailable") {
          setOverview(null);
        }
      });
    ds.getJson<{ mismatch?: boolean; warning?: string | null; frontend?: { commit?: string } }>(
      "/api/build-info",
      ac.signal,
    )
      .then((info) => {
        if (info.mismatch) setBuildMismatch(info.warning || "Frontend dist commit differs from HEAD");
        else setBuildMismatch(null);
      })
      .catch(() => undefined);
    return () => ac.abort();
  }, [ds]);

  const board = overview?.gate_board || {};
  const current = overview?.gate_status?.current;
  const pkg = overview?.active_submitted_package;
  const feCommit = (overview?.frontend_build as { commit?: string } | null | undefined)?.commit;

  return (
    <header className="top-status" aria-label="Repository status">
      <div className="brand">QuantSilico</div>
      <div className="top-status-primary">
        <span className="chip" title="Branch">
          {overview?.branch || "…"}
        </span>
        <span className="chip mono" title="Repository commit">
          repo {short(overview?.commit)}
        </span>
        {feCommit ? (
          <span className="chip mono" title="Frontend build commit">
            ui {short(feCommit)}
          </span>
        ) : (
          <span className="chip muted" title="No dist/build-info.json yet">
            ui not built
          </span>
        )}
        <span className="chip accent" title="Submitted candidate">
          {pkg?.candidate || "package…"}
        </span>
        <span className="chip">{overview?.research_phase || "phase…"}</span>
      </div>
      <div className="top-status-gates" aria-label="Current gates">
        <span className="chip">
          LEARNING <StatusBadge value={current?.learning_readiness || board.LEARNING_READINESS_GATE || "…"} />
        </span>
        <span className="chip">
          PORTAL <StatusBadge value={current?.portal_submission || board.PORTAL_SUBMISSION_GATE || "…"} />
        </span>
        <span className="chip">
          PROMO <StatusBadge value={current?.learned_promotion || board.LEARNED_PROMOTION_GATE || "NONE"} />
        </span>
        <span className="chip">jobs {overview?.active_jobs?.length ?? 0}</span>
      </div>
      {buildMismatch ? <div className="top-warning">{buildMismatch}</div> : null}
    </header>
  );
}
