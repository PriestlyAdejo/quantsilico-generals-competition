import { FormEvent, useEffect, useRef, useState } from "react";
import { useDataSource } from "../data/DataSourceContext";
import type { CapabilitiesResponse, JobRecord } from "../data/types";
import { ApiError } from "../data/types";
import { PageHeader, Panel } from "../components/data-display/Panel";
import { BackendUnavailable, ErrorState, LoadingState } from "../components/feedback/States";
import { StatusBadge } from "../components/status/StatusBadge";

export default function ArenaPage() {
  const ds = useDataSource();
  const [caps, setCaps] = useState<CapabilitiesResponse | null>(null);
  const [candidates, setCandidates] = useState<string[]>([]);
  const [candidate, setCandidate] = useState("heuristic_v2f_plus_planner_terminal_form");
  const [opponent, setOpponent] = useState("expander");
  const [seed, setSeed] = useState(0);
  const [job, setJob] = useState<JobRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendDown, setBackendDown] = useState(false);
  const launching = useRef(false);

  useEffect(() => {
    const ac = new AbortController();
    Promise.all([
      ds.getCapabilities(ac.signal),
      ds.getJson<{ candidates: string[] }>("/api/jobs/allowlist", ac.signal),
    ])
      .then(([c, allow]) => {
        setCaps(c);
        setCandidates(allow.candidates || []);
      })
      .catch((err: unknown) => {
        if (err instanceof ApiError && err.kind === "backend_unavailable") setBackendDown(true);
        else setError(String(err));
      });
    return () => ac.abort();
  }, [ds]);

  if (backendDown) return <BackendUnavailable />;
  if (!caps) return <LoadingState />;

  const launchCap = caps.capabilities.arena_match_launch;
  const enabled = launchCap?.enabled ?? false;

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!enabled || launching.current) return;
    launching.current = true;
    setError(null);
    try {
      const result = await ds.launchMatch({
        candidate,
        opponent,
        seed,
        max_turns: 50,
        record_replay: true,
      });
      setJob(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      launching.current = false;
    }
  }

  return (
    <div>
      <PageHeader
        title="Arena"
        subtitle="Allowlisted local evaluator jobs only. No shell strings from the browser."
      />
      {!enabled ? (
        <div className="warning" style={{ marginBottom: "0.75rem" }}>
          Disabled: {launchCap?.reason || "Arena launch unavailable"}
        </div>
      ) : null}
      <Panel title="Launch match">
        <form className="form-grid" onSubmit={onSubmit}>
          <label>
            Candidate
            <select value={candidate} onChange={(e) => setCandidate(e.target.value)} disabled={!enabled}>
              {candidates.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label>
            Opponent
            <select value={opponent} onChange={(e) => setOpponent(e.target.value)} disabled={!enabled}>
              {candidates.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label>
            Seed
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              disabled={!enabled}
            />
          </label>
          <button className="btn primary" type="submit" disabled={!enabled}>
            Launch
          </button>
        </form>
        {error ? <ErrorState title="Launch failed" detail={error} /> : null}
      </Panel>
      {job ? (
        <Panel title="Job result">
          <div className="row">
            <span className="mono">{job.job_id}</span>
            <StatusBadge value={job.state} />
          </div>
          {job.state === "COMPLETED" ? (
            <div style={{ marginTop: "0.75rem" }}>
              <strong>MATCH COMPLETE</strong>
              {job.replay_status === "REPLAY_NOT_RECORDED" || !job.replay_id ? (
                <div className="muted">REPLAY NOT RECORDED</div>
              ) : (
                <div>
                  Replay: <a href={`/replay/${job.replay_id}`}>{job.replay_id}</a>
                </div>
              )}
            </div>
          ) : null}
          {job.error ? <div className="pre">{job.error}</div> : null}
          {job.match_record ? (
            <div className="pre" style={{ marginTop: "0.75rem" }}>
              {JSON.stringify(job.match_record, null, 2)}
            </div>
          ) : null}
        </Panel>
      ) : null}
    </div>
  );
}
