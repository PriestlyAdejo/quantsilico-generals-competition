import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useDataSource } from "../data/DataSourceContext";
import type { CapabilitiesResponse, JobRecord } from "../data/types";
import { ApiError } from "../data/types";
import { GeneralsBoard, type BoardFrame } from "../components/board/GeneralsBoard";
import { PageHeader, Panel } from "../components/data-display/Panel";
import { RawRecordDrawer } from "../components/feedback/RawRecordDrawer";
import { BackendUnavailable, ErrorState, LoadingState } from "../components/feedback/States";
import { StatusBadge } from "../components/status/StatusBadge";

function waitingBoard(seed: number): BoardFrame {
  const n = 18;
  const typeGrid = Array.from({ length: n }, (_, r) =>
    Array.from({ length: n }, (_, c) => ((r + c + seed) % 11 === 0 ? 0 : 1)),
  );
  return { mapKey: `waiting-${seed}`, height: n, width: n, typeGrid };
}

export default function ArenaPage() {
  const ds = useDataSource();
  const [caps, setCaps] = useState<CapabilitiesResponse | null>(null);
  const [candidates, setCandidates] = useState<string[]>([]);
  const [candidate, setCandidate] = useState("heuristic_v2f_plus_planner_terminal_fix");
  const [opponent, setOpponent] = useState("official_expander");
  const [seed, setSeed] = useState(0);
  const [maxTurns, setMaxTurns] = useState(50);
  const [job, setJob] = useState<JobRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendDown, setBackendDown] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [rawOpen, setRawOpen] = useState(false);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
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
        if (allow.candidates?.includes("official_expander")) setOpponent("official_expander");
        else if (allow.candidates?.includes("expander")) setOpponent("expander");
      })
      .catch((err: unknown) => {
        if (err instanceof ApiError && err.kind === "backend_unavailable") setBackendDown(true);
        else setError(String(err));
      });
    return () => ac.abort();
  }, [ds]);

  useEffect(() => {
    if (!job || !["QUEUED", "RUNNING"].includes(job.state) || !startedAt) return;
    const t = window.setInterval(() => setElapsed(Math.floor((Date.now() - startedAt) / 1000)), 500);
    return () => window.clearInterval(t);
  }, [job, startedAt]);

  useEffect(() => {
    if (!job || !["QUEUED", "RUNNING"].includes(job.state)) return;
    const ac = new AbortController();
    const poll = window.setInterval(() => {
      ds.getJob(job.job_id, ac.signal)
        .then(setJob)
        .catch(() => undefined);
    }, 1500);
    return () => {
      ac.abort();
      window.clearInterval(poll);
    };
  }, [ds, job?.job_id, job?.state]);

  const board = useMemo(() => waitingBoard(seed), [seed]);

  if (backendDown) return <BackendUnavailable />;
  if (!caps) return <LoadingState />;

  const launchCap = caps.capabilities.arena_match_launch;
  const enabled = (launchCap?.enabled ?? false) && !demoMode;

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!enabled || launching.current) return;
    launching.current = true;
    setError(null);
    setStartedAt(Date.now());
    setElapsed(0);
    try {
      const result = await ds.launchMatch({
        candidate,
        opponent,
        seed,
        max_turns: maxTurns,
        record_replay: true,
      });
      setJob(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      launching.current = false;
    }
  }

  const running = job && ["QUEUED", "RUNNING"].includes(job.state);
  const match = (job?.match_record || {}) as Record<string, unknown>;

  return (
    <div className="page-arena">
      <PageHeader
        eyebrow="$ ARENA /"
        title="Arena"
        subtitle="Allowlisted local evaluator jobs. Production mode never fabricates live board telemetry."
      />
      <div className="mode-toggle row gap">
        <button type="button" className={!demoMode ? "btn primary" : "btn ghost"} onClick={() => setDemoMode(false)}>
          OFFICIAL EVALUATOR
        </button>
        <button type="button" className={demoMode ? "btn primary" : "btn ghost"} onClick={() => setDemoMode(true)}>
          DEMO (visual only)
        </button>
      </div>
      {demoMode ? (
        <div className="banner warning">DEMO ADAPTER — Synthetic frontend behaviour. Does not launch evaluator jobs or write evidence.</div>
      ) : null}
      {!enabled && !demoMode ? (
        <div className="banner warning">Disabled: {launchCap?.reason || "Arena launch unavailable"}</div>
      ) : null}

      <div className="workspace-3col">
        <aside className="inspector">
          <Panel title="Configuration">
            <form className="form-stack" onSubmit={onSubmit}>
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
                <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} disabled={!enabled} />
              </label>
              <label>
                Max turns
                <input
                  type="number"
                  value={maxTurns}
                  min={1}
                  max={1200}
                  onChange={(e) => setMaxTurns(Number(e.target.value))}
                  disabled={!enabled}
                />
              </label>
              <button className="btn primary" type="submit" disabled={!enabled}>
                Launch
              </button>
            </form>
            {error ? <ErrorState title="Launch failed" detail={error} /> : null}
          </Panel>
        </aside>

        <section className="board-stage" aria-label="Match board">
          <GeneralsBoard frame={board} />
          {running ? (
            <p className="board-caption muted">
              Job {job.state} · elapsed {elapsed}s · live board telemetry not emitted by this evaluator. Waiting board is
              static (not synthetic animation).
            </p>
          ) : (
            <p className="board-caption muted">Board stage ready. Mid-match cell updates appear only if the backend emits them.</p>
          )}
        </section>

        <aside className="inspector">
          <Panel
            title="Telemetry"
            actions={
              job ? (
                <button type="button" className="btn ghost" onClick={() => setRawOpen(true)}>
                  Raw
                </button>
              ) : null
            }
          >
            {job ? (
              <div className="stack">
                <div className="row between">
                  <span className="mono">{job.job_id}</span>
                  <StatusBadge value={job.state} />
                </div>
                <dl className="kv">
                  <dt>Candidate</dt>
                  <dd className="mono">{job.candidate}</dd>
                  <dt>Opponent</dt>
                  <dd className="mono">{job.opponent}</dd>
                  <dt>Seed</dt>
                  <dd>{job.seed}</dd>
                  {match.winner != null ? (
                    <>
                      <dt>Winner</dt>
                      <dd>{String(match.winner)}</dd>
                    </>
                  ) : null}
                  {match.turns != null ? (
                    <>
                      <dt>Turns</dt>
                      <dd>{String(match.turns)}</dd>
                    </>
                  ) : null}
                </dl>
                {job.state === "COMPLETED" ? (
                  <div>
                    <strong>MATCH COMPLETE</strong>
                    {job.replay_id ? (
                      <div>
                        Replay: <Link to={`/replay/${job.replay_id}`}>{job.replay_id}</Link>
                      </div>
                    ) : (
                      <div className="muted">REPLAY NOT RECORDED</div>
                    )}
                  </div>
                ) : null}
                {job.error ? <p className="failure">{job.error}</p> : null}
                {job.notes?.length ? (
                  <ul className="log-tail">
                    {job.notes.map((n, i) => (
                      <li key={i}>{n}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="muted">Log tail empty until evaluator notes arrive.</p>
                )}
              </div>
            ) : (
              <p className="muted">Launch a match to populate job state, elapsed time, and result.</p>
            )}
          </Panel>
        </aside>
      </div>

      <RawRecordDrawer
        open={rawOpen}
        onClose={() => setRawOpen(false)}
        title="Arena job record"
        recordId={job?.job_id}
        schema="JobRecord"
        provenance="LOCAL_EVALUATOR"
        raw={job}
        formatted={
          job ? (
            <dl className="kv">
              <dt>State</dt>
              <dd>{job.state}</dd>
              <dt>Replay</dt>
              <dd>{job.replay_id || "NOT RECORDED"}</dd>
            </dl>
          ) : null
        }
      />
    </div>
  );
}
