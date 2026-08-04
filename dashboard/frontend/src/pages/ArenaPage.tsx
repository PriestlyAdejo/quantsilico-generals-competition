import React, { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { useDataSource } from "../app/DataSourceProvider";
import GeneralsBoard from "../components/board/GeneralsBoard";
import DataSourceBadge from "../components/status/DataSourceBadge";
import PageHeader from "../components/typography/PageHeader";
import Panel from "../components/data-display/Panel";
import { generateBoard } from "../utils/gameBoard";
import { BoardState } from "../types/match";
import { SUBMITTED_CANDIDATE_ID } from "../types/common";
import { toast } from "sonner";
import { CapabilityDisabledError } from "../services/apiErrors";

type JobDict = {
  job_id?: string;
  state?: string;
  candidate?: string;
  opponent?: string;
  seed?: number;
  max_turns?: number;
  created_at?: string;
  updated_at?: string;
  error?: string | null;
  notes?: string[];
  match_record?: Record<string, unknown> | null;
  replay_id?: string | null;
  replay_status?: string | null;
};

const isDemoMode = import.meta.env.VITE_DASHBOARD_DATA_MODE === "demo";

export default function ArenaPage() {
  const ds = useDataSource();
  const navigate = useNavigate();
  const launchLock = useRef(false);

  const [candidates, setCandidates] = useState<string[]>([]);
  const [candidate, setCandidate] = useState(SUBMITTED_CANDIDATE_ID);
  const [opponent, setOpponent] = useState("expander");
  const [seed, setSeed] = useState(7);
  const [maxTurns, setMaxTurns] = useState(50);
  const [preview, setPreview] = useState<BoardState>(() => generateBoard(18, 18, 0));
  const [job, setJob] = useState<JobDict | null>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    if (isDemoMode) return;
    (async () => {
      try {
        const allow = await ds.getJson<{ candidates?: string[] }>("/api/jobs/allowlist");
        const list = (allow.candidates ?? []).filter(Boolean).sort();
        setCandidates(list);
        if (list.includes(SUBMITTED_CANDIDATE_ID)) setCandidate(SUBMITTED_CANDIDATE_ID);
        else if (list[0]) setCandidate(list[0]);
        if (list.includes("expander")) setOpponent("expander");
        else if (list.includes("hunter")) setOpponent("hunter");
        else if (list.find((x: string) => x !== candidate)) {
          setOpponent(list.find((x: string) => x !== SUBMITTED_CANDIDATE_ID) ?? list[0]);
        }
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Failed to load allowlist");
      }
    })();
  }, [ds]);

  useEffect(() => {
    setPreview(generateBoard(18, 18, seed));
  }, [seed]);

  const pollJob = useCallback(async (jobId: string) => {
    const started = Date.now();
    const hardTimeoutMs = 180_000;
    while (Date.now() - started < hardTimeoutMs) {
      const latest = await ds.getJson<JobDict>(`/api/jobs/${encodeURIComponent(jobId)}`);
      setJob(latest);
      const state = String(latest.state ?? "").toUpperCase();
      if (["COMPLETED", "FAILED", "CANCELLED"].includes(state)) {
        setRunning(false);
        if (state === "COMPLETED" && latest.replay_id) {
          navigate(`/replay/${encodeURIComponent(latest.replay_id)}`);
        }
        return;
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
    setRunning(false);
    toast.error("Arena smoke timed out");
  }, [ds, navigate]);

  const handleLaunch = async () => {
    if (running || launchLock.current) return;
    launchLock.current = true;
    setRunning(true);
    try {
      if (isDemoMode) {
        // Preserve Figma demo animation only in explicit demo mode
        const matchId = await ds.createDemoMatch({
          player1: "cnn_agent",
          player2: "heuristic",
          mapPreset: "standard",
          mapSize: 18,
          speedMultiplier: 1,
          label: "DEMO arena",
        });
        toast.message(`DEMO match ${matchId}`);
        setRunning(false);
        launchLock.current = false;
        return;
      }
      const created = await ds.getJson<JobDict>("/api/jobs/match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_type: "MATCH",
          candidate,
          opponent,
          seed,
          max_turns: maxTurns,
          record_replay: true,
        }),
      });
      setJob(created);
      const id = created.job_id;
      if (!id) throw new Error("No job_id returned");
      await pollJob(id);
    } catch (e) {
      setRunning(false);
      const message =
        e instanceof CapabilityDisabledError || e instanceof Error ? e.message : "Arena launch failed";
      toast.error(message);
    } finally {
      launchLock.current = false;
    }
  };

  const elapsed =
    job?.created_at && job?.updated_at
      ? `${Math.max(0, (Date.parse(job.updated_at) - Date.parse(job.created_at)) / 1000).toFixed(1)}s`
      : "NOT RECORDED";

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        eyebrow="arena/"
        title="Arena"
        subtitle={isDemoMode ? "DEMO mode — synthetic Figma match animation." : "Allowlisted local evaluator matches."}
      />
      {isDemoMode && <DataSourceBadge kind="DEMO" />}
      {!isDemoMode && (
        <p className="px-1 mb-2 text-[#6F7C89] font-mono text-[10px]">
          Production mode · <a className="text-[#22D3EE] hover:underline" href="/documentation/arena">About Arena</a>
        </p>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr_280px] gap-4 flex-1 min-h-0 p-1">
        <Panel title="Configuration" eyebrow="config/">
          <div className="space-y-3 text-xs" style={{ fontFamily: "var(--font-mono)" }}>
            <label className="block text-[#6F7C89]">
              Candidate
              <select
                className="mt-1 w-full bg-[#0C1116] border border-[#1E2630] text-[#EAF0F6] px-2 py-1.5"
                value={candidate}
                onChange={(e) => setCandidate(e.target.value)}
                disabled={isDemoMode}
              >
                {(candidates.length ? candidates : [SUBMITTED_CANDIDATE_ID]).map((id) => (
                  <option key={id} value={id}>{id}</option>
                ))}
              </select>
            </label>
            <label className="block text-[#6F7C89]">
              Opponent
              <select
                className="mt-1 w-full bg-[#0C1116] border border-[#1E2630] text-[#EAF0F6] px-2 py-1.5"
                value={opponent}
                onChange={(e) => setOpponent(e.target.value)}
                disabled={isDemoMode}
              >
                {(candidates.length ? candidates : ["expander", "hunter"]).map((id) => (
                  <option key={id} value={id}>{id}</option>
                ))}
              </select>
            </label>
            <label className="block text-[#6F7C89]">
              Seed
              <input
                type="number"
                className="mt-1 w-full bg-[#0C1116] border border-[#1E2630] text-[#EAF0F6] px-2 py-1.5"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
              />
            </label>
            <label className="block text-[#6F7C89]">
              Max turns
              <input
                type="number"
                className="mt-1 w-full bg-[#0C1116] border border-[#1E2630] text-[#EAF0F6] px-2 py-1.5"
                value={maxTurns}
                min={1}
                max={200}
                onChange={(e) => setMaxTurns(Number(e.target.value))}
              />
            </label>
            <button
              type="button"
              onClick={handleLaunch}
              disabled={running}
              className="w-full py-2 text-xs font-bold uppercase tracking-wider border border-[#FFB000] text-[#FFB000] hover:bg-[#FFB000] hover:text-[#090D11] disabled:opacity-50"
            >
              {running ? "Running…" : "Launch match"}
            </button>
          </div>
        </Panel>

        <Panel title="Board workspace" eyebrow="board/" className="min-w-0">
          {!job && (
            <>
              <p className="text-[#6F7C89] font-mono text-[10px] mb-2">Map preview (not live telemetry)</p>
              <div className="min-w-0 overflow-hidden max-w-xl mx-auto">
                <GeneralsBoard board={preview} variant="arena" />
              </div>
            </>
          )}
          {job && (
            <>
              <p className="text-[#FFB000] font-mono text-[11px] mb-2">
                LIVE BOARD TELEMETRY NOT EMITTED BY THIS EVALUATOR
              </p>
              <div className="min-w-0 overflow-hidden max-w-xl mx-auto opacity-80">
                <GeneralsBoard board={preview} variant="arena" />
              </div>
            </>
          )}
        </Panel>

        <Panel title="Job status" eyebrow="job/">
          {!job ? (
            <p className="text-[#6F7C89] font-mono text-xs">No job launched yet.</p>
          ) : (
            <div className="space-y-1 text-[11px]" style={{ fontFamily: "var(--font-mono)" }}>
              <div className="text-[#6F7C89]">Job ID: <span className="text-[#FFB000]">{job.job_id}</span></div>
              <div className="text-[#6F7C89]">State: <span className="text-[#EAF0F6]">{job.state}</span></div>
              <div className="text-[#6F7C89]">Elapsed: <span className="text-[#EAF0F6]">{elapsed}</span></div>
              <div className="text-[#6F7C89]">Candidate: <span className="text-[#EAF0F6]">{job.candidate}</span></div>
              <div className="text-[#6F7C89]">Opponent: <span className="text-[#EAF0F6]">{job.opponent}</span></div>
              <div className="text-[#6F7C89]">Seed: <span className="text-[#EAF0F6]">{job.seed}</span></div>
              <div className="text-[#6F7C89]">Max turns: <span className="text-[#EAF0F6]">{job.max_turns}</span></div>
              <div className="text-[#6F7C89]">Replay: <span className="text-[#EAF0F6]">{job.replay_id ?? job.replay_status ?? "NOT RECORDED"}</span></div>
              {job.error && <div className="text-[#F85149]">Error: {job.error}</div>}
              {job.match_record && (
                <pre className="mt-2 max-h-40 overflow-auto text-[10px] text-[#8593A1]">
                  {JSON.stringify(job.match_record, null, 2)}
                </pre>
              )}
              {Array.isArray(job.notes) && job.notes.length > 0 && (
                <div className="mt-2 text-[#8593A1]">{job.notes.map((n, i) => <div key={i}>{n}</div>)}</div>
              )}
              {String(job.state).toUpperCase() === "COMPLETED" && !job.replay_id && (
                <div className="mt-2 text-[#FFB000]">MATCH COMPLETE · REPLAY NOT RECORDED</div>
              )}
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
