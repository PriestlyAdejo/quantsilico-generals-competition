import React, { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../app/components/ui/tabs";
import { Slider } from "../app/components/ui/slider";
import { Play, Pause } from "lucide-react";
import { useDataSource } from "../app/DataSourceProvider";
import { useReplayPlayer } from "../hooks/useReplayPlayer";
import GeneralsBoard from "../components/board/GeneralsBoard";
import DataSourceBadge from "../components/status/DataSourceBadge";
import PageHeader from "../components/typography/PageHeader";
import Panel from "../components/data-display/Panel";
import LoadingState from "../components/feedback/LoadingState";
import { ReplayRecord } from "../types/replay";

const CHART_STYLE = { fontFamily: "var(--font-mono)", fontSize: 10, fill: "#6F7C89" };
const SPEED_OPTIONS = [0.5, 1, 2, 4];

export default function ReplayLabPage() {
  const { replayId } = useParams<{ replayId?: string }>();
  const ds = useDataSource();
  const navigate = useNavigate();
  const [replay, setReplay] = useState<ReplayRecord | null | undefined>(undefined);
  const [registry, setRegistry] = useState<ReplayRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const list = await ds.listReplays();
      if (cancelled) return;
      setRegistry(list);
      if (!replayId) {
        if (list.length === 0) {
          setReplay(null);
          setError("empty");
          return;
        }
        navigate(`/replay/${encodeURIComponent(list[list.length - 1].id)}`, { replace: true });
        return;
      }
      const found = await ds.getReplayById(replayId);
      if (cancelled) return;
      if (!found) {
        setReplay(null);
        setError("missing");
        return;
      }
      setError(null);
      setReplay(found);
    })();
    return () => { cancelled = true; };
  }, [ds, replayId, navigate]);

  if (replay === undefined) return <LoadingState />;

  if (error === "empty") {
    return (
      <div className="p-6 space-y-4">
        <PageHeader eyebrow="replay/" title="Replay Lab" subtitle="No recorded replays yet." />
        <Panel title="Empty registry" eyebrow="state/">
          <p className="text-[#8593A1] font-mono text-xs mb-3">Replays are produced by Arena match jobs with recording enabled.</p>
          <Link to="/arena" className="text-[#22D3EE] font-mono text-xs hover:underline">Open Arena</Link>
        </Panel>
      </div>
    );
  }

  if (error === "missing" || replay === null) {
    return (
      <div className="p-6 space-y-4">
        <PageHeader eyebrow="replay/" title="Replay Lab" subtitle={`Replay not found: ${replayId}`} />
        <Panel title="Recover" eyebrow="selector/">
          <p className="text-[#8593A1] font-mono text-xs mb-3">Choose a valid replay or load the latest.</p>
          <div className="flex flex-wrap gap-2 mb-3">
            {registry.slice(-8).reverse().map((r) => (
              <button
                key={r.id}
                type="button"
                className="px-2 py-1 border border-[#1E2630] text-[#CDD6DF] font-mono text-[10px] hover:border-[#FFB000]"
                onClick={() => navigate(`/replay/${encodeURIComponent(r.id)}`)}
              >
                {r.id}
              </button>
            ))}
          </div>
          {registry.length > 0 && (
            <button
              type="button"
              className="px-3 py-1.5 border border-[#FFB000] text-[#FFB000] font-mono text-xs"
              onClick={() => navigate(`/replay/${encodeURIComponent(registry[registry.length - 1].id)}`)}
            >
              Load latest replay
            </button>
          )}
        </Panel>
      </div>
    );
  }

  return <ReplayViewer replay={replay} registry={registry} onSelect={(id) => navigate(`/replay/${encodeURIComponent(id)}`)} />;
}

function ReplayViewer({
  replay,
  registry,
  onSelect,
}: {
  replay: ReplayRecord;
  registry: ReplayRecord[];
  onSelect: (id: string) => void;
}) {
  const player = useReplayPlayer(replay.frames, 1);
  const { currentTurn, currentFrame, playing, speed, setSpeed, seekTo, togglePlay } = player;
  const armiesData = replay.frames.map((f) => ({ turn: f.turn, P1: f.p1Armies, P2: f.p2Armies }));
  const currentDecision = replay.decisions[currentTurn] ?? replay.decisions[0];
  const eventTurns = new Set(replay.events.map((e) => e.turn));
  const hasFrames = replay.frames.length > 0;

  return (
    <div>
      <PageHeader
        eyebrow="replay/"
        title={replay.label ?? `Replay ${replay.id}`}
        subtitle={`${replay.config.player1} vs ${replay.config.player2} — ${replay.outcome}`}
      />
      <DataSourceBadge kind={replay.kind} />
      <p className="mb-2 text-[#6F7C89] font-mono text-[10px]">
        <a className="text-[#22D3EE] hover:underline" href="/documentation/replay">About Replay Lab</a>
      </p>
      <div className="my-3">
        <label className="text-[#6F7C89] font-mono text-[10px]">
          Replay selector
          <select
            className="block mt-1 bg-[#0C1116] border border-[#1E2630] text-[#EAF0F6] font-mono text-xs px-2 py-1.5 rounded-sm min-w-[16rem]"
            value={replay.id}
            onChange={(e) => onSelect(e.target.value)}
          >
            {registry.map((r) => (
              <option key={r.id} value={r.id}>{r.label ?? r.id}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="flex gap-4">
        <div className="flex-1 min-w-0 space-y-4">
          <Panel title="Board" eyebrow="board/">
            {hasFrames && currentFrame ? (
              <div className="min-w-0 overflow-hidden">
                <GeneralsBoard board={currentFrame.board} variant="replay" />
              </div>
            ) : (
              <div className="min-h-40 flex flex-col items-center justify-center text-[#6F7C89] gap-2" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
                <span>REPLAY FRAMES NOT RECORDED</span>
                <span className="text-[10px]">Match metadata remains available in the inspector.</span>
              </div>
            )}
          </Panel>

          {hasFrames && (
            <div className="flex items-center gap-3 bg-[#11161C] border border-[#1E2630] rounded-sm px-4 py-2">
              <button type="button" onClick={togglePlay} className="text-[#FFB000]">
                {playing ? <Pause size={14} /> : <Play size={14} />}
              </button>
              <Slider
                value={[currentTurn]}
                min={0}
                max={Math.max(replay.frames.length - 1, 0)}
                step={1}
                onValueChange={(v) => seekTo(v[0] ?? 0)}
                className="flex-1"
              />
              <div className="flex gap-1">
                {SPEED_OPTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setSpeed(s)}
                    className={`px-1.5 py-0.5 text-[10px] font-mono border ${speed === s ? "border-[#FFB000] text-[#FFB000]" : "border-[#1E2630] text-[#6F7C89]"}`}
                  >
                    {s}x
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="w-80 flex-shrink-0 space-y-4">
          <Panel title="Inspector" eyebrow="meta/">
            <div className="space-y-1 text-[11px]" style={{ fontFamily: "var(--font-mono)" }}>
              <div className="text-[#6F7C89]">ID: <span className="text-[#CDD6DF]">{replay.id}</span></div>
              <div className="text-[#6F7C89]">Turns: <span className="text-[#CDD6DF]">{replay.totalTurns}</span></div>
              <div className="text-[#6F7C89]">Outcome: <span className="text-[#CDD6DF]">{replay.outcome}</span></div>
              <div className="text-[#6F7C89]">Events: <span className="text-[#CDD6DF]">{replay.events.length}</span></div>
              <div className="text-[#6F7C89]">Event markers: <span className="text-[#CDD6DF]">{eventTurns.size}</span></div>
              {currentDecision && (
                <div className="text-[#6F7C89]">Decision turn: <span className="text-[#CDD6DF]">{currentDecision.turn}</span></div>
              )}
            </div>
          </Panel>
          {armiesData.length > 0 && (
            <Panel title="Armies" eyebrow="charts/">
              <ResponsiveContainer width="100%" height={120}>
                <LineChart data={armiesData}>
                  <CartesianGrid stroke="#1E2630" strokeDasharray="3 3" />
                  <XAxis dataKey="turn" tick={CHART_STYLE} />
                  <YAxis tick={CHART_STYLE} />
                  <Tooltip />
                  <ReferenceLine x={currentTurn} stroke="#FFB000" />
                  <Line type="monotone" dataKey="P1" stroke="#3B82F6" dot={false} />
                  <Line type="monotone" dataKey="P2" stroke="#EF4444" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Panel>
          )}
          <Tabs defaultValue="events">
            <TabsList>
              <TabsTrigger value="events">Events</TabsTrigger>
              <TabsTrigger value="raw">Raw</TabsTrigger>
            </TabsList>
            <TabsContent value="events">
              <div className="max-h-40 overflow-auto text-[10px] font-mono text-[#8593A1] space-y-1">
                {replay.events.length === 0 ? "NOT RECORDED" : replay.events.map((e, i) => (
                  <div key={i}>t{e.turn}: {e.label}</div>
                ))}
              </div>
            </TabsContent>
            <TabsContent value="raw">
              <pre className="max-h-40 overflow-auto text-[10px] font-mono text-[#6F7C89]">{JSON.stringify({ id: replay.id, totalTurns: replay.totalTurns, frames: replay.frames.length }, null, 2)}</pre>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
