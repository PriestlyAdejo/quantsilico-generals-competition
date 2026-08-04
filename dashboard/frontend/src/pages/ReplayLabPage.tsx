import React, { useEffect, useState } from "react";
import { useParams } from "react-router";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../app/components/ui/tabs";
import { Slider } from "../app/components/ui/slider";
import { Play, Pause, SkipBack, SkipForward } from "lucide-react";
import { useDataSource } from "../app/DataSourceProvider";
import { useReplayPlayer } from "../hooks/useReplayPlayer";
import GeneralsBoard from "../components/board/GeneralsBoard";
import DataSourceBadge from "../components/status/DataSourceBadge";
import PageHeader from "../components/typography/PageHeader";
import Panel from "../components/data-display/Panel";
import LoadingState from "../components/feedback/LoadingState";
import ErrorState from "../components/feedback/ErrorState";
import { ReplayRecord } from "../types/replay";

const CHART_STYLE = { fontFamily: "var(--font-mono)", fontSize: 10, fill: "#6F7C89" };
const SPEED_OPTIONS = [0.5, 1, 2, 4];

export default function ReplayLabPage() {
  const { replayId } = useParams<{ replayId?: string }>();
  const ds = useDataSource();
  const [replay, setReplay] = useState<ReplayRecord | null | undefined>(undefined);

  useEffect(() => {
    const id = replayId ?? "replay-demo-001";
    ds.getReplayById(id).then((r) => setReplay(r));
  }, [ds, replayId]);

  if (replay === undefined) return <LoadingState />;
  if (replay === null) return <ErrorState error={`Replay not found: ${replayId ?? "replay-demo-001"}`} />;

  return <ReplayViewer replay={replay} />;
}

function ReplayViewer({ replay }: { replay: ReplayRecord }) {
  const player = useReplayPlayer(replay.frames, 1);
  const { currentTurn, currentFrame, playing, speed, setSpeed, seekTo, togglePlay } = player;

  const armiesData = replay.frames.map((f) => ({
    turn: f.turn,
    P1: f.p1Armies,
    P2: f.p2Armies,
  }));

  const currentDecision = replay.decisions[currentTurn] ?? replay.decisions[0];

  // Event turns for scrubber markers
  const eventTurns = new Set(replay.events.map((e) => e.turn));

  return (
    <div>
      <PageHeader
        eyebrow="replay/"
        title={replay.label ?? `Replay ${replay.id}`}
        subtitle={`${replay.config.player1} vs ${replay.config.player2} — ${replay.outcome}`}
      />
      <DataSourceBadge kind={replay.kind} />

      <div className="flex gap-4">
        {/* Board + controls (center) */}
        <div className="flex-1 min-w-0 space-y-4">
          <Panel title="Board" eyebrow="board/">
            {currentFrame ? (
              <div className="overflow-auto">
                <GeneralsBoard board={currentFrame.board} />
              </div>
            ) : (
              <div className="h-40 flex items-center justify-center text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
                No frame data.
              </div>
            )}
          </Panel>

          {/* Controls bar */}
          <div className="flex items-center gap-3 bg-[#11161C] border border-[#1E2630] rounded-sm px-4 py-2">
            <button
              onClick={togglePlay}
              className="flex items-center justify-center w-8 h-8 border border-[#FFB000] text-[#FFB000] rounded-sm hover:bg-[#FFB000] hover:text-[#090D11] transition-colors flex-shrink-0"
            >
              {playing ? <Pause size={14} /> : <Play size={14} />}
            </button>

            {/* Speed selector */}
            <div className="flex gap-1">
              {SPEED_OPTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => setSpeed(s)}
                  className={`px-2 py-0.5 text-xs rounded-sm border transition-colors ${
                    speed === s
                      ? "border-[#FFB000] text-[#FFB000] bg-[#161C24]"
                      : "border-[#1E2630] text-[#6F7C89] hover:text-[#CDD6DF]"
                  }`}
                  style={{ fontFamily: "var(--font-mono)" }}
                >
                  {s}×
                </button>
              ))}
            </div>

            <span
              className="text-[#8593A1] ml-auto"
              style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}
            >
              Turn {currentTurn} / {replay.totalTurns - 1}
            </span>
          </div>

          {/* Scrubber */}
          <div className="bg-[#11161C] border border-[#1E2630] rounded-sm px-4 py-3 relative">
            <Slider
              min={0}
              max={Math.max(0, replay.frames.length - 1)}
              step={1}
              value={[currentTurn]}
              onValueChange={([v]) => seekTo(v)}
              className="w-full"
            />
            {/* Event markers */}
            <div className="relative mt-1" style={{ height: 8 }}>
              {replay.frames.length > 1 &&
                [...eventTurns].map((t) => (
                  <div
                    key={t}
                    className="absolute w-1.5 h-1.5 rounded-full bg-[#FFB000]"
                    style={{
                      left: `${(t / (replay.frames.length - 1)) * 100}%`,
                      transform: "translateX(-50%)",
                      top: 0,
                    }}
                    title={replay.events.find((e) => e.turn === t)?.label}
                  />
                ))}
            </div>
          </div>

          {/* Army chart */}
          <Panel title="P1 vs P2 Armies" eyebrow="chart/">
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={armiesData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E2630" />
                <XAxis dataKey="turn" tick={CHART_STYLE} axisLine={false} tickLine={false} />
                <YAxis tick={CHART_STYLE} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: "#11161C", border: "1px solid #1E2630", fontFamily: "var(--font-mono)", fontSize: 10 }}
                  labelStyle={{ color: "#EAF0F6" }}
                />
                <ReferenceLine x={currentTurn} stroke="#FFB000" strokeWidth={1} strokeDasharray="4 2" />
                <Line type="monotone" dataKey="P1" stroke="#3B82F6" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="P2" stroke="#EF4444" dot={false} strokeWidth={1.5} />
              </LineChart>
            </ResponsiveContainer>
          </Panel>
        </div>

        {/* Inspector right */}
        <div className="w-[300px] flex-shrink-0">
          <Panel className="h-full">
            <Tabs defaultValue="decision">
              <TabsList className="w-full grid grid-cols-3 bg-[#0C1116] border border-[#1E2630] rounded-sm mb-3 h-auto p-0.5">
                {["decision", "state", "raw"].map((tab) => (
                  <TabsTrigger
                    key={tab}
                    value={tab}
                    className="text-[9px] uppercase tracking-wider py-1 data-[state=active]:bg-[#161C24] data-[state=active]:text-[#FFB000] text-[#6F7C89] rounded-sm"
                    style={{ fontFamily: "var(--font-mono)" }}
                  >
                    {tab}
                  </TabsTrigger>
                ))}
              </TabsList>
              <TabsList className="w-full grid grid-cols-4 bg-[#0C1116] border border-[#1E2630] rounded-sm mb-4 h-auto p-0.5">
                {["memory", "beliefs", "risk", "explain"].map((tab) => (
                  <TabsTrigger
                    key={tab}
                    value={tab}
                    className="text-[9px] uppercase tracking-wider py-1 data-[state=active]:bg-[#161C24] data-[state=active]:text-[#FFB000] text-[#6F7C89] rounded-sm"
                    style={{ fontFamily: "var(--font-mono)" }}
                  >
                    {tab}
                  </TabsTrigger>
                ))}
              </TabsList>

              <TabsContent value="decision">
                {currentDecision ? (
                  <div className="space-y-2">
                    <InspectorRow label="SRC" value={`(${currentDecision.srcRow}, ${currentDecision.srcCol})`} />
                    <InspectorRow label="DST" value={`(${currentDecision.dstRow}, ${currentDecision.dstCol})`} />
                    <InspectorRow label="ARMIES MOVED" value={currentDecision.armiesMoved} />
                    <InspectorRow label="POLICY LOGIT" value={currentDecision.policyLogit.toFixed(3)} />
                    <InspectorRow label="VALUE EST." value={currentDecision.valueEstimate.toFixed(3)} />
                  </div>
                ) : (
                  <span className="text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>No decision data.</span>
                )}
              </TabsContent>

              <TabsContent value="state">
                {currentFrame ? (
                  <div className="space-y-2">
                    <InspectorRow label="DIMENSIONS" value={`${currentFrame.board.width}×${currentFrame.board.height}`} />
                    <InspectorRow label="P1 ARMIES" value={currentFrame.p1Armies} />
                    <InspectorRow label="P2 ARMIES" value={currentFrame.p2Armies} />
                    <InspectorRow label="P1 LAND" value={currentFrame.p1Land} />
                    <InspectorRow label="P2 LAND" value={currentFrame.p2Land} />
                    <InspectorRow label="TURN" value={currentFrame.turn} />
                  </div>
                ) : (
                  <span className="text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>No state data.</span>
                )}
              </TabsContent>

              <TabsContent value="memory">
                <span className="text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>(Pass 2)</span>
              </TabsContent>
              <TabsContent value="beliefs">
                <span className="text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>(Pass 2)</span>
              </TabsContent>
              <TabsContent value="risk">
                <span className="text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>(Pass 2)</span>
              </TabsContent>
              <TabsContent value="explain">
                <span className="text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>(Pass 2)</span>
              </TabsContent>

              <TabsContent value="raw">
                <pre
                  className="text-[#8593A1] overflow-auto max-h-96 text-[9px] leading-relaxed"
                  style={{ fontFamily: "var(--font-mono)" }}
                >
                  {JSON.stringify(currentFrame, null, 2)}
                </pre>
              </TabsContent>
            </Tabs>

            {/* Events list */}
            <div className="mt-4 border-t border-[#1E2630] pt-3">
              <div className="text-[#6F7C89] uppercase tracking-widest mb-2" style={{ fontFamily: "var(--font-mono)", fontSize: 9 }}>
                Events
              </div>
              <div className="space-y-1 max-h-40 overflow-y-auto">
                {replay.events.filter((e) => e.turn <= currentTurn).map((e, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span className="text-[#FFB000] flex-shrink-0" style={{ fontFamily: "var(--font-mono)", fontSize: 9 }}>
                      T{e.turn}
                    </span>
                    <span className="text-[#8593A1]" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>
                      {e.label}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function InspectorRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between py-1 border-b border-[#1E2630]">
      <span className="text-[#6F7C89] uppercase tracking-widest" style={{ fontFamily: "var(--font-mono)", fontSize: 9 }}>
        {label}
      </span>
      <span className="text-[#EAF0F6]" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>
        {value}
      </span>
    </div>
  );
}
