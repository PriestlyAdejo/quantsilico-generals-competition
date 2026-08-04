import React, { useState, useRef, useCallback } from "react";
import { useNavigate } from "react-router";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { useDataSource } from "../app/DataSourceProvider";
import GeneralsBoard from "../components/board/GeneralsBoard";
import DataSourceBadge from "../components/status/DataSourceBadge";
import PageHeader from "../components/typography/PageHeader";
import Panel from "../components/data-display/Panel";
import LoadingState from "../components/feedback/LoadingState";
import { generateBoard } from "../utils/gameBoard";
import { BoardState, MatchFrame, PlayerSlot, MapPreset } from "../types/match";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "../app/components/ui/select";
import { Slider } from "../app/components/ui/slider";
import { toast } from "sonner";
import { CapabilityDisabledError } from "../services/apiErrors";

const CHART_STYLE = { fontFamily: "var(--font-mono)", fontSize: 10, fill: "#6F7C89" };

interface Config {
  player1: PlayerSlot;
  player2: PlayerSlot;
  mapPreset: MapPreset;
  mapSize: 18 | 19 | 20 | 21;
  speedMultiplier: number;
}

export default function ArenaPage() {
  const ds = useDataSource();
  const navigate = useNavigate();

  const [config, setConfig] = useState<Config>({
    player1: "cnn_agent",
    player2: "heuristic",
    mapPreset: "standard",
    mapSize: 18,
    speedMultiplier: 1,
  });
  const [running, setRunning] = useState(false);
  const [currentBoard, setCurrentBoard] = useState<BoardState | null>(null);
  const [frames, setFrames] = useState<MatchFrame[]>([]);
  const [events, setEvents] = useState<string[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const matchIdRef = useRef<string | null>(null);
  const turnRef = useRef(0);

  const stopInterval = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
  }, []);

  const handleRun = async () => {
    if (running) return;
    setRunning(true);
    setFrames([]);
    setEvents(["Match started"]);
    turnRef.current = 0;

    try {
      const matchId = await ds.createDemoMatch({
        ...config,
        label: `Arena — ${config.player1} vs ${config.player2}`,
      });
      matchIdRef.current = matchId;

      const intervalMs = Math.round(600 / config.speedMultiplier);
      intervalRef.current = setInterval(async () => {
        const turn = ++turnRef.current;
        const board = generateBoard(config.mapSize, config.mapSize, turn);
        const newEvents = turn % 5 === 0 ? [`Turn ${turn}: Territory expanded`] : [];
        const frame: MatchFrame = {
          turn,
          board,
          p1Armies: 20 + turn * 2,
          p2Armies: 18 + turn,
          p1Land: 10 + turn,
          p2Land: 9 + turn,
          events: newEvents,
        };
        await ds.appendMatchFrame(matchId, frame);
        setCurrentBoard(board);
        setFrames((prev) => [...prev, frame]);
        if (newEvents.length) setEvents((prev) => [...prev, ...newEvents]);

        if (turn >= 30) {
          stopInterval();
          await ds.completeDemoMatch(matchId, "player1_win");
          const replayId = await ds.createReplayFromMatch(matchId);
          setRunning(false);
          navigate(`/replay/${replayId}`);
        }
      }, intervalMs);
    } catch (e) {
      setRunning(false);
      const message =
        e instanceof CapabilityDisabledError
          ? e.message
          : e instanceof Error
            ? e.message
            : "Arena run failed";
      toast.error(message);
    }
  };

  const latestFrame = frames[frames.length - 1] ?? null;
  const armiesData = frames.map((f) => ({ turn: f.turn, P1: f.p1Armies, P2: f.p2Armies }));
  const landData = frames.map((f) => ({ turn: f.turn, P1: f.p1Land, P2: f.p2Land }));

  return (
    <div className="flex flex-col h-full">
      <PageHeader eyebrow="arena/" title="Arena" subtitle="Run demo matches between agents." />
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Config Panel */}
        <div className="w-[280px] flex-shrink-0 space-y-4">
          <Panel title="Match Config" eyebrow="config/">
            <div className="space-y-3">
              <DataSourceBadge kind="DEMO" />
              <Field label="PLAYER 1">
                <Select value={config.player1} onValueChange={(v) => setConfig((c) => ({ ...c, player1: v as PlayerSlot }))}>
                  <SelectTrigger className="bg-[#0C1116] border-[#1E2630] text-[#CDD6DF] h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#11161C] border-[#1E2630] text-[#CDD6DF] text-xs">
                    <SelectItem value="cnn_agent">Recurrent CNN — DEMO</SelectItem>
                    <SelectItem value="heuristic">Heuristic</SelectItem>
                    <SelectItem value="random">Legal Random</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
              <Field label="PLAYER 2">
                <Select value={config.player2} onValueChange={(v) => setConfig((c) => ({ ...c, player2: v as PlayerSlot }))}>
                  <SelectTrigger className="bg-[#0C1116] border-[#1E2630] text-[#CDD6DF] h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#11161C] border-[#1E2630] text-[#CDD6DF] text-xs">
                    <SelectItem value="cnn_agent">Recurrent CNN — DEMO</SelectItem>
                    <SelectItem value="heuristic">Heuristic</SelectItem>
                    <SelectItem value="random">Legal Random</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
              <Field label="MAP PRESET">
                <Select value={config.mapPreset} onValueChange={(v) => setConfig((c) => ({ ...c, mapPreset: v as MapPreset }))}>
                  <SelectTrigger className="bg-[#0C1116] border-[#1E2630] text-[#CDD6DF] h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#11161C] border-[#1E2630] text-[#CDD6DF] text-xs">
                    <SelectItem value="standard">Standard</SelectItem>
                    <SelectItem value="islands">Islands</SelectItem>
                    <SelectItem value="maze">Maze</SelectItem>
                    <SelectItem value="tournament">Tournament</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
              <Field label="MAP SIZE">
                <Select value={String(config.mapSize)} onValueChange={(v) => setConfig((c) => ({ ...c, mapSize: parseInt(v) as 18 | 19 | 20 | 21 }))}>
                  <SelectTrigger className="bg-[#0C1116] border-[#1E2630] text-[#CDD6DF] h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#11161C] border-[#1E2630] text-[#CDD6DF] text-xs">
                    <SelectItem value="18">18</SelectItem>
                    <SelectItem value="19">19</SelectItem>
                    <SelectItem value="20">20</SelectItem>
                    <SelectItem value="21">21</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
              <Field label={`SPEED: ${config.speedMultiplier}×`}>
                <Slider
                  min={0.5}
                  max={4}
                  step={0.5}
                  value={[config.speedMultiplier]}
                  onValueChange={([v]) => setConfig((c) => ({ ...c, speedMultiplier: v }))}
                  className="mt-1"
                />
              </Field>
              <button
                onClick={handleRun}
                disabled={running}
                className="w-full mt-2 py-2 text-xs font-bold uppercase tracking-wider rounded-sm border border-[#FFB000] text-[#FFB000] bg-[#161C24] hover:bg-[#FFB000] hover:text-[#090D11] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                {running ? "RUNNING…" : "RUN MATCH"}
              </button>
            </div>
          </Panel>
        </div>

        {/* Board Center */}
        <div className="flex-1 min-w-0 flex flex-col gap-4">
          <Panel title="Board" className="flex-1">
            {currentBoard ? (
              <div className="overflow-auto">
                <GeneralsBoard board={currentBoard} />
              </div>
            ) : (
              <div className="flex items-center justify-center h-48">
                <span className="text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
                  Configure and click RUN MATCH to start.
                </span>
              </div>
            )}
          </Panel>

          {/* Charts row */}
          {frames.length > 1 && (
            <div className="grid grid-cols-2 gap-4">
              <Panel title="Armies">
                <ResponsiveContainer width="100%" height={120}>
                  <LineChart data={armiesData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E2630" />
                    <XAxis dataKey="turn" tick={CHART_STYLE} axisLine={false} tickLine={false} />
                    <YAxis tick={CHART_STYLE} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2630", fontFamily: "var(--font-mono)", fontSize: 10 }} />
                    <Line type="monotone" dataKey="P1" stroke="#3B82F6" dot={false} strokeWidth={1.5} />
                    <Line type="monotone" dataKey="P2" stroke="#EF4444" dot={false} strokeWidth={1.5} />
                  </LineChart>
                </ResponsiveContainer>
              </Panel>
              <Panel title="Land">
                <ResponsiveContainer width="100%" height={120}>
                  <LineChart data={landData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E2630" />
                    <XAxis dataKey="turn" tick={CHART_STYLE} axisLine={false} tickLine={false} />
                    <YAxis tick={CHART_STYLE} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: "#11161C", border: "1px solid #1E2630", fontFamily: "var(--font-mono)", fontSize: 10 }} />
                    <Line type="monotone" dataKey="P1" stroke="#3B82F6" dot={false} strokeWidth={1.5} />
                    <Line type="monotone" dataKey="P2" stroke="#EF4444" dot={false} strokeWidth={1.5} />
                  </LineChart>
                </ResponsiveContainer>
              </Panel>
            </div>
          )}
        </div>

        {/* Telemetry Right */}
        <div className="w-[260px] flex-shrink-0">
          <Panel title="Telemetry" eyebrow="telem/" className="h-full">
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <Stat label="TURN" value={latestFrame?.turn ?? 0} />
                <Stat label="P1 LAND" value={latestFrame?.p1Land ?? 0} />
                <Stat label="P1 ARMIES" value={latestFrame?.p1Armies ?? 0} />
                <Stat label="P2 ARMIES" value={latestFrame?.p2Armies ?? 0} />
              </div>
              <div className="border-t border-[#1E2630] pt-3">
                <div className="text-[#6F7C89] uppercase tracking-widest mb-2" style={{ fontFamily: "var(--font-mono)", fontSize: 9 }}>
                  Events
                </div>
                <div className="space-y-1 max-h-48 overflow-y-auto">
                  {events.slice(-20).map((ev, i) => (
                    <div key={i} className="text-[#8593A1]" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>
                      {ev}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-[#6F7C89] uppercase tracking-widest mb-1" style={{ fontFamily: "var(--font-mono)", fontSize: 9 }}>
        {label}
      </div>
      {children}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-[#0C1116] border border-[#1E2630] rounded-sm px-2 py-1.5">
      <div className="text-[#6F7C89] uppercase tracking-widest" style={{ fontFamily: "var(--font-mono)", fontSize: 8 }}>
        {label}
      </div>
      <div className="text-[#EAF0F6] font-bold" style={{ fontFamily: "var(--font-mono)", fontSize: 13 }}>
        {value}
      </div>
    </div>
  );
}
