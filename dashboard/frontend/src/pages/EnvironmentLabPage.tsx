import React, { useCallback, useEffect, useState } from "react";
import GeneralsBoard from "../components/board/GeneralsBoard";
import DataSourceBadge from "../components/status/DataSourceBadge";
import PageHeader from "../components/typography/PageHeader";
import Panel from "../components/data-display/Panel";
import { generateBoard, applyMove } from "../utils/gameBoard";
import { BoardState, CellState, PlayerSlot, MapPreset } from "../types/match";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "../app/components/ui/select";
import { useDataSource } from "../app/DataSourceProvider";
import { toast } from "sonner";

type LabMode = "OFFICIAL" | "DEMO";

interface LabConfig {
  player1: PlayerSlot;
  player2: PlayerSlot;
  mapPreset: MapPreset;
}

type SessionPublic = {
  session_id?: string;
  seed?: number;
  map_preset?: string;
  created_at?: string;
  expires_at?: string;
  action_count?: number;
  turn?: number;
  events?: string[];
  board?: { width?: number; height?: number; cells?: unknown; turn?: number };
  telemetry?: {
    p1_armies?: number;
    p2_armies?: number;
    p1_land?: number;
    p2_land?: number;
    fog_pct?: number;
  };
  closed?: boolean;
  limits?: { max_concurrent?: number; ttl_s?: number; max_actions?: number };
};

const isForcedDemo = import.meta.env.VITE_DASHBOARD_DATA_MODE === "demo";

function countFog(board: BoardState): number {
  let fog = 0;
  let total = 0;
  for (const row of board.cells) {
    for (const cell of row) {
      total++;
      if (!cell.visible) fog++;
    }
  }
  return total > 0 ? Math.round((fog / total) * 100) : 0;
}

function sumArmies(board: BoardState, owner: "player1" | "player2"): number {
  return board.cells.flat().filter((c) => c.owner === owner).reduce((s, c) => s + c.armies, 0);
}

function mapApiBoard(session: SessionPublic): BoardState | null {
  const board = session.board;
  if (!board || !Array.isArray(board.cells)) return null;
  const cells = board.cells as CellState[][];
  const height = board.height ?? cells.length;
  const width = board.width ?? (cells[0]?.length ?? 0);
  if (!height || !width) return null;
  return {
    width,
    height,
    cells,
    turn: board.turn ?? session.turn ?? 0,
  };
}

function emptyBoard(size = 18): BoardState {
  return {
    width: size,
    height: size,
    turn: 0,
    cells: Array.from({ length: size }, () =>
      Array.from({ length: size }, () => ({
        terrain: "plain" as const,
        owner: "neutral" as const,
        armies: 0,
        visible: true,
      })),
    ),
  };
}

export default function EnvironmentLabPage() {
  const ds = useDataSource();
  const [mode, setMode] = useState<LabMode>(isForcedDemo ? "DEMO" : "OFFICIAL");
  const [config, setConfig] = useState<LabConfig>({
    player1: "manual",
    player2: "heuristic",
    mapPreset: "standard",
  });
  const [seed, setSeed] = useState(0);
  const [turn, setTurn] = useState(0);
  const [board, setBoard] = useState<BoardState>(() => generateBoard(18, 18, 0));
  const [selectedSrc, setSelectedSrc] = useState<{ row: number; col: number } | null>(null);
  const [events, setEvents] = useState<string[]>(["Lab initialised."]);
  const [session, setSession] = useState<SessionPublic | null>(null);
  const [busy, setBusy] = useState(false);
  const [officialError, setOfficialError] = useState<string | null>(null);

  const applySession = useCallback((s: SessionPublic) => {
    setSession(s);
    const mapped = mapApiBoard(s);
    if (mapped) {
      setBoard(mapped);
      setTurn(mapped.turn);
    }
    setEvents(Array.isArray(s.events) && s.events.length ? s.events : ["Session ready."]);
    setSelectedSrc(null);
    setOfficialError(null);
  }, []);

  const createOfficialSession = useCallback(async () => {
    setBusy(true);
    setOfficialError(null);
    try {
      if (session?.session_id) {
        try {
          await ds.getJson(`/api/environment/sessions/${encodeURIComponent(session.session_id)}`, {
            method: "DELETE",
          });
        } catch {
          /* ignore close errors */
        }
      }
      const created = await ds.getJson<SessionPublic>("/api/environment/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed, map_preset: config.mapPreset, ttl_s: 900 }),
      });
      applySession(created);
      toast.message(`Official session ${created.session_id ?? "created"}`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Session create failed";
      setOfficialError(msg);
      toast.error(msg);
      setBoard(emptyBoard());
      setEvents([`OFFICIAL session unavailable: ${msg}`]);
    } finally {
      setBusy(false);
    }
  }, [applySession, config.mapPreset, ds, seed, session?.session_id]);

  useEffect(() => {
    if (mode !== "OFFICIAL" || isForcedDemo) return;
    void createOfficialSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- create once on mode enter
  }, [mode]);

  useEffect(() => {
    if (mode === "DEMO") {
      setSession(null);
      setOfficialError(null);
      setTurn(0);
      setBoard(generateBoard(18, 18, 0));
      setEvents(["DEMO adapter initialised — synthetic boards only."]);
      setSelectedSrc(null);
    }
  }, [mode]);

  const handleCellClick = useCallback(
    async (row: number, col: number) => {
      if (config.player1 !== "manual") return;
      if (!selectedSrc) {
        const cell = board.cells[row]?.[col];
        if (cell && cell.owner === "player1" && cell.armies > 1) {
          setSelectedSrc({ row, col });
        }
        return;
      }
      if (selectedSrc.row === row && selectedSrc.col === col) {
        setSelectedSrc(null);
        return;
      }

      if (mode === "DEMO") {
        const newBoard = applyMove(board, selectedSrc.row, selectedSrc.col, row, col);
        const nextTurn = turn + 1;
        setBoard({ ...newBoard, turn: nextTurn });
        setTurn(nextTurn);
        setEvents((prev) => [
          ...prev,
          `Turn ${nextTurn}: Moved from (${selectedSrc.row},${selectedSrc.col}) → (${row},${col})`,
        ]);
        setSelectedSrc(null);
        return;
      }

      if (!session?.session_id) {
        toast.error("No official session");
        return;
      }
      setBusy(true);
      try {
        const updated = await ds.getJson<SessionPublic>(
          `/api/environment/sessions/${encodeURIComponent(session.session_id)}/step`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              src_row: selectedSrc.row,
              src_col: selectedSrc.col,
              dst_row: row,
              dst_col: col,
            }),
          },
        );
        applySession(updated);
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Illegal or failed step";
        setOfficialError(msg);
        toast.error(msg);
        setSelectedSrc(null);
      } finally {
        setBusy(false);
      }
    },
    [applySession, board, config.player1, ds, mode, selectedSrc, session?.session_id, turn],
  );

  const handleStepForward = async () => {
    if (mode === "DEMO") {
      const nextTurn = turn + 1;
      setBoard(generateBoard(18, 18, nextTurn));
      setTurn(nextTurn);
      setEvents((prev) => [...prev, `Step → turn ${nextTurn}`]);
      setSelectedSrc(null);
      return;
    }
    toast.message("Official mode advances via legal cell moves (or reset).");
  };

  const handleReset = async () => {
    if (mode === "DEMO") {
      setTurn(0);
      setBoard(generateBoard(18, 18, 0));
      setEvents(["DEMO lab reset."]);
      setSelectedSrc(null);
      return;
    }
    if (!session?.session_id) {
      await createOfficialSession();
      return;
    }
    setBusy(true);
    try {
      const updated = await ds.getJson<SessionPublic>(
        `/api/environment/sessions/${encodeURIComponent(session.session_id)}/reset`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ seed, map_preset: config.mapPreset }),
        },
      );
      applySession(updated);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Reset failed";
      setOfficialError(msg);
      toast.error(msg);
    } finally {
      setBusy(false);
    }
  };

  const fogPct =
    mode === "OFFICIAL" && session?.telemetry?.fog_pct != null
      ? Math.round(session.telemetry.fog_pct * 100)
      : countFog(board);
  const p1Armies =
    mode === "OFFICIAL" && session?.telemetry?.p1_armies != null
      ? session.telemetry.p1_armies
      : sumArmies(board, "player1");
  const p2Armies =
    mode === "OFFICIAL" && session?.telemetry?.p2_armies != null
      ? session.telemetry.p2_armies
      : sumArmies(board, "player2");

  const heatMax = Math.max(...board.cells.flat().map((c) => c.armies), 1);
  const ttlRemaining =
    session?.expires_at != null
      ? Math.max(0, Math.round((Date.parse(session.expires_at) - Date.now()) / 1000))
      : null;

  return (
    <div>
      <PageHeader
        eyebrow="environment-lab/"
        title="Environment Lab"
        subtitle={
          mode === "OFFICIAL"
            ? "Official GeneralsEnv interactive sessions."
            : "DEMO adapter — synthetic boards only."
        }
      />
      {mode === "DEMO" && <DataSourceBadge kind="DEMO" />}
      <p className="px-1 mb-2 text-[#6F7C89] font-mono text-[10px]">
        <a className="text-[#22D3EE] hover:underline" href="/documentation/env-official">
          About Environment Lab
        </a>
        {" · "}
        Guardrails: max 2 concurrent · TTL 15–60 min · max 5,000 actions · records under var/ (not committed)
      </p>
      <div className="flex gap-4">
        <div className="w-[240px] flex-shrink-0 space-y-3">
          <Panel title="Simulator" eyebrow="config/">
            <div className="space-y-3">
              <Field label="MODE">
                <Select
                  value={mode}
                  onValueChange={(v) => setMode(v as LabMode)}
                  disabled={isForcedDemo}
                >
                  <SelectTrigger className="bg-[#0C1116] border-[#1E2630] text-[#CDD6DF] h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#11161C] border-[#1E2630] text-[#CDD6DF] text-xs">
                    <SelectItem value="OFFICIAL">Official Environment</SelectItem>
                    <SelectItem value="DEMO">Demo Adapter</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
              {mode === "OFFICIAL" && (
                <Field label="SEED">
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value) || 0)}
                    className="w-full h-8 px-2 bg-[#0C1116] border border-[#1E2630] text-[#CDD6DF] text-xs font-mono rounded-sm"
                  />
                </Field>
              )}
              <Field label="PLAYER 1">
                <Select value={config.player1} onValueChange={(v) => setConfig((c) => ({ ...c, player1: v as PlayerSlot }))}>
                  <SelectTrigger className="bg-[#0C1116] border-[#1E2630] text-[#CDD6DF] h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#11161C] border-[#1E2630] text-[#CDD6DF] text-xs">
                    <SelectItem value="manual">Manual</SelectItem>
                    <SelectItem value="heuristic">Heuristic</SelectItem>
                    <SelectItem value="cnn_agent">CNN Agent</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
              <Field label="PLAYER 2">
                <Select value={config.player2} onValueChange={(v) => setConfig((c) => ({ ...c, player2: v as PlayerSlot }))}>
                  <SelectTrigger className="bg-[#0C1116] border-[#1E2630] text-[#CDD6DF] h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#11161C] border-[#1E2630] text-[#CDD6DF] text-xs">
                    <SelectItem value="heuristic">Heuristic</SelectItem>
                    <SelectItem value="cnn_agent">CNN Agent</SelectItem>
                    <SelectItem value="random">Random</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
              <Field label="MAP">
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
              <div className="flex gap-2 mt-2">
                <button
                  onClick={() => void handleStepForward()}
                  disabled={busy}
                  className="flex-1 py-1.5 text-xs font-bold uppercase tracking-wider rounded-sm border border-[#1E2630] text-[#8593A1] hover:border-[#FFB000] hover:text-[#FFB000] transition-colors disabled:opacity-40"
                  style={{ fontFamily: "var(--font-mono)" }}
                >
                  Step →
                </button>
                <button
                  onClick={() => void handleReset()}
                  disabled={busy}
                  className="flex-1 py-1.5 text-xs font-bold uppercase tracking-wider rounded-sm border border-[#1E2630] text-[#8593A1] hover:border-[#F85149] hover:text-[#F85149] transition-colors disabled:opacity-40"
                  style={{ fontFamily: "var(--font-mono)" }}
                >
                  Reset
                </button>
              </div>
              {mode === "OFFICIAL" && (
                <button
                  onClick={() => void createOfficialSession()}
                  disabled={busy}
                  className="w-full py-1.5 text-xs font-bold uppercase tracking-wider rounded-sm border border-[#22D3EE] text-[#22D3EE] hover:bg-[#22D3EE]/10 transition-colors disabled:opacity-40"
                  style={{ fontFamily: "var(--font-mono)" }}
                >
                  New Session
                </button>
              )}
              {config.player1 === "manual" && (
                <div className="text-[#6F7C89] mt-2" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>
                  Click a P1 cell to select source, then click destination.
                  {selectedSrc && (
                    <span className="block text-[#FFB000] mt-1">
                      Selected: ({selectedSrc.row},{selectedSrc.col})
                    </span>
                  )}
                </div>
              )}
              {mode === "OFFICIAL" && session?.session_id && (
                <div className="text-[#6F7C89] mt-2 space-y-1" style={{ fontFamily: "var(--font-mono)", fontSize: 9 }}>
                  <div>Session: <span className="text-[#CDD6DF]">{session.session_id.slice(0, 8)}…</span></div>
                  <div>Actions: {session.action_count ?? 0} / {session.limits?.max_actions ?? 5000}</div>
                  <div>TTL remaining: {ttlRemaining != null ? `${ttlRemaining}s` : "NOT RECORDED"}</div>
                </div>
              )}
              {officialError && (
                <div className="text-[#F85149] mt-2" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>
                  {officialError}
                </div>
              )}
            </div>
          </Panel>
        </div>

        <div className="flex-1 min-w-0">
          <Panel title={`Turn ${turn}`} eyebrow="board/">
            <div className="overflow-hidden min-w-0">
              <GeneralsBoard
                board={board}
                selectedSrc={selectedSrc ?? undefined}
                onCellClick={config.player1 === "manual" && !busy ? (r, c) => void handleCellClick(r, c) : undefined}
              />
            </div>
          </Panel>
        </div>

        <div className="w-[220px] flex-shrink-0 space-y-3">
          <Panel title="Env Stats" eyebrow="telemetry/">
            <div className="space-y-2">
              <Stat label="TURN" value={turn} />
              <Stat label="P1 ARMIES" value={p1Armies} />
              <Stat label="P2 ARMIES" value={p2Armies} />
              <Stat label="FOG COVERAGE" value={`${fogPct}%`} />
            </div>
            <div className="mt-3 border-t border-[#1E2630] pt-3">
              <div className="text-[#6F7C89] uppercase tracking-widest mb-2" style={{ fontFamily: "var(--font-mono)", fontSize: 9 }}>
                Events
              </div>
              <div className="space-y-1 max-h-40 overflow-y-auto">
                {events.slice(-15).map((ev, i) => (
                  <div key={i} className="text-[#8593A1]" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>
                    {ev}
                  </div>
                ))}
              </div>
            </div>
          </Panel>
        </div>
      </div>

      <div className="mt-4">
        <Panel title="Vectorised Env — Army Heatmap" eyebrow="vec-env/">
          <div className="overflow-auto">
            <div
              className="inline-grid gap-px"
              style={{ gridTemplateColumns: `repeat(${board.width}, 16px)` }}
            >
              {board.cells.flat().map((cell, i) => {
                const intensity = cell.armies / heatMax;
                const r = Math.round(255 * Math.min(intensity, 1));
                return (
                  <div
                    key={i}
                    title={`${cell.armies}`}
                    style={{
                      width: 16,
                      height: 16,
                      backgroundColor: `rgba(${r}, ${Math.round(r * 0.69)}, 0, ${0.2 + intensity * 0.8})`,
                      border: "1px solid #1E2630",
                    }}
                  />
                );
              })}
            </div>
          </div>
          <div className="mt-2 text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 9 }}>
            Amber intensity = army count (max {heatMax})
          </div>
        </Panel>
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
    <div className="flex items-center justify-between py-1 border-b border-[#1E2630]">
      <span className="text-[#6F7C89] uppercase tracking-widest" style={{ fontFamily: "var(--font-mono)", fontSize: 9 }}>
        {label}
      </span>
      <span className="text-[#EAF0F6] font-bold" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
        {value}
      </span>
    </div>
  );
}
