import React, { useState, useCallback } from "react";
import GeneralsBoard from "../components/board/GeneralsBoard";
import DataSourceBadge from "../components/status/DataSourceBadge";
import PageHeader from "../components/typography/PageHeader";
import Panel from "../components/data-display/Panel";
import { generateBoard, applyMove } from "../utils/gameBoard";
import { BoardState, CellState, PlayerSlot, MapPreset } from "../types/match";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "../app/components/ui/select";

interface LabConfig {
  player1: PlayerSlot;
  player2: PlayerSlot;
  mapPreset: MapPreset;
}

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

export default function EnvironmentLabPage() {
  const [config, setConfig] = useState<LabConfig>({
    player1: "manual",
    player2: "heuristic",
    mapPreset: "standard",
  });
  const [turn, setTurn] = useState(0);
  const [board, setBoard] = useState<BoardState>(() => generateBoard(18, 18, 0));
  const [selectedSrc, setSelectedSrc] = useState<{ row: number; col: number } | null>(null);
  const [events, setEvents] = useState<string[]>(["Lab initialised."]);

  const handleCellClick = useCallback(
    (row: number, col: number) => {
      if (config.player1 !== "manual") return;
      if (!selectedSrc) {
        // Select source — must own the cell
        const cell = board.cells[row]?.[col];
        if (cell && cell.owner === "player1" && cell.armies > 1) {
          setSelectedSrc({ row, col });
        }
      } else {
        // Apply move
        if (selectedSrc.row === row && selectedSrc.col === col) {
          setSelectedSrc(null);
          return;
        }
        const newBoard = applyMove(board, selectedSrc.row, selectedSrc.col, row, col);
        const nextTurn = turn + 1;
        setBoard({ ...newBoard, turn: nextTurn });
        setTurn(nextTurn);
        setEvents((prev) => [
          ...prev,
          `Turn ${nextTurn}: Moved from (${selectedSrc.row},${selectedSrc.col}) → (${row},${col})`,
        ]);
        setSelectedSrc(null);
      }
    },
    [config.player1, board, selectedSrc, turn],
  );

  const handleStepForward = () => {
    const nextTurn = turn + 1;
    setBoard(generateBoard(18, 18, nextTurn));
    setTurn(nextTurn);
    setEvents((prev) => [...prev, `Step → turn ${nextTurn}`]);
    setSelectedSrc(null);
  };

  const handleReset = () => {
    setTurn(0);
    setBoard(generateBoard(18, 18, 0));
    setEvents(["Lab reset."]);
    setSelectedSrc(null);
  };

  const fogPct = countFog(board);
  const p1Armies = sumArmies(board, "player1");
  const p2Armies = sumArmies(board, "player2");

  // Heat-map grid: army counts (0–40 range, amber intensity)
  const heatMax = Math.max(...board.cells.flat().map((c) => c.armies), 1);

  return (
    <div>
      <PageHeader eyebrow="environment-lab/" title="Environment Lab" subtitle="Interactive board exploration." />
      <DataSourceBadge kind="DEMO" />
      <div className="flex gap-4">
        {/* Controls */}
        <div className="w-[240px] flex-shrink-0 space-y-3">
          <Panel title="Simulator" eyebrow="config/">
            <div className="space-y-3">
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
                  onClick={handleStepForward}
                  className="flex-1 py-1.5 text-xs font-bold uppercase tracking-wider rounded-sm border border-[#1E2630] text-[#8593A1] hover:border-[#FFB000] hover:text-[#FFB000] transition-colors"
                  style={{ fontFamily: "var(--font-mono)" }}
                >
                  Step →
                </button>
                <button
                  onClick={handleReset}
                  className="flex-1 py-1.5 text-xs font-bold uppercase tracking-wider rounded-sm border border-[#1E2630] text-[#8593A1] hover:border-[#F85149] hover:text-[#F85149] transition-colors"
                  style={{ fontFamily: "var(--font-mono)" }}
                >
                  Reset
                </button>
              </div>
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
            </div>
          </Panel>
        </div>

        {/* Board */}
        <div className="flex-1 min-w-0">
          <Panel title={`Turn ${turn}`} eyebrow="board/">
            <div className="overflow-auto">
              <GeneralsBoard
                board={board}
                selectedSrc={selectedSrc ?? undefined}
                onCellClick={config.player1 === "manual" ? handleCellClick : undefined}
              />
            </div>
          </Panel>
        </div>

        {/* Telemetry */}
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

      {/* Heat-map */}
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
