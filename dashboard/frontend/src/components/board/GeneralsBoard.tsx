import React, { useMemo, useState, useRef, useCallback } from "react";
import { BoardState, CellState } from "../../types/match";
import { coordinateLabel } from "../../utils/gameBoard";

interface Props {
  board: BoardState;
  selectedSrc?: { row: number; col: number };
  selectedDst?: { row: number; col: number };
  onCellClick?: (row: number, col: number) => void;
  className?: string;
  interactive?: boolean;
  attributionOverlay?: number[][];
  beliefOverlay?: number[][];
  changedCells?: { row: number; col: number }[];
  riskRegion?: { row: number; col: number }[];
  pathCells?: { row: number; col: number }[];
  boardSummary?: string;
}

function cellFill(cell: CellState): string {
  if (!cell.visible) return "#090D11";
  switch (cell.terrain) {
    case "mountain":
      return "#1E2630";
    case "city":
      if (cell.owner === "player1") return "rgba(59,130,246,0.3)";
      if (cell.owner === "player2") return "rgba(239,68,68,0.3)";
      return "#1E2630";
    case "general":
      if (cell.owner === "player1") return "rgba(59,130,246,0.5)";
      if (cell.owner === "player2") return "rgba(239,68,68,0.5)";
      return "#1E2630";
    default:
      if (cell.owner === "player1") return "rgba(59,130,246,0.2)";
      if (cell.owner === "player2") return "rgba(239,68,68,0.2)";
      return "#0C1116";
  }
}

function cellSymbol(cell: CellState): string | null {
  if (!cell.visible) return "?";
  switch (cell.terrain) {
    case "mountain": return "▲";
    case "city": return "⬡";
    case "general": return "★";
    default: return null;
  }
}

interface TooltipState {
  row: number;
  col: number;
  x: number;
  y: number;
}

export default function GeneralsBoard({
  board, selectedSrc, selectedDst, onCellClick, className = "",
  interactive = true, attributionOverlay, beliefOverlay,
  changedCells, riskRegion, pathCells, boardSummary,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  const CELL = 28;
  const svgW = board.width * CELL;
  const svgH = board.height * CELL;

  // Static terrain layer — only changes when board dimensions change
  const terrainLayer = useMemo(() => {
    return board.cells.flatMap((row, r) =>
      row.map((cell, c) => {
        const sym = cellSymbol(cell);
        const fill = cellFill(cell);
        return (
          <g key={`t-${r}-${c}`}>
            <rect
              x={c * CELL}
              y={r * CELL}
              width={CELL}
              height={CELL}
              fill={fill}
              stroke="#1E2630"
              strokeWidth={0.5}
            />
            {sym && (
              <text
                x={c * CELL + CELL / 2}
                y={r * CELL + CELL / 2 + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={cell.terrain === "general" ? 12 : 10}
                fill={cell.terrain === "general" ? (cell.owner === "player1" ? "#3B82F6" : cell.owner === "player2" ? "#EF4444" : "#6F7C89") : "#6F7C89"}
                style={{ pointerEvents: "none", userSelect: "none" }}
              >
                {sym}
              </text>
            )}
          </g>
        );
      })
    );
  }, [board.width, board.height]);

  // Dynamic overlay: army counts and ownership coloring update every frame
  const dynamicLayer = useMemo(() => {
    return board.cells.flatMap((row, r) =>
      row.map((cell, c) => {
        const fill = cellFill(cell);
        const isSrc = selectedSrc?.row === r && selectedSrc?.col === c;
        const isDst = selectedDst?.row === r && selectedDst?.col === c;
        return (
          <g key={`d-${r}-${c}`}>
            <rect
              x={c * CELL}
              y={r * CELL}
              width={CELL}
              height={CELL}
              fill={fill}
              stroke={isSrc ? "#FFB000" : isDst ? "#22D3EE" : "#1E2630"}
              strokeWidth={isSrc || isDst ? 2 : 0.5}
              style={{ cursor: onCellClick ? "pointer" : "default" }}
              onClick={() => onCellClick?.(r, c)}
              onMouseEnter={(e) => {
                const rect = containerRef.current?.getBoundingClientRect();
                if (!rect) return;
                setTooltip({ row: r, col: c, x: e.clientX - rect.left, y: e.clientY - rect.top });
              }}
              onMouseLeave={() => setTooltip(null)}
            />
            {cell.visible && cell.armies > 0 && (
              <text
                x={c * CELL + CELL / 2}
                y={r * CELL + CELL / 2 + (cellSymbol(cell) ? 5 : 1)}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={8}
                fill="white"
                style={{ pointerEvents: "none", userSelect: "none" }}
              >
                {cell.armies}
              </text>
            )}
          </g>
        );
      })
    );
  }, [board, selectedSrc, selectedDst, onCellClick]);

  const hoveredCell =
    tooltip != null ? board.cells[tooltip.row]?.[tooltip.col] : null;

  return (
    <div ref={containerRef} className={`relative overflow-auto ${className}`}>
      <svg
        width={svgW}
        height={svgH}
        style={{ display: "block", imageRendering: "pixelated" }}
      >
        <g>{terrainLayer}</g>
        <g>{dynamicLayer}</g>
        {attributionOverlay && board.cells.map((row, r) =>
          row.map((_, c) => {
            const v = attributionOverlay[r]?.[c] ?? 0;
            if (v < 0.05) return null;
            return <rect key={`attr-${r}-${c}`} x={c * CELL} y={r * CELL} width={CELL} height={CELL} fill={`rgba(255,176,0,${v * 0.7})`} pointerEvents="none" />;
          })
        )}
        {beliefOverlay && board.cells.map((row, r) =>
          row.map((_, c) => {
            const v = beliefOverlay[r]?.[c] ?? 0;
            if (v < 0.05) return null;
            return <rect key={`belief-${r}-${c}`} x={c * CELL} y={r * CELL} width={CELL} height={CELL} fill={`rgba(34,211,238,${v * 0.5})`} pointerEvents="none" />;
          })
        )}
        {riskRegion?.map(({ row, col }) =>
          <rect key={`risk-${row}-${col}`} x={col * CELL} y={row * CELL} width={CELL} height={CELL} fill="rgba(248,81,73,0.25)" stroke="rgba(248,81,73,0.5)" strokeWidth={1} pointerEvents="none" />
        )}
        {pathCells?.map(({ row, col }) =>
          <rect key={`path-${row}-${col}`} x={col * CELL} y={row * CELL} width={CELL} height={CELL} fill="rgba(63,185,80,0.2)" stroke="rgba(63,185,80,0.5)" strokeWidth={1} pointerEvents="none" />
        )}
        {changedCells?.map(({ row, col }) =>
          <rect key={`changed-${row}-${col}`} x={col * CELL} y={row * CELL} width={CELL} height={CELL} fill="none" stroke="#FFB000" strokeWidth={1.5} pointerEvents="none" />
        )}
      </svg>
      {boardSummary && <p className="sr-only">{boardSummary}</p>}
      {tooltip && hoveredCell && (
        <div
          className="absolute z-50 pointer-events-none bg-[#11161C] border border-[#1E2630] rounded-sm px-2 py-1.5 shadow-lg"
          style={{
            left: tooltip.x + 10,
            top: tooltip.y + 10,
            fontFamily: "var(--font-mono)",
            fontSize: 10,
          }}
        >
          <div className="text-[#FFB000] font-bold mb-0.5">
            {coordinateLabel(tooltip.row, tooltip.col)}
          </div>
          <div className="text-[#8593A1]">Terrain: <span className="text-[#CDD6DF]">{hoveredCell.terrain}</span></div>
          <div className="text-[#8593A1]">Owner: <span className="text-[#CDD6DF]">{hoveredCell.owner}</span></div>
          <div className="text-[#8593A1]">Armies: <span className="text-[#CDD6DF]">{hoveredCell.armies}</span></div>
        </div>
      )}
    </div>
  );
}
