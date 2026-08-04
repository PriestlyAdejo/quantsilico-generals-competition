import React, { useEffect, useState, useCallback } from "react";
import { useDataSource } from "../app/DataSourceProvider";
import { PopulationEntry, PayoffMatrix } from "../types/population";
import DataSourceBadge from "../components/status/DataSourceBadge";
import DateTimeCell from "../components/data-display/DateTimeCell";
import { shortDisplayName } from "../utils/displayNames";

function cellColor(v: number | null): string {
  if (v === null) return "#1A1F28";
  if (v < 0.4) return `rgba(248,81,73,${0.4 + (0.4 - v) * 1.2})`;
  if (v > 0.6) return `rgba(63,185,80,${0.4 + (v - 0.6) * 1.2})`;
  return `rgba(133,147,161,${0.3 + Math.abs(v - 0.5) * 0.5})`;
}

export default function PopulationPage() {
  const ds = useDataSource();
  const [entries, setEntries] = useState<PopulationEntry[]>([]);
  const [matrix, setMatrix] = useState<PayoffMatrix | null>(null);
  const [hovered, setHovered] = useState<{ r: number; c: number } | null>(null);
  const [focusCell, setFocusCell] = useState<{ r: number; c: number }>({ r: 0, c: 0 });

  useEffect(() => {
    ds.getPopulationSummary().then(({ entries: e, matrix: m }) => {
      setEntries(e);
      setMatrix(m);
    });
  }, [ds]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (!matrix) return;
    const n = matrix.agents.length;
    setFocusCell(prev => {
      if (e.key === "ArrowRight") return { r: prev.r, c: Math.min(prev.c + 1, n - 1) };
      if (e.key === "ArrowLeft") return { r: prev.r, c: Math.max(prev.c - 1, 0) };
      if (e.key === "ArrowDown") return { r: Math.min(prev.r + 1, n - 1), c: prev.c };
      if (e.key === "ArrowUp") return { r: Math.max(prev.r - 1, 0), c: prev.c };
      return prev;
    });
    if (["ArrowRight","ArrowLeft","ArrowDown","ArrowUp"].includes(e.key)) e.preventDefault();
  }, [matrix]);

  if (!matrix) return (
    <div className="p-6">
      <p className="text-[#FFB000] font-mono text-xs uppercase tracking-widest mb-1">$ population/</p>
      <h1 className="text-2xl font-bold text-[#EAF0F6]" style={{ fontFamily: "var(--font-heading)" }}>Population</h1>
      <p className="text-[#8593A1] text-sm mt-4">Loading…</p>
    </div>
  );

  const n = matrix.agents.length;
  const cellSize = 72;

  const focusedValue = matrix.matrix[focusCell.r]?.[focusCell.c];
  const hoveredValue = hovered ? matrix.matrix[hovered.r]?.[hovered.c] : null;

  return (
    <div className="p-6 space-y-6">
      <header>
        <p className="text-[#FFB000] font-mono text-xs uppercase tracking-widest mb-1">$ population/</p>
        <h1 className="text-2xl font-bold text-[#EAF0F6]" style={{ fontFamily: "var(--font-heading)" }}>Population</h1>
        <p className="text-[#8593A1] text-sm mt-1">
          Preferential Fictitious Self-Play (PFSP) population — payoff matrix and sampling weights.
        </p>
        <p className="text-[#6F7C89] font-mono text-[10px] mt-1">
          <a className="text-[#22D3EE] hover:underline" href="/documentation/population">About Population / PFSP</a>
          {" · "}
          <a className="text-[#22D3EE] hover:underline" href="/documentation/glossary">Glossary</a>
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section>
          <div className="flex items-center justify-between mb-3">
            <p className="text-[#8593A1] font-mono text-xs uppercase tracking-widest">Payoff Matrix</p>
            <span className="text-[#6F7C89] font-mono text-xs">
              {matrix.suite} — <DateTimeCell iso={matrix.updatedAt} />
            </span>
          </div>
          <p className="text-[#6F7C89] font-mono text-xs mb-3">Row = agent, Column = opponent. Value = row agent win rate. Use arrow keys to navigate.</p>

          <div className="overflow-auto" onKeyDown={handleKeyDown} tabIndex={0} role="grid" aria-label="Payoff matrix" style={{ outline: "none" }}>
            <div style={{ display: "grid", gridTemplateColumns: `80px repeat(${n}, ${cellSize}px)` }}>
              <div />
              {matrix.agents.map((a, c) => (
                <div key={c} className="text-center py-1 border-b border-[#1E2630]" style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: "#6F7C89", textTransform: "uppercase" }}>
                  {shortDisplayName(a, 10)}
                </div>
              ))}
              {matrix.matrix.map((row, r) => (
                <React.Fragment key={r}>
                  <div className="flex items-center pr-2 border-r border-[#1E2630]" style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: "#8593A1" }} title={matrix.agents[r]}>
                    {shortDisplayName(matrix.agents[r], 12)}
                  </div>
                  {row.map((val, c) => {
                    const isFocused = focusCell.r === r && focusCell.c === c;
                    const isHovered = hovered?.r === r && hovered?.c === c;
                    const isSelf = r === c;
                    return (
                      <div
                        key={c}
                        role="gridcell"
                        tabIndex={-1}
                        aria-label={val === null ? `${matrix.agents[r]} vs ${matrix.agents[c]}: MISSING` : `${matrix.agents[r]} vs ${matrix.agents[c]}: ${(val * 100).toFixed(0)}%`}
                        onClick={() => setFocusCell({ r, c })}
                        onMouseEnter={() => setHovered({ r, c })}
                        onMouseLeave={() => setHovered(null)}
                        style={{
                          width: cellSize,
                          height: cellSize - 8,
                          backgroundColor: isSelf ? "#0A0E13" : cellColor(val),
                          border: isFocused ? "2px solid #FFB000" : isHovered ? "1px solid #2D3748" : "1px solid transparent",
                          cursor: "pointer",
                          position: "relative",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        {isSelf ? (
                          <span style={{ color: "#2D3748", fontFamily: "var(--font-mono)", fontSize: 10 }}>—</span>
                        ) : val === null ? (
                          <>
                            <svg width={cellSize} height={cellSize - 8} style={{ position: "absolute", top: 0, left: 0 }}>
                              <defs>
                                <pattern id={`hatch-${r}-${c}`} patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">
                                  <line x1="0" y1="0" x2="0" y2="6" stroke="#1E2630" strokeWidth="1.5" />
                                </pattern>
                              </defs>
                              <rect width={cellSize} height={cellSize - 8} fill={`url(#hatch-${r}-${c})`} />
                            </svg>
                            <span style={{ position: "relative", color: "#4A5568", fontFamily: "var(--font-mono)", fontSize: 8, textTransform: "uppercase", textAlign: "center" }}>MISSING</span>
                          </>
                        ) : (
                          <span style={{ color: "#EAF0F6", fontFamily: "var(--font-mono)", fontSize: 12, fontWeight: 700 }}>
                            {(val * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                    );
                  })}
                </React.Fragment>
              ))}
            </div>
          </div>

          {(hoveredValue !== null || focusedValue !== null) && (
            <div className="mt-2 p-2 border border-[#1E2630] rounded-sm bg-[#0C1116]">
              {hovered ? (
                <p className="text-[#CDD6DF] font-mono text-xs">
                  <span className="text-[#8593A1]">{matrix.agents[hovered.r]}</span>
                  <span className="text-[#4A5568] mx-1">vs</span>
                  <span className="text-[#8593A1]">{matrix.agents[hovered.c]}</span>
                  <span className="mx-2 text-[#FFB000] font-bold">
                    {hoveredValue === null ? "MISSING" : `${(hoveredValue * 100).toFixed(1)}%`}
                  </span>
                </p>
              ) : (
                <p className="text-[#CDD6DF] font-mono text-xs">
                  Focused: <span className="text-[#8593A1]">{matrix.agents[focusCell.r]}</span> vs <span className="text-[#8593A1]">{matrix.agents[focusCell.c]}</span>
                  <span className="mx-2 text-[#FFB000] font-bold">{focusedValue === null ? "MISSING" : `${(focusedValue * 100).toFixed(1)}%`}</span>
                </p>
              )}
            </div>
          )}

          <div className="flex items-center gap-3 mt-2" style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: "#6F7C89" }}>
            <span>■ <span style={{ color: "#F85149" }}>Red</span> = losing</span>
            <span>■ <span style={{ color: "#8593A1" }}>Grey</span> = neutral</span>
            <span>■ <span style={{ color: "#3FB950" }}>Green</span> = winning</span>
            <span>▦ = MISSING</span>
          </div>
        </section>

        <section>
          <p className="text-[#8593A1] font-mono text-xs uppercase tracking-widest mb-3">PFSP Sampling Weights</p>
          <div className="space-y-2">
            {entries.map(entry => (
              <div key={entry.id} className="flex items-center gap-3">
                <div className="w-32 truncate">
                  <span className="text-[#CDD6DF]" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>{entry.name}</span>
                </div>
                <div className="flex-1 h-5 bg-[#1E2630] rounded-sm overflow-hidden relative">
                  <div
                    className="h-full rounded-sm transition-all"
                    style={{ width: `${Number.isFinite(entry.pfspWeight) ? entry.pfspWeight * 100 : 0}%`, backgroundColor: entry.isMainAgent ? "#FFB000" : "#22D3EE", opacity: 0.8 }}
                  />
                  <span className="absolute right-2 top-0 h-full flex items-center text-[#8593A1]" style={{ fontFamily: "var(--font-mono)", fontSize: 9 }}>
                    {Number.isFinite(entry.pfspWeight) ? `${(entry.pfspWeight * 100).toFixed(0)}%` : "—"}
                  </span>
                </div>
                <DataSourceBadge kind={entry.kind} pill />
              </div>
            ))}
          </div>

          <p className="text-[#8593A1] font-mono text-xs uppercase tracking-widest mt-6 mb-3">Agents</p>
          <div className="space-y-2">
            {entries.map(entry => (
              <div key={entry.id} className="border border-[#1E2630] rounded-sm p-3 flex items-center justify-between">
                <div>
                  <p className="text-[#CDD6DF]" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>
                    {entry.name}
                    {entry.isMainAgent && <span className="ml-2 text-[#FFB000]" style={{ fontSize: 9 }}>★ MAIN</span>}
                  </p>
                  <p className="text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 9 }}>
                    {Number.isFinite(entry.gamesPlayed) ? `${entry.gamesPlayed} games` : "games count NOT RECORDED"}
                    {" — "}
                    {Number.isFinite(entry.winRate) ? `${(entry.winRate * 100).toFixed(0)}% win rate` : "win rate NOT RECORDED"}
                  </p>
                </div>
                <DataSourceBadge kind={entry.kind} pill />
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
