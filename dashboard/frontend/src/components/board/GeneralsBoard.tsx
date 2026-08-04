type Cell = number;

export type BoardFrame = {
  mapKey: string;
  height: number;
  width: number;
  typeGrid: Cell[][];
  ownerGrid?: Cell[][];
  armyGrid?: Cell[][];
};

/** Terrain must remount when map identity changes, not only when H×W match. */
export function GeneralsBoard({ frame }: { frame: BoardFrame }) {
  const { mapKey, height, width, typeGrid, ownerGrid, armyGrid } = frame;
  const cell = 20;
  const summary = `Board ${width}×${height}, map ${mapKey}`;

  return (
    <div>
      <div className="board-root" aria-label={summary}>
        <svg
          key={mapKey}
          viewBox={`0 0 ${width * cell} ${height * cell}`}
          role="img"
          aria-label={summary}
        >
          {/* terrain */}
          {typeGrid.map((row, r) =>
            row.map((t, c) => {
              const fill =
                t === 0 ? "#1a2330" /* mountain/void-ish */ :
                t === 2 ? "#243041" /* city-ish */ :
                t === 3 || t === 4 ? "#2a3340" /* structure */ :
                "#0f1620";
              return (
                <rect
                  key={`${mapKey}-t-${r}-${c}`}
                  x={c * cell}
                  y={r * cell}
                  width={cell}
                  height={cell}
                  fill={fill}
                  stroke="#1e2630"
                  strokeWidth={0.5}
                />
              );
            }),
          )}
          {/* ownership */}
          {ownerGrid?.map((row, r) =>
            row.map((o, c) => {
              if (!o) return null;
              const fill = o === 1 ? "rgba(61,139,253,0.35)" : "rgba(248,81,73,0.35)";
              return (
                <rect
                  key={`${mapKey}-o-${r}-${c}`}
                  x={c * cell}
                  y={r * cell}
                  width={cell}
                  height={cell}
                  fill={fill}
                />
              );
            }),
          )}
          {/* units */}
          {armyGrid?.map((row, r) =>
            row.map((a, c) => {
              if (!a) return null;
              return (
                <text
                  key={`${mapKey}-a-${r}-${c}`}
                  x={c * cell + cell / 2}
                  y={r * cell + cell / 2 + 3}
                  textAnchor="middle"
                  fontSize={8}
                  fill="#eaf0f6"
                  fontFamily="var(--font-mono)"
                >
                  {a}
                </text>
              );
            }),
          )}
        </svg>
      </div>
      <div className="board-summary">{summary}</div>
    </div>
  );
}
