type SeriesPoint = Record<string, number | string | undefined>;

type Props = {
  title: string;
  points: SeriesPoint[];
  seriesKeys: string[];
  missing?: string[];
  note?: string | null;
  xKey?: string;
};

const COLORS = ["#c47a12", "#2f5d50", "#8b3a3a", "#3a4a6b", "#6b4a3a"];

/** Minimal SVG multi-series chart; empty points render NOT RECORDED. */
export function MetricChart({
  title,
  points,
  seriesKeys,
  missing = [],
  note,
  xKey = "update",
}: Props) {
  const width = 420;
  const height = 160;
  const pad = 24;

  if (!points.length) {
    return (
      <div className="chart-block">
        <h4>{title}</h4>
        <p className="muted">NOT RECORDED — no producer points yet</p>
        {note ? <p className="muted">{note}</p> : null}
      </div>
    );
  }

  const xs = points.map((p, i) => Number(p[xKey] ?? i));
  const series = seriesKeys.filter((k) => points.some((p) => typeof p[k] === "number"));
  const ys = series.flatMap((k) => points.map((p) => Number(p[k])).filter((v) => Number.isFinite(v)));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys, 0);
  const maxY = Math.max(...ys, 1e-6);
  const sx = (x: number) => pad + ((x - minX) / Math.max(maxX - minX, 1e-9)) * (width - 2 * pad);
  const sy = (y: number) => height - pad - ((y - minY) / Math.max(maxY - minY, 1e-9)) * (height - 2 * pad);

  return (
    <div className="chart-block">
      <h4>{title}</h4>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title} className="metric-chart">
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#bbb" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#bbb" />
        {series.map((key, si) => {
          const d = points
            .map((p, i) => {
              const y = Number(p[key]);
              if (!Number.isFinite(y)) return null;
              const x = Number(p[xKey] ?? i);
              return `${sx(x)},${sy(y)}`;
            })
            .filter(Boolean)
            .join(" ");
          return (
            <polyline
              key={key}
              fill="none"
              stroke={COLORS[si % COLORS.length]}
              strokeWidth={2}
              points={d}
            />
          );
        })}
      </svg>
      <div className="chart-legend muted">
        {series.map((k, i) => (
          <span key={k} style={{ color: COLORS[i % COLORS.length], marginRight: "0.75rem" }}>
            {k}
          </span>
        ))}
      </div>
      {missing.length ? (
        <p className="muted">NOT RECORDED fields: {missing.join(", ")}</p>
      ) : null}
      {note ? <p className="muted">{note}</p> : null}
    </div>
  );
}
