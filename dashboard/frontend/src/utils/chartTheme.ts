/* Single Recharts theme object — no per-chart hex codes allowed in components */
export const chartTheme = {
  series: ["#FFB000", "#22D3EE", "#3FB950", "#F85149", "#8B98A5", "#C084FC"],
  positive: "#3FB950",
  negative: "#F85149",
  neutral: "#8593A1",
  benchmark: "#5A6570",
  grid: "#1B222B",
  axis: "#5A6570",
  axisLabel: "#8593A1",
  tooltip: {
    bg: "#161C24",
    border: "#1E2630",
    text: "#EAF0F6",
  },
} as const;

export const rechartsDefaults = {
  margin: { top: 4, right: 8, bottom: 4, left: 0 },
  cartesianGrid: { strokeDasharray: "3 3", stroke: chartTheme.grid, vertical: false },
  axisStyle: { tick: { fill: chartTheme.axisLabel, fontSize: 11, fontFamily: "'JetBrains Mono', monospace" } },
};
