import React, { useEffect, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router";
import { useDataSource } from "../app/DataSourceProvider";
import { ExperimentRecord } from "../types/experiment";
import FilterBar from "../components/forms/FilterBar";
import DataSourceBadge from "../components/status/DataSourceBadge";
import { fmtWDL, fmtPct } from "../utils/formatting";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";
import { chartTheme, rechartsDefaults } from "../utils/chartTheme";

export default function ExperimentsPage() {
  const ds = useDataSource();
  const { experimentId } = useParams<{ experimentId?: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const [experiments, setExperiments] = useState<ExperimentRecord[]>([]);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<string[]>([]);

  const compareParam = searchParams.get("compare");

  useEffect(() => {
    ds.listExperiments().then(setExperiments);
  }, [ds]);

  useEffect(() => {
    if (compareParam) setSelected(compareParam.split(","));
    else if (experimentId) setSelected([experimentId]);
  }, [compareParam, experimentId]);

  const filtered = experiments.filter(e =>
    !search || e.label.toLowerCase().includes(search.toLowerCase()) || e.candidate.toLowerCase().includes(search.toLowerCase())
  );

  const selectedRecords = selected.map(id => experiments.find(e => e.id === id)).filter((e): e is ExperimentRecord => !!e);

  const handleSelect = (id: string) => {
    const next = selected.includes(id)
      ? selected.filter(s => s !== id)
      : selected.length < 3 ? [...selected, id] : [id];
    setSelected(next);
    if (next.length > 0) {
      const primary = next[0];
      navigate(`/experiments/${primary}?compare=${next.join(",")}`);
    } else {
      navigate("/experiments");
    }
  };

  const comparisonData = selectedRecords.length > 0 ? selectedRecords.map(e => ({
    name: e.candidate.length > 12 ? e.candidate.slice(0, 12) + "…" : e.candidate,
    wins: e.wdl.wins,
    draws: e.wdl.draws,
    losses: e.wdl.losses,
    discovery: e.discoveryRate ?? 0,
  })) : [];

  return (
    <div className="p-6 space-y-6">
      <header>
        <p className="text-[#FFB000] font-mono text-xs uppercase tracking-widest mb-1">$ experiments/</p>
        <h1 className="text-2xl font-bold text-[#EAF0F6]" style={{ fontFamily: "var(--font-heading)" }}>Experiments</h1>
        <p className="text-[#8593A1] text-sm mt-1">Evaluation run history — select up to 3 rows to compare.</p>
      </header>

      {selectedRecords.length > 1 && (
        <section className="border border-[#1E2630] rounded-sm p-4 bg-[#0C1116]">
          <p className="text-[#FFB000] font-mono text-xs uppercase tracking-widest mb-3">Comparison</p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            {selectedRecords.map(e => (
              <div key={e.id} className="border border-[#1E2630] rounded-sm p-3">
                <DataSourceBadge kind={e.kind} pill />
                <p className="text-[#CDD6DF] font-mono text-xs mt-1">{e.candidate}</p>
                <p className="text-[#FFB000] font-mono text-lg font-bold">{fmtWDL(e.wdl, e.wdlAvailability)}</p>
                <p className="text-[#8593A1] font-mono text-xs">
                  Discovery: {e.discoveryRate != null ? fmtPct(e.discoveryRate) : "NOT MEASURED"}
                </p>
                <p className="text-[#8593A1] font-mono text-xs">
                  Gate: <span className={e.discoveryGate === "PASSED" ? "text-[#3FB950]" : e.discoveryGate === "FAILED" ? "text-[#F85149]" : "text-[#6F7C89]"}>{e.discoveryGate}</span>
                </p>
              </div>
            ))}
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData} {...rechartsDefaults}>
                <CartesianGrid {...rechartsDefaults.cartesianGrid} />
                <XAxis dataKey="name" {...rechartsDefaults.axisStyle} />
                <YAxis {...rechartsDefaults.axisStyle} />
                <Tooltip contentStyle={{ background: chartTheme.tooltip.bg, border: `1px solid ${chartTheme.tooltip.border}`, color: chartTheme.tooltip.text }} />
                <Legend />
                <Bar dataKey="wins" fill={chartTheme.positive} name="Wins" />
                <Bar dataKey="draws" fill={chartTheme.neutral} name="Draws" />
                <Bar dataKey="losses" fill={chartTheme.negative} name="Losses" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      <FilterBar search={search} onSearch={setSearch} placeholder="Search candidates, opponents…" />

      <div className="border border-[#1E2630] rounded-sm overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="bg-[#0C1116] border-b border-[#1E2630]">
              {["", "Label", "Candidate", "Result", "Discovery", "Gate", "Date", "Kind"].map(h => (
                <th key={h} className="px-3 py-2 text-left" style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "#6F7C89", textTransform: "uppercase", letterSpacing: "0.06em" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map(exp => {
              const isSelected = selected.includes(exp.id);
              return (
                <tr
                  key={exp.id}
                  onClick={() => handleSelect(exp.id)}
                  className={`border-b border-[#1E2630] cursor-pointer transition-colors ${isSelected ? "bg-[#1A2030]" : "hover:bg-[#0C1116]"}`}
                >
                  <td className="px-3 py-2.5 w-6">
                    <div className={`w-3 h-3 rounded-sm border ${isSelected ? "border-[#FFB000] bg-[#FFB000]" : "border-[#2D3748]"}`} />
                  </td>
                  <td className="px-3 py-2.5"><span className="text-[#CDD6DF]" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>{exp.label.length > 40 ? exp.label.slice(0, 40) + "…" : exp.label}</span></td>
                  <td className="px-3 py-2.5"><span className="text-[#8593A1]" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>{exp.candidate.length > 20 ? exp.candidate.slice(0, 20) + "…" : exp.candidate}</span></td>
                  <td className="px-3 py-2.5"><span className="text-[#FFB000] font-bold" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>{fmtWDL(exp.wdl, exp.wdlAvailability)}</span></td>
                  <td className="px-3 py-2.5"><span className="text-[#CDD6DF]" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>{exp.discoveryRate != null ? fmtPct(exp.discoveryRate) : <span className="text-[#4A5568]">NOT MEASURED</span>}</span></td>
                  <td className="px-3 py-2.5">
                    <span className={`font-mono text-xs ${exp.discoveryGate === "PASSED" ? "text-[#3FB950]" : exp.discoveryGate === "FAILED" ? "text-[#F85149]" : "text-[#6F7C89]"}`}>{exp.discoveryGate}</span>
                  </td>
                  <td className="px-3 py-2.5"><span className="text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>{exp.dateLabel}</span></td>
                  <td className="px-3 py-2.5"><DataSourceBadge kind={exp.kind} pill /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
