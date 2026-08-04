import React, { useEffect, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router";
import { useDataSource } from "../app/DataSourceProvider";
import { ModelRecord, ModelArchitecture, ModelLifecycle, CompetitiveRole } from "../types/model";
import DataSourceBadge from "../components/status/DataSourceBadge";
import { fmtWDL, fmtK } from "../utils/formatting";
import RawRecordDrawer from "../components/feedback/RawRecordDrawer";

const archColors: Record<ModelArchitecture, string> = {
  heuristic: "#FFB000",
  mlp_control: "#22D3EE",
  recurrent_cnn: "#8593A1",
  recurrent_graph_belief: "#818CF8",
  graph_belief_pyg_research: "#A78BFA",
};

const lifecycleColor: Record<ModelLifecycle, string> = {
  SCAFFOLDED: "#4A5568",
  SMOKE_TESTED: "#6F7C89",
  TRAINED: "#8593A1",
  EVALUATED: "#22D3EE",
  REJECTED: "#F85149",
  REJECTED_INCOMPATIBLE: "#F85149",
};

const roleColor: Record<CompetitiveRole, string> = {
  NONE: "#4A5568",
  BASELINE: "#8593A1",
  CHALLENGER: "#FFB000",
  CHAMPION: "#3FB950",
};

export default function ModelsPage() {
  const ds = useDataSource();
  const { modelId } = useParams<{ modelId?: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [drawerRecord, setDrawerRecord] = useState<ModelRecord | null>(null);

  useEffect(() => { ds.listModels().then(setModels); }, [ds]);

  const compareParam = searchParams.get("compare");
  useEffect(() => {
    if (compareParam) setSelected(compareParam.split(","));
    else if (modelId) setSelected([modelId]);
  }, [compareParam, modelId]);

  const selectedModels = selected.map(id => models.find(m => m.id === id)).filter((m): m is ModelRecord => !!m);

  const handleSelect = (id: string) => {
    const next = selected.includes(id) ? selected.filter(s => s !== id) : selected.length < 3 ? [...selected, id] : [id];
    setSelected(next);
    if (next.length > 0) navigate(`/models/${next[0]}?compare=${next.join(",")}`);
    else navigate("/models");
  };

  return (
    <div className="p-6 space-y-6">
      <header>
        <p className="text-[#FFB000] font-mono text-xs uppercase tracking-widest mb-1">$ models/</p>
        <h1 className="text-2xl font-bold text-[#EAF0F6]" style={{ fontFamily: "var(--font-heading)" }}>Models</h1>
        <p className="text-[#8593A1] text-sm mt-1">Architecture registry — lifecycle, role, and delivery status.</p>
      </header>

      {selectedModels.length > 1 && (
        <section className="border border-[#1E2630] rounded-sm p-4 bg-[#0C1116]">
          <p className="text-[#FFB000] font-mono text-xs uppercase tracking-widest mb-3">Comparison</p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {selectedModels.map(m => (
              <div key={m.id} className="border border-[#1E2630] rounded-sm p-3">
                <DataSourceBadge kind={m.kind} pill />
                <p className="text-[#CDD6DF] font-mono text-xs mt-1">{m.name}</p>
                <div className="mt-1.5 flex flex-wrap gap-1">
                  <Tag label={m.lifecycle} color={lifecycleColor[m.lifecycle]} />
                  <Tag label={m.role} color={roleColor[m.role]} />
                  <Tag label={m.deliveryStatus} color="#4A5568" />
                </div>
                <p className="text-[#8593A1] font-mono text-xs mt-2">WDL: {fmtWDL(m.wdl)}</p>
                <p className="text-[#8593A1] font-mono text-xs">Params: {fmtK(m.parameters) || "N/A"}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      <div className="border border-[#1E2630] rounded-sm overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="bg-[#0C1116] border-b border-[#1E2630]">
              {["", "Name", "Arch", "Lifecycle", "Role", "Delivery", "WDL", "Kind", ""].map(h => (
                <th key={h} className="px-3 py-2 text-left" style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "#6F7C89", textTransform: "uppercase", letterSpacing: "0.06em" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map(model => {
              const isSelected = selected.includes(model.id);
              return (
                <tr key={model.id} onClick={() => handleSelect(model.id)} className={`border-b border-[#1E2630] cursor-pointer transition-colors ${isSelected ? "bg-[#1A2030]" : "hover:bg-[#0C1116]"}`}>
                  <td className="px-3 py-2.5 w-6">
                    <div className={`w-3 h-3 rounded-sm border ${isSelected ? "border-[#FFB000] bg-[#FFB000]" : "border-[#2D3748]"}`} />
                  </td>
                  <td className="px-3 py-2.5">
                    <span className="text-[#CDD6DF]" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>{model.name}</span>
                    {model.blockerReason && <p className="text-[#F85149]" style={{ fontFamily: "var(--font-mono)", fontSize: 9 }}>⚠ {model.blockerReason}</p>}
                  </td>
                  <td className="px-3 py-2.5">
                    <span className="font-bold" style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: archColors[model.architecture] }}>{model.architecture}</span>
                  </td>
                  <td className="px-3 py-2.5"><Tag label={model.lifecycle} color={lifecycleColor[model.lifecycle]} /></td>
                  <td className="px-3 py-2.5"><Tag label={model.role} color={roleColor[model.role]} /></td>
                  <td className="px-3 py-2.5"><Tag label={model.deliveryStatus} color="#4A5568" /></td>
                  <td className="px-3 py-2.5"><span className="text-[#FFB000] font-bold" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>{fmtWDL(model.wdl)}</span></td>
                  <td className="px-3 py-2.5"><DataSourceBadge kind={model.kind} pill /></td>
                  <td className="px-3 py-2.5">
                    <button onClick={e => { e.stopPropagation(); setDrawerRecord(model); }} className="text-[#6F7C89] hover:text-[#FFB000] font-mono text-xs transition-colors">RAW</button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <RawRecordDrawer
        open={!!drawerRecord}
        onClose={() => setDrawerRecord(null)}
        title={drawerRecord?.name ?? ""}
        kind={drawerRecord?.kind}
        record={drawerRecord as unknown as Record<string, unknown> ?? {}}
      />
    </div>
  );
}

function Tag({ label, color }: { label: string; color: string }) {
  return (
    <span className="inline-flex items-center px-1.5 py-0.5 rounded-sm border" style={{ color, borderColor: `${color}66`, fontFamily: "var(--font-mono)", fontSize: 9, letterSpacing: "0.06em", textTransform: "uppercase" }}>{label}</span>
  );
}
