import React, { useEffect, useState } from "react";
import { useDataSource } from "../app/DataSourceProvider";
import { ExplanationRecord, CounterfactualRecord, FaithfulnessStatus } from "../types/explanation";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../app/components/ui/tabs";
import DataSourceBadge from "../components/status/DataSourceBadge";
import GeneralsBoard from "../components/board/GeneralsBoard";
import { generateBoard } from "../utils/gameBoard";

const faithColors: Record<FaithfulnessStatus, string> = {
  VERIFIED: "#3FB950", PARTIAL: "#FFB000", EXPERIMENTAL: "#22D3EE",
  FAILED: "#F85149", NOT_EVALUATED: "#6F7C89",
};

export default function ExplainabilityPage() {
  const ds = useDataSource();
  const [explanations, setExplanations] = useState<ExplanationRecord[]>([]);
  const [counterfactuals, setCounterfactuals] = useState<CounterfactualRecord[]>([]);
  const [selected, setSelected] = useState<ExplanationRecord | null>(null);
  const [horizon, setHorizon] = useState(4);
  const demoBoard = generateBoard(18, 18, 12);

  useEffect(() => {
    ds.listExplanations().then(list => {
      setExplanations(list);
      if (list.length > 0) {
        setSelected(list[0]);
        ds.getCounterfactuals(list[0].id).then(setCounterfactuals);
      }
    });
  }, [ds]);

  return (
    <div className="p-6 space-y-4">
      <header>
        <p className="text-[#FFB000] font-mono text-xs uppercase tracking-widest mb-1">$ explainability/</p>
        <h1 className="text-2xl font-bold text-[#EAF0F6]" style={{ fontFamily: "var(--font-heading)" }}>Explainability</h1>
      </header>

      {selected && <DataSourceBadge kind={selected.kind} />}

      <Tabs defaultValue="decision">
        <TabsList className="bg-[#0C1116] border border-[#1E2630]">
          {["decision", "why-cell", "why-not", "next", "concepts", "counterfactuals", "faithfulness"].map(t => (
            <TabsTrigger key={t} value={t} className="data-[state=active]:bg-[#1E2630] data-[state=active]:text-[#FFB000] text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 10, textTransform: "uppercase" }}>
              {t.replace(/-/g, " ")}
            </TabsTrigger>
          ))}
        </TabsList>

        <TabsContent value="decision">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
            <div>
              <p className="text-[#8593A1] font-mono text-xs uppercase mb-2">Board State — Turn {selected?.turn ?? 0}</p>
              <div style={{ maxWidth: 360 }}>
                <GeneralsBoard board={demoBoard} interactive={false} />
              </div>
            </div>
            <div>
              <p className="text-[#8593A1] font-mono text-xs uppercase mb-2">Top Features</p>
              {selected?.topFeatures.map((f, i) => (
                <div key={i} className="flex items-center gap-3 mb-2">
                  <span className="w-36 text-[#CDD6DF] truncate" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>{f.name}</span>
                  <div className="flex-1 h-4 bg-[#1E2630] rounded-sm overflow-hidden">
                    <div className="h-full bg-[#22D3EE] rounded-sm" style={{ width: `${f.weight * 100}%`, opacity: 0.8 }} />
                  </div>
                  <span className="text-[#8593A1] w-10 text-right" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>{(f.weight * 100).toFixed(0)}%</span>
                </div>
              ))}
              <p className="text-[#8593A1] font-mono text-xs uppercase mt-4 mb-2">Method</p>
              <p className="text-[#CDD6DF] font-mono text-sm">{selected?.method ?? "—"}</p>
              <p className="text-[#8593A1] font-mono text-xs mt-4">Faithfulness: <span style={{ color: faithColors[selected?.faithfulness ?? "NOT_EVALUATED"] }}>{selected?.faithfulness ?? "—"}</span></p>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="why-cell">
          <div className="mt-4">
            <p className="text-[#8593A1] font-mono text-xs mb-3">Attribution overlay — higher intensity = more influential cell.</p>
            <div style={{ maxWidth: 360 }}>
              <GeneralsBoard board={demoBoard} interactive={false} attributionOverlay={selected?.saliencyMap} />
            </div>
          </div>
        </TabsContent>

        <TabsContent value="why-not">
          <div className="mt-4 space-y-3">
            <p className="text-[#8593A1] font-mono text-xs">Counterfactual alternatives — actions that were considered but not taken.</p>
            {counterfactuals.length === 0 ? (
              <p className="text-[#4A5568] font-mono text-xs">No counterfactuals recorded for this decision.</p>
            ) : counterfactuals.map(cf => (
              <div key={cf.id} className="border border-[#1E2630] rounded-sm p-3">
                <p className="text-[#CDD6DF] font-mono text-xs">
                  ({cf.altAction.srcRow},{cf.altAction.srcCol}) → ({cf.altAction.dstRow},{cf.altAction.dstCol})
                </p>
                <p className="text-[#8593A1] font-mono text-xs">Value: {cf.altValueEstimate.toFixed(3)} ({cf.difference > 0 ? "+" : ""}{cf.difference.toFixed(3)} vs chosen)</p>
                {cf.notes && <p className="text-[#6F7C89] font-mono text-xs mt-1">{cf.notes}</p>}
              </div>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="next">
          <div className="mt-4">
            <div className="flex items-center gap-3 mb-4">
              <p className="text-[#8593A1] font-mono text-xs">Horizon:</p>
              {[1, 2, 4, 8, 16, 32].map(h => (
                <button key={h} onClick={() => setHorizon(h)} className={`px-3 py-1 rounded-sm border font-mono text-xs transition-colors ${horizon === h ? "border-[#FFB000] text-[#FFB000]" : "border-[#1E2630] text-[#6F7C89] hover:border-[#2D3748]"}`}>
                  {h}T
                </button>
              ))}
            </div>
            <p className="text-[#6F7C89] font-mono text-xs">Horizon predictions at +{horizon} turns. Demo — synthetic projection.</p>
            <div style={{ maxWidth: 360, marginTop: 12 }}>
              <GeneralsBoard board={generateBoard(18, 18, 12 + horizon)} interactive={false} />
            </div>
          </div>
        </TabsContent>

        <TabsContent value="concepts">
          <div className="mt-4 space-y-3">
            <p className="text-[#8593A1] font-mono text-xs">High-level concept activations (DEMO — not yet implemented for imported evidence).</p>
            {["army_concentration", "general_proximity", "city_control", "fog_frontier", "expansion_momentum"].map(concept => (
              <div key={concept} className="flex items-center gap-3">
                <span className="w-44 text-[#CDD6DF]" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>{concept}</span>
                <div className="flex-1 h-4 bg-[#1E2630] rounded-sm overflow-hidden">
                  <div className="h-full bg-[#8593A1] rounded-sm" style={{ width: `${20 + Math.random() * 60}%`, opacity: 0.6 }} />
                </div>
                <DataSourceBadge kind="DEMO" pill />
              </div>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="counterfactuals">
          <div className="mt-4 space-y-3">
            {counterfactuals.length === 0 ? (
              <p className="text-[#4A5568] font-mono text-xs">No counterfactuals recorded.</p>
            ) : counterfactuals.map(cf => (
              <div key={cf.id} className="border border-[#1E2630] rounded-sm p-3 bg-[#0C1116]">
                <p className="text-[#FFB000] font-mono text-sm font-bold">Alt: ({cf.altAction.srcRow},{cf.altAction.srcCol})→({cf.altAction.dstRow},{cf.altAction.dstCol})</p>
                <div className="grid grid-cols-2 gap-2 mt-2">
                  <div><p className="text-[#6F7C89] font-mono text-xs">Alt value</p><p className="text-[#CDD6DF] font-mono text-sm">{cf.altValueEstimate.toFixed(3)}</p></div>
                  <div><p className="text-[#6F7C89] font-mono text-xs">Δ vs chosen</p><p className={`font-mono text-sm ${cf.difference >= 0 ? "text-[#3FB950]" : "text-[#F85149]"}`}>{cf.difference >= 0 ? "+" : ""}{cf.difference.toFixed(3)}</p></div>
                </div>
                {cf.notes && <p className="text-[#8593A1] font-mono text-xs mt-2">{cf.notes}</p>}
              </div>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="faithfulness">
          <div className="mt-4">
            <div className="border border-[#1E2630] rounded-sm overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr className="bg-[#0C1116] border-b border-[#1E2630]">
                    {["Method", "Status", "Score", "Notes"].map(h => (
                      <th key={h} className="px-3 py-2 text-left" style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "#6F7C89", textTransform: "uppercase" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {selected?.faithfulnessChecks?.map((check, i) => (
                    <tr key={i} className="border-b border-[#1E2630]">
                      <td className="px-3 py-2.5 font-mono text-xs text-[#CDD6DF]">{check.method}</td>
                      <td className="px-3 py-2.5">
                        <span className="font-mono text-xs font-bold" style={{ color: faithColors[check.status] }}>{check.status}</span>
                      </td>
                      <td className="px-3 py-2.5 font-mono text-xs text-[#8593A1]">{check.score != null ? check.score.toFixed(2) : "—"}</td>
                      <td className="px-3 py-2.5 font-mono text-xs text-[#6F7C89]">{check.notes ?? "—"}</td>
                    </tr>
                  )) ?? <tr><td colSpan={4} className="px-3 py-4 text-center text-[#4A5568] font-mono text-xs">No checks recorded.</td></tr>}
                </tbody>
              </table>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
