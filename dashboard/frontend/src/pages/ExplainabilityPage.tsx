import React, { useEffect, useState } from "react";
import { useDataSource } from "../app/DataSourceProvider";
import { ExplanationRecord, CounterfactualRecord, FaithfulnessStatus } from "../types/explanation";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../app/components/ui/tabs";
import DataSourceBadge from "../components/status/DataSourceBadge";
import GeneralsBoard from "../components/board/GeneralsBoard";
import { shortDisplayName } from "../utils/displayNames";

const faithColors: Record<FaithfulnessStatus, string> = {
  VERIFIED: "#3FB950", PARTIAL: "#FFB000", EXPERIMENTAL: "#22D3EE",
  FAILED: "#F85149", NOT_EVALUATED: "#6F7C89",
};

function EmptyState({ message }: { message: string }) {
  return (
    <div className="mt-4 border border-[#1E2630] rounded-sm p-4 bg-[#0C1116]">
      <p className="text-[#8593A1] font-mono text-xs">{message}</p>
      <p className="text-[#6F7C89] font-mono text-[10px] mt-2">
        No synthetic or DEMO projection is shown. Import a recorded explanation to populate this view.
      </p>
    </div>
  );
}

export default function ExplainabilityPage() {
  const ds = useDataSource();
  const [explanations, setExplanations] = useState<ExplanationRecord[]>([]);
  const [counterfactuals, setCounterfactuals] = useState<CounterfactualRecord[]>([]);
  const [selected, setSelected] = useState<ExplanationRecord | null>(null);
  const [horizon, setHorizon] = useState(4);

  useEffect(() => {
    ds.listExplanations().then(list => {
      // Prefer latest by turn when ids are opaque; otherwise first recorded entry.
      const sorted = [...list].sort((a, b) => (b.turn ?? 0) - (a.turn ?? 0));
      setExplanations(sorted);
      if (sorted.length > 0) {
        setSelected(sorted[0]);
        ds.getCounterfactuals(sorted[0].id).then(setCounterfactuals);
      }
    });
  }, [ds]);

  const hasSaliency = Boolean(selected?.saliencyMap?.length);
  const hasFeatures = Boolean(selected?.topFeatures?.length);

  return (
    <div className="p-6 space-y-4">
      <header>
        <p className="text-[#FFB000] font-mono text-xs uppercase tracking-widest mb-1">$ explainability/</p>
        <h1 className="text-2xl font-bold text-[#EAF0F6]" style={{ fontFamily: "var(--font-heading)" }}>Explainability</h1>
        <p className="text-[#6F7C89] font-mono text-[10px] mt-1">
          <a className="text-[#22D3EE] hover:underline" href="/documentation/explainability">About Explainability</a>
        </p>
      </header>

      {explanations.length > 0 ? (
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-[#6F7C89] font-mono text-[10px] uppercase">Record</label>
          <select
            className="bg-[#0C1116] border border-[#1E2630] text-[#CDD6DF] font-mono text-xs px-2 py-1 rounded-sm"
            value={selected?.id ?? ""}
            onChange={(e) => {
              const next = explanations.find((x) => x.id === e.target.value) ?? null;
              setSelected(next);
              if (next) ds.getCounterfactuals(next.id).then(setCounterfactuals);
              else setCounterfactuals([]);
            }}
          >
            {explanations.map((ex) => (
              <option key={ex.id} value={ex.id}>
                {shortDisplayName(ex.id)} · turn {ex.turn}
              </option>
            ))}
          </select>
          {selected && <DataSourceBadge kind={selected.kind} />}
        </div>
      ) : (
        <p className="text-[#6F7C89] font-mono text-xs">No explanation records imported.</p>
      )}

      <Tabs defaultValue="decision">
        <TabsList className="bg-[#0C1116] border border-[#1E2630]">
          {["decision", "why-cell", "why-not", "next", "concepts", "counterfactuals", "faithfulness"].map(t => (
            <TabsTrigger key={t} value={t} className="data-[state=active]:bg-[#1E2630] data-[state=active]:text-[#FFB000] text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 10, textTransform: "uppercase" }}>
              {t.replace(/-/g, " ")}
            </TabsTrigger>
          ))}
        </TabsList>

        <TabsContent value="decision">
          {!selected ? (
            <EmptyState message="No explanation record selected." />
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
              <div>
                <p className="text-[#8593A1] font-mono text-xs uppercase mb-2">Board State — Turn {selected.turn}</p>
                {hasSaliency ? (
                  <div className="flex justify-center">
                    <GeneralsBoard
                      variant="explainability"
                      board={{
                        width: selected.saliencyMap[0]?.length ?? 0,
                        height: selected.saliencyMap.length,
                        turn: selected.turn,
                        cells: selected.saliencyMap.map((row) =>
                          row.map(() => ({
                            terrain: "plain" as const,
                            owner: "neutral" as const,
                            armies: 0,
                            visible: true,
                          })),
                        ),
                      }}
                      interactive={false}
                      attributionOverlay={selected.saliencyMap}
                    />
                  </div>
                ) : (
                  <EmptyState message="Board snapshot NOT AVAILABLE for this record. Attribution overlay requires a recorded saliency map." />
                )}
              </div>
              <div>
                <p className="text-[#8593A1] font-mono text-xs uppercase mb-2">Top Features</p>
                {hasFeatures ? (
                  selected.topFeatures.map((f, i) => (
                    <div key={i} className="flex items-center gap-3 mb-2">
                      <span className="w-36 text-[#CDD6DF] truncate" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>{f.name}</span>
                      <div className="flex-1 h-4 bg-[#1E2630] rounded-sm overflow-hidden">
                        <div className="h-full bg-[#22D3EE] rounded-sm" style={{ width: `${f.weight * 100}%`, opacity: 0.8 }} />
                      </div>
                      <span className="text-[#8593A1] w-10 text-right" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>{(f.weight * 100).toFixed(0)}%</span>
                    </div>
                  ))
                ) : (
                  <p className="text-[#6F7C89] font-mono text-xs">Feature attributions NOT RECORDED for this explanation.</p>
                )}
                <p className="text-[#8593A1] font-mono text-xs uppercase mt-4 mb-2">Method</p>
                <p className="text-[#CDD6DF] font-mono text-sm">{selected.method || "NOT RECORDED"}</p>
                <p className="text-[#8593A1] font-mono text-xs mt-4">
                  Faithfulness:{" "}
                  <span style={{ color: faithColors[selected.faithfulness ?? "NOT_EVALUATED"] }}>
                    {selected.faithfulness ?? "NOT_EVALUATED"}
                  </span>
                </p>
              </div>
            </div>
          )}
        </TabsContent>

        <TabsContent value="why-cell">
          {!selected || !hasSaliency ? (
            <EmptyState message="Cell attribution NOT AVAILABLE — no saliency map on the selected record." />
          ) : (
            <div className="mt-4">
              <p className="text-[#8593A1] font-mono text-xs mb-3">Attribution overlay — higher intensity = more influential cell.</p>
              <div className="flex justify-center">
                <GeneralsBoard
                  variant="explainability"
                  board={{
                    width: selected.saliencyMap[0]?.length ?? 0,
                    height: selected.saliencyMap.length,
                    turn: selected.turn,
                    cells: selected.saliencyMap.map((row) =>
                      row.map(() => ({
                        terrain: "plain" as const,
                        owner: "neutral" as const,
                        armies: 0,
                        visible: true,
                      })),
                    ),
                  }}
                  interactive={false}
                  attributionOverlay={selected.saliencyMap}
                />
              </div>
            </div>
          )}
        </TabsContent>

        <TabsContent value="why-not">
          <div className="mt-4 space-y-3">
            <p className="text-[#8593A1] font-mono text-xs">Counterfactual alternatives — actions that were considered but not taken.</p>
            {counterfactuals.length === 0 ? (
              <EmptyState message="No counterfactuals recorded for this decision." />
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
            <EmptyState message={`Horizon predictions at +${horizon} turns are NOT AVAILABLE. No recorded forward projection exists for this explanation.`} />
          </div>
        </TabsContent>

        <TabsContent value="concepts">
          <EmptyState message="Concept activations are NOT AVAILABLE. This tab only shows imported concept scores — random DEMO bars are disabled." />
        </TabsContent>

        <TabsContent value="counterfactuals">
          <div className="mt-4 space-y-3">
            {counterfactuals.length === 0 ? (
              <EmptyState message="No counterfactuals recorded." />
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
                  )) ?? (
                    <tr>
                      <td colSpan={4} className="px-3 py-4 text-center text-[#4A5568] font-mono text-xs">
                        No faithfulness checks recorded.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
