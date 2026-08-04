import { ExplanationRecord, CounterfactualRecord } from "../../types/explanation";

export const demoExplanation: ExplanationRecord = {
  id: "expl-demo-001",
  kind: "DEMO",
  matchId: "replay-demo-001",
  turn: 12,
  method: "gradient_saliency",
  saliencyMap: Array.from({ length: 18 }, (_, r) =>
    Array.from({ length: 18 }, (_, c) =>
      Math.max(0, Math.min(1, 0.3 + (r + c) * 0.015))
    )
  ),
  beliefMap: Array.from({ length: 18 }, (_, r) =>
    Array.from({ length: 18 }, (_, c) =>
      Math.max(0, Math.min(1, 0.5 - Math.abs(r - 9) * 0.03 - Math.abs(c - 9) * 0.03))
    )
  ),
  topFeatures: [
    { name: "army_advantage", weight: 0.42 },
    { name: "general_distance", weight: 0.31 },
    { name: "city_ownership", weight: 0.18 },
    { name: "fog_boundary", weight: 0.09 },
  ],
  faithfulness: "PARTIAL",
  faithfulnessChecks: [
    { method: "roar", status: "PARTIAL", score: 0.61, notes: "Moderate faithfulness on held-out decisions" },
    { method: "perturbation", status: "EXPERIMENTAL", notes: "Under development" },
    { method: "linear_proxy", status: "NOT_EVALUATED" },
    { method: "insertion", status: "NOT_EVALUATED" },
  ],
  notes: "Demo explanation — synthetic gradient saliency.",
};

export const demoCounterfactuals: CounterfactualRecord[] = [
  {
    id: "cf-demo-001",
    explanationId: "expl-demo-001",
    altAction: { srcRow: 3, srcCol: 4, dstRow: 3, dstCol: 5 },
    altValueEstimate: 0.52,
    difference: -0.08,
    notes: "Alternative lateral move — lower value estimate.",
  },
  {
    id: "cf-demo-002",
    explanationId: "expl-demo-001",
    altAction: { srcRow: 3, srcCol: 4, dstRow: 2, dstCol: 4 },
    altValueEstimate: 0.48,
    difference: -0.12,
    notes: "Retreat — significantly lower value estimate.",
  },
];
