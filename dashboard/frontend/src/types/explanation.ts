import { DataSourceKind, ID } from "./common";

export type FaithfulnessStatus = "VERIFIED" | "PARTIAL" | "EXPERIMENTAL" | "FAILED" | "NOT_EVALUATED";

export interface FaithfulnessCheck {
  method: string;
  status: FaithfulnessStatus;
  score?: number;
  notes?: string;
}

export interface CounterfactualRecord {
  id: ID;
  explanationId: ID;
  altAction: { srcRow: number; srcCol: number; dstRow: number; dstCol: number };
  altValueEstimate: number;
  difference: number;
  notes?: string;
}

export interface ExplanationRecord {
  id: ID;
  kind: DataSourceKind;
  matchId: ID;
  turn: number;
  method: string;
  saliencyMap: number[][];
  beliefMap?: number[][];
  topFeatures: { name: string; weight: number }[];
  faithfulness: FaithfulnessStatus;
  faithfulnessChecks?: FaithfulnessCheck[];
  horizonPredictions?: { horizon: number; boardState: number[][] }[];
  counterfactualIds?: ID[];
  notes?: string;
}
