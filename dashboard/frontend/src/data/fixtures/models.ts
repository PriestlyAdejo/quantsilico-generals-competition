import { ModelRecord } from "../../types/model";
import { SCHEMA_VERSION } from "../../types/common";

export const heuristicModel: ModelRecord = {
  id: "model-heuristic-v2f",
  kind: "IMPORTED_PROJECT_EVIDENCE",
  schemaVersion: SCHEMA_VERSION,
  name: "heuristic_v2f_plus_planner_terminal_fix",
  architecture: "heuristic",
  checkpoint: "heuristic-v2f-evaluated",
  lifecycle: "EVALUATED",
  role: "CHALLENGER",
  deliveryStatus: "NOT_RECORDED",
  parameters: null,
  trainingSteps: null,
  wdl: { wins: 21, draws: 27, losses: 0 },
  wdlAvailability: "RECORDED",
  discoveryRate: 0.438,
  promotionState: "BLOCKED",
  blockerReason: "DISCOVERY DEVELOPMENT GATE — 0.438 < threshold",
  createdAt: "2024-11-06T00:00:00.000Z",
  notes: "Current candidate. Not a learned model — rule-based heuristic with planner.",
};

export const recurrentCnnModel: ModelRecord = {
  id: "model-recurrent-cnn-001",
  kind: "DEMO",
  schemaVersion: SCHEMA_VERSION,
  name: "Recurrent CNN — Demo",
  architecture: "recurrent_cnn",
  checkpoint: "ckpt-rcnn-smoke-step-5k",
  lifecycle: "SMOKE_TESTED",
  role: "NONE",
  deliveryStatus: "NOT_APPLICABLE",
  parameters: 2_400_000,
  trainingSteps: 5000,
  wdl: { wins: 1, draws: 3, losses: 6 },
  wdlAvailability: "RECORDED",
  createdAt: new Date().toISOString(),
  notes: "Demo — synthetic smoke-test data only.",
};

export const graphBeliefModel: ModelRecord = {
  id: "model-graph-belief-001",
  kind: "DEMO",
  schemaVersion: SCHEMA_VERSION,
  name: "Graph Belief PyG — Research",
  architecture: "graph_belief_pyg_research",
  checkpoint: "ckpt-gbpyg-scaffold",
  lifecycle: "SCAFFOLDED",
  role: "NONE",
  deliveryStatus: "NOT_APPLICABLE",
  parameters: 5_100_000,
  trainingSteps: null,
  wdl: null,
  wdlAvailability: "MISSING",
  createdAt: new Date().toISOString(),
  notes: "Demo — architecture scaffolded, no training recorded.",
};

export const mlpControlModel: ModelRecord = {
  id: "model-mlp-control-001",
  kind: "DEMO",
  schemaVersion: SCHEMA_VERSION,
  name: "MLP Control",
  architecture: "mlp_control",
  checkpoint: "ckpt-mlp-scaffold",
  lifecycle: "SCAFFOLDED",
  role: "NONE",
  deliveryStatus: "NOT_APPLICABLE",
  parameters: 850_000,
  trainingSteps: null,
  wdl: null,
  wdlAvailability: "MISSING",
  createdAt: new Date().toISOString(),
  notes: "Demo — control baseline, not yet trained.",
};

export const allModels: ModelRecord[] = [
  heuristicModel,
  recurrentCnnModel,
  graphBeliefModel,
  mlpControlModel,
];
