import { ExperimentRecord } from "../../types/experiment";
import { SCHEMA_VERSION } from "../../types/common";

export const currentExperiment: ExperimentRecord = {
  id: "exp-current-heuristic",
  kind: "IMPORTED_PROJECT_EVIDENCE",
  schemaVersion: SCHEMA_VERSION,
  label: "heuristic_v2f_plus_planner_terminal_fix — Phase 9Q development",
  candidate: "heuristic_v2f_plus_planner_terminal_fix",
  opponent: "hunter_heuristic",
  suite: "development",
  lifecycle: "COMPLETE",
  wdl: { wins: 21, draws: 27, losses: 0 },
  discoveryRate: 0.438,
  conversionRate: 1.0,
  terminalTurnP50: 175,
  terminalTurnP95: 330,
  discoveryGate: "FAILED",
  developmentGate: "PENDING",
  observedAt: "2024-11-06T00:00:00.000Z",
  dateLabel: "2024-11-06",
  notes: "Discovery gate FAILED: 0.438 < threshold.",
};

export const historicalExpanderExperiment: ExperimentRecord = {
  id: "exp-historical-expander",
  kind: "IMPORTED_PROJECT_EVIDENCE",
  schemaVersion: SCHEMA_VERSION,
  label: "CNN-v3-Expander — Historical experiment",
  candidate: "CNN-v3-Expander",
  opponent: "hunter_heuristic",
  suite: "development",
  lifecycle: "COMPLETE",
  wdl: { wins: 11, draws: 37, losses: 0 },
  discoveryRate: 0.44,
  conversionRate: 0.53,
  terminalTurnP50: 182,
  terminalTurnP95: 341,
  discoveryGate: "PASSED",
  developmentGate: "PASSED",
  observedAt: null,
  dateLabel: "Historical — exact timestamp not recorded",
  notes: "Historical Expander experiment result.",
};

export const demoExperiment: ExperimentRecord = {
  id: "exp-demo-001",
  kind: "DEMO",
  schemaVersion: SCHEMA_VERSION,
  label: "Demo Experiment — Synthetic",
  candidate: "demo_cnn_v2",
  opponent: "legal_random",
  suite: "smoke",
  lifecycle: "RUNNING",
  wdl: { wins: 3, draws: 2, losses: 1 },
  discoveryRate: 0.3,
  discoveryGate: "PENDING",
  developmentGate: "PENDING",
  observedAt: null,
  dateLabel: "Demo — synthetic session data",
};

export const allExperiments: ExperimentRecord[] = [
  currentExperiment,
  historicalExpanderExperiment,
  demoExperiment,
];
