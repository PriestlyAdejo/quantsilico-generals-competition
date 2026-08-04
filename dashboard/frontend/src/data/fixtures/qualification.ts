import { QualCandidate, QualificationSummary } from "../../types/qualification";

export const heuristicCandidate: QualCandidate = {
  id: "cand-heuristic-v2f",
  kind: "IMPORTED_PROJECT_EVIDENCE",
  name: "heuristic_v2f_plus_planner_terminal_fix",
  checkpoint: "heuristic-v2f-evaluated",
  phase9q: {
    currentStep: "development",
    steps: [
      { step: "screening",    status: "complete", completedAt: "2024-11-01" },
      { step: "development",  status: "active" },
      { step: "holdout",      status: "pending" },
      { step: "package",      status: "pending" },
      { step: "linux_parity", status: "pending" },
      { step: "upload_ready", status: "pending" },
      { step: "portal",       status: "pending" },
    ],
  },
  screeningWDL:    { wins: 21, draws: 27, losses: 0 },
  developmentWDL:  { wins: 21, draws: 27, losses: 0 },
  discoveryRate:   0.438,
  conversionRate:  1.0,
  failureClass:    "DISCOVERY_DEVELOPMENT_GATE",
  terminalTurnP50: 175,
  terminalTurnP95: 330,
  submittedAt: "2024-11-06",
  notes: "Current candidate. Discovery gate FAILED: 0.438 < threshold. PPO not started.",
};

export const expanderCandidate: QualCandidate = {
  id: "cand-expander-v3",
  kind: "IMPORTED_PROJECT_EVIDENCE",
  name: "CNN-v3-Expander",
  checkpoint: "ckpt-expander-v3-step-480k",
  phase9q: {
    currentStep: "upload_ready",
    steps: [
      { step: "screening",    status: "complete", completedAt: "2024-10-20" },
      { step: "development",  status: "complete", completedAt: "2024-10-28" },
      { step: "holdout",      status: "complete", completedAt: "2024-11-02" },
      { step: "package",      status: "complete", completedAt: "2024-11-04" },
      { step: "linux_parity", status: "complete", completedAt: "2024-11-05" },
      { step: "upload_ready", status: "active" },
      { step: "portal",       status: "pending" },
    ],
  },
  screeningWDL:    { wins: 11, draws: 37, losses: 0 },
  developmentWDL:  { wins: 11, draws: 37, losses: 0 },
  discoveryRate:   0.44,
  conversionRate:  0.53,
  failureClass:    "DEATHTOUCH_NOT_EXPLOITED",
  terminalTurnP50: 182,
  terminalTurnP95: 341,
  submittedAt: "2024-10-18",
  notes: "Historical — Expander experiment. Exact timestamp not recorded.",
};

export const demoCandidate: QualCandidate = {
  id: "cand-demo-baseline",
  kind: "DEMO",
  name: "CNN-v2-Baseline",
  checkpoint: "ckpt-baseline-v2-step-200k",
  phase9q: {
    currentStep: "screening",
    steps: [
      { step: "screening",    status: "active" },
      { step: "development",  status: "pending" },
      { step: "holdout",      status: "pending" },
      { step: "package",      status: "pending" },
      { step: "linux_parity", status: "pending" },
      { step: "upload_ready", status: "pending" },
      { step: "portal",       status: "pending" },
    ],
  },
  screeningWDL:    { wins: 4, draws: 9, losses: 2 },
  developmentWDL:  { wins: 0, draws: 0, losses: 0 },
  discoveryRate:   0.21,
  terminalTurnP50: 210,
  terminalTurnP95: 390,
  submittedAt: "2024-11-06",
};

export const allCandidates: QualCandidate[] = [heuristicCandidate, expanderCandidate, demoCandidate];

export const qualSummary: QualificationSummary = {
  totalCandidates: 3,
  passed: 0,
  failed: 1,
  inProgress: 1,
  phase9qCurrent: "development",
  expanderRecord: { wins: 11, draws: 37, losses: 0 },
  discoveryRate: 0.438,
  conversionRate: 1.0,
};
