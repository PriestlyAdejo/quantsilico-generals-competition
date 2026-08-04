import { QualCandidate, QualificationSummary } from "../../types/qualification";

export const heuristicCandidate: QualCandidate = {
  id: "cand-heuristic-v2f",
  kind: "IMPORTED_PROJECT_EVIDENCE",
  name: "heuristic_v2f_plus_planner_terminal_fix",
  checkpoint: "heuristic-v2f-evaluated",
  phase9q: {
    currentStep: "development",
    steps: [
      { step: "screening", status: "complete", completedAt: "2024-11-01", label: "Screening Evaluation" },
      { step: "development", status: "active", label: "Development Evaluation" },
      { step: "holdout", status: "pending", label: "Holdout Evaluation" },
      { step: "package", status: "pending", label: "Package Build" },
      { step: "linux_parity", status: "pending", label: "Linux Validation" },
      { step: "upload_ready", status: "pending", label: "Upload Ready" },
      { step: "portal", status: "pending", label: "Portal Accepted" },
    ],
  },
  screeningWDL: null,
  developmentWDL: { wins: 21, draws: 27, losses: 0 },
  screeningAvailability: "MISSING",
  developmentAvailability: "RECORDED",
  discovery: { value: 0.438, availability: "RECORDED", suite: "development" },
  conversion: { value: 1.0, availability: "RECORDED", suite: "development" },
  failureClass: "DISCOVERY_DEVELOPMENT_GATE",
  terminalTurnP50: null,
  terminalTurnP95: null,
  submittedAt: "2024-11-06",
  notes: "Current candidate. Discovery gate FAILED: 0.438 < threshold.",
};

export const expanderCandidate: QualCandidate = {
  id: "cand-expander-v3",
  kind: "IMPORTED_PROJECT_EVIDENCE",
  name: "CNN-v3-Expander",
  checkpoint: "ckpt-expander-v3-step-480k",
  phase9q: {
    currentStep: "upload_ready",
    steps: [
      { step: "screening", status: "complete", label: "Screening Evaluation" },
      { step: "development", status: "complete", label: "Development Evaluation" },
      { step: "holdout", status: "complete", label: "Holdout Evaluation" },
      { step: "package", status: "complete", label: "Package Build" },
      { step: "linux_parity", status: "complete", label: "Linux Validation" },
      { step: "upload_ready", status: "active", label: "Upload Ready" },
      { step: "portal", status: "pending", label: "Portal Accepted" },
    ],
  },
  screeningWDL: { wins: 11, draws: 37, losses: 0 },
  developmentWDL: { wins: 11, draws: 37, losses: 0 },
  screeningAvailability: "RECORDED",
  developmentAvailability: "RECORDED",
  discovery: { value: 0.44, availability: "RECORDED" },
  conversion: { value: 0.53, availability: "RECORDED" },
  failureClass: "DEATHTOUCH_NOT_EXPLOITED",
  terminalTurnP50: 182,
  terminalTurnP95: 341,
  submittedAt: "2024-10-18",
  notes: "Historical — Expander experiment.",
};

export const demoCandidate: QualCandidate = {
  id: "cand-demo-baseline",
  kind: "DEMO",
  name: "CNN-v2-Baseline",
  checkpoint: "ckpt-baseline-v2-step-200k",
  phase9q: {
    currentStep: "screening",
    steps: [
      { step: "screening", status: "active", label: "Screening Evaluation" },
      { step: "development", status: "pending", label: "Development Evaluation" },
      { step: "holdout", status: "pending", label: "Holdout Evaluation" },
      { step: "package", status: "pending", label: "Package Build" },
      { step: "linux_parity", status: "pending", label: "Linux Validation" },
      { step: "upload_ready", status: "pending", label: "Upload Ready" },
      { step: "portal", status: "pending", label: "Portal Accepted" },
    ],
  },
  screeningWDL: { wins: 4, draws: 9, losses: 2 },
  developmentWDL: null,
  screeningAvailability: "RECORDED",
  developmentAvailability: "MISSING",
  discovery: { value: 0.21, availability: "RECORDED" },
  conversion: { value: null, availability: "MISSING" },
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
  discoveryAvailability: "RECORDED",
  conversionAvailability: "RECORDED",
};
