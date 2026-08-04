import { TrainingBlockedState, TrainingRun } from "../../types/training";

export const trainingBlockedState: TrainingBlockedState = {
  reason: "Development gate not passed. Phase 9Q development evaluation incomplete.",
  gateFailedAt: "2024-11-03T14:22:00Z",
  requiredAction: "Pass Phase 9Q development evaluation (≥10W or ≥30D, 0 losses) before launching new training runs.",
};

export const demoTrainingRun: TrainingRun = {
  id: "run-demo-smoke",
  kind: "DEMO",
  label: "SMOKE — Sanity check",
  preset: "smoke",
  status: "idle",
  totalSteps: 10_000,
  currentStep: 0,
  metrics: [],
  blockedReason: undefined,
};

export const completedRuns: TrainingRun[] = [
  {
    id: "run-expander-dev",
    kind: "IMPORTED_PROJECT_EVIDENCE",
    label: "Expander development run",
    preset: "dev",
    status: "complete",
    checkpoint: "ckpt-expander-v3-step-480k",
    totalSteps: 480_000,
    currentStep: 480_000,
    metrics: [],
    startedAt: "2024-10-09T08:00:00Z",
    estimatedCompletion: "2024-10-22T18:00:00Z",
  },
];
