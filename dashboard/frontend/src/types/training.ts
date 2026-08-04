import { DataSourceKind, ID } from "./common";

export type TrainingStatus = "blocked" | "idle" | "running" | "paused" | "complete" | "failed";
export type TrainingPreset = "smoke" | "dev" | "full" | "ablation";

export interface TrainingMetric {
  step: number;
  policyLoss: number;
  valueLoss: number;
  entropy: number;
  klDiv: number;
  gradNorm: number;
  reward: number;
  winRate: number;
  drawRate: number;
  lossRate: number;
  stepsPerSec: number;
  gpuUtil: number;
}

export interface TrainingRun {
  id: ID;
  kind: DataSourceKind;
  label: string;
  preset: TrainingPreset;
  status: TrainingStatus;
  blockedReason?: string;
  checkpoint?: string;
  totalSteps: number;
  currentStep: number;
  metrics: TrainingMetric[];
  startedAt?: string;
  estimatedCompletion?: string;
}

export interface TrainingBlockedState {
  reason: string;
  gateFailedAt: string;
  requiredAction: string;
}
