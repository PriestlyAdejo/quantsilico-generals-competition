import { DataSourceKind, ID, WDL } from "./common";

export type Phase9QStep =
  | "screening"
  | "development"
  | "holdout"
  | "package"
  | "linux_parity"
  | "upload_ready"
  | "portal";

export type StepStatus = "complete" | "active" | "pending" | "failed";

export interface Phase9QState {
  currentStep: Phase9QStep;
  steps: { step: Phase9QStep; status: StepStatus; completedAt?: string }[];
}

export interface QualCandidate {
  id: ID;
  kind: DataSourceKind;
  name: string;
  checkpoint: string;
  phase9q: Phase9QState;
  screeningWDL: WDL;
  developmentWDL: WDL;
  holdoutWDL?: WDL;
  discoveryRate: number;
  conversionRate?: number;
  failureClass?: string;
  terminalTurnP50: number;
  terminalTurnP95: number;
  submittedAt: string;
  notes?: string;
}

export interface QualificationSummary {
  totalCandidates: number;
  passed: number;
  failed: number;
  inProgress: number;
  phase9qCurrent: Phase9QStep;
  expanderRecord: WDL;
  discoveryRate: number;
  conversionRate: number;
}
