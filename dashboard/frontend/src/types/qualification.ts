import { DataSourceKind, ID, WDL, MetricAvailability, AvailableMetric } from "./common";

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
  steps: { step: Phase9QStep; status: StepStatus; completedAt?: string; label?: string }[];
}

export interface QualStageInfo {
  id: string;
  label: string;
  internalId: string;
  status: StepStatus;
  explains: string;
  evidence: string;
  passMeans: string;
  failMeans: string;
  blocksNext: boolean;
  perspective: string;
  reasons?: string[];
}

export interface QualCandidate {
  id: ID;
  kind: DataSourceKind;
  name: string;
  checkpoint: string;
  phase9q: Phase9QState;
  /** Null when availability is not RECORDED */
  screeningWDL: WDL | null;
  developmentWDL: WDL | null;
  holdoutWDL?: WDL | null;
  screeningAvailability: MetricAvailability;
  developmentAvailability: MetricAvailability;
  discovery: AvailableMetric<number>;
  conversion: AvailableMetric<number>;
  failureClass?: string;
  terminalTurnP50: number | null;
  terminalTurnP95: number | null;
  submittedAt: string | null;
  notes?: string;
  stages?: QualStageInfo[];
}

export interface QualificationSummary {
  totalCandidates: number;
  passed: number;
  failed: number;
  inProgress: number;
  phase9qCurrent: Phase9QStep;
  expanderRecord: WDL | null;
  discoveryRate: number | null;
  conversionRate: number | null;
  discoveryAvailability: MetricAvailability;
  conversionAvailability: MetricAvailability;
}
