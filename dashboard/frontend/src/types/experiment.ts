import { DataSourceKind, ID, WDL } from "./common";

export type ExperimentLifecycle = "PLANNED" | "RUNNING" | "COMPLETE" | "FAILED" | "ARCHIVED";
export type GateStatus = "PASSED" | "FAILED" | "PENDING" | "NOT_EVALUATED";

export interface ExperimentRecord {
  id: ID;
  kind: DataSourceKind;
  schemaVersion: string;
  label: string;
  candidate: string;
  opponent: string;
  suite: string;
  lifecycle: ExperimentLifecycle;
  /** Null when the manifest does not contain evaluation WDL. */
  wdl: WDL | null;
  wdlAvailability: import("./common").MetricAvailability;
  discoveryRate?: number;
  conversionRate?: number;
  terminalTurnP50?: number;
  terminalTurnP95?: number;
  discoveryGate: GateStatus;
  developmentGate: GateStatus;
  startedAt?: string;
  completedAt?: string;
  observedAt: string | null;
  dateLabel: string;
  notes?: string;
}

export interface ExperimentFilter {
  candidate?: string;
  opponent?: string;
  suite?: string;
  lifecycle?: ExperimentLifecycle;
  kind?: DataSourceKind;
}

export interface ExperimentComparison {
  ids: string[];
  records: ExperimentRecord[];
}
