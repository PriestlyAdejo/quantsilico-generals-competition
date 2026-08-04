import { DataSourceKind, ID, WDL } from "./common";

export interface ActiveJob {
  id: ID;
  label: string;
  progress: number;
  status: "running" | "complete" | "failed";
}

export interface ExperimentTimelineEntry {
  id: ID;
  label: string;
  startedAt: string;
  completedAt: string | null;
  status: "complete" | "running" | "planned";
  dateLabel?: string;
}

export interface ApplicationStatusRecord {
  schemaVersion: string;
  id: ID;
  kind: DataSourceKind;
  branch: string;
  engineSha: string;
  hardware: string;
  currentCandidate: string;
  currentSubmittedBaseline: string | null;
  currentChampion: string | null;
  currentPhase: string;
  updatedAt: string;
}

export interface OverviewRecord {
  schemaVersion: string;
  id: ID;
  kind: DataSourceKind;
  wdlHistory: { week: string; label: string; wdl: WDL; kind: DataSourceKind; dateLabel?: string }[];
  qualificationFunnel: { stage: string; count: number }[];
  experimentTimeline: ExperimentTimelineEntry[];
  activeJobs: ActiveJob[];
  currentResult: WDL | null;
  discoveryRate: number;
  conversionRate: number;
  ppoStatus: "NOT_STARTED" | "RUNNING" | "COMPLETE";
  blocker: string | null;
  currentCandidate: string;
}
