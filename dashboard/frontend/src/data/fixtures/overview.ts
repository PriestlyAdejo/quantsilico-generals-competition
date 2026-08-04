import { OverviewRecord, ApplicationStatusRecord } from "../../types/overview";
import { SCHEMA_VERSION } from "../../types/common";

export const applicationStatus: ApplicationStatusRecord = {
  schemaVersion: SCHEMA_VERSION,
  id: "app-status-001",
  kind: "IMPORTED_PROJECT_EVIDENCE",
  branch: "feature/full-research-platform-v0",
  engineSha: "9e3b9d13cca51caa1bb07db48bb85c9e90ce0462",
  hardware: "RTX 3070 Laptop GPU",
  currentCandidate: "heuristic_v2f_plus_planner_terminal_fix",
  currentSubmittedBaseline: null,
  currentChampion: null,
  currentPhase: "Phase 9Q",
  updatedAt: "2024-11-06T00:00:00.000Z",
};

export const overviewRecord: OverviewRecord = {
  schemaVersion: SCHEMA_VERSION,
  id: "overview-001",
  kind: "IMPORTED_PROJECT_EVIDENCE",
  currentCandidate: "heuristic_v2f_plus_planner_terminal_fix",
  currentResult: { wins: 21, draws: 27, losses: 0 },
  discoveryRate: 0.438,
  conversionRate: 1.0,
  ppoStatus: "NOT_STARTED",
  blocker: "DISCOVERY DEVELOPMENT GATE",
  wdlHistory: [
    {
      week: "Historical",
      label: "CNN-v3-Expander",
      wdl: { wins: 11, draws: 37, losses: 0 },
      kind: "IMPORTED_PROJECT_EVIDENCE",
      dateLabel: "Historical — exact timestamp not recorded",
    },
    {
      week: "Current",
      label: "heuristic_v2f_plus_planner_terminal_fix",
      wdl: { wins: 21, draws: 27, losses: 0 },
      kind: "IMPORTED_PROJECT_EVIDENCE",
      dateLabel: "2024-11-06",
    },
  ],
  qualificationFunnel: [
    { stage: "Submitted", count: 14 },
    { stage: "Screening", count: 11 },
    { stage: "Development", count: 6 },
    { stage: "Holdout", count: 3 },
    { stage: "Packaged", count: 2 },
    { stage: "Upload Ready", count: 1 },
  ],
  experimentTimeline: [
    {
      id: "exp-historical-expander",
      label: "CNN-v3-Expander — 11W/37D/0L",
      startedAt: "2024-01-01",
      completedAt: "2024-01-01",
      status: "complete",
      dateLabel: "Historical — exact timestamp not recorded",
    },
    {
      id: "exp-current-heuristic",
      label: "heuristic_v2f_plus_planner_terminal_fix — 21W/27D/0L",
      startedAt: "2024-11-06",
      completedAt: null,
      status: "running",
    },
  ],
  activeJobs: [
    { id: "job-001", label: "Phase 9Q development evaluation", progress: 1.0, status: "complete" },
    { id: "job-002", label: "Discovery gate assessment", progress: 1.0, status: "complete" },
  ],
};
