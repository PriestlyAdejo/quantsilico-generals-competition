import { RepositoryStatus, CommitRecord, CiRun, EnvironmentLock } from "../../types/repository";
import { SCHEMA_VERSION } from "../../types/common";

export const repoStatus: RepositoryStatus = {
  id: "repo-status-001",
  kind: "IMPORTED_PROJECT_EVIDENCE",
  schemaVersion: SCHEMA_VERSION,
  branch: "feature/full-research-platform-v0",
  engineSha: "9e3b9d13cca51caa1bb07db48bb85c9e90ce0462",
  hardware: "RTX 3070 Laptop GPU",
  linuxParityStatus: "passed",
  testStatus: "passed",
  packageStatus: "NOT_BUILT",
  updatedAt: "2024-11-06T00:00:00.000Z",
};

export const recentCommits: CommitRecord[] = [
  {
    id: "commit-001",
    kind: "DEMO",
    sha: "9e3b9d1",
    message: "feat: add planner terminal fix to heuristic agent",
    author: "researcher",
    committedAt: new Date(Date.now() - 86400000).toISOString(),
    branch: "feature/full-research-platform-v0",
  },
  {
    id: "commit-002",
    kind: "DEMO",
    sha: "a1c2e3f",
    message: "fix: correct discovery gate threshold comparison",
    author: "researcher",
    committedAt: new Date(Date.now() - 172800000).toISOString(),
    branch: "feature/full-research-platform-v0",
  },
  {
    id: "commit-003",
    kind: "DEMO",
    sha: "b3d4f5a",
    message: "test: add unit tests for planner terminal conditions",
    author: "researcher",
    committedAt: new Date(Date.now() - 259200000).toISOString(),
    branch: "feature/full-research-platform-v0",
  },
];

export const ciRuns: CiRun[] = [
  {
    id: "ci-001",
    kind: "DEMO",
    commitSha: "9e3b9d1",
    status: "passed",
    suite: "smoke_tests",
    durationSecs: 42,
    startedAt: new Date(Date.now() - 86400000).toISOString(),
  },
  {
    id: "ci-002",
    kind: "DEMO",
    commitSha: "a1c2e3f",
    status: "passed",
    suite: "unit_tests",
    durationSecs: 87,
    startedAt: new Date(Date.now() - 172800000).toISOString(),
  },
];

export const environmentLocks: EnvironmentLock[] = [
  {
    id: "lock-001",
    kind: "DEMO",
    name: "gpu_training_slot",
    lockedBy: "phase9q_eval",
    lockedAt: new Date(Date.now() - 3600000).toISOString(),
    reason: "Phase 9Q evaluation — slot reserved",
  },
];
