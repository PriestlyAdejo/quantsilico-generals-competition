import { DataSourceKind, ID } from "./common";

export type CiStatus = "passed" | "failed" | "running" | "skipped";

export type RepoEvidenceStatus =
  | "PASS"
  | "FAIL"
  | "NOT_RUN"
  | "NOT_RECORDED"
  | "NOT_CONFIGURED"
  | "NOT_APPLICABLE"
  | "UNKNOWN"
  | CiStatus;

export interface CommitRecord {
  id: ID;
  kind: DataSourceKind;
  sha: string;
  message: string;
  author: string;
  committedAt: string;
  branch: string;
}

export interface CiRun {
  id: ID;
  kind: DataSourceKind;
  commitSha: string;
  status: CiStatus;
  suite: string;
  durationSecs?: number;
  startedAt: string;
}

export interface EnvironmentLock {
  id: ID;
  kind: DataSourceKind;
  name: string;
  lockedBy: string;
  lockedAt: string;
  reason: string;
}

export interface RepositoryStatus {
  id: ID;
  kind: DataSourceKind;
  schemaVersion: string;
  branch: string;
  engineSha: string;
  hardware: string;
  linuxParityStatus: RepoEvidenceStatus | string;
  testStatus: RepoEvidenceStatus | string;
  packageStatus: "NOT_BUILT" | "BUILT" | "VALIDATED" | string;
  updatedAt: string;
}
