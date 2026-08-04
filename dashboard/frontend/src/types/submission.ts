import { DataSourceKind, ID } from "./common";

export type PackageStage =
  | "CANDIDATE_SELECTED"
  | "PACKAGE_BUILT"
  | "WINDOWS_VALIDATED"
  | "LINUX_VALIDATED"
  | "UPLOAD_READY"
  | "MANUALLY_SUBMITTED"
  | "PORTAL_ACCEPTED"
  | "QUALIFIED";

export type PackageAction =
  | "simulate_build"
  | "simulate_validate_windows"
  | "simulate_validate_linux"
  | "mark_upload_ready";

export interface PipelineStep {
  stage: PackageStage;
  label: string;
  status: "complete" | "active" | "pending" | "blocked";
  completedAt?: string;
  blockerReason?: string;
}

export interface SubmissionPackage {
  id: ID;
  kind: DataSourceKind;
  candidateName: string;
  checkpoint: string;
  sha256: string | null;
  sizeBytes: number | null;
  builtAt: string | null;
  validatedWindowsAt: string | null;
  validatedLinuxAt: string | null;
  notes?: string;
}

export interface SubmissionPipeline {
  id: ID;
  kind: DataSourceKind;
  currentStage: PackageStage;
  steps: PipelineStep[];
  activePackage: SubmissionPackage | null;
  updatedAt: string;
}
