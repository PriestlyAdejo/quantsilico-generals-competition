import { SubmissionPipeline, SubmissionPackage } from "../../types/submission";

export const currentPackage: SubmissionPackage = {
  id: "pkg-heuristic-v2f",
  kind: "IMPORTED_PROJECT_EVIDENCE",
  candidateName: "heuristic_v2f_plus_planner_terminal_fix",
  checkpoint: "heuristic-v2f-evaluated",
  sha256: null,
  sizeBytes: null,
  builtAt: null,
  validatedWindowsAt: null,
  validatedLinuxAt: null,
  notes: "Package not yet built — blocked by discovery development gate.",
};

export const submissionPipeline: SubmissionPipeline = {
  id: "pipeline-001",
  kind: "IMPORTED_PROJECT_EVIDENCE",
  currentStage: "CANDIDATE_SELECTED",
  steps: [
    { stage: "CANDIDATE_SELECTED", label: "Candidate Selected", status: "active" },
    { stage: "PACKAGE_BUILT", label: "Package Built", status: "pending" },
    { stage: "WINDOWS_VALIDATED", label: "Windows Validated", status: "pending" },
    { stage: "LINUX_VALIDATED", label: "Linux Validated", status: "pending" },
    { stage: "UPLOAD_READY", label: "Upload Ready", status: "pending" },
    { stage: "MANUALLY_SUBMITTED", label: "Manually Submitted", status: "pending" },
    { stage: "PORTAL_ACCEPTED", label: "Portal Accepted", status: "pending" },
    { stage: "QUALIFIED", label: "Qualified", status: "pending" },
  ],
  activePackage: currentPackage,
  updatedAt: "2024-11-06T00:00:00.000Z",
};
