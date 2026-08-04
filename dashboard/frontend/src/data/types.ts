export type Capability = { enabled: boolean; reason: string };

export type CapabilitiesResponse = {
  schema_version: number;
  capabilities: Record<string, Capability>;
};

export type GateCurrent = {
  learning_readiness: string;
  heuristic_development: string;
  pre_ppo_submission: string;
  portal_submission: string;
  learned_promotion: string;
};

export type GateStatusSplit = {
  schema_version: number;
  kind: string;
  current: GateCurrent;
  historical_observations: Record<string, unknown>[];
};

export type OverviewResponse = {
  schema_version: number;
  branch: string;
  commit: string;
  dirty: boolean;
  engine_commit: string;
  research_phase: string;
  active_submitted_package: SubmissionPackage;
  heuristic_baseline: string;
  learned_champion: string | null;
  learned_champion_note: string;
  gate_status: GateStatusSplit;
  gate_board: Record<string, string>;
  metrics?: Record<string, string | number | null | undefined>;
  active_jobs: JobRecord[];
  learning_smoke?: Record<string, unknown>;
  frontend_build?: Record<string, unknown> | null;
  candidate_identity?: Record<string, unknown>;
};

export type SubmissionPackage = {
  candidate: string;
  package_path: string;
  package_sha256: string;
  config_hash?: string;
  authoritative_policy_source_commit: string;
  embedded_bot_commit: string;
  embedded_metadata_status: string;
  repository_completion_commit: string;
  lifecycle?: string;
  portal_gate_name?: string;
  portal_verdict?: string;
  metadata_note?: string;
  windows_validation?: string;
  linux_parity?: string;
};

export type JobRecord = {
  schema_version?: number;
  job_id: string;
  job_type: string;
  state: string;
  candidate?: string;
  opponent?: string;
  seed?: number;
  match_record?: Record<string, unknown> | null;
  replay_id?: string | null;
  replay_status?: string | null;
  error?: string | null;
  notes?: string[];
};

export type ChartProvenance = {
  manifestId: string;
  kind: string;
  observedAt?: string | null;
  seedSplit?: string | null;
  architecture?: string | null;
  candidate?: string | null;
  missingFields?: string[];
};

export type ApiErrorKind = "backend_unavailable" | "schema_mismatch" | "http" | "not_found";

export class ApiError extends Error {
  kind: ApiErrorKind;
  status?: number;
  constructor(kind: ApiErrorKind, message: string, status?: number) {
    super(message);
    this.kind = kind;
    this.status = status;
  }
}
