import { DataSourceKind, ID, WDL } from "./common";

export type ModelArchitecture =
  | "heuristic"
  | "mlp_control"
  | "recurrent_cnn"
  | "recurrent_graph_belief"
  | "graph_belief_pyg_research";

export type ModelLifecycle =
  | "SCAFFOLDED"
  | "SMOKE_TESTED"
  | "TRAINED"
  | "EVALUATED"
  | "REJECTED"
  | "REJECTED_INCOMPATIBLE";

export type CompetitiveRole = "BASELINE" | "CHALLENGER" | "CHAMPION" | "NONE";

export type DeliveryStatus =
  | "NOT_PACKAGED"
  | "PACKAGED"
  | "UPLOAD_READY"
  | "SUBMITTED"
  | "NOT_APPLICABLE"
  | "NOT_RECORDED";

export interface ModelRecord {
  id: ID;
  kind: DataSourceKind;
  schemaVersion: string;
  name: string;
  architecture: ModelArchitecture;
  checkpoint: string;
  lifecycle: ModelLifecycle;
  role: CompetitiveRole;
  deliveryStatus: DeliveryStatus;
  /** Null when parameter count is not recorded on the model DTO. */
  parameters: number | null;
  trainingSteps: number | null;
  /** Null when evaluation WDL is not recorded. */
  wdl: WDL | null;
  wdlAvailability: import("./common").MetricAvailability;
  eloEstimate?: number;
  discoveryRate?: number;
  promotionState?: "BLOCKED" | "ELIGIBLE" | "PROMOTED" | "NONE";
  blockerReason?: string;
  parentId?: ID;
  createdAt: string;
  notes?: string;
}
