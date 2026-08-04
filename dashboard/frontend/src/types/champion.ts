import { DataSourceKind, ID } from "./common";

export type ChecklistStatus = "PASS" | "FAIL" | "BLOCKED" | "PENDING" | "NOT_EVALUATED";

export interface ChecklistRow {
  gate: string;
  status: ChecklistStatus;
  detail: string;
  blockerReason?: string;
}

export interface PromotionChecklist {
  id: ID;
  kind: DataSourceKind;
  candidateId: ID;
  rows: ChecklistRow[];
  overallStatus: ChecklistStatus;
  promotionAllowed: boolean;
}

export interface ChampionWorkspace {
  id: ID;
  kind: DataSourceKind;
  schemaVersion: string;
  currentChampion: string | null;
  currentCandidate: string;
  currentSubmittedBaseline: string | null;
  checklist: PromotionChecklist;
  updatedAt: string;
}
