import { ID } from "./common";

export interface PortalObservation {
  id: ID;
  kind: "OFFICIAL_PORTAL_OBSERVATION";
  candidateName: string;
  rank?: number;
  score?: number;
  observedAt: string;
  notes?: string;
}

export interface ManualSubmissionRecord {
  id: ID;
  kind: "MANUALLY_RECORDED" | "DEMO";
  candidateName: string;
  submittedAt: string;
  method: string;
  confirmedAt?: string;
  notes?: string;
}
