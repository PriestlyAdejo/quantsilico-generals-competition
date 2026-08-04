import { PortalObservation, ManualSubmissionRecord } from "../../types/competition";

export const portalObservations: PortalObservation[] = [];

export const manualRecords: ManualSubmissionRecord[] = [
  {
    id: "manual-demo-001",
    kind: "DEMO",
    candidateName: "demo_cnn_v2_baseline",
    submittedAt: new Date().toISOString(),
    method: "portal_web_ui",
    notes: "Demo record — synthetic. No actual portal submission has been recorded.",
  },
];
