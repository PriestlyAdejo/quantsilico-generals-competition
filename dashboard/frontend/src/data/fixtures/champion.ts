import { ChampionWorkspace, PromotionChecklist } from "../../types/champion";
import { SCHEMA_VERSION } from "../../types/common";

export const promotionChecklist: PromotionChecklist = {
  id: "checklist-heuristic-v2f",
  kind: "IMPORTED_PROJECT_EVIDENCE",
  candidateId: "cand-heuristic-v2f",
  rows: [
    { gate: "Screening", status: "PASS", detail: "21W/27D/0L screening result recorded." },
    {
      gate: "Discovery Development Gate",
      status: "FAIL",
      detail: "Discovery rate 0.438 — below required threshold.",
      blockerReason: "0.438 < minimum threshold. Downstream gates blocked.",
    },
    { gate: "Holdout Evaluation", status: "PENDING", detail: "Not started — blocked by discovery gate." },
    { gate: "Linux Parity", status: "PENDING", detail: "Not started." },
    {
      gate: "Package Build",
      status: "BLOCKED",
      detail: "Cannot package until discovery gate passes.",
      blockerReason: "Upstream gate FAILED.",
    },
    { gate: "PPO Training", status: "PENDING", detail: "PPO NOT STARTED." },
  ],
  overallStatus: "FAIL",
  promotionAllowed: false,
};

export const championWorkspace: ChampionWorkspace = {
  id: "champion-workspace-001",
  kind: "IMPORTED_PROJECT_EVIDENCE",
  schemaVersion: SCHEMA_VERSION,
  currentChampion: null,
  currentCandidate: "heuristic_v2f_plus_planner_terminal_fix",
  currentSubmittedBaseline: null,
  checklist: promotionChecklist,
  updatedAt: "2024-11-06T00:00:00.000Z",
};
