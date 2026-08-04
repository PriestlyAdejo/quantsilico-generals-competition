import { DataSourceKind, ID } from "./common";
import { MatchFrame, MatchConfig } from "./match";

export interface ReplayEvent {
  turn: number;
  type: "capture" | "general_captured" | "army_move" | "city_taken" | "fog_lifted";
  label: string;
  player?: "player1" | "player2";
}

export interface DecisionRecord {
  turn: number;
  srcRow: number;
  srcCol: number;
  dstRow: number;
  dstCol: number;
  armiesMoved: number;
  policyLogit: number;
  valueEstimate: number;
  topKActions: { row: number; col: number; logit: number }[];
}

export interface ReplayRecord {
  id: ID;
  kind: DataSourceKind;
  matchId: ID;
  config: MatchConfig;
  frames: MatchFrame[];
  events: ReplayEvent[];
  decisions: DecisionRecord[];
  outcome: "player1_win" | "player2_win" | "draw";
  totalTurns: number;
  createdAt: string;
  label?: string;
}
