import { DataSourceKind, ID, WDL } from "./common";

export type CellTerrain = "plain" | "mountain" | "city" | "general" | "obstacle";
export type CellOwner = "neutral" | "player1" | "player2" | "fog";

export interface CellState {
  terrain: CellTerrain;
  owner: CellOwner;
  armies: number;
  visible: boolean;
}

export interface BoardState {
  width: number;
  height: number;
  cells: CellState[][];
  turn: number;
}

export interface MatchFrame {
  turn: number;
  board: BoardState;
  p1Armies: number;
  p2Armies: number;
  p1Land: number;
  p2Land: number;
  events: string[];
}

export type PlayerSlot = "cnn_agent" | "heuristic" | "manual" | "random";
export type MapPreset = "standard" | "islands" | "maze" | "tournament";

export interface MatchConfig {
  player1: PlayerSlot;
  player2: PlayerSlot;
  mapPreset: MapPreset;
  mapSize: 18 | 19 | 20 | 21;
  speedMultiplier: number;
  label?: string;
}

export interface MatchRecord {
  id: ID;
  kind: DataSourceKind;
  config: MatchConfig;
  frames: MatchFrame[];
  outcome: "player1_win" | "player2_win" | "draw" | "in_progress";
  totalTurns: number;
  wdl?: WDL;
  startedAt: string;
  completedAt?: string;
}
