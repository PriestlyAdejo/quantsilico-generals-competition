import { DataSourceKind, ID } from "./common";

export type PopulationSuite = "PFSP_LATEST" | "DEMO_SUITE" | "HISTORICAL";

export interface PopulationEntry {
  id: ID;
  kind: DataSourceKind;
  name: string;
  checkpoint: string;
  payoffs: (number | null)[];
  pfspWeight: number;
  gamesPlayed: number;
  winRate: number;
  isMainAgent: boolean;
}

export interface PayoffMatrix {
  agents: string[];
  matrix: (number | null)[][];
  suite: PopulationSuite;
  updatedAt: string;
}
