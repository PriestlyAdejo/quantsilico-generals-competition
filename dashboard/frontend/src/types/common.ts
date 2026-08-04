export type DataSourceKind =
  | "DEMO"
  | "IMPORTED_PROJECT_EVIDENCE"
  | "OFFICIAL_PORTAL_OBSERVATION"
  | "MANUALLY_RECORDED";

export const SCHEMA_VERSION = "2.0.0";

export interface Timestamped {
  createdAt: string;
  updatedAt?: string;
}

export interface WithProvenance {
  kind: DataSourceKind;
}

export type ID = string;

export interface ConfidenceInterval {
  lower: number;
  upper: number;
  confidence: number;
}

export interface WDL {
  wins: number;
  draws: number;
  losses: number;
}

export function wdlTotal(wdl: WDL): number {
  return wdl.wins + wdl.draws + wdl.losses;
}

export function wdlWinRate(wdl: WDL): number {
  const t = wdlTotal(wdl);
  return t === 0 ? 0 : wdl.wins / t;
}
