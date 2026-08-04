export type DataSourceKind =
  | "DEMO"
  | "IMPORTED_PROJECT_EVIDENCE"
  | "OFFICIAL_PORTAL_OBSERVATION"
  | "MANUALLY_RECORDED";

export type MetricAvailability =
  | "RECORDED"
  | "MISSING"
  | "NOT_APPLICABLE"
  | "SCHEMA_UNSUPPORTED";

export const SCHEMA_VERSION = "2.0.0";

/** Authoritative submitted heuristic — never alias to terminal_form. */
export const SUBMITTED_CANDIDATE_ID = "heuristic_v2f_plus_planner_terminal_fix";

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

export interface AvailableMetric<T> {
  value: T | null;
  availability: MetricAvailability;
  source?: string;
  suite?: string;
  note?: string;
}

export function wdlTotal(wdl: WDL): number {
  return wdl.wins + wdl.draws + wdl.losses;
}

export function wdlWinRate(wdl: WDL): number {
  const t = wdlTotal(wdl);
  return t === 0 ? 0 : wdl.wins / t;
}

export function recordedWdl(wins: number, draws: number, losses: number, source?: string): AvailableMetric<WDL> {
  return {
    value: { wins, draws, losses },
    availability: "RECORDED",
    source,
  };
}

export function missingMetric<T = never>(note?: string): AvailableMetric<T> {
  return { value: null, availability: "MISSING", note };
}
