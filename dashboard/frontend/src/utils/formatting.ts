import { WDL, wdlTotal, type AvailableMetric, type MetricAvailability } from "../types/common";

export function fmtWDL(wdl: WDL | null | undefined, availability?: MetricAvailability): string {
  if (availability === "MISSING" || availability === "NOT_APPLICABLE" || availability === "SCHEMA_UNSUPPORTED") {
    return availability === "NOT_APPLICABLE" ? "NOT APPLICABLE" : "NOT RECORDED";
  }
  if (!wdl) return "NOT RECORDED";
  return `${wdl.wins}W / ${wdl.draws}D / ${wdl.losses}L`;
}

export function fmtAvailableWdl(m: AvailableMetric<WDL> | null | undefined): string {
  if (!m) return "NOT RECORDED";
  return fmtWDL(m.value, m.availability);
}

export function fmtWinRate(wdl: WDL | null | undefined, availability?: MetricAvailability): string {
  if (availability === "MISSING" || !wdl) return "—";
  const t = wdlTotal(wdl);
  if (t === 0 && availability === "RECORDED") return "0.0%";
  if (t === 0) return "—";
  return `${((wdl.wins / t) * 100).toFixed(1)}%`;
}

/** Rate in [0,1]. Missing stays honest. */
export function fmtPct(
  v: number | null | undefined,
  availability: MetricAvailability = v == null ? "MISSING" : "RECORDED",
): string {
  if (availability !== "RECORDED" || v == null || !Number.isFinite(v)) {
    if (availability === "NOT_APPLICABLE") return "NOT APPLICABLE";
    return "NOT RECORDED";
  }
  return `${(v * 100).toFixed(1)}%`;
}

export function fmtAvailablePct(m: AvailableMetric<number> | null | undefined): string {
  if (!m) return "NOT RECORDED";
  return fmtPct(m.value, m.availability);
}

export function fmtK(n: number | null | undefined, availability: MetricAvailability = n == null ? "MISSING" : "RECORDED"): string {
  if (availability !== "RECORDED" || n == null || !Number.isFinite(n)) return "NOT RECORDED";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "NOT RECORDED";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "NOT RECORDED";
  return d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
}

export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return "NOT RECORDED";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "NOT RECORDED";
  return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}

export function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

export function fmtMissingLabel(availability: MetricAvailability = "MISSING"): string {
  switch (availability) {
    case "NOT_APPLICABLE":
      return "NOT APPLICABLE";
    case "SCHEMA_UNSUPPORTED":
      return "SCHEMA UNSUPPORTED";
    case "RECORDED":
      return "RECORDED";
    default:
      return "NOT RECORDED";
  }
}
