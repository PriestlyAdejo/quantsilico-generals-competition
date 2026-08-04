import { WDL, wdlTotal } from "../types/common";

export function fmtWDL(wdl: WDL): string {
  return `${wdl.wins}W / ${wdl.draws}D / ${wdl.losses}L`;
}

export function fmtWinRate(wdl: WDL): string {
  const t = wdlTotal(wdl);
  if (t === 0) return "—";
  return `${((wdl.wins / t) * 100).toFixed(1)}%`;
}

export function fmtPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

export function fmtK(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

export function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
}

export function fmtTime(iso: string): string {
  return new Date(iso).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}

export function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}
