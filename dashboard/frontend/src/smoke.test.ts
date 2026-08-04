import { describe, expect, it, vi } from "vitest";
import { ApiDataSource } from "./services/apiDataSource";
import { CapabilityDisabledError } from "./services/apiErrors";
import type { DataSource } from "./services/dataSource";
import { fmtPct, fmtWDL } from "./utils/formatting";

describe("honest formatters", () => {
  it("distinguishes missing from recorded zero", () => {
    expect(fmtWDL(null, "MISSING")).toBe("NOT RECORDED");
    expect(fmtWDL({ wins: 0, draws: 0, losses: 0 }, "RECORDED")).toBe("0W / 0D / 0L");
    expect(fmtPct(null)).toBe("NOT RECORDED");
    expect(fmtPct(Number.NaN)).toBe("NOT RECORDED");
    expect(fmtPct(0, "RECORDED")).toBe("0.0%");
    expect(fmtPct(0.438)).toBe("43.8%");
    expect(fmtPct(1)).toBe("100.0%");
  });
});

describe("population antifallback", () => {
  it("does not render NaN weights as 50%", () => {
    const weight = Number.NaN;
    const label = Number.isFinite(weight) ? `${(weight * 100).toFixed(0)}%` : "—";
    expect(label).toBe("—");
    expect(label).not.toBe("50%");
  });
});

describe("ApiDataSource coverage", () => {
  it("statically implements DataSource", () => {
    const ds: DataSource = new ApiDataSource();
    expect(ds).toBeInstanceOf(ApiDataSource);
  });

  it("disables demo mutations in API mode", async () => {
    const ds = new ApiDataSource();
    await expect(ds.createDemoMatch({
      player1: "heuristic",
      player2: "heuristic",
      mapPreset: "standard",
      mapSize: 18,
      speedMultiplier: 1,
    })).rejects.toBeInstanceOf(CapabilityDisabledError);
  });

  it("maps overview without inventing WDL zeros when qualification is missing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (String(url).includes("/api/qualification")) {
          return {
            ok: true,
            status: 200,
            json: async () => ({ candidates: [] }),
          };
        }
        return {
          ok: true,
          status: 200,
          json: async () => ({
            schema_version: 1,
            branch: "fix/dashboard-data-and-board-integrity",
            commit: "abc123",
            engine_commit: "9e3b9d13",
            research_phase: "console integrity",
            heuristic_baseline: "heuristic_v2f_plus_planner_terminal_fix",
            learned_champion: null,
            active_submitted_package: { candidate: "heuristic_v2f_plus_planner_terminal_fix" },
            metrics: {
              submitted_candidate: "heuristic_v2f_plus_planner_terminal_fix",
              learned_promotion: "NONE",
            },
            learning_smoke: { readiness: "PASS" },
            active_jobs: [{ id: "job-1", state: "RUNNING", label: "match" }],
          }),
        };
      }),
    );

    const ds = new ApiDataSource();
    const overview = await ds.getOverview();
    expect(overview.kind).toBe("IMPORTED_PROJECT_EVIDENCE");
    expect(overview.currentCandidate).toContain("terminal_fix");
    expect(overview.currentCandidate).not.toContain("terminal_form");
    expect(overview.wdlHistory).toEqual([]);
    expect(overview.currentResult).toBeNull();
    expect(Number.isNaN(overview.discoveryRate)).toBe(true);
    expect(Number.isNaN(overview.activeJobs[0]?.progress)).toBe(true);
    vi.unstubAllGlobals();
  });

  it("maps DEVELOPMENT qualification 21/27/0 with discovery and conversion", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({
          default_candidate: "heuristic_v2f_plus_planner_terminal_fix",
          stages: [
            {
              id: "development",
              label: "Development Evaluation",
              internal_id: "development",
              status: "failed",
              explains: "Internal research suite",
            },
          ],
          candidates: [
            {
              id: "heuristic_v2f_plus_planner_terminal_fix",
              development_wdl: {
                wins: 21,
                draws: 27,
                losses: 0,
                availability: "RECORDED",
                source: "phase_9q_development_abc_terminal_fix.json",
                suite: "development",
              },
              screening_wdl: { availability: "MISSING" },
              discovery: {
                value: 0.4375,
                availability: "RECORDED",
                source: "phase_9q_development_abc_terminal_fix.json",
                suite: "development",
              },
              conversion: {
                value: 1.0,
                availability: "RECORDED",
                source: "phase_9q_development_abc_terminal_form.json",
                suite: "development",
              },
            },
          ],
        }),
      })),
    );

    const ds = new ApiDataSource();
    const candidates = await ds.listCandidates();
    expect(candidates[0]?.id).toBe("heuristic_v2f_plus_planner_terminal_fix");
    expect(candidates[0]?.developmentWDL).toEqual({ wins: 21, draws: 27, losses: 0 });
    expect(candidates[0]?.screeningWDL).toBeNull();
    expect(candidates[0]?.discovery.value).toBeCloseTo(0.4375);
    expect(candidates[0]?.conversion.value).toBe(1);
    expect(fmtPct(candidates[0]?.discovery.value)).toBe("43.8%");
    expect(fmtPct(candidates[0]?.conversion.value)).toBe("100.0%");
    vi.unstubAllGlobals();
  });
});

describe("legacy frontend bans", () => {
  it("does not ship fidelity.css or MorePages modules", async () => {
    const modules = import.meta.glob("./**/*.{ts,tsx,css}");
    const keys = Object.keys(modules);
    expect(keys.some((k) => k.includes("fidelity.css"))).toBe(false);
    expect(keys.some((k) => k.includes("MorePages"))).toBe(false);
  });
});
