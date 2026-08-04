import { describe, expect, it, vi } from "vitest";
import { ApiDataSource } from "./services/apiDataSource";
import { CapabilityDisabledError } from "./services/apiErrors";
import type { DataSource } from "./services/dataSource";

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

  it("maps overview without inventing WDL history", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({
          schema_version: 1,
          branch: "fix/exact-figma-frontend-port",
          commit: "abc123",
          engine_commit: "9e3b9d13",
          research_phase: "console port",
          heuristic_baseline: "heuristic_v2f_plus_planner_terminal_form",
          learned_champion: null,
          active_submitted_package: { candidate: "heuristic_v2f_plus_planner_terminal_form" },
          metrics: {
            submitted_candidate: "heuristic_v2f_plus_planner_terminal_form",
            learned_promotion: "NONE",
          },
          learning_smoke: { readiness: "PASS" },
          active_jobs: [],
        }),
      })),
    );

    const ds = new ApiDataSource();
    const overview = await ds.getOverview();
    expect(overview.kind).toBe("IMPORTED_PROJECT_EVIDENCE");
    expect(overview.currentCandidate).toContain("heuristic_v2f");
    expect(overview.wdlHistory).toEqual([]);
    expect(overview.currentResult).toEqual({ wins: 0, draws: 0, losses: 0 });
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
