import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { ApiDataSource } from "./data/ApiDataSource";
import { GeneralsBoard } from "./components/board/GeneralsBoard";
import { StatusBadge } from "./components/status/StatusBadge";
import { DataSourceContext } from "./data/DataSourceContext";
import OverviewPage from "./pages/OverviewPage";

function mockOverview() {
  return {
    schema_version: 1,
    branch: "feature/full-research-platform-v0",
    commit: "b030abca83620e7b0ca508bfbb33ba17a2795ae8",
    dirty: false,
    engine_commit: "9e3b9d13",
    research_phase: "test",
    active_submitted_package: {
      candidate: "heuristic_v2f_plus_planner_terminal_fix",
      package_path: "x.zip",
      package_sha256: "abc",
      authoritative_policy_source_commit: "027ff5d",
      embedded_bot_commit: "ee06778",
      embedded_metadata_status: "STALE",
      repository_completion_commit: "26954e6",
      metadata_note: "note",
    },
    heuristic_baseline: "heuristic_v2f_plus_planner_terminal_fix",
    learned_champion: null,
    learned_champion_note: "NO LEARNED CHAMPION",
    gate_status: {
      schema_version: 1,
      kind: "GATE_STATUS_SPLIT",
      current: {
        learning_readiness: "PASS",
        heuristic_development: "FAIL",
        pre_ppo_submission: "PASS",
        portal_submission: "PASS",
        learned_promotion: "NONE",
      },
      historical_observations: [
        {
          source: "UPLOAD_RECORD",
          learning_readiness: "PENDING_AT_RECORD_TIME",
          observed_at: "2026-08-04",
        },
      ],
    },
    gate_board: {
      HEURISTIC_DEVELOPMENT_GATE: "FAIL",
      PRE_PPO_SUBMISSION_GATE: "PASS",
      PORTAL_SUBMISSION_GATE: "PASS",
      LEARNING_READINESS_GATE: "PASS",
      LEARNED_PROMOTION_GATE: "NONE",
    },
    metrics: {
      learning_readiness: "PASS",
      cnn_latency: "PASS",
      graph_latency: "PASS",
    },
    active_jobs: [],
  };
}

describe("console smoke", () => {
  it("StatusBadge maps FAIL", () => {
    render(<StatusBadge value="FAIL" />);
    expect(screen.getByText("FAIL").className).toContain("fail");
  });

  it("GeneralsBoard remounts terrain when mapKey changes at same size", () => {
    const base = {
      height: 2,
      width: 2,
      typeGrid: [
        [1, 1],
        [1, 0],
      ],
    };
    const { rerender, container } = render(<GeneralsBoard frame={{ ...base, mapKey: "map-a" }} />);
    const first = container.querySelector("svg");
    rerender(<GeneralsBoard frame={{ ...base, mapKey: "map-b", typeGrid: [[0, 0], [0, 1]] }} />);
    const second = container.querySelector("svg");
    expect(first).not.toBe(second);
    expect(screen.getByText(/map map-b/)).toBeTruthy();
  });

  it("ApiDataSource surfaces backend unavailable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.reject(new TypeError("network"))),
    );
    const ds = new ApiDataSource();
    await expect(ds.getOverview()).rejects.toMatchObject({ kind: "backend_unavailable" });
    vi.unstubAllGlobals();
  });

  it("Overview uses current readiness PASS and is not primary JSON", async () => {
    const overview = mockOverview();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (String(url).includes("/api/overview")) {
          return {
            ok: true,
            status: 200,
            json: async () => overview,
          };
        }
        return { ok: false, status: 404, json: async () => ({ detail: "missing" }) };
      }),
    );
    render(
      <MemoryRouter>
        <DataSourceContext.Provider value={new ApiDataSource()}>
          <OverviewPage />
        </DataSourceContext.Provider>
      </MemoryRouter>,
    );
    expect(await screen.findByText("Overview")).toBeTruthy();
    expect(screen.getAllByText("PASS").length).toBeGreaterThan(0);
    expect(screen.queryByText(/"schema_version": 1/)).toBeNull();
    expect(screen.getAllByText(/PENDING_AT_RECORD_TIME/).length).toBeGreaterThan(0);
    vi.unstubAllGlobals();
  });
});
