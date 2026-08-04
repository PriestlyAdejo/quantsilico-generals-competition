import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { ApiDataSource } from "./data/ApiDataSource";
import { GeneralsBoard } from "./components/board/GeneralsBoard";
import { StatusBadge } from "./components/status/StatusBadge";

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
    const { rerender, container } = render(
      <GeneralsBoard frame={{ ...base, mapKey: "map-a" }} />,
    );
    const first = container.querySelector("svg");
    rerender(
      <GeneralsBoard
        frame={{ ...base, mapKey: "map-b", typeGrid: [[0, 0], [0, 1]] }}
      />,
    );
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
});
