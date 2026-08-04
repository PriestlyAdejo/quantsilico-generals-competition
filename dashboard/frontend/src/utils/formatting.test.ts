import { describe, expect, it } from "vitest";
import { fmtDate, fmtDateTime, fmtTime } from "../utils/formatting";
import { shortDisplayName } from "../utils/displayNames";

describe("fmtDate / fmtTime", () => {
  it("formats valid ISO timestamps", () => {
    const iso = "2026-08-04T12:34:00.000Z";
    expect(fmtDate(iso)).not.toBe("NOT RECORDED");
    expect(fmtTime(iso)).not.toBe("NOT RECORDED");
    expect(fmtDateTime(iso)).toContain(fmtDate(iso));
  });

  it("never returns Invalid Date", () => {
    expect(fmtDate("not-a-date")).toBe("NOT RECORDED");
    expect(fmtTime("")).toBe("NOT RECORDED");
    expect(fmtDateTime("NOT RECORDED")).toBe("NOT RECORDED");
    expect(fmtDate(null)).toBe("NOT RECORDED");
  });
});

describe("shortDisplayName", () => {
  it("preserves short ids", () => {
    expect(shortDisplayName("abc")).toBe("abc");
  });

  it("truncates long ids with ellipsis", () => {
    const long = "heuristic_v2f_plus_planner_terminal_fix";
    const short = shortDisplayName(long, 20);
    expect(short.length).toBeLessThanOrEqual(20);
    expect(short).toContain("…");
    expect(shortDisplayName(null)).toBe("NOT RECORDED");
  });
});
