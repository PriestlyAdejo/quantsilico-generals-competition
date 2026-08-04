import type { CapabilitiesResponse, OverviewResponse, JobRecord } from "./types";

export interface DataSource {
  getCapabilities(signal?: AbortSignal): Promise<CapabilitiesResponse>;
  getOverview(signal?: AbortSignal): Promise<OverviewResponse>;
  getJson<T = Record<string, unknown>>(path: string, signal?: AbortSignal): Promise<T>;
  launchMatch(
    body: {
      candidate: string;
      opponent: string;
      seed: number;
      max_turns?: number;
      record_replay?: boolean;
    },
    signal?: AbortSignal,
  ): Promise<JobRecord>;
  getJob(jobId: string, signal?: AbortSignal): Promise<JobRecord>;
}
