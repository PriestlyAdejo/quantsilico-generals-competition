import type { DataSource } from "./DataSource";
import type { CapabilitiesResponse, JobRecord, OverviewResponse } from "./types";
import { ApiError } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.headers || {}),
      },
    });
  } catch {
    throw new ApiError("backend_unavailable", "BACKEND UNAVAILABLE");
  }

  if (res.status === 404) {
    throw new ApiError("not_found", "Record not found", 404);
  }
  if (!res.ok) {
    const detail = await res.text();
    throw new ApiError("http", detail || `HTTP ${res.status}`, res.status);
  }

  const data = (await res.json()) as T & { schema_version?: number };
  if (data && typeof data === "object" && "schema_version" in data) {
    const v = Number((data as { schema_version?: number }).schema_version);
    if (!Number.isFinite(v) || v < 1) {
      throw new ApiError("schema_mismatch", "Unsupported or missing schema_version");
    }
  }
  return data;
}

export class ApiDataSource implements DataSource {
  getCapabilities(signal?: AbortSignal) {
    return request<CapabilitiesResponse>("/api/capabilities", { signal });
  }

  getOverview(signal?: AbortSignal) {
    return request<OverviewResponse>("/api/overview", { signal });
  }

  getJson<T = Record<string, unknown>>(path: string, signal?: AbortSignal) {
    return request<T>(path, { signal });
  }

  launchMatch(
    body: {
      candidate: string;
      opponent: string;
      seed: number;
      max_turns?: number;
      record_replay?: boolean;
    },
    signal?: AbortSignal,
  ) {
    return request<JobRecord>("/api/jobs/match", {
      method: "POST",
      signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_type: "MATCH", ...body }),
    });
  }

  getJob(jobId: string, signal?: AbortSignal) {
    return request<JobRecord>(`/api/jobs/${encodeURIComponent(jobId)}`, { signal });
  }
}
