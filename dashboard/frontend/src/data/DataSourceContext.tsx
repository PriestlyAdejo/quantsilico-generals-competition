import { createContext, useContext } from "react";
import type { DataSource } from "./DataSource";
import { ApiDataSource } from "./ApiDataSource";

const mode = (import.meta.env.VITE_DASHBOARD_DATA_MODE as string | undefined) || "api";

function createDefaultSource(): DataSource {
  // Demo mode is explicit opt-in only. Never fall back silently.
  if (mode === "demo") {
    // Lazy: demo is not the production path; ApiDataSource remains default.
    // A dedicated DemoDataSource can be added later without changing call sites.
    console.warn("VITE_DASHBOARD_DATA_MODE=demo requested; production still uses ApiDataSource until DemoDataSource is registered.");
  }
  return new ApiDataSource();
}

export const dataSource = createDefaultSource();
export const DataSourceContext = createContext<DataSource>(dataSource);

export function useDataSource(): DataSource {
  return useContext(DataSourceContext);
}
