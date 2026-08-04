import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { DataSource } from "../services/dataSource";
import { ApiDataSource } from "../services/apiDataSource";
import LoadingState from "../components/feedback/LoadingState";
import ErrorState from "../components/feedback/ErrorState";

const DataSourceContext = createContext<DataSource | null>(null);

function resolveMode(): "api" | "demo" {
  const raw = import.meta.env.VITE_DASHBOARD_DATA_MODE;
  return raw === "demo" ? "demo" : "api";
}

export function DataSourceProvider({ children }: { children: React.ReactNode }) {
  const mode = resolveMode();
  const [ds, setDs] = useState<DataSource | null>(() => (mode === "api" ? new ApiDataSource() : null));
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (mode !== "demo") return;
    let cancelled = false;
    (async () => {
      try {
        const { MockDataSource } = await import("../services/mockDataSource");
        if (!cancelled) setDs(new MockDataSource());
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load demo data source");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [mode]);

  const value = useMemo(() => ds, [ds]);

  if (error) return <ErrorState error={error} />;
  if (!value) return <LoadingState />;

  return <DataSourceContext.Provider value={value}>{children}</DataSourceContext.Provider>;
}

export function useDataSource(): DataSource {
  const ctx = useContext(DataSourceContext);
  if (!ctx) throw new Error("useDataSource must be used within DataSourceProvider");
  return ctx;
}
