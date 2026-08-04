/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_DASHBOARD_DATA_MODE?: "api" | "demo";
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
