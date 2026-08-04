import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { execSync } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

function buildInfoPlugin() {
  return {
    name: "quantsilico-build-info",
    closeBundle() {
      let branch = "unknown";
      let commit = "unknown";
      try {
        branch = execSync("git branch --show-current", { encoding: "utf8" }).trim();
        commit = execSync("git rev-parse HEAD", { encoding: "utf8" }).trim();
      } catch {
        /* ignore */
      }
      const outDir = resolve(__dirname, "dist");
      mkdirSync(outDir, { recursive: true });
      writeFileSync(
        resolve(outDir, "build-info.json"),
        JSON.stringify(
          {
            schema_version: 1,
            kind: "FRONTEND_BUILD_INFO",
            branch,
            commit,
            built_at: new Date().toISOString(),
          },
          null,
          2,
        ) + "\n",
        "utf8",
      );
    },
  };
}

export default defineConfig({
  plugins: [react(), buildInfoPlugin()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8765",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  test: {
    environment: "jsdom",
  },
});
