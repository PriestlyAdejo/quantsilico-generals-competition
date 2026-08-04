# Exact Figma frontend port — deviations

Recovery marker: local tag `dashboard-before-exact-figma-port-1d3d836` @ `1d3d836`.

## Preserved exactly

- Reachable Figma application graph from `src/main.tsx` (84 files; UI closure: command, dialog, select, sheet, slider, tabs, utils).
- Golden `fonts.css` (Montserrat / Raleway / JetBrains Mono via Google Fonts) unchanged for parity.
- Page markup, shell geometry, Tailwind v4 theme for normal loaded DEMO mode.
- Production remains React 19 (golden reference may use React 18 locally only).

## Intentional production deltas

| Area | Deviation | Why |
| --- | --- | --- |
| Data source | `ApiDataSource` default; `MockDataSource` only if `VITE_DASHBOARD_DATA_MODE=demo` (dynamic import) | Real repository evidence; no silent mock in production |
| Arena / Training mutations | Capability-disabled + toast catch | No unsafe demo mutation path on API mode |
| Submission actions | Demo package actions throw `CapabilityDisabledError` | Uploads remain manual by design |
| Overview WDL / rates | Empty / zero when API lacks competitive WDL | Do not invent metrics |
| Replay boards | Metadata stub when private replay JSON has no frames | Evidence files are outcome summaries, not frame dumps |
| Docs section bodies | Stub text pointing at repo docs | `/api/documentation` is index-only |
| TypeScript | `noUnusedLocals`/`noUnusedParameters` off | Preserve exact golden sources without mass unused-import edits |
| Strict Mode | Wrapped in `StrictMode` in `main.tsx` | Production React 19 requirement |
| Build identity | `dist/build-info.json` via Vite plugin | SHA verification |

## Old frontend removal

Verified absent under `dashboard/frontend/src`:

- `fidelity.css`
- `MorePages`
- sparse JSON-primary page modules

`JSON.stringify` remains only in ReplayLab raw frame panel and RawRecordDrawer (intentional inspector affordances).

## Deferred (separate tasks)

- Font localisation / self-hosting after visual approval
- Deterministic Playwright golden-vs-production screenshot harness (Chromium, fixed viewport, fonts ready, animations off)
- Wiring Arena to allowlisted `/api/jobs/match` without inventing board frames
